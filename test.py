import requests
from requests.auth import HTTPBasicAuth
import xml.etree.ElementTree as ET

jenkins_url = "http://localhost:8080"
username = "admin"
api_token = "11e5e64070b538a20ab25a210086a35d8d"

jenkins_jobs = []
auth = HTTPBasicAuth(username, api_token)
def list_jobs():
    response = requests.get(
        f"{jenkins_url}/api/json",
        auth=auth
    )
    if response.status_code == 200:
        jobs = response.json().get("jobs", [])
        for job in jobs:
            print(f"Job: {job['name']} - URL: {job['url']}")
            if job['name'][:10] == 'auto-task-':
                print("Found!")
                config_url = job['url'] + "config.xml"
                response = requests.get(config_url, auth=auth)

                if response.status_code != 200:
                    print(f"Failed to fetch config: {response.status_code} {response.text}")
                    exit()

                xml_root = ET.fromstring(response.text)

                triggers = xml_root.find("triggers")

                if triggers is None:
                    triggers = ET.SubElement(xml_root, "triggers", attrib={"class": "vector"})

                for child in list(triggers):
                    if "TimerTrigger" in child.tag:
                        triggers.remove(child)

                new_trigger = ET.SubElement(triggers, "hudson.triggers.TimerTrigger")
                spec = ET.SubElement(new_trigger, "spec")
                spec.text = "H 10 * * *" 

                new_config_xml = ET.tostring(xml_root, encoding="unicode")

                update_url = config_url
                headers = {"Content-Type": "application/xml"}

                update_response = requests.post(update_url, data=new_config_xml, headers=headers, auth=auth)

                if update_response.status_code == 200:
                    print(f"Job schedule updated successfully.")
                else:
                    print(f"Failed to update job: {update_response.status_code} {update_response.text}")

    else:
        print("Failed to get jobs", response.status_code, response.text)

list_jobs()
