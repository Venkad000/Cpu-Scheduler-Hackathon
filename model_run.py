from model import CpuSchedulingEnv
from prophet import Prophet
import pandas as pd
import gym
from stable_baselines3 import PPO
from gym import spaces
import numpy as np
import jenkins
from datetime import datetime
import time
import requests
from requests.auth import HTTPBasicAuth
import xml.etree.ElementTree as ET

jenkins_url = "http://localhost:8080"
username = "admin"
api_token = "redacted"

numberOfTasks = 0

def get_number_of_jobs():
    num = 0
    response = requests.get(
        f"{jenkins_url}/api/json",
        auth=HTTPBasicAuth(username, api_token)
    )
    if response.status_code == 200:
        jobs = response.json().get("jobs", [])
        for job in jobs:
            if job['name'][:10] == 'auto-task-':
                # numberOfTasks += 1
                num += 1
            # print(f"Job: {job['name']} - URL: {job['url']}")
    else:
        print("Failed to get jobs", response.status_code, response.text)
    return num

numberOfTasks = get_number_of_jobs()
print("number of teasks : ", numberOfTasks)

# Prepare data
df = pd.read_csv('cpu_mem_usage.csv') 
df['timestamp'] = pd.to_datetime(df['Time'])
df = df.rename(columns={'Time': 'ds', 'CPU (%)': 'y'})

# Train model
model = Prophet()
model.fit(df)

# Predict next 24 hours
future = model.make_future_dataframe(periods=24, freq='H')
forecast = model.predict(future)

# Extract predicted values
predicted_cpu = forecast[['ds', 'yhat']].tail(24)
predicted_values = predicted_cpu['yhat'].values

# Suppose you have a raw CPU usage percentage array (24 hours)

# raw_cpu_usage = np.array([45, 30, 20, 25, 35, 60, 70, 80, 90, 85, 60, 50, 40, 30, 25, 20, 15, 10, 5, 10, 15, 20, 25, 30])
raw_cpu_usage = predicted_values
# Normalize it between 0 and 1
normalized_cpu_usage = raw_cpu_usage / 100.0

# Create a list of such predictions if you have multiple days/samples
cpu_predictions_list = [normalized_cpu_usage]  # you can add more arrays

# Instantiate environment with your real data
env = CpuSchedulingEnv(cpu_predictions_list=cpu_predictions_list, max_tasks=numberOfTasks+1)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

model.save("cpu_scheduling_ppo_model")

def evaluate_model(model, env, episodes=5):
    all_schedules = []
    all_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        schedule = []
        total_reward = 0

        while not done:
          action, _ = model.predict(obs)
          obs, reward, done, _ = env.step(action)
          if not done:
              schedule.append(action)
          total_reward += reward


        all_schedules.append(schedule)
        all_rewards.append(total_reward)

    return all_schedules, all_rewards

schedules, rewards = evaluate_model(model, env, episodes=1000)

print(rewards.index(max(rewards)))
print(schedules[rewards.index(max(rewards))])

print(sorted(schedules[rewards.index(max(rewards))]))
timestamps = sorted(schedules[rewards.index(max(rewards))])
print(timestamps)

jenkins_jobs = []
auth = HTTPBasicAuth(username, api_token)

def list_jobs():
    i = 0
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
                print(i)
                spec.text = f"H {timestamps[i]} * * *" 

                new_config_xml = ET.tostring(xml_root, encoding="unicode")

                update_url = config_url
                headers = {"Content-Type": "application/xml"}

                update_response = requests.post(update_url, data=new_config_xml, headers=headers, auth=auth)

                if update_response.status_code == 200:
                    print(f"Job schedule updated successfully.")
                else:
                    print(f"Failed to update job: {update_response.status_code} {update_response.text}")
                i += 1
    else:
        print("Failed to get jobs", response.status_code, response.text)

list_jobs()
