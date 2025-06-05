import psutil
import time
import csv

# Set the output CSV file
output_file = 'cpu_mem_usage.csv'

cpu_usage = psutil.cpu_percent(interval=1)
memory_usage = psutil.virtual_memory().percent

# Format the current time
current_time = time.strftime('%Y-%m-%d %H:%M:%S')

# Write the data to the CSV file
with open(output_file, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), cpu_usage, memory_usage])