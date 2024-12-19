from time import sleep
import requests


url = "http://localhost:9696/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client).json()

print(response)

while True:
    sleep(0.1)
    response = requests.post(url, json=client).json()
    print(response)