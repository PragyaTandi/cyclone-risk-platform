import requests
import json

with open("../risk_prediction.json", "r") as file:
    risk_data = json.load(file)

response = requests.post(
    "http://127.0.0.1:5000/advisory",
    json=risk_data
)

print("Status code:", response.status_code)
print("Response:")
print(response.json())