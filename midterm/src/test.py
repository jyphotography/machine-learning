import requests
import json
test_json = {"BEDS": 4, "BATH": 2, "PROPERTYSQFT": 2184, "SUBLOCALITY": "Queens County"}
url = 'http://127.0.0.1:5000/predict'
response = requests.post(url, json=test_json)
result = response.json()

print(json.dumps(result, indent=2))