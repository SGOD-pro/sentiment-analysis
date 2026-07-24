import requests
import json
import sys

BASE_URL = "http://localhost:8000/api"
batch_id = "18d46434-aa5a-4033-9d73-f847ab1688e2"

print("\nFetching Categories Summary...")
r = requests.get(f"{BASE_URL}/categories/summary?batch_id={batch_id}")
print(json.dumps(r.json(), indent=2))

print("\nFetching Issues Distribution...")
r = requests.get(f"{BASE_URL}/issues/distribution?batch_id={batch_id}")
print(json.dumps(r.json(), indent=2))

print("\nFetching Reviews (Page 1)...")
r = requests.get(f"{BASE_URL}/reviews?batch_id={batch_id}&limit=2")
print(json.dumps(r.json(), indent=2))
