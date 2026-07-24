import requests
import time
import sys

BASE_URL = "http://localhost:8000/api"

print("Uploading CSV...")
with open("../test_data/mixed_categories_reviews.csv", "rb") as f:
    resp = requests.post(
        f"{BASE_URL}/upload",
        files={"file": ("mixed_categories_reviews.csv", f, "text/csv")},
        data={"text_col": "text", "category_col": "category", "date_col": "date"}
    )
    
if not resp.ok:
    print(f"Failed to upload: {resp.text}")
    sys.exit(1)

data = resp.json()
if not data.get("success"):
    print(f"Upload API failed: {data}")
    sys.exit(1)

batch_id = data["data"]["batch_id"]
print(f"Uploaded successfully. Batch ID: {batch_id}")

print("Polling batch status...")
for _ in range(360):
    try:
        r = requests.get(f"{BASE_URL}/batches/{batch_id}/status")
        status_data = r.json()
        status = status_data['data']['status']
        print(f"Status: {status} ({status_data['data'].get('processed_count', 0)} / {status_data['data'].get('total_reviews', 0)})")
        if status in ('done', 'failed'):
            break
    except Exception as e:
        print(f"Could not check status: {e}")
    time.sleep(10)

print("\nFetching Categories Summary...")
r = requests.get(f"{BASE_URL}/categories/summary?batch_id={batch_id}")
print(r.json())

print("\nFetching Issues Distribution...")
r = requests.get(f"{BASE_URL}/issues/distribution?batch_id={batch_id}")
print(r.json())

print("\nFetching Reviews (Page 1)...")
r = requests.get(f"{BASE_URL}/reviews?batch_id={batch_id}&limit=2")
print(r.json())
