import urllib.request
import urllib.parse
import json
import time
import uuid

API_URL = "https://i2txad9309.execute-api.ap-south-1.amazonaws.com"
# Read csv
with open('test-data-large.csv', 'rb') as f:
    csv_data = f.read()

# Build multipart form data manually
boundary = uuid.uuid4().hex
body = (
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="file"; filename="test-data.csv"\r\n'
    f'Content-Type: text/csv\r\n\r\n'
).encode('utf-8') + csv_data + (
    f'\r\n--{boundary}\r\n'
    f'Content-Disposition: form-data; name="text_col"\r\n\r\n'
    f'review\r\n'
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="date_col"\r\n\r\n'
    f'date\r\n'
    f'--{boundary}\r\n'
    f'Content-Disposition: form-data; name="category_col"\r\n\r\n'
    f'category\r\n'
    f'--{boundary}--\r\n'
).encode('utf-8')

headers = {
    'Content-Type': f'multipart/form-data; boundary={boundary}',
    'Content-Length': str(len(body))
}

req = urllib.request.Request(f"{API_URL}/api/upload", data=body, headers=headers, method="POST")

print("Uploading...")
start_upload = time.time()
try:
    res = urllib.request.urlopen(req)
    data = json.loads(res.read())
    upload_duration = time.time() - start_upload
    print(f"Upload response received in {upload_duration:.2f} seconds:", data)
    batch_id = data["data"]["batch_id"]
except Exception as e:
    print("Upload failed:", e)
    if hasattr(e, 'read'):
        print(e.read())
    exit(1)

# Now poll
print(f"Polling batch {batch_id}...")
status = "processing"
polls = []
start_time = time.time()

while status == "processing":
    req = urllib.request.Request(f"{API_URL}/api/batches/{batch_id}/status")
    try:
        res = urllib.request.urlopen(req)
        s_data = json.loads(res.read())
        batch_data = s_data["data"]
        status = batch_data["status"]
        processed = batch_data["processed_count"]
        total = batch_data["total_reviews"]
        polls.append(f"{processed} (status: {status})")
        time.sleep(0.1)  # poll very fast to catch any intermediate
        if time.time() - start_time > 30:
            print("Timeout polling!")
            break
    except Exception as e:
        print("Poll failed:", e)
        time.sleep(0.5)

print("\nPoll sequence:")
print(", ".join(polls))
