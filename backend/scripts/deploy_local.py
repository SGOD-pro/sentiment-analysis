"""
Deploy built SAM artifacts to local Floci environment.

Deploys:
1. bge-text-embeder (from .aws-sam/build/BgeTextEmbedderFunction)
2. sentimetric-backend-api (from .aws-sam/build/BackendFunction)
"""

import io
import json
import os
import zipfile
import boto3
from botocore.exceptions import ClientError

ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "test")
AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".aws-sam", "build"))

def get_lambda_client():
    return boto3.client(
        "lambda",
        endpoint_url=ENDPOINT_URL,
        region_name=REGION,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )

def make_zip(dir_path: str) -> bytes:
    """Create a zip in memory from a directory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, dir_path)
                zf.write(full_path, rel_path)
    return buf.getvalue()

def deploy_function(client, func_name: str, build_folder: str, handler: str, memory: int = 512):
    print(f"\n--- Deploying {func_name} from {build_folder} ---")
    folder_path = os.path.join(BUILD_DIR, build_folder)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Build folder not found: {folder_path}")

    print(f"Creating zip for {build_folder}...")
    zip_bytes = make_zip(folder_path)
    print(f"Zip created. Size: {len(zip_bytes) / (1024 * 1024):.2f} MB")

    try:
        client.get_function(FunctionName=func_name)
        print(f"Function {func_name} exists, updating code...")
        client.update_function_code(FunctionName=func_name, ZipFile=zip_bytes)
        client.update_function_configuration(
            FunctionName=func_name,
            Handler=handler,
            MemorySize=memory,
            Timeout=30,
        )
        print(f"Function {func_name} updated successfully.")
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ResourceNotFoundException", "404"):
            print(f"Function {func_name} does not exist, creating...")
            client.create_function(
                FunctionName=func_name,
                Runtime="python3.13",
                Role="arn:aws:iam::000000000000:role/dummy-role",
                Handler=handler,
                Code={"ZipFile": zip_bytes},
                MemorySize=memory,
                Timeout=30,
            )
            print(f"Function {func_name} created successfully.")
        else:
            raise

def cleanup_old_function(client, old_name: str = "sentimetric-ml-inference"):
    try:
        client.delete_function(FunctionName=old_name)
        print(f"Deleted deprecated function: {old_name}")
    except ClientError:
        pass

def main():
    client = get_lambda_client()
    cleanup_old_function(client, "sentimetric-ml-inference")
    
    # Deploy bge-text-embeder
    deploy_function(
        client,
        func_name="bge-text-embeder",
        build_folder="BgeTextEmbedderFunction",
        handler="handler.lambda_handler",
        memory=512,
    )

    # Deploy sentimetric-backend-api
    deploy_function(
        client,
        func_name="sentimetric-backend-api",
        build_folder="BackendFunction",
        handler="lambda_handler.handler",
        memory=256,
    )

    print("\n=== VERIFYING DEPLOYED FUNCTIONS ===")
    funcs = client.list_functions()["Functions"]
    for f in funcs:
        print(f"  - {f['FunctionName']} (Memory: {f['MemorySize']}MB, Runtime: {f['Runtime']})")

    # Test direct invocation of bge-text-embeder
    print("\nTesting invocation of bge-text-embeder...")
    payload = json.dumps({"texts": ["Excellent battery life and very fast shipping!"]})
    res = client.invoke(FunctionName="bge-text-embeder", Payload=payload.encode())
    resp_payload = json.loads(res["Payload"].read())
    print("Invocation status:", res["StatusCode"])
    if "body" in resp_payload:
        body = json.loads(resp_payload["body"]) if isinstance(resp_payload["body"], str) else resp_payload["body"]
        embeddings = body.get("embeddings", [])
        print(f"Successfully received {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}!")
    elif "embeddings" in resp_payload:
        embeddings = resp_payload["embeddings"]
        print(f"Successfully received {len(embeddings)} embeddings of dimension {len(embeddings[0]) if embeddings else 0}!")
    else:
        print("Response payload:", resp_payload)

if __name__ == "__main__":
    main()
