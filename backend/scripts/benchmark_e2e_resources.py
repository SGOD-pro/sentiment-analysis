"""
End-to-end benchmark and system resource consumption monitor (pure stdlib).

Measures:
- CPU utilization across backend and Lambda
- Memory consumption (RSS, limit percentage)
- Inference latency (total, per review, cold vs warm)
- Package & artifact storage footprint
"""

import json
import os
import subprocess
import time
import requests

BASE_URL = "http://localhost:8000/api"
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test-data.csv"))
BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".aws-sam", "build"))

def get_docker_stats():
    try:
        out = subprocess.check_output(["docker", "stats", "--no-stream", "--format", "{{json .}}"]).decode()
        containers = []
        for line in out.strip().split("\n"):
            if line:
                containers.append(json.loads(line))
        return containers
    except Exception as e:
        return [{"error": str(e)}]

def get_proc_memory(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            lines = f.readlines()
        res = {}
        for line in lines:
            if line.startswith("VmRSS:"):
                res["rss_kb"] = int(line.split()[1])
            elif line.startswith("VmSize:"):
                res["vms_kb"] = int(line.split()[1])
            elif line.startswith("Threads:"):
                res["threads"] = int(line.split()[1])
        res["rss_mb"] = round(res.get("rss_kb", 0) / 1024, 2)
        res["vms_mb"] = round(res.get("vms_kb", 0) / 1024, 2)
        return res
    except Exception as e:
        return {"error": str(e)}

def get_system_memory():
    with open("/proc/meminfo") as f:
        lines = f.readlines()
    info = {}
    for line in lines:
        parts = line.split(":")
        if len(parts) == 2:
            key = parts[0].strip()
            val = parts[1].strip().split()[0]
            if val.isdigit():
                info[key] = int(val)
    total_gb = round(info.get("MemTotal", 0) / (1024 * 1024), 2)
    avail_gb = round(info.get("MemAvailable", 0) / (1024 * 1024), 2)
    return total_gb, avail_gb

def find_uvicorn_pid():
    try:
        out = subprocess.check_output(["pgrep", "-f", "uvicorn src.main:app"]).decode().strip()
        pids = [int(p) for p in out.split() if p.isdigit()]
        return pids[-1] if pids else None
    except Exception:
        return None

def get_dir_size_mb(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return round(total / (1024 * 1024), 2)

def run_benchmark():
    print("==================================================")
    print("📊 SENTRIX SYSTEM RESOURCE CONSUMPTION & E2E BENCHMARK")
    print("==================================================")

    # 1. Static Artifact & Package Sizes
    print("\n[1] Deployment Package & Artifact Sizes:")
    backend_pkg_size = get_dir_size_mb(os.path.join(BUILD_DIR, "BackendFunction"))
    lambda_pkg_size = get_dir_size_mb(os.path.join(BUILD_DIR, "BgeTextEmbedderFunction"))
    onnx_model_size = get_dir_size_mb(os.path.join(BUILD_DIR, "BgeTextEmbedderFunction", "artifacts", "bge_onnx_quantized"))
    mlp_weights_size = os.path.getsize(os.path.join(BUILD_DIR, "BackendFunction", "artifacts", "mlp_weights.npz")) / (1024 * 1024)
    centroids_size = os.path.getsize(os.path.join(BUILD_DIR, "BackendFunction", "artifacts", "issue_centroids.npy")) / 1024

    print(f"  • BackendFunction SAM Package     : {backend_pkg_size:.2f} MB")
    print(f"  • BgeTextEmbedderFunction Package : {lambda_pkg_size:.2f} MB (Budget: 250MB)")
    print(f"  • Quantized BGE ONNX Model        : {onnx_model_size:.2f} MB")
    print(f"  • Pure Numpy MLP Classifier       : {mlp_weights_size:.2f} MB")
    print(f"  • KMeans Issue Centroids Matrix   : {centroids_size:.2f} KB")

    # 2. Baseline Resource Usage
    uvicorn_pid = find_uvicorn_pid()
    total_mem_gb, avail_mem_gb = get_system_memory()
    print(f"\n[2] Pre-Inference Baseline Resources (Uvicorn PID: {uvicorn_pid}):")
    baseline_backend = get_proc_memory(uvicorn_pid) if uvicorn_pid else {}
    print(f"  • Backend Memory RSS : {baseline_backend.get('rss_mb', 'N/A')} MB")
    print(f"  • System Total RAM   : {total_mem_gb} GB")
    print(f"  • Available RAM      : {avail_mem_gb} GB")

    docker_baseline = get_docker_stats()
    for c in docker_baseline:
        print(f"  • Docker Container [{c.get('Name')}]: Mem={c.get('MemUsage')}, CPU={c.get('CPUPerc')}")

    # 3. Execution of Batch Processing (140 Reviews)
    print("\n[3] Triggering Batch Processing via /api/upload...")
    t_upload_start = time.perf_counter()
    with open(CSV_PATH, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/upload",
            files={"file": ("test-data.csv", f, "text/csv")},
            data={"text_col": "review", "category_col": "category", "date_col": "date"}
        )
    t_upload_ack = time.perf_counter() - t_upload_start
    assert r.status_code == 200, f"Upload failed: {r.text}"
    batch_id = r.json()["data"]["batch_id"]
    print(f"  • Batch ID created       : {batch_id}")
    print(f"  • Upload & S3 Ingestion  : {t_upload_ack*1000:.2f} ms")

    # Polling & Performance Sampling
    print("\n[4] Polling Batch Inference & Monitoring Load...")
    t_poll_start = time.perf_counter()
    peak_backend_rss = baseline_backend.get("rss_mb", 0)
    final_status = None
    processed_count = 0

    while time.perf_counter() - t_poll_start < 60:
        st_res = requests.get(f"{BASE_URL}/batches/{batch_id}/status").json()
        data = st_res.get("data", {})
        status = data.get("status")
        processed_count = data.get("processed_count", 0)

        if uvicorn_pid:
            m = get_proc_memory(uvicorn_pid)
            if m.get("rss_mb", 0) > peak_backend_rss:
                peak_backend_rss = m["rss_mb"]

        if status in ("done", "failed"):
            final_status = status
            break
        time.sleep(0.2)

    t_inference_duration = time.perf_counter() - t_poll_start
    assert final_status == "done", f"Inference failed with status: {final_status}"

    processed_count = int(processed_count)
    print(f"  • Total Inference Duration : {t_inference_duration:.2f} s for {processed_count} reviews")
    print(f"  • Throughput               : {processed_count / t_inference_duration:.2f} reviews/sec")
    print(f"  • Average Latency / Review : {(t_inference_duration / processed_count)*1000:.2f} ms/review")

    # 4. Post-Inference Resource Stats
    print("\n[5] Post-Inference Resource Consumption:")
    current_backend = get_proc_memory(uvicorn_pid) if uvicorn_pid else {}
    print(f"  • Backend Peak Memory (RSS)  : {peak_backend_rss} MB")
    print(f"  • Backend Final Memory (RSS) : {current_backend.get('rss_mb', 'N/A')} MB")

    docker_post = get_docker_stats()
    for c in docker_post:
        print(f"  • Docker Container [{c.get('Name')}]: Mem={c.get('MemUsage')}, CPU={c.get('CPUPerc')}")

    # 5. Endpoint Query Latencies
    print("\n[6] API Endpoint Response Times (Read Path):")
    endpoints = [
        ("GET /api/categories/summary", f"{BASE_URL}/categories/summary?batch_id={batch_id}"),
        ("GET /api/issues/distribution", f"{BASE_URL}/issues/distribution?batch_id={batch_id}"),
        ("GET /api/trends", f"{BASE_URL}/trends?batch_id={batch_id}"),
        ("GET /api/reviews?limit=25", f"{BASE_URL}/reviews?batch_id={batch_id}&limit=25"),
        ("GET /api/batches/stats", f"{BASE_URL}/batches/stats"),
    ]
    for label, url in endpoints:
        t0 = time.perf_counter()
        res = requests.get(url)
        lat = (time.perf_counter() - t0) * 1000
        assert res.status_code == 200
        print(f"  • {label:<30} : {lat:6.2f} ms")

    print("\n==================================================")
    print("✅ BENCHMARK COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_benchmark()
