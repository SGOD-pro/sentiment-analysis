"""
Full-Power Peak Multi-Threaded Stress Test for AWS Lambda Embedding Engine.

Conducts an exhaustive stress and peak throughput benchmark purely against the
AWS Lambda function ('bge-text-embeder'), with zero local ML computation on this host machine.

Stages:
1. Concurrency Scaling Ramp (identifies throughput & latency scaling per concurrency tier).
2. High-Volume Sustained Blast across 1,200+ real customer reviews at peak concurrency.
3. Sub-millisecond latency distribution analytics (Min, Max, Mean, P50, P90, P95, P99).
4. High-capacity botocore connection pooling (max_pool_connections=64, adaptive retries, jittered backoff).
5. Mathematical integrity & vector verification under maximum concurrent stress.
"""

import concurrent.futures
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT_DIR, "test_data", "mixed_categories_reviews.csv")

# Benchmark Configuration
TARGET_COUNT = int(os.environ.get("TARGET_COUNT", "1200"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "20"))
# AWS account concurrency quota is 10. Defaulting to 8 stays safely within quota while maximizing concurrency.
PEAK_CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
RUN_RAMP = os.environ.get("RUN_RAMP", "true").lower() in ("true", "1", "yes")

# AWS Lambda Configuration
AWS_PROFILE = os.environ.get("AWS_PROFILE", "aws")
AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
RAW_ENDPOINT_URL = os.environ.get("AWS_ENDPOINT_URL", "")
ENDPOINT_URL = (
    None
    if RAW_ENDPOINT_URL.strip().lower() in ("", "none", "aws", "false")
    else RAW_ENDPOINT_URL
)

LAMBDA_FUNCTION_NAME = os.environ.get("LAMBDA_FUNCTION_NAME", "bge-text-embeder")

# High-performance client configuration
BOTO_CONFIG = Config(
    max_pool_connections=64,
    retries={"max_attempts": 6, "mode": "adaptive"},
    connect_timeout=15,
    read_timeout=90,
)


def get_lambda_client() -> boto3.client:
    """Create a high-concurrency optimized boto3 Lambda client."""
    session_kwargs = {"region_name": AWS_REGION}
    if AWS_PROFILE:
        try:
            session = boto3.Session(profile_name=AWS_PROFILE, **session_kwargs)
        except Exception:
            session = boto3.Session(**session_kwargs)
    else:
        session = boto3.Session(**session_kwargs)

    client_kwargs = {"config": BOTO_CONFIG}
    if ENDPOINT_URL:
        client_kwargs["endpoint_url"] = ENDPOINT_URL
        if "AWS_ACCESS_KEY_ID" not in os.environ and not AWS_PROFILE:
            client_kwargs["aws_access_key_id"] = "test"
        if "AWS_SECRET_ACCESS_KEY" not in os.environ and not AWS_PROFILE:
            client_kwargs["aws_secret_access_key"] = "test"

    return session.client("lambda", **client_kwargs)


def invoke_lambda_embed(texts: List[str], client: Optional[boto3.client] = None) -> np.ndarray:
    """
    Invoke the AWS Lambda function to generate normalized 384-dim embeddings.
    Includes exponential backoff with jitter to handle AWS account concurrency limits gracefully.
    """
    if client is None:
        client = get_lambda_client()

    payload_bytes = json.dumps({"texts": texts}).encode("utf-8")

    max_attempts = 6
    resp = None
    for attempt in range(max_attempts):
        try:
            resp = client.invoke(
                FunctionName=LAMBDA_FUNCTION_NAME,
                InvocationType="RequestResponse",
                Payload=payload_bytes,
            )
            break
        except ClientError as err:
            err_code = err.response.get("Error", {}).get("Code", "")
            if err_code in ("TooManyRequestsException", "429", "ThrottlingException", "EC2ThrottledException"):
                if attempt == max_attempts - 1:
                    raise
                backoff = (0.5 * (2 ** attempt)) + (random.random() * 0.5)
                time.sleep(backoff)
            else:
                raise
        except Exception as err:
            if "TooManyRequestsException" in str(err) or "Rate Exceeded" in str(err):
                if attempt == max_attempts - 1:
                    raise
                backoff = (0.5 * (2 ** attempt)) + (random.random() * 0.5)
                time.sleep(backoff)
            else:
                raise

    if resp is None:
        raise RuntimeError("Failed to obtain response from AWS Lambda after retries.")

    raw_payload = resp["Payload"].read()
    try:
        payload_data = json.loads(raw_payload)
    except Exception as err:
        raise RuntimeError(f"Failed to parse Lambda response payload: {raw_payload[:250]}") from err

    if "body" in payload_data:
        body = payload_data["body"]
        if isinstance(body, str):
            body = json.loads(body)
        embeddings = body.get("embeddings")
    elif "embeddings" in payload_data:
        embeddings = payload_data["embeddings"]
    elif "errorMessage" in payload_data:
        raise RuntimeError(f"Lambda execution error: {payload_data['errorMessage']}")
    else:
        raise RuntimeError(f"Unexpected Lambda response structure: {list(payload_data.keys())}")

    return np.array(embeddings, dtype=np.float32)


def get_process_memory_mb() -> float:
    """Measure this script's host RAM RSS usage in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 2)
    except Exception:
        pass
    return 0.0


def load_dataset(count: int = TARGET_COUNT) -> List[str]:
    """Load verified non-empty reviews from CSV."""
    print(f"Loading {count} reviews from {os.path.relpath(CSV_PATH, ROOT_DIR)}...")
    texts = []
    with open(CSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("text") or "").strip()
            if len(t) >= 15:
                texts.append(t)
                if len(texts) >= count:
                    break

    if len(texts) < count:
        raise ValueError(f"Expected at least {count} reviews, found {len(texts)}")

    lengths = [len(t) for t in texts]
    print(f"✓ Loaded {len(texts)} reviews successfully!")
    print(f"  • Min text length : {min(lengths)} characters")
    print(f"  • Max text length : {max(lengths)} characters")
    print(f"  • Avg text length : {sum(lengths)/len(lengths):.1f} characters")
    return texts


def chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def worker_task(args: Tuple[List[str], int, boto3.client]) -> Tuple[int, np.ndarray, float]:
    """Worker task executing batch on Lambda with elapsed duration."""
    batch, batch_idx, client = args
    t0 = time.perf_counter()
    arr = invoke_lambda_embed(batch, client=client)
    duration = time.perf_counter() - t0
    return batch_idx, arr, duration


def compute_percentiles(latencies: List[float]) -> Dict[str, float]:
    """Compute statistical percentiles from latency distribution."""
    if not latencies:
        return {}
    s = sorted(latencies)
    n = len(s)

    def p(pct):
        k = (n - 1) * (pct / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] * (c - k) + s[c] * (k - f)

    return {
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
        "p50": p(50),
        "p90": p(90),
        "p95": p(95),
        "p99": p(99),
    }


def run_concurrency_ramp_benchmark(sample_texts: List[str], concurrencies: List[int], client: boto3.client) -> List[Dict]:
    """
    Ramp-up stage testing multiple concurrency levels to identify the
    throughput scaling curve and peak saturation ceiling.
    """
    print("\n==================================================================")
    print("🔥 STAGE 1: CONCURRENCY RAMP & PEAK SATURATION TEST")
    print("==================================================================")
    print(f"Testing concurrency levels: {concurrencies}")
    print("Evaluating throughput & latency scaling per concurrency tier...")

    ramp_results = []
    sub_batch_size = BATCH_SIZE

    for c in concurrencies:
        num_test_batches = c
        total_items = num_test_batches * sub_batch_size
        ramp_slice = (sample_texts * (math.ceil(total_items / len(sample_texts))))[:total_items]
        chunks = chunk_list(ramp_slice, sub_batch_size)

        t_start = time.perf_counter()
        latencies = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
            futures = [
                executor.submit(worker_task, (chunk, idx, client))
                for idx, chunk in enumerate(chunks)
            ]
            for f in concurrent.futures.as_completed(futures):
                try:
                    _, _, duration = f.result()
                    latencies.append(duration)
                except Exception as exc:
                    print(f"  [Concurrency {c}] Worker notice: {exc}")

        total_time = time.perf_counter() - t_start
        throughput = total_items / total_time if total_time > 0 else 0
        pcts = compute_percentiles(latencies)

        row = {
            "concurrency": c,
            "batches": num_test_batches,
            "items": total_items,
            "wall_time": total_time,
            "throughput": throughput,
            "mean_lat": pcts.get("mean", 0.0),
            "p50": pcts.get("p50", 0.0),
            "p95": pcts.get("p95", 0.0),
            "p99": pcts.get("p99", 0.0),
        }
        ramp_results.append(row)

        print(
            f"  • Workers: {c:2d} | Items: {total_items:4d} | "
            f"Wall Time: {total_time:5.2f}s | "
            f"Throughput: {throughput:6.1f} rev/s | "
            f"P50: {pcts.get('p50', 0):5.2f}s | P95: {pcts.get('p95', 0):5.2f}s"
        )

    return ramp_results


def run_sustained_peak_blast(texts: List[str], concurrency: int, client: boto3.client) -> Tuple[np.ndarray, Dict]:
    """
    Stage 2: Sustained full-blast load on all target reviews (1,200+)
    at the designated concurrency level.
    """
    chunks = chunk_list(texts, BATCH_SIZE)
    num_batches = len(chunks)

    print("\n==================================================================")
    print(f"⚡ STAGE 2: SUSTAINED MAXIMUM-LOAD PEAK BLAST ({len(texts)} REVIEWS)")
    print("==================================================================")
    print(f"Target Review Volume   : {len(texts)} reviews")
    print(f"Total Batch Invocations: {num_batches} batches ({BATCH_SIZE} reviews/batch)")
    print(f"Worker Concurrency     : {concurrency} parallel threads")
    print("Botocore Conn Pool     : 64 active connections")
    print("Firing requests...")

    t0 = time.perf_counter()
    batch_results = [None] * num_batches
    batch_latencies = [0.0] * num_batches
    errors = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_idx = {
            executor.submit(worker_task, (chunk, idx, client)): idx
            for idx, chunk in enumerate(chunks)
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                batch_idx, emb_array, duration = future.result()
                batch_results[batch_idx] = emb_array
                batch_latencies[batch_idx] = duration
                completed += 1

                elapsed = time.perf_counter() - t0
                current_rate = sum(len(chunks[i]) for i in range(completed)) / elapsed if elapsed > 0 else 0
                pct_done = (completed / num_batches) * 100

                bar_len = 30
                filled_len = int(bar_len * completed // num_batches)
                bar = "█" * filled_len + "░" * (bar_len - filled_len)

                sys.stdout.write(
                    f"\r  [{bar}] {completed}/{num_batches} ({pct_done:5.1f}%) | "
                    f"Speed: {current_rate:5.1f} rev/s | "
                    f"Last: {duration:4.2f}s | "
                    f"Elapsed: {elapsed:5.1f}s"
                )
                sys.stdout.flush()
            except Exception as exc:
                errors.append((idx, str(exc)))
                print(f"\n❌ Batch {idx} failed: {exc}")

    print()
    total_time = time.perf_counter() - t0

    if errors:
        print(f"\n⚠️ Encountered {len(errors)} failed batches during stress run!")

    valid_results = [r for r in batch_results if r is not None]
    all_embeddings = np.vstack(valid_results) if valid_results else np.empty((0, 384), dtype=np.float32)

    valid_latencies = [lat for lat in batch_latencies if lat > 0.0]
    pcts = compute_percentiles(valid_latencies)

    stats = {
        "total_reviews": len(texts),
        "total_time": total_time,
        "throughput": len(texts) / total_time if total_time > 0 else 0,
        "effective_item_latency_ms": (total_time / len(texts)) * 1000 if texts else 0,
        "success_rate": ((num_batches - len(errors)) / num_batches) * 100 if num_batches else 0,
        "errors": len(errors),
        "percentiles": pcts,
    }

    return all_embeddings, stats


def validate_embeddings(all_embeddings: np.ndarray, expected_count: int, client: boto3.client):
    """Rigorous mathematical integrity and semantic verification."""
    print("\n==================================================================")
    print("📐 STAGE 3: VECTOR MATHEMATICAL INTEGRITY & QUALITY (POST-STRESS)")
    print("==================================================================")
    # 1. Dimensions check
    assert all_embeddings.shape == (expected_count, 384), (
        f"Shape mismatch: expected ({expected_count}, 384), got {all_embeddings.shape}"
    )
    print(f"1. Matrix Dimension Validation : PASSED ({all_embeddings.shape})")

    # 2. Finite numbers
    is_finite = bool(np.all(np.isfinite(all_embeddings)))
    assert is_finite, "Corrupted float detected: NaN or Inf present in embeddings!"
    print("2. Numeric Stability Check     : PASSED (Zero NaNs, Zero Infs)")

    # 3. L2 Normalization (Unit vectors)
    norms = np.linalg.norm(all_embeddings, axis=1)
    norm_min = float(np.min(norms))
    norm_max = float(np.max(norms))
    norm_mean = float(np.mean(norms))
    print(f"3. L2 Unit Normalization       : PASSED (Min={norm_min:.5f}, Max={norm_max:.5f}, Mean={norm_mean:.5f})")
    assert np.allclose(norms, 1.0, atol=1e-3), "Embeddings are not unit normalized!"

    # 4. Semantic Discrimination
    print("4. Semantic Discrimination Test:")
    val_samples = [
        "Absolutely loved this! High quality, works great, highly recommended.",
        "Very good product, excellent quality, works like a charm!",
        "Terrible item, arrived broken and damaged, completely useless waste of money.",
    ]
    val_embeddings = invoke_lambda_embed(val_samples, client=client)
    sim_pos_pos = float(np.dot(val_embeddings[0], val_embeddings[1]))
    sim_pos_neg = float(np.dot(val_embeddings[0], val_embeddings[2]))
    print(f"    - Cosine Similarity (Positive vs Similar Positive)  : {sim_pos_pos:.4f}")
    print(f"    - Cosine Similarity (Positive vs Negative Complaint): {sim_pos_neg:.4f}")
    assert sim_pos_pos > sim_pos_neg, "Semantic discrimination failed!"
    assert sim_pos_pos > 0.65, "Expected high cosine similarity for similar reviews!"
    print("    => Vector quality and semantic discrimination verified 100%!")


def main():
    print("==================================================================")
    print("🚀 BGE TEXT EMBEDDER FULL POTENTIAL PEAK STRESS TEST")
    print("==================================================================")
    print(f"Target Lambda Function : {LAMBDA_FUNCTION_NAME}")
    print(f"Target AWS Endpoint    : {ENDPOINT_URL or 'Real AWS Cloud'}")
    print(f"Target AWS Region      : {AWS_REGION}")
    print(f"AWS Profile            : {AWS_PROFILE or '(Default / None)'}")
    print(f"Target Review Volume   : {TARGET_COUNT} reviews")
    print(f"Batch Size             : {BATCH_SIZE} reviews per invocation")
    print(f"Peak Concurrency Tier  : {PEAK_CONCURRENCY} worker threads")
    print("Execution Architecture : 100% AWS Lambda (Zero local ML compute)")
    print("==================================================================\n")

    mem_initial = get_process_memory_mb()
    print(f"Initial Script Memory RSS: {mem_initial:.2f} MB")

    client = get_lambda_client()

    # 0. Warm-up & Connectivity Check
    print("\nWarming up AWS Lambda containers...")
    t0_warm = time.perf_counter()
    try:
        warm_res = invoke_lambda_embed(["Warmup ping verifying AWS Lambda runtime readiness."], client=client)
        assert warm_res.shape == (1, 384)
        t_warm_ms = (time.perf_counter() - t0_warm) * 1000
        print(f"✓ AWS Lambda '{LAMBDA_FUNCTION_NAME}' warm & ready! ({t_warm_ms:.1f} ms, vector shape {warm_res.shape})")
    except Exception as exc:
        print(f"❌ Could not establish connection to Lambda '{LAMBDA_FUNCTION_NAME}': {exc}")
        sys.exit(1)

    # 1. Load Dataset
    texts = load_dataset(TARGET_COUNT)

    # 2. Concurrency Ramp (Tiers scaled to AWS account limits)
    ramp_stats = []
    if RUN_RAMP:
        if PEAK_CONCURRENCY >= 16:
            ramp_concurrencies = [4, 8, 12, 16]
        elif PEAK_CONCURRENCY >= 8:
            ramp_concurrencies = [1, 2, 4, 8]
        else:
            ramp_concurrencies = [1, PEAK_CONCURRENCY]
        ramp_stats = run_concurrency_ramp_benchmark(texts, ramp_concurrencies, client)

    # 3. Sustained Maximum Peak Blast
    all_embeddings, stats = run_sustained_peak_blast(texts, PEAK_CONCURRENCY, client)

    # 4. Mathematical Validation
    validate_embeddings(all_embeddings, TARGET_COUNT, client)

    # 5. Executive Peak Scorecard
    pcts = stats["percentiles"]
    mem_final = get_process_memory_mb()

    print("\n==================================================================")
    print("📊 EXECUTIVE PEAK PERFORMANCE SCORECARD")
    print("==================================================================")
    print(f"  • Total Reviews Processed : {stats['total_reviews']:,} reviews")
    print(f"  • Sustained Throughput    : {stats['throughput']:.2f} reviews / second")
    print(f"  • Effective Latency       : {stats['effective_item_latency_ms']:.2f} ms / review")
    print(f"  • Total Wall-Clock Time   : {stats['total_time']:.2f} seconds")
    print(f"  • Batch Success Rate      : {stats['success_rate']:.1f}% ({stats['errors']} failed)")
    print(f"  • Batch Latency (P50)     : {pcts.get('p50', 0):.2f} s / batch ({BATCH_SIZE} reviews)")
    print(f"  • Batch Latency (P90)     : {pcts.get('p90', 0):.2f} s / batch")
    print(f"  • Batch Latency (P95)     : {pcts.get('p95', 0):.2f} s / batch")
    print(f"  • Batch Latency (P99)     : {pcts.get('p99', 0):.2f} s / batch")
    print(f"  • Local Memory Footprint  : {mem_final:.2f} MB (Delta: +{mem_final - mem_initial:.2f} MB)")

    if ramp_stats:
        print("\n📈 CONCURRENCY SCALING SUMMARY:")
        print("  | Threads | Throughput (rev/s) | Wall Time (s) | P50 Latency (s) | P95 Latency (s) |")
        print("  |---------|-------------------|---------------|-----------------|-----------------|")
        for r in ramp_stats:
            print(
                f"  |   {r['concurrency']:2d}    |     {r['throughput']:6.1f}        |    {r['wall_time']:6.2f}     |     {r['p50']:6.2f}      |     {r['p95']:6.2f}      |"
            )
        best_tier = max(ramp_stats, key=lambda x: x["throughput"])
        print(f"\n🏆 Optimal Peak Tier: {best_tier['throughput']:.1f} reviews/sec at {best_tier['concurrency']} concurrent threads!")

    print("==================================================================")
    print("🎉 FULL-POTENTIAL PEAK STRESS TEST COMPLETED SUCCESSFULLY!")
    print("==================================================================\n")


if __name__ == "__main__":
    main()
