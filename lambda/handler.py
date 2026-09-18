"""
Embedding Lambda handler.

Purpose: High-throughput 384-dimensional text embedding generator using quantized BGE ONNX.
Input: {"texts": ["review 1", "review 2", ...]}
Output: {"statusCode": 200, "body": '{"embeddings": [[... 384 floats ...], ...]}'}

Optimizations:
1. Tokenize-once pipeline: all texts are tokenized once into token IDs without redundant passes.
2. Length-aware micro-batching: reviews sorted by sequence length before batching to eliminate
   excessive [PAD] token overhead, with strictly guaranteed original order restoration.
3. Thread-contention elimination: intra-op and inter-op threads tuned for constrained container vCPUs.
4. Production instrumentation: structured CloudWatch JSON logging of padding ratio, sequence lengths,
   tokenizer, ONNX inference, pooling, and normalization latencies.
5. Cold-start pre-warming: pre-compiles execution graph and allocates memory arena during Init.
"""

import concurrent.futures
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        for key in record.__dict__:
            if (
                key not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__
                and key not in ("message", "msg")
            ):
                entry[key] = record.__dict__[key]
        if record.exc_info and record.exc_info[0]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


logger = logging.getLogger("ml_embedding")
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JSONFormatter())
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"),
)

# Configuration driven by environment variables with evidence-backed safe defaults
EMBED_MAX_LENGTH = int(os.environ.get("EMBED_MAX_LENGTH", "256"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "16"))
EMBED_MAX_WORKERS = int(os.environ.get("EMBED_MAX_WORKERS", "1"))
ORT_INTRA_OP_THREADS = int(os.environ.get("ORT_INTRA_OP_THREADS", "1"))
ORT_INTER_OP_THREADS = int(os.environ.get("ORT_INTER_OP_THREADS", "1"))

# 1. Initialize Tokenizer (unpadded for single-pass length-aware batching)
tokenizer = Tokenizer.from_file(f"{ARTIFACT_DIR}/bge_onnx_quantized/tokenizer.json")
tokenizer.no_padding()
tokenizer.enable_truncation(max_length=EMBED_MAX_LENGTH)

# 2. Optimized ONNX Runtime Session
sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = True
sess_options.enable_cpu_mem_arena = True
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = ORT_INTRA_OP_THREADS
sess_options.inter_op_num_threads = ORT_INTER_OP_THREADS

onnx_session = ort.InferenceSession(
    f"{ARTIFACT_DIR}/bge_onnx_quantized/model_quantized.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)

# 3. Optional ThreadPool for Multi-Core Micro-Batching (if EMBED_MAX_WORKERS > 1)
_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
if EMBED_MAX_WORKERS > 1:
    _executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=EMBED_MAX_WORKERS,
        thread_name_prefix="bge_worker",
    )

# 4. Cold Start Optimization: Pre-warm execution graph during Lambda Init phase
try:
    _warm_enc = tokenizer.encode("[PAD]")
    _w_ids = np.array([_warm_enc.ids], dtype=np.int64)
    _w_mask = np.array([_warm_enc.attention_mask], dtype=np.int64)
    _w_types = np.zeros_like(_w_ids)
    onnx_session.run(
        None,
        {
            "input_ids": _w_ids,
            "attention_mask": _w_mask,
            "token_type_ids": _w_types,
        },
    )
    logger.info("cold_start_warmup_complete")
except Exception as _exc:
    logger.warning("cold_start_warmup_failed", extra={"error": str(_exc)})


def tokenize_batch(texts: List[str]) -> Tuple[List[Tuple[int, List[int], int]], float]:
    """
    Tokenize all texts once into token IDs without global padding.
    Returns:
        indexed_items: List of (original_index, token_ids, token_length)
        tokenizer_ms: Time taken in milliseconds
    """
    t_tok_start = time.perf_counter()
    encodings = tokenizer.encode_batch(texts)
    tokenizer_ms = (time.perf_counter() - t_tok_start) * 1000
    indexed_items = [(i, enc.ids, len(enc.ids)) for i, enc in enumerate(encodings)]
    return indexed_items, tokenizer_ms


def build_length_aware_batches(
    indexed_items: List[Tuple[int, List[int], int]],
    batch_size: int = EMBED_BATCH_SIZE,
) -> List[List[Tuple[int, List[int], int]]]:
    """
    Sort items by token length and group into homogenous sequence-length micro-batches.
    Minimizes attention padding inflation from heterogeneous review lengths.
    """
    sorted_items = sorted(indexed_items, key=lambda x: x[2])
    return [sorted_items[i : i + batch_size] for i in range(0, len(sorted_items), batch_size)]


def run_onnx_micro_batch(
    micro_batch: List[Tuple[int, List[int], int]]
) -> Dict[str, Any]:
    """
    Execute inference on a single length-aware micro-batch:
    1. Pad array only to the maximum length of this specific micro-batch.
    2. Run ONNX model.
    3. Perform CLS pooling (matching BGE training architecture).
    4. Perform L2 unit normalization.
    """
    bsz = len(micro_batch)
    real_lengths = [x[2] for x in micro_batch]
    max_l = max(real_lengths)
    min_l = min(real_lengths)
    mean_l = float(np.mean(real_lengths))
    real_tokens = sum(real_lengths)
    padded_tokens = bsz * max_l
    padding_ratio = 1.0 - (real_tokens / padded_tokens) if padded_tokens else 0.0

    # Dynamic NumPy padding to current micro-batch max length
    input_ids = np.zeros((bsz, max_l), dtype=np.int64)
    attention_mask = np.zeros((bsz, max_l), dtype=np.int64)
    for r_i, item in enumerate(micro_batch):
        l = item[2]
        input_ids[r_i, :l] = item[1]
        attention_mask[r_i, :l] = 1
    token_type_ids = np.zeros_like(input_ids)

    # 1. ONNX Inference
    t_onnx_start = time.perf_counter()
    outputs = onnx_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    onnx_ms = (time.perf_counter() - t_onnx_start) * 1000

    # 2. CLS Pooling (index 0 token hidden state)
    t_pool_start = time.perf_counter()
    last_hidden_state = outputs[0]
    cls_embeddings = last_hidden_state[:, 0, :]
    pooling_ms = (time.perf_counter() - t_pool_start) * 1000

    # 3. L2 Normalization (unit vectors)
    t_norm_start = time.perf_counter()
    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    norm_embeddings = cls_embeddings / np.clip(norms, 1e-9, None)
    normalization_ms = (time.perf_counter() - t_norm_start) * 1000

    metrics = {
        "batch_size": bsz,
        "max_seq_len": max_l,
        "min_seq_len": min_l,
        "mean_seq_len": mean_l,
        "real_token_count": real_tokens,
        "padded_token_count": padded_tokens,
        "padding_ratio": padding_ratio,
        "onnx_ms": onnx_ms,
        "pooling_ms": pooling_ms,
        "normalization_ms": normalization_ms,
        "onnx_threads": ORT_INTRA_OP_THREADS,
        "worker_count": EMBED_MAX_WORKERS,
    }

    logger.debug("micro_batch_complete", extra=metrics)

    return {
        "indices": [x[0] for x in micro_batch],
        "embeddings": norm_embeddings,
        "metrics": metrics,
    }


def embed_texts(texts: List[str], batch_size: int = EMBED_BATCH_SIZE) -> np.ndarray:
    """
    Generate normalized 384-dimensional CLS embeddings for input texts with:
    1. Single-pass tokenization.
    2. Length-aware micro-batching.
    3. Restored original input ordering.
    4. Comprehensive CloudWatch JSON instrumentation.
    """
    n = len(texts)
    if n == 0:
        return np.empty((0, 384), dtype=np.float32)

    t_embed_start = time.perf_counter()

    # Step 1: Tokenize all texts once
    indexed_items, tok_ms = tokenize_batch(texts)

    # Step 2: Build length-aware micro-batches
    # If batch is very small (<=4), process as a single direct batch without sorting overhead
    if n <= 4:
        batches = [indexed_items]
    else:
        batches = build_length_aware_batches(indexed_items, batch_size=batch_size)

    reconstructed = np.empty((n, 384), dtype=np.float32)
    tot_onnx_ms = 0.0
    tot_pool_ms = 0.0
    tot_norm_ms = 0.0
    tot_real_tokens = 0
    tot_padded_tokens = 0

    # Step 3: Run batches (sequentially or across workers if configured)
    if _executor is not None and len(batches) > 1:
        futures = [_executor.submit(run_onnx_micro_batch, b) for b in batches]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            m = res["metrics"]
            tot_onnx_ms += m["onnx_ms"]
            tot_pool_ms += m["pooling_ms"]
            tot_norm_ms += m["normalization_ms"]
            tot_real_tokens += m["real_token_count"]
            tot_padded_tokens += m["padded_token_count"]
            for idx, orig_i in enumerate(res["indices"]):
                reconstructed[orig_i] = res["embeddings"][idx]
    else:
        for b in batches:
            res = run_onnx_micro_batch(b)
            m = res["metrics"]
            tot_onnx_ms += m["onnx_ms"]
            tot_pool_ms += m["pooling_ms"]
            tot_norm_ms += m["normalization_ms"]
            tot_real_tokens += m["real_token_count"]
            tot_padded_tokens += m["padded_token_count"]
            for idx, orig_i in enumerate(res["indices"]):
                reconstructed[orig_i] = res["embeddings"][idx]

    t_embed_end = time.perf_counter()
    total_embedding_ms = (t_embed_end - t_embed_start) * 1000
    overall_padding_ratio = (
        1.0 - (tot_real_tokens / tot_padded_tokens) if tot_padded_tokens else 0.0
    )

    logger.info(
        "embed_texts",
        extra={
            "input_count": n,
            "tokenizer_ms": tok_ms,
            "onnx_ms": tot_onnx_ms,
            "pooling_ms": tot_pool_ms,
            "normalization_ms": tot_norm_ms,
            "total_embedding_ms": total_embedding_ms,
            "real_token_count": tot_real_tokens,
            "padded_token_count": tot_padded_tokens,
            "padding_ratio": overall_padding_ratio,
            "batch_size": batch_size,
            "num_micro_batches": len(batches),
            "onnx_threads": ORT_INTRA_OP_THREADS,
            "worker_count": EMBED_MAX_WORKERS,
        },
    )

    return reconstructed


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda entry point for text embedding.
    Expects: {"texts": ["review 1", "review 2", ...]}
    Or warmup ping: {"warmup": true} / {"action": "warmup"}
    Returns: {"statusCode": 200, "body": '{"embeddings": [[... 384 floats ...], ...]}'}
    """
    t_handler_start = time.perf_counter()

    # CloudWatch / EventBridge warmup ping support
    if event.get("warmup") or event.get("action") == "warmup":
        logger.info("warmup_ping_received")
        return {"statusCode": 200, "body": json.dumps({"status": "warm"})}

    texts = event.get("texts", [])
    if not texts:
        return {"statusCode": 400, "body": json.dumps({"error": "no texts provided"})}

    embeddings = embed_texts(texts)

    t_handler_end = time.perf_counter()
    duration_handler_ms = (t_handler_end - t_handler_start) * 1000
    logger.info(
        "lambda_handler",
        extra={
            "duration_ms": duration_handler_ms,
            "batch_size": len(texts),
        },
    )

    return {"statusCode": 200, "body": json.dumps({"embeddings": embeddings.tolist()})}


if __name__ == "__main__":
    test_event = {
        "texts": [
            "this broke after two days, complete waste of money",
            "works exactly as described, very happy with it",
            "it's fine, does the job, nothing special",
        ]
    }
    response = lambda_handler(test_event, None)
    body = json.loads(response["body"])
    emb = body.get("embeddings", [])
    print(f"Generated {len(emb)} embeddings of dimension {len(emb[0]) if emb else 0}")
    assert len(emb) == 3
    assert len(emb[0]) == 384
    print("Sanity check passed: Embedding Lambda works correctly!")
