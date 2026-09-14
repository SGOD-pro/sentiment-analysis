"""
Embedding Lambda handler.

Purpose: High-throughput text embedding generator using quantized BGE ONNX model.
Input: {"texts": ["review 1", "review 2", ...]}
Output: {"statusCode": 200, "body": '{"embeddings": [[... 384 floats ...], ...]}'}

Optimizations:
1. Module-level pre-initialization (cold start graph compilation & memory arena allocation during Lambda Init).
2. ThreadPool micro-batching across CPU cores to saturate vCPUs and avoid attention padding inflation.
3. Memory arena and pattern reuse enabled to prevent malloc/free thrashing.
4. Warmup ping handler support for EventBridge scheduled pings.
"""

import concurrent.futures
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone

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
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

ARTIFACT_DIR = os.environ.get(
    "ARTIFACT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"),
)

# 1. Initialize Tokenizer
tokenizer = Tokenizer.from_file(f"{ARTIFACT_DIR}/bge_onnx_quantized/tokenizer.json")
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=256)

# 2. Optimized ONNX Runtime Session
sess_options = ort.SessionOptions()
sess_options.enable_mem_pattern = True
sess_options.enable_cpu_mem_arena = True
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = min(2, os.cpu_count() or 1)
sess_options.inter_op_num_threads = 1

onnx_session = ort.InferenceSession(
    f"{ARTIFACT_DIR}/bge_onnx_quantized/model_quantized.onnx",
    sess_options=sess_options,
    providers=["CPUExecutionProvider"],
)

# 3. Thread Pool for Multi-Core Micro-Batching
MAX_WORKERS = min(4, max(2, os.cpu_count() or 2))
executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=MAX_WORKERS,
    thread_name_prefix="bge_worker",
)

# 4. Cold Start Optimization: Pre-warm execution graph during Lambda Init phase
# Lambda provides burst vCPU during initialization; executing a single dummy pass
# compiles ONNX operator kernels and allocates memory arena before requests arrive.
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


def _embed_chunk(chunk: list[str]) -> np.ndarray:
    """Process a single chunk of texts through tokenizer and ONNX session."""
    t_tok_start = time.perf_counter()
    encodings = tokenizer.encode_batch(chunk)
    duration_tok_ms = (time.perf_counter() - t_tok_start) * 1000

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    t_run_start = time.perf_counter()
    outputs = onnx_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    duration_run_ms = (time.perf_counter() - t_run_start) * 1000

    last_hidden_state = outputs[0]
    cls_embeddings = last_hidden_state[:, 0, :]
    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    embeddings = cls_embeddings / np.clip(norms, 1e-9, None)

    logger.info(
        "embed_chunk",
        extra={
            "count": len(chunk),
            "tokenizer_ms": duration_tok_ms,
            "session_run_ms": duration_run_ms,
        },
    )
    return embeddings


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generate normalized 384-dimensional CLS embeddings for input texts.
    Employs ThreadPool micro-batching for batches > 4 to:
    1. Saturate available vCPUs concurrently.
    2. Prevent massive sequence padding inflation across heterogeneous reviews.
    """
    t_embed_start = time.perf_counter()
    n = len(texts)

    if n <= 4 or MAX_WORKERS <= 1:
        embeddings = _embed_chunk(texts)
        workers_used = 1
    else:
        # Split into balanced micro-batches across workers (min 4 per chunk to amortize thread overhead)
        chunk_size = max(4, math.ceil(n / MAX_WORKERS))
        chunks = [texts[i : i + chunk_size] for i in range(0, n, chunk_size)]
        results = list(executor.map(_embed_chunk, chunks))
        embeddings = np.vstack(results)
        workers_used = min(len(chunks), MAX_WORKERS)

    t_embed_end = time.perf_counter()
    duration_embed_texts_ms = (t_embed_end - t_embed_start) * 1000

    logger.info(
        "embed_texts",
        extra={
            "duration_ms": duration_embed_texts_ms,
            "batch_size": n,
            "workers": workers_used,
        },
    )

    return embeddings


def lambda_handler(event, context):
    """
    Lambda entry point for text embedding.
    Expects: {"texts": ["review 1", "review 2", ...]}
    Or warmup ping: {"warmup": true} / {"action": "warmup"}
    Returns: {"statusCode": 200, "body": '{"embeddings": [[...], ...]}'}
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
