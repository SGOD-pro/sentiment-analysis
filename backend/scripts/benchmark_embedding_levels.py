"""
Level 1 Single Lambda Execution Benchmark.

Compares Baseline (Old) vs Optimized (New) BGE Embedding Pipeline inside Lambda.
Measures:
- Preprocessing time
- Tokenization time
- ONNX Inference time
- Pooling time
- Normalization time
- Total embedding time
- Real vs Padded tokens & Padding Ratio
Across 10, 20, 50, 100, and 300 real customer reviews.
"""

import csv
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from services.text_preprocessing import text_preprocessing

ARTIFACT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "lambda", "artifacts", "bge_onnx_quantized"
)

# ── Old Implementation (Baseline) ──
tok_old = Tokenizer.from_file(os.path.join(ARTIFACT_DIR, "tokenizer.json"))
tok_old.enable_padding(pad_id=0, pad_token="[PAD]")
tok_old.enable_truncation(max_length=256)

sess_old_opts = ort.SessionOptions()
sess_old_opts.intra_op_num_threads = 2
sess_old_opts.inter_op_num_threads = 1
sess_old = ort.InferenceSession(
    os.path.join(ARTIFACT_DIR, "model_quantized.onnx"),
    sess_options=sess_old_opts,
    providers=["CPUExecutionProvider"],
)


def run_old_pipeline(texts: List[str]) -> Dict[str, Any]:
    t_start = time.perf_counter()

    # Preprocessing
    t0 = time.perf_counter()
    cleaned = [text_preprocessing(t) or t for t in texts]
    prep_ms = (time.perf_counter() - t0) * 1000

    # Old handler divides by 2 workers
    n = len(cleaned)
    chunk_size = max(4, math.ceil(n / 2)) if n > 4 else n
    chunks = [cleaned[i : i + chunk_size] for i in range(0, n, chunk_size)]

    tok_ms = 0.0
    onnx_ms = 0.0
    pool_ms = 0.0
    norm_ms = 0.0
    real_tokens = 0
    padded_tokens = 0
    embs = []

    for c in chunks:
        t0 = time.perf_counter()
        encs = tok_old.encode_batch(c)
        tok_ms += (time.perf_counter() - t0) * 1000

        max_l = len(encs[0].ids)
        bsz = len(c)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        types = np.zeros_like(ids)
        real_tokens += sum(int(np.sum(e.attention_mask)) for e in encs)
        padded_tokens += bsz * max_l

        t0 = time.perf_counter()
        out = sess_old.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
        )
        onnx_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cls_e = out[0][:, 0, :]
        pool_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        norms = np.linalg.norm(cls_e, axis=1, keepdims=True)
        norm_e = cls_e / np.clip(norms, 1e-9, None)
        norm_ms += (time.perf_counter() - t0) * 1000
        embs.append(norm_e)

    total_ms = (time.perf_counter() - t_start) * 1000
    pad_ratio = 1.0 - (real_tokens / padded_tokens) if padded_tokens else 0.0
    return {
        "count": len(texts),
        "prep_ms": prep_ms,
        "tok_ms": tok_ms,
        "onnx_ms": onnx_ms,
        "pool_ms": pool_ms,
        "norm_ms": norm_ms,
        "total_ms": total_ms,
        "pad_ratio": pad_ratio,
        "real_tokens": real_tokens,
        "padded_tokens": padded_tokens,
    }


# ── New Implementation (Optimized) ──
tok_new = Tokenizer.from_file(os.path.join(ARTIFACT_DIR, "tokenizer.json"))
tok_new.no_padding()
tok_new.enable_truncation(max_length=256)

sess_new_opts = ort.SessionOptions()
sess_new_opts.enable_mem_pattern = True
sess_new_opts.enable_cpu_mem_arena = True
sess_new_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_new_opts.intra_op_num_threads = 1
sess_new_opts.inter_op_num_threads = 1
sess_new = ort.InferenceSession(
    os.path.join(ARTIFACT_DIR, "model_quantized.onnx"),
    sess_options=sess_new_opts,
    providers=["CPUExecutionProvider"],
)


def run_new_pipeline(texts: List[str], micro_bsz: int = 16) -> Dict[str, Any]:
    t_start = time.perf_counter()

    # Preprocessing
    t0 = time.perf_counter()
    cleaned = [text_preprocessing(t) or t for t in texts]
    prep_ms = (time.perf_counter() - t0) * 1000

    # Tokenize once
    t0 = time.perf_counter()
    encs = tok_new.encode_batch(cleaned)
    tok_ms = (time.perf_counter() - t0) * 1000

    indexed = [(i, enc.ids, len(enc.ids)) for i, enc in enumerate(encs)]
    indexed.sort(key=lambda x: x[2])  # sort by length

    batches = [
        indexed[i : i + micro_bsz] for i in range(0, len(indexed), micro_bsz)
    ]

    onnx_ms = 0.0
    pool_ms = 0.0
    norm_ms = 0.0
    real_tokens = 0
    padded_tokens = 0
    reconstructed = np.empty((len(texts), 384), dtype=np.float32)

    for b in batches:
        bsz = len(b)
        max_l = max(x[2] for x in b)
        real_tokens += sum(x[2] for x in b)
        padded_tokens += bsz * max_l

        ids = np.zeros((bsz, max_l), dtype=np.int64)
        mask = np.zeros((bsz, max_l), dtype=np.int64)
        for r_i, item in enumerate(b):
            l = item[2]
            ids[r_i, :l] = item[1]
            mask[r_i, :l] = 1
        types = np.zeros_like(ids)

        t0 = time.perf_counter()
        out = sess_new.run(
            None,
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
        )
        onnx_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cls_e = out[0][:, 0, :]
        pool_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        norms = np.linalg.norm(cls_e, axis=1, keepdims=True)
        norm_e = cls_e / np.clip(norms, 1e-9, None)
        norm_ms += (time.perf_counter() - t0) * 1000

        for r_i, item in enumerate(b):
            reconstructed[item[0]] = norm_e[r_i]

    total_ms = (time.perf_counter() - t_start) * 1000
    pad_ratio = 1.0 - (real_tokens / padded_tokens) if padded_tokens else 0.0
    return {
        "count": len(texts),
        "prep_ms": prep_ms,
        "tok_ms": tok_ms,
        "onnx_ms": onnx_ms,
        "pool_ms": pool_ms,
        "norm_ms": norm_ms,
        "total_ms": total_ms,
        "pad_ratio": pad_ratio,
        "real_tokens": real_tokens,
        "padded_tokens": padded_tokens,
    }


def main():
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "test_data",
        "sampled_categories",
        "Automotive_sample.csv",
    )
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    all_texts = [r["text"] for r in rows if "text" in r]

    # Warmup
    run_old_pipeline(all_texts[:10])
    run_new_pipeline(all_texts[:10])

    print("=" * 105)
    print("LEVEL 1 BENCHMARK: OLD (BASELINE) vs NEW (OPTIMIZED) EMBEDDING PIPELINE")
    print("=" * 105)
    header = (
        f"{'Reviews':>7} | {'Impl':>4} | {'Total(ms)':>9} | {'Prep(ms)':>8} | "
        f"{'Tok(ms)':>7} | {'ONNX(ms)':>8} | {'Pool(ms)':>8} | {'Norm(ms)':>8} | "
        f"{'PadRatio':>8} | {'Speedup':>7}"
    )
    print(header)
    print("-" * 105)

    counts = [10, 20, 50, 100, 300]
    for count in counts:
        t_slice = all_texts[:count]
        m_old = run_old_pipeline(t_slice)
        m_new = run_new_pipeline(t_slice)
        speedup = m_old["total_ms"] / m_new["total_ms"]

        line_old = (
            f"{count:7d} | OLD  | {m_old['total_ms']:9.1f} | {m_old['prep_ms']:8.2f} | "
            f"{m_old['tok_ms']:7.2f} | {m_old['onnx_ms']:8.1f} | {m_old['pool_ms']:8.2f} | "
            f"{m_old['norm_ms']:8.2f} | {m_old['pad_ratio']*100:7.1f}% |     1.0x"
        )
        line_new = (
            f"{count:7d} | NEW  | {m_new['total_ms']:9.1f} | {m_new['prep_ms']:8.2f} | "
            f"{m_new['tok_ms']:7.2f} | {m_new['onnx_ms']:8.1f} | {m_new['pool_ms']:8.2f} | "
            f"{m_new['norm_ms']:8.2f} | {m_new['pad_ratio']*100:7.1f}% | {speedup:6.2f}x"
        )
        print(line_old)
        print(line_new)
        print("-" * 105)


if __name__ == "__main__":
    main()
