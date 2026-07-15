# Memory.md — Project State Snapshot

Update this file at the end of every working session.
Purpose: load full project context in one file, not by reading every folder.

---

## Current Phase

Phase 7 — Backend API (In Progress)

Lambda handler is working and locally tested.
Backend FastAPI + API Gateway not yet built.
Dashboard not yet built.

---

## What Is Done (do not rebuild)

### ML Pipeline — complete, do not touch

Embedding: BGE-small-en-v1.5, frozen, quantized ONNX INT8
Sentiment: MLP (256→128→3) + asymmetric threshold (pos_margin=0.30, neg_margin=0.0)
Issue detection: KMeans K=15, distance_threshold=0.70

All artifacts saved at:
`/content/drive/MyDrive/lambda_deploy_artifacts/`

Files:
- `bge_onnx_quantized/model_quantized.onnx` (~35MB) — ships to Lambda
- `bge_onnx_quantized/tokenizer.json` + supporting files
- `mlp_weights.npz` (~0.5MB)
- `issue_centroids.npy` (~22KB)
- `config.json` (<1KB)

`bge_onnx_fp32/` is a build artifact — NOT shipped to Lambda.

Lambda handler: `lambda_handler_final.py` — tested end-to-end, output verified.

Dataset: `bge_clean_metadata.parquet` + `bge_clean_embeddings.npy`
Splits: `clean_train_idx_v4.npy` / `clean_test_idx_v4.npy`
Frozen eval: `difficult_neutral_eval_FROZEN.csv` (300 rows, never train on this)
MLP weights: `final_mlp_state.pt`

### Confirmed metrics (leak-free, v4 split)

MLP + asymmetric threshold:
- Macro F1: 0.7930
- Neg recall: 0.7915
- Neutral recall: 0.7426
- Frozen slice neutral recall: 0.2967

DistilBERT and finetuned BGE both tested — both lost. Do not revisit.

---

## What Is Not Done (next to build)

### Phase 7 — Backend (start here)

FastAPI app with these routes:
```
POST /api/upload         — accept CSV, map columns, queue batch
GET  /api/batches/:id    — poll processing status
GET  /api/trends         — sentiment trend by week/month
GET  /api/categories/summary
GET  /api/issues/distribution
GET  /api/reviews        — paginated, filterable
GET  /api/reviews/:id
```

Lambda client: call inference Lambda in batches of 50-100 reviews.

DynamoDB tables: Reviews, Batches, Aggregates (see Architecture.md).

### Phase 8 — Frontend

React + TypeScript + shadcn/ui + Recharts + Tailwind v4.1

Upload page:
- CSV upload
- User selects: text column, category column (optional), date column (optional)
- Any extra CSV columns stored and shown in review feed table
- Preview before submit

Dashboard page:
- Alert strip (rule-based, negative spike detection)
- Summary cards (total reviews, % negative, top issue tag)
- Sentiment trend chart (Recharts LineChart, 3 series)
- Category breakdown table (sortable)
- Issue distribution chart (Recharts BarChart)

Review feed page:
- Paginated table
- Filters: sentiment, category, issue_tag, date range, confidence threshold
- All columns from original CSV shown alongside sentiment + issue_tag
- Sort by date or confidence margin

---

## Key Decisions (full detail in Decisions.md)

001: Freeze encoder — 13 experiments proved finetuning never beat frozen BGE + MLP
004: Asymmetric threshold — missing negative is costlier than false neutral
005: Confidence-margin audit sampling — not random, targeted
006: Protect negative recall — threshold only softens positive predictions

---

## Known Limitations (full detail in PROJECT.md)

- Neutral class ceiling ~0.75 F1 — structural, not fixable by tuning
- Cross-category clustering contaminates some issue clusters with domain signal (music → audio cluster)
  Fix designed (per-category clustering, 500+ row gate) but not yet built
- Out-of-distribution text untested (trained on Amazon reviews only)

---

## File Map (where things actually are)

```
Drive/MyDrive/
  Dataset/embeddings_output/
    bge_clean_metadata.parquet     ← main dataset (corrected labels)
    bge_clean_embeddings.npy       ← BGE embeddings for all rows
    clean_train_idx_v4.npy
    clean_test_idx_v4.npy
    final_mlp_state.pt             ← trained MLP weights (PyTorch)
    difficult_neutral_eval_FROZEN.csv
    audit_registry.csv             ← all 9,990 reviewed row IDs
    issue_kmeans_model.joblib
    cluster_naming_worksheet.csv
    negative_reviews_clustered.csv

  lambda_deploy_artifacts/
    bge_onnx_quantized/            ← ships to Lambda
    mlp_weights.npz
    issue_centroids.npy
    config.json

  bge-small/                       ← raw PyTorch BGE model (local backup)
```

---

## Rules summary (full detail in Rules.md)

- One embedding pass, shared across all downstream tasks
- Lambda: numpy + onnxruntime + tokenizers only. No torch, no transformers, no sklearn
- Frontend: shadcn/ui components, Recharts charts, Tailwind v4.1 styling
- Write minimum code needed. No wrappers, no abstraction without reason
- CSV ingestion must be column-flexible — user maps columns at upload, never hardcode column names
- Never hardcode thresholds, paths, URLs, AWS IDs — everything in config
- No print() in production — use structured logging
- Cold start: 3-5s expected. Warm inference: <500ms per batch
