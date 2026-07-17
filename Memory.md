# Memory.md — Project State Snapshot

Update this file at the end of every working session.
Purpose: load full project context in one file, not by reading every folder.

---

## Current Phase

Phase 8 — Frontend (Complete, verified)

Lambda handler is working and locally tested.
Backend FastAPI is complete — all routes built and tested (60 tests passing).
Dashboard, Reviews, Reports pages built and verified — all filters wired.

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

### Backend API — complete, all 60 tests passing

Stack: FastAPI + boto3 + pydantic-settings, uv package manager, moto for tests.

Routes:
```
POST /api/upload              — CSV upload, S3 storage, batch creation, background processing
GET  /api/batches/:id/status  — batch processing status
GET  /api/batches/stats        — aggregate batch processing metrics
GET  /api/trends              — weekly sentiment counts from Aggregates
GET  /api/categories/summary  — ranked category list by sentiment score
GET  /api/issues/distribution — issue tag counts for negative reviews
GET  /api/reviews             — paginated, filterable review list
GET  /api/reviews/:id         — single review detail
GET  /health                  — health check
```

Key files: `backend/main.py`, `config.py`, `database.py`, `logger.py`, `models.py`
Routers: `routers/{upload,batches,trends,categories,issues,reviews}.py`
Services: `services/{batch_processor,lambda_client,text_preprocessing}.py`
Tests: `tests/test_{health,upload,batch_processor,batches,trends,categories,issues,reviews,filter_regression,text_preprocessing}.py`

Aggregates key patterns: `TREND#{category}#{week}`, `CAT#{category}`, `ISSUE#{tag}#{week}`
Batch processor: chunks → Lambda (concurrent via ThreadPoolExecutor), batch_write_item for DynamoDB,
  text_preprocessing (HTML strip, URL removal, capitalize after fullstop) applied before inference,
  processing_duration_seconds stored in Batches table.
All config from env vars (pydantic-settings). No hardcoded values. Structured JSON logging.

---

## What Is Not Done (next to build)

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

## Bugs Fixed

### Filter wiring (Phase 8 fix)

Root cause: (a) + (c)
- (a) Frontend: Date Range `<Select>` on Dashboard/Reports/Reviews was static UI — `onChange` never
  wired to state that feeds `useEffect` dependency array. Filters updated sidebar but never triggered
  re-fetch.
- (c) Backend: `/api/categories/summary` accepted `from`/`to` params but queried `CAT#` keys which
  have no date dimension — date filtering was silently ignored.

Fix:
- Frontend: replaced static Select with `DateRangeFilter` component (shadcn Calendar + Popover).
  All filter state (dateRange, category, sentiment, confidence) now feeds `useEffect` deps that
  trigger API re-fetches with correct query params.
- Backend: `categories.py` now uses `TREND#` keys (which contain week info) when `from`/`to` are
  provided, falls back to `CAT#` fast path when no date range given.
- Sentiment filter on Dashboard now toggles trend chart series visibility (show only selected).
- Confidence filter no longer incorrectly zeroes stat card totals — stats always show real totals.

Regression tests: `test_filter_regression.py` (9 tests) covering all 4 filtered endpoints.

### Batch processing speed (Phase 8 fix)

- Lambda invocations changed from sequential `for` loop to concurrent `ThreadPoolExecutor`.
- DynamoDB writes changed from individual `put_item` to `batch_writer` (up to 25 per call).
- Text preprocessing added before inference (HTML strip, URL removal, hex/UUID removal).
- Timing instrumentation added (structured logging: S3 read, Lambda, DynamoDB phases).
- `processing_duration_seconds` now stored in Batches table.

### Reports page metrics (Phase 8 fix)

- Removed fake "System Uptime" card.
- Replaced "Avg Processing Time" (was "—") with real "Avg Batch Time" from `/api/batches/stats`.
- Added "Batches Processed" count card.

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
