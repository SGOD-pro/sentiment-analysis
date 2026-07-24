# Memory.md — Project State Snapshot

Update this file at the end of every working session. This was found
EMPTY during Phase 13 planning — meaning Phase 11 and Phase 12's real
work (deployment debugging, feedback loop implementation) was never
recorded here. Read this file first in any new session before reading
anything else. If it's ever found stale/wrong, fix it immediately, don't
work around it.

---

## Current Phase

Phase 13.2 — Confidence-Score Dashboard Signal (Next)

Phase 13.1 (Per-Category Issue Clustering) is COMPLETE. We have added `per_category_clustering.py` to calculate local centroids, updated `export_mlp_and_clusters.py` to output a unified `issue_centroids.npz`, and updated the Lambda inference, backend processing, DynamoDB schemas (using `ISSUE#{tag}#{source}#{week}` format), and frontend Issue Distribution chart (now a PieChart showing `cluster_source` tooltips).

Phases 1-12 are complete or in-progress-but-functional. The core
platform (sentiment + issue detection + dashboard + corrections
feedback loop) is deployed and working. Phase 13 addresses documented
known weaknesses, not new features.

---

## What Is Actually Done and Deployed

### ML Pipeline (Phases 2-6) — complete, do not retrain from scratch

- BGE-small-en-v1.5 (frozen) → 2-layer MLP (256→128→3) + asymmetric
  confidence threshold (pos_margin=0.30, neg_margin=0.0)
- Metrics (v4 split, leak-free): neg_recall=0.7915, neutral_recall=0.7426,
  frozen_recall=0.2967 (300-row frozen eval slice, never trained on)
- 13 architecture experiments run and documented — MLP on frozen
  embeddings beat DistilBERT finetune and finetuned-BGE-encoder in all
  of them. Do not re-attempt these without new evidence.
- Issue detection: KMeans K=15 on 68,967 negative reviews, distance
  threshold 0.70. 13 clusters usably named, clusters 6 and 11 are noise
  ("other"). Known limitation: cross-category signal contamination
  (see Phase 13.1).
- Deployment artifacts: quantized ONNX (~35MB) + mlp_weights.npz (~0.5MB)
  + issue_centroids.npy (~22KB) + config.json — all in
  lambda/artifacts/, synced from S3 at deploy time, NOT committed to git.

### Two-Lambda Architecture (Phase 11) — deployed and verified

- `BackendFunction`: FastAPI + Mangum, 256MB, behind API Gateway HTTP API
- `MLInferenceFunction`: ONNX+MLP+KMeans, 1024MB (raised from 512MB
  during Phase 11 debugging — OOM occurred at 512MB, see below), NO
  API Gateway route, invoked only by BackendFunction via
  boto3.client("lambda").invoke() with FunctionName resolved from
  ML_INFERENCE_FUNCTION_NAME env var (injected by template.yaml's
  !Ref MLInferenceFunction — never hardcode this name anywhere)
- Local dev: lambda_client.py branches on AWS_ENDPOINT_URL containing
  "localhost" — imports lambda/handler.py directly in-process, runs
  the REAL model locally (not a mock), no AWS Lambda involved
- CORS: configured ONLY on API Gateway (SentiMetricHttpApi), via
  FrontendOrigin parameter, two-pass (deploy 1: "*", deploy 2 after
  Vercel URL exists: real origin via --parameter-overrides). Confirm
  template.yaml's Parameters block actually has FrontendOrigin wired
  to AllowOrigins via !Ref before assuming CORS tightening works — this
  was found REVERTED to hardcoded "*" once already mid-session, check
  it's still correctly wired before next CORS-related work.

### Known deploy issues found and fixed during Phase 11 (don't re-break these)

1. `requirements.txt` must sit INSIDE the CodeUri folder for each
   function (backend/deploy/src/requirements.txt AND
   lambda/requirements.txt), not next to template.yaml — SAM build
   silently skips dependency install otherwise ("requirements.txt file
   not found. Continuing the build without dependencies" — this failure
   mode does NOT stop the build, only shows as a runtime import error).
2. MLInferenceFunction OOM'd at 512MB — raised to 1024MB. Session.run
   duration is 7-22 seconds per batch of 50 under real Lambda
   concurrency — this is now the known performance baseline, not
   necessarily fully optimized (SessionOptions with disabled memory
   arena/pattern were added, timing instrumentation added — confirm
   these actually reduced variance before assuming it's optimal).
3. Unused transitive dependencies (hf_xet, huggingface_hub, httpx,
   httpcore) get pulled in by tokenizers/optimum tooling but aren't
   needed at runtime — pruned in CI post-build.
4. startup_check.py needs dynamodb:ListTables and lambda:GetFunction
   IAM permissions on BackendFunction's execution role — these are NOT
   granted by DynamoDBCrudPolicy/LambdaInvokePolicy (different action
   set) and needed an explicit inline Statement block added to
   template.yaml.
5. AWS::DynamoDB::Table resources are NOT yet in template.yaml (Phase 11
   task, marked as a "gap to close" — check if this got resolved or if
   tables are still a manual pre-deploy step).

### Feedback Loop (Phase 12) — implemented, verification incomplete

- PATCH /api/reviews/{review_id}/correct — upserts corrections,
  no-op guard, cache invalidation
- CorrectionsTable in DynamoDB (PK: correction_id, GSI on review_id)
- export_corrections.py → CSV matching label_correction_loop.py's format
- retrain_with_corrections.py — merges corrections into v4 training
  data, retrains MLP (same architecture/seed as production), gates on
  BOTH neg_recall >= 0.7915 AND frozen_recall >= 0.2967, only swaps
  mlp_weights.npz if both pass
- Trust layers added: session-based anomaly detection (>70% same label
  from one session with 3+ corrections = excluded), confidence-based
  flagging (high-confidence overrides logged for scrutiny), minimum
  volume gate (2+ independent sessions required for NEW text before
  training inclusion — existing v4 text originally allowed single-
  correction inclusion, this was flagged as a real gap and SHOULD be
  tightened to match, verify this was actually done)

**IMPORTANT — verification status:** Phase 12's trust layers were
implemented and imports/types check out, but were NOT verified against
real adversarial test cases as of last session (no confirmed test of:
suspicious-session exclusion actually firing, tie-breaking logic for
disagreeing corrections, confidence_margin preservation across repeat
corrections on the same review). Do not assume these work correctly
until that verification pass is actually run and results reported.

---

## Data File Locations (Google Drive, not in git)

```
Drive/MyDrive/Dataset/embeddings_output/
  bge_clean_metadata.parquet       ← authoritative dataset, v4 corrected labels
  bge_clean_metadata_corrected.parquet  ← POSSIBLE DUPLICATE, unresolved -
                                            check which is authoritative
                                            before using either in retraining
  bge_clean_embeddings.npy         ← BGE embeddings, ~203k rows, 384-dim
  clean_train_idx_v4.npy           ← use this, not v2 or v3
  clean_test_idx_v4.npy            ← use this, not v2 or v3 (v2/v3 kept
                                       for historical reference only)
  difficult_neutral_eval_FROZEN.csv  ← 300 rows, NEVER train on this,
                                         NEVER modify
  audit_registry.csv               ← all 9,990+ reviewed row IDs across
                                        every correction round
  final_mlp_state.pt               ← production MLP weights (PyTorch)
  issue_kmeans_model.joblib
  cluster_naming_worksheet.csv

Drive/MyDrive/lambda_deploy_artifacts/  (also mirrored to S3)
  bge_onnx_quantized/               ← ships to Lambda
  bge_onnx_fp32/                    ← build intermediate, NEVER ships,
                                        always excluded via --exclude flag
  mlp_weights.npz
  issue_centroids.npy
  config.json
```

S3: s3://sentimetric-prod-storage/ml-artifacts/lambda_deploy_artifacts/
(source of truth for CI/CD deploys — synced down fresh every deploy,
never committed to git, never trust a local copy without re-syncing)

---

## Confirmed Metrics (cite these exactly, don't approximate)

MLP + asymmetric threshold, v4 split, production model:
- Macro F1: 0.7930
- Negative recall: 0.7915
- Neutral recall: 0.7426
- Frozen slice neutral recall: 0.2967

These are the retraining gate thresholds in retrain_with_corrections.py.
Any future retrain must clear or exceed BOTH neg_recall and frozen_recall
to replace the production model.

---

## What's Next (Phase 13, ranked)

1. Per-category issue clustering (highest value/lowest complexity —
   fixes cross-category signal contamination in issue tags)
2. Confidence-score dashboard signal (existing data, unused — surface it)
3. Compositional "mixed_but_neutral" sentiment — genuinely unsolved,
   timeboxed research, ordinal regression reframing untested
4. Category-specific model variants (v2 scope, don't start before 1-3)

Explicitly NOT next: BERTopic, semantic search, knowledge graph,
recommendation engine — v3 items, don't pull forward, don't add scope
that doesn't address a documented weakness.

---

## Rules Summary (full detail in Rules.md)

- One embedding pass, shared across sentiment + issue detection
- Lambda: numpy + onnxruntime + tokenizers ONLY, verified via CI check
  that asserts torch/transformers/sklearn are absent from the package
- Frontend: shadcn/ui components, Recharts charts, Tailwind v4.1 only
- CSV ingestion is column-flexible — never hardcode column names,
  user maps columns at upload time
- Every function requires a test case — no exceptions for "small" ones
  (this rule exists BECAUSE a small untested function caused the
  filter-wiring bug in Phase 8)
- Never hardcode thresholds, paths, URLs, AWS IDs — config/parameters only
- No print() in production — structured JSON logging only