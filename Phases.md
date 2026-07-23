# Project Phases

## Phase 1

Project Planning

Status

Completed

Deliverables

Requirement analysis

Architecture

Folder structure

Rules

Output

Project documentation

---

## Phase 2

Dataset Pipeline

Status

Completed

Tasks

Load Amazon Reviews

Cleaning

Preprocessing

Export datasets

Output

Processed dataset

---

## Phase 3

Embedding Pipeline

Status

Completed

Tasks

BGE embedding

Batch generation

Caching

Output

Embedding vectors

---

## Phase 4

Title

Sentiment Model Development & Label Audit

Status

Completed

Goal

Build a production-ready sentiment classifier with reliable evaluation and calibrated decision boundaries.

Tasks

- Train baseline models
- Train final MLP classifier
- Manual label audit (Round 1)
- Manual label audit (Round 2)
- Review confidence-margin analysis
- Correct dataset labels
- Build label taxonomy
- Threshold calibration
- Frozen evaluation validation
- Final model export

Deliverables

- Final MLP model
- Label audit reports
- Updated training dataset
- Confidence thresholds
- Frozen evaluation benchmark
- Exported model weights

Output

mlp_weights.npz

Lessons Learned

- Label quality had greater impact than increasing model complexity.
- Confidence-margin sampling identified more valuable review candidates than random sampling.
- Neutral sentiment remains structurally ambiguous even after label correction.

---

## Phase 5

Issue Detection

Completed

Tasks

Negative filtering

KMeans

Cluster naming

Centroid export

Output

issue_centroids.npy

---

## Phase 6

Model Export

Completed

Tasks

ONNX

INT8 Quantization

Packaging

Output

Lambda artifacts

---

## Phase 7

Backend API

In Progress

Tasks

Lambda

API Gateway

Validation

Logging

Authentication

Output

REST API

---

## Phase 8

Dashboard

Pending

Tasks

React

Charts

Filters

Analytics

Export

Output

Web Dashboard

---

## Phase 9

Optimization

Pending

Tasks

Performance

Caching

Monitoring

Cost reduction

---

## Phase 10

Production

Pending

Tasks

Deployment

Monitoring

CI/CD

Documentation

---

## Phase 11

Title

Two-Lambda Deployment Orchestration

Status

In Progress

Goal

Deploy BackendFunction and MLInferenceFunction as two properly right-sized
Lambda functions with matching local dev and CI behavior, per Deployment.md.

Tasks

- Restructure backend/ into src/ layout matching template.yaml's Handler path
- Create backend/src/lambda_handler.py (Mangum wrapper), add mangum dependency
- Consolidate ML handler to single location: lambda/handler.py (delete ml/ copy)
- Fix lambda_client.py local-bypass import path for new src/ folder depth
- Right-size BackendFunction to 256MB (down from 512MB — no ML libraries loaded)
- Add AWS::DynamoDB::Table resources to template.yaml (currently manual pre-step)
- Switch test-backend CI job to LocalStack instead of real AWS secrets
- Write backend/scripts/local_bootstrap.py (creates local S3 bucket + DynamoDB tables)
- Full verification pass per Deployment.md checklist (7 steps)
- Confirm MLInferenceFunction has zero API Gateway routes (unreachable externally)

Deliverables

- backend/src/ restructured and working locally
- lambda/handler.py as single source of truth for ML inference
- template.yaml with right-sized memory per function + DynamoDB table resources
- CI running tests against LocalStack, not real AWS
- Deployment.md verification checklist fully passed against real AWS

Output

Deployed API Gateway URL, both Lambda functions live in AWS Console

---

## Phase 12

Title

Human Feedback Loop for Sentiment Correction

Status

In Progress

Goal

Let users correct wrong sentiment predictions directly in the Reviews page,
capturing (review_id, batch_id, text, label, manual_label, date) in a
Corrections DynamoDB table for future model retraining — reusing the same
confidence-margin audit methodology already proven during initial model
development.

Tasks

- [x] CorrectionsTable in template.yaml (PK: correction_id, GSI: review-corrections-index)
- [x] config.py — dynamodb_corrections_table field
- [x] database.py — corrections added to Tables dataclass
- [x] routers/corrections.py — PATCH /api/reviews/{review_id}/correct (upsert, no-op guard, cache invalidation)
- [x] cache.py — cache_delete_prefix helper
- [x] main.py — corrections router registered
- [x] backend/scripts/export_corrections.py — DynamoDB → CSV (text, label, manual_label, date, review_id, batch_id)
- [x] ml/retrain_with_corrections.py — merge corrections + retrain MLP + compare benchmark + swap artifacts if better
- [x] frontend/types/index.ts — Correction interface, correction? on Review
- [x] frontend/api/client.ts — correctReview()
- [x] frontend/pages/Reviews.tsx — CorrectionPanel in dialog, correction badge on cards, optimistic state

Deliverables

- PATCH /api/reviews/:id/correct endpoint
- Corrections DynamoDB table (SAM-managed)
- Inline correction UI in Reviews page (dialog + card badge)
- export_corrections.py — produces CSV for retraining pipeline
- retrain_with_corrections.py — full retrain-compare-swap pipeline

Output (retraining pipeline)

Run export_corrections.py → retrain_with_corrections.py → if metrics improve,
lambda/artifacts/mlp_weights.npz is replaced → CI/CD deploys updated model automatically.

---

## Phase 13

Title

Post-Launch Quality Improvements (Ranked by Value/Complexity)

Status

Pending

Goal

Close the documented, known gaps in the current system rather than adding
new surface area. Every item here addresses a weakness already identified
and written down in PROJECT.md's "Known Limitations" section — this phase
does not introduce new scope, it resolves existing scope.

Explicitly NOT in this phase: BERTopic, semantic search, knowledge graph,
recommendation engine. These are v3 items that would add visible surface
area without touching any of the four real weaknesses below. Do not pull
them forward into this phase regardless of how appealing they look —
a documented weakness fixed is worth more than a new feature added, both
technically and as an interview answer.

### 13.1 — Per-Category Issue Clustering (highest value, lowest complexity)

Problem: cross-category KMeans clustering mixes product-domain signal
with complaint-type signal (music reviews cluster together because
they're about music, not because they share a complaint type). See
PROJECT.md's "Issue Detection: Cross-Category Signal Contamination".

Tasks:
- Cluster negative reviews within each product category separately, for
  categories with 500+ negative reviews (volume gate — smaller categories
  don't have enough data to cluster meaningfully on their own)
- Cross-category clustering (existing, live) becomes the fallback for
  categories below the volume gate and for reviews with no category
  mapped at upload
- `cluster_source` field (already in Reviews table schema) gets populated:
  `per_category` or `cross_category_fallback` — this field exists today
  but nothing writes a real value to it yet
- Re-run cluster naming exercise for each category above the volume gate
  (manual step, budget real time — this is the part that doesn't scale
  to "just run a script")
- Update issue distribution endpoint/dashboard to show cluster_source
  so a user can see which issue tags are category-specific vs fallback

Deliverable: issue tags that reflect actual complaint types per category,
not contaminated by product-domain vocabulary.

### 13.2 — Confidence-Score Dashboard Signal (medium value, medium complexity)

Problem: `confidence_margin` and per-class probabilities are already
stored on every review, but never surfaced to the user. This is real
data sitting unused.

Tasks:
- Add "average confidence" metric per category on the Dashboard
  (aggregate confidence_margin across reviews in that category/period)
- Flag categories with low average confidence as "predictions less
  reliable here" — simple threshold-based badge, not new ML work
- Consider surfacing this in the Review Feed too: a visual indicator
  (already partially planned in UI_UX.md's "low-confidence flag") for
  individual reviews below a threshold

Deliverable: business owners can see where the model is less certain,
not just what it predicted — turns existing stored data into decision-
relevant signal with zero new model training.

### 13.3 — Compositional ("mixed_but_neutral") Sentiment (real value, harder)

Problem: 20.8% of the model's confident-mistake cases (from the original
audit taxonomy) are reviews with genuine compositional sentiment — "loved
it until X broke" — that no tested intervention fixed. Upweighting,
threshold adjustment, and retraining all failed to move this specific
bucket. This is the one genuinely unsolved ML problem in the project,
not a missing feature.

Tasks (research-flavored, not a guaranteed win):
- Revisit the ordinal regression reframing hypothesis (predicting the
  underlying star rating as a continuous/ordinal target rather than
  hard 3-class classification) — this was proposed but never properly
  tested with a clean, isolated experiment
- If pursued: needs its own frozen eval comparison against the current
  MLP baseline (neg_recall 0.7915, frozen_recall 0.2967) using the exact
  same v4 split and frozen slice, not a new evaluation methodology
- Document the outcome either way — a properly-run experiment that
  confirms ordinal regression doesn't help is still a real, useful,
  defensible result (see PROJECT.md's existing pattern of documenting
  negative results, e.g. the DistilBERT and finetuned-BGE experiments)

Deliverable: either a measurable improvement to neutral recall past
~0.75, or a rigorously documented negative result explaining why the
ceiling holds — both are legitimate outcomes for this phase.

### 13.4 — Category-Specific Model Variants (v2 scope, harder, real production ML)

Problem: per-category F1 varies meaningfully (Software: 0.707,
Pet_Supplies: 0.709 vs. better-performing categories) — a single
universal model may be underperforming on specific categories that
would benefit from specialized handling.

Tasks:
- Identify categories with F1 significantly below the overall average
  using existing per-category evaluation data
- Design a lightweight specialization approach (e.g. a small per-category
  bias/calibration layer on top of the shared MLP, not a fully separate
  model per category — avoid the deployment/maintenance cost of N
  separate models)
- Leverage existing CI/CD infrastructure (already supports conditional
  model swaps via the Phase 12 retraining pipeline) to support deploying
  category-aware variants without new architectural work

Deliverable: measurable F1 improvement on the specific underperforming
categories, without regressing the categories that already perform well.
Explicitly v2/stretch — do not start this before 13.1-13.3 are resolved.

---

## Prioritization Note

Sequence: 13.1 → 13.2 → 13.3 → 13.4, in that order, unless a specific
business need reorders it. 13.1 and 13.2 are the highest-value,
lowest-risk items and should ship first. 13.3 is a real research
question with no guaranteed outcome — timebox it rather than letting it
block 13.4. 13.4 should not start until the earlier three are either
shipped or explicitly deprioritized with a written reason in this file.