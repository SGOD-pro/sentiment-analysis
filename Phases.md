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

Human Feedback Loop for Sentiment Correction (v2 — Deferred)

Status

Deferred to v2 — not started

Goal

Let users correct wrong sentiment predictions directly in the Reviews page,
capturing (batch_id, review_id, text, original_label, manual_label,
corrected_by, corrected_at) for future model retraining — reusing the same
confidence-margin audit methodology already proven during initial model
development (see PROJECT.md's Manual Label Audit section).

Why deferred, not built now

This is a genuine, valuable ML pipeline feature, not scope creep — but it
requires: a new DynamoDB table (Corrections) or extending Reviews with a
correction sub-object, a new API endpoint (PATCH /api/reviews/:id/correct),
frontend UI for inline correction, and a data-export path for eventually
feeding corrections back into a retraining pipeline. This is a full phase
of its own work, not a quick addition, and v1 is not blocked on it.

Planned tasks (v2, not now)

- Corrections table/field design (batch_id, text, original_label,
  manual_label, corrected_by, corrected_at)
- PATCH /api/reviews/:id/correct endpoint
- Reviews page UI: inline correction control per review card
- Export script: pull corrections, format for the existing ml/ audit
  pipeline (same taxonomy structure as the original label_correction_loop.py)
- Retraining trigger strategy (manual batch export vs continuous)

Output (v2)

A documented, reusable feedback loop that closes the ML lifecycle loop —
legitimate to describe as "production ML pipeline with human-in-the-loop
retraining" once built, not before.