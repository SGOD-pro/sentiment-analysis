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
