# Architectural Decisions

---

## Decision 001

Freeze encoder

Reason

Smaller deployment

Better reproducibility

Evidence

- Sentiment training experiment log (13 controlled experiments)
- evaluate_v4_and_frozen.py
- confidence_threshold_test.py
- PROJECT.md ("The Sentiment Model")

Date

...

---

## Decision 002

Use ONNX

Reason

Lower latency

Smaller package

---

Decision 003

Nearest centroid

Reason

No labeled issue dataset

Fast inference

---

## Decision 004

Asymmetric threshold

Reason

Negative recall more important

---

## Decision 005

Title

Manual label audit via confidence-margin sampling

Status

Accepted

Date

2026-07

Reason

Random sampling primarily surfaces easy examples and provides limited information about model weaknesses.

Instead, reviews were selected using prediction confidence margins, prioritizing uncertain predictions and confident disagreements between the model and labels.

Outcome

- Two targeted audit rounds
- 9,990 manually reviewed reviews
- Label taxonomy established:
    - mislabel
    - genuinely_mixed
    - model_wrong
- Produced significantly higher-value corrections than random sampling.

Evidence

- Manual audit documentation
- Phase 4 training logs
- PROJECT.md ("Manual label audit")

---

## Decision 006

Title

Asymmetric confidence threshold

Status

Accepted

Date

2026-07

Reason

Business analysis determined that missing a genuinely negative review is more costly than incorrectly classifying uncertain positive reviews.

Instead of using a symmetric confidence threshold across classes, separate thresholds were introduced.

Decision

- Protect negative recall
- Allow uncertain positive predictions to fall back to neutral
- Keep negative predictions unless confidence is genuinely insufficient

Outcome

- Negative recall preserved
- Neutral recall improved
- Better alignment with business requirements

Evidence

- Threshold evaluation experiments
- Confidence-margin analysis
- PROJECT.md ("The asymmetric threshold")