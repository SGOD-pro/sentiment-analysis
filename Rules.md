# Project Rules

## Purpose

Rules that every future feature, module, and contributor must follow.

Violation of these rules requires updating Decisions.md before implementation.

---

# 1. General Rules

- Production-first implementation.
- Every feature must have a measurable purpose.
- No placeholder code.
- No experimental code inside production modules.
- One responsibility per module.
- Prefer readability over clever code.

---

# 2. Architecture Rules

- Shared embedding pipeline.
- Embedding generated only once.
- Downstream tasks consume same embedding.
- Lambda performs inference only.
- No training code inside deployment.

---

# 3. Library Rules

Allowed

Deployment

- numpy
- onnxruntime
- tokenizers

Offline Training

- pandas
- scikit-learn

Notes

- Training-only libraries must never be included in the Lambda deployment package.

Forbidden

- torch in Lambda
- transformers in Lambda
- sentence-transformers in Lambda
- tensorflow

---

# 4. Performance Rules

Cold Start Target

Expected cold start:
3–5 seconds
(Depends on Lambda memory allocation and ONNX model loading.)

Warm inference:
Typically under 500ms for small batches after initialization.

The model is loaded once during container initialization and reused for subsequent invocations.

Inference Performance

Target warm inference latency:

- Single review: typically <300ms
- Small batches (10–50 reviews): measured per request, not per review

Performance should always be evaluated at the request level because embedding generation is batched.

Memory:
<512MB

Deployment package:
<250MB

---

# 5. Error Handling Rules

Never silently fail.

Always

- validate input
- return structured error
- log internal errors
- never expose stacktrace

Response format

{
    success,
    error_code,
    message,
    data
}

---

# 6. Logging Rules

Every important operation should log

- timestamp
- module
- execution time
- error if any

No print() in production.

---

# 7. Testing Rules

Every ML module requires

- unit test
- integration test
- inference validation

Every API requires

- success case
- invalid input
- empty input
- malformed JSON

---

# 8. Documentation Rules

Every module must contain

Purpose

Input

Output

Dependencies

Example

Complexity

---

# 9. Folder Rules

No training artifacts inside frontend.

No frontend code inside backend.

No business logic inside UI.

---

# 10. Configuration Rules

Never hardcode

thresholds

paths

URLs

AWS IDs

Everything lives inside config.

---

# 11. Security Rules

Validate uploads.

Limit upload size.

Sanitize text.

Never trust user input.

---

# 12. Deployment Rules

Training

↓

Export

↓

Validation

↓

Packaging

↓

Deployment

No skipping validation.

---

# 14. Frontend Rules

Framework: React + TypeScript only.

UI Components: shadcn/ui only.
- Toasts → shadcn/ui Toast
- Alerts → shadcn/ui Alert
- Dialogs → shadcn/ui Dialog
- Tables → shadcn/ui Table
- No other component libraries.

Charts: Recharts only. No D3, no Chart.js.

Styling: Tailwind CSS v4.1 only.
- All tokens defined in CSS via @theme.
- No inline styles.
- No styled-components.
- No CSS modules.

Code rules:
- Write as little code as possible to do the job correctly.
- No unnecessary abstraction.
- No wrapper components that add no behavior.
- No extra state if props suffice.
- No comments explaining what the code obviously does.
- Every component has exactly one responsibility.

---

# 15. CSV Flexibility Rules

The upload pipeline must never assume fixed column names.

- User selects: text column, category column (optional), date column (optional)
- Any additional columns from the CSV are stored as-is and surfaced in the review feed table
- The dashboard adapts to whatever columns exist — no hardcoded "category" or "rating" assumptions
- Column mapping is set at upload time, stored with the batch, used consistently throughout

Every model must have

Model Version

Dataset Version

Threshold Version

Embedding Version

Issue Cluster Version

