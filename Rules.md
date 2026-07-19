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

Every function requires a test case. No exceptions for "small" or
"obvious" functions — a one-line function with no test is exactly where
silent regressions hide (see the filter-wiring bug found in Phase 8,
which existed precisely because query-param handling had no test asserting
the params actually affected the query).

Every ML module requires

- unit test
- integration test
- inference validation

Every API endpoint requires

- success case
- invalid input
- empty input
- malformed JSON
- filter/query-param cases: assert the response DATA changes when a
  filter changes, not just that the request doesn't error. A 200 response
  with unfiltered data passing a test that only checks status code is a
  false pass — this exact failure mode shipped once already.

Every service function (lambda_client.py, batch_processor.py, etc.)
requires a test for both branches when the function has environment-
dependent behavior — e.g. lambda_client.invoke_lambda() must have a test
covering the localhost/LocalStack bypass path AND a mocked test covering
the real boto3.invoke() path, not just whichever one is easiest to test.

Local dev and CI must run the SAME test suite against the SAME LocalStack
setup — see Deployment.md. A test that only passes in CI but not locally
(or vice versa) indicates an environment-detection bug, not an acceptable
inconsistency.

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