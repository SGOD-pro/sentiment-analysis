# Deployment.md — Local Dev & AWS Orchestration Strategy

Two Lambda functions. One deployment target (AWS). One local dev strategy
that mirrors production behavior without requiring AWS credentials for
day-to-day coding.

---

## The Core Problem This Solves

`BackendFunction` calls `MLInferenceFunction` via `boto3.client("lambda").invoke()`
in production — a real AWS SDK call resolved by IAM + function name, no URL
involved (see Architecture.md's Lambda-to-Lambda section).

Locally, there is no real Lambda runtime to invoke. `lambda_client.py` must
behave identically in shape (same function signature, same return type)
while actually running the ML code as a plain Python import in dev, and a
real cross-Lambda `boto3.invoke()` in production. This file defines exactly
how that branch is decided and configured, in both environments, so nobody
has to guess again.

---

## Environment Matrix

| Environment | `AWS_ENDPOINT_URL` | ML inference happens via | Requires AWS credentials? |
|---|---|---|---|
| Local dev (default) | unset / `http://localhost:4566` | Direct Python import of `lambda/handler.py` | No |
| CI (pytest) | `http://localhost:4566` (LocalStack) | Direct Python import of `lambda/handler.py` | No (LocalStack mock creds) |
| Staging/Prod | unset (real AWS) | `boto3.client("lambda").invoke()` | Yes (IAM role) |

**The branch condition in `lambda_client.py` is exactly this table, encoded
in one `if`:** `if settings.aws_endpoint_url and "localhost" in settings.aws_endpoint_url`.
Nothing else should determine this branch. Do not add a second flag, do not
add an `ENVIRONMENT == "development"` check elsewhere that could disagree
with this one — one source of truth for "are we really talking to AWS."

---

## Local Dev Setup (No AWS Account Needed)

### 1. Environment file

`backend/.env` (copy from `.env.example`, never commit the real one):

```
ENVIRONMENT=development
AWS_ENDPOINT_URL=http://localhost:4566
AWS_DEFAULT_REGION=ap-south-1
S3_BUCKET_NAME=sentimetric-dev
DYNAMODB_REVIEWS_TABLE=ReviewsDev
DYNAMODB_BATCHES_TABLE=BatchesDev
DYNAMODB_AGGREGATES_TABLE=AggregatesDev
ML_INFERENCE_FUNCTION_NAME=unused-in-local-mode
LOG_LEVEL=DEBUG
```

`ML_INFERENCE_FUNCTION_NAME` is present but genuinely unused when the
localhost branch fires — `lambda_client.py` never reads it in that path.
Kept in `.env` only so the same `config.py` schema works in both
environments without conditional required-fields logic.

### 2. LocalStack for S3 + DynamoDB (not for Lambda-to-Lambda)

LocalStack mocks S3 and DynamoDB locally so `database.py` and the upload
flow work without hitting real AWS. It does NOT need to mock Lambda — the
ML inference bypass in `lambda_client.py` skips LocalStack's Lambda service
entirely and runs `lambda/handler.py` as a direct import instead. This is
simpler and faster than trying to make LocalStack emulate a real Lambda
Docker container for every code change during dev.

```bash
docker run -d -p 4566:4566 -e SERVICES=s3,dynamodb localstack/localstack
```

Create local tables/bucket once (see `backend/scripts/local_bootstrap.py`,
build this as part of Phase 11 — see Phases.md).

### 3. Run the backend

```bash
cd backend
uv sync
uv run uvicorn src.main:app --reload --port 8000
```

`GET /health` should respond immediately — this route never touches
`MLInferenceFunction` or `lambda/handler.py` at all, confirming the thin
function stays thin even locally.

Upload a CSV through `/api/upload` — this exercises the full local-bypass
path: `upload.py` → `batch_processor.py` → `lambda_client.invoke_lambda()`
→ direct import of `lambda/handler.py` → real ONNX/MLP inference running
in-process, no AWS Lambda involved, results written to LocalStack DynamoDB.

**This means local dev runs the REAL model, not a mock.** The only thing
mocked is the network boundary (no real Lambda invocation), not the ML
logic itself. This is deliberate — catches real prediction bugs during
local dev, not just plumbing bugs.

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev
```

`.env.local`:
```
VITE_API_BASE_URL=http://localhost:8000
```

---

## Staging/Production Setup (Real AWS)

### 1. One-time manual setup (not automated, done once per AWS account)

- Create the S3 bucket (`aws s3 mb s3://sentimetric-prod-storage`)
- Upload ML artifacts to S3 (see the artifact sync note in `ci_cd_pipeline.yml`'s
  comments — one-time `aws s3 sync` from your local `lambda_deploy_artifacts/`)
- Confirm IAM user permissions (already verified — see Rules.md IAM section)

### 2. Everything else is automated via CI/CD

`sam deploy` creates both Lambda functions, the DynamoDB tables (via
`AWS::DynamoDB::Table` resources — see note below), the S3 bucket
reference, API Gateway, and all IAM roles/policies from one `template.yaml`.
No manual AWS Console clicking required after the one-time setup above.

**Gap to close:** current `template.yaml` does not yet declare DynamoDB
table resources — it only references table names via `Parameters` and
grants CRUD policy on those names. This means the tables must currently
be created manually before first deploy, OR the template needs
`AWS::DynamoDB::Table` resources added. Decide and implement in Phase 11
(see Phases.md) — recommended: add table resources to the template so
`sam deploy` is fully self-contained and reproducible from zero.

### 3. The real Lambda-to-Lambda call

In production, `lambda_client.py`'s non-localhost branch fires:

```python
client = boto3.client("lambda", region_name=settings.aws_region)
response = client.invoke(
    FunctionName=settings.lambda_function_name,  # resolved from
                                                    # ML_INFERENCE_FUNCTION_NAME
                                                    # env var, injected by
                                                    # template.yaml's
                                                    # !Ref MLInferenceFunction
    InvocationType="RequestResponse",
    Payload=json.dumps({"texts": texts}).encode(),
)
```

No URL. No CORS. Authorized entirely by `BackendFunction`'s IAM execution
role (`LambdaInvokePolicy` in `template.yaml`), resolved by function name,
not network address. See Architecture.md's "Lambda-to-Lambda Orchestration"
section for the full request path diagram.

---

## CI/CD Orchestration (Both Environments, One Pipeline)

The existing `ci_cd_pipeline.yml` already separates concerns correctly —
extending it here to be explicit about how local-shaped tests and
real-AWS deploys coexist in one workflow.

```
detect-changes
      │
      ├──► test-backend (runs against LocalStack, NOT real AWS)
      │         - starts LocalStack service container in the runner
      │         - runs pytest suite, including a real call through
      │           lambda_client.py's localhost branch (imports
      │           lambda/handler.py directly, same as local dev)
      │         - this is the ONE place CI proves the ML handler code
      │           actually produces correct predictions, without ever
      │           touching real AWS Lambda
      │
      ├──► test-ml (verifies lambda/handler.py in isolation)
      │         - installs ONLY onnxruntime/numpy/tokenizers
      │         - asserts torch/transformers/sklearn are absent
      │         - runs handler.py's own local test block directly
      │
      ├──► test-frontend (lint, type-check, build)
      │
      ├──► deploy-backend (needs test-backend + test-ml both passing)
      │         - THIS is where real AWS credentials are used
      │         - pulls ML artifacts from S3 (not LocalStack)
      │         - sam build + sam deploy — creates/updates BOTH
      │           Lambda functions in one CloudFormation stack
      │         - this is the ONLY job in the entire pipeline that
      │           touches real AWS Lambda deployment
      │
      └──► deploy-frontend (needs test-frontend passing)
                - resolves the real API Gateway URL from CloudFormation
                - injects it into the Vercel build
```

**Key principle:** CI tests never deploy anything and never require real
AWS Lambda to exist yet — they run entirely against LocalStack (S3,
DynamoDB) plus direct Python imports (ML handler), exactly mirroring what
a developer's laptop does. Only the `deploy-*` jobs touch real AWS, and
only after tests pass. This means a broken AWS deploy never blocks a
developer from working locally, and a local-only bug is always caught
before it reaches the deploy step.

### Add LocalStack to `test-backend` job (gap to close)

Current `ci_cd_pipeline.yml`'s `test-backend` job runs against real AWS
via secrets (see `env:` block in the existing workflow) — this should
switch to LocalStack for the same reason local dev uses it: tests should
not require real AWS credentials or touch real AWS resources just to
verify the API works.

```yaml
test-backend:
  services:
    localstack:
      image: localstack/localstack
      ports:
        - 4566:4566
      env:
        SERVICES: s3,dynamodb
  env:
    AWS_ENDPOINT_URL: http://localhost:4566
    AWS_ACCESS_KEY_ID: test
    AWS_SECRET_ACCESS_KEY: test
    AWS_DEFAULT_REGION: ap-south-1
    S3_BUCKET_NAME: sentimetric-test
    DYNAMODB_REVIEWS_TABLE: ReviewsTest
    DYNAMODB_BATCHES_TABLE: BatchesTest
    DYNAMODB_AGGREGATES_TABLE: AggregatesTest
```

Real AWS secrets (`AWS_ACCESS_KEY_ID` referencing the actual account) are
only used in `deploy-backend` and `deploy-frontend` jobs going forward —
not in `test-backend`. This is a real change from the current workflow
and should be implemented in Phase 11.

---

## Verification Checklist (Run Before Calling v1 Deployed)

1. `docker run` LocalStack, run backend locally, upload a CSV, confirm
   real sentiment predictions come back (proves local-bypass path works)
2. Push a commit, confirm CI runs `test-backend` against LocalStack
   (not real AWS), confirm it passes without requiring deploy secrets
3. Manually trigger `deploy-backend` via `workflow_dispatch`, confirm
   both Lambda functions appear in AWS Console after `sam deploy`
4. Check `BackendFunction`'s IAM execution role in AWS Console — confirm
   it has an inline or attached policy granting `lambda:InvokeFunction`
   scoped to `MLInferenceFunction`'s ARN
5. Hit the real API Gateway URL's `/health` — confirm fast response
   (proves `BackendFunction` at 256MB, no ONNX loaded, is genuinely fast)
6. Upload a real CSV through the deployed API Gateway URL, confirm
   results appear correctly (proves the real cross-Lambda `boto3.invoke()`
   path works, not just the local bypass)
7. Confirm `MLInferenceFunction` has zero entries in API Gateway routes
   (proves it's genuinely unreachable from outside)
