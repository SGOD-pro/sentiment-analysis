# Sentrix: AI Customer Review Analytics & Issue Detection

> A production-grade NLP pipeline that transforms raw Amazon product reviews into structured business intelligence — sentiment trends, issue detection, and actionable category breakdowns — all deployable on AWS Lambda within a strict 250MB budget.

## Business Impact & What We Built

Most sentiment analysis projects stop at "positive / negative / neutral" and call it done. This one doesn't. 

We built a platform that answers the questions a business owner actually needs answered on a Monday morning:
- **Which of my product categories had a spike in negative reviews this week?**
- **When customers complain, what are they complaining about specifically?**
- **Is the neutral sentiment on my new product launch genuinely mixed, or just weak positive?**

The core of the system is a **shared embedding pipeline** that feeds three downstream tasks simultaneously from a single inference pass. The entire pipeline is deployed serverless on AWS Lambda and operates at near-instant speed.

---

## The Story: How We Built It (And Why)

### 1. The Sentiment Model & The Asymmetric Threshold
Our goal was to accurately predict Sentiment. We started with a frozen BGE-small-en-v1.5 encoder and trained a 2-layer MLP on top of the embeddings. 

Instead of using a standard `argmax` that treats all prediction errors equally, we applied a business-centric **asymmetric threshold**. For a business dashboard, missing a genuinely negative review was identified as the costliest error. When our model predicts "positive" but its confidence margin over the second-place class is below 0.30, it defaults to "neutral". Negative predictions are never softened. 
**Result:** Negative recall is protected at ~0.795 while neutral recall improved measurably.

### 2. Why ONNX?
AWS Lambda has severe deployment size limits (250MB unzipped max). We couldn't use the standard HuggingFace `transformers` or `PyTorch` libraries because their massive dependency footprints (1GB+) would instantly exceed the limit.

Instead, we exported the BGE-small encoder to the **ONNX format** and quantized it to INT8. This shrank the embedding model down to **~35MB** and allowed us to run it using only `onnxruntime` (a very lightweight C++ backend). This easily fits within our Lambda budget and dramatically speeds up both inference and cold starts.

### 3. Why we didn't just save the MLP model (Saving raw weights instead)
Even though we trained the MLP classifier in PyTorch, we didn't save the full model object (`torch.save`) because loading it in Lambda would still require importing PyTorch. 

Instead, we manually extracted the raw weights and biases from the trained PyTorch MLP into a tiny **`mlp_weights.npz`** file (~0.8MB). At inference time, we load these weights and run the forward pass manually using pure `numpy`:
`np.maximum(0, X @ W1 + b1) @ W2 + b2`
This keeps our Lambda footprint incredibly small and blazing fast, relying on nothing but standard math operations.

### 4. Why Clustering for Issue Detection instead of a new Classification Model?
We needed a way to tag *why* customers were leaving negative reviews (e.g., "sizing_and_fit", "breakage_and_damage"). We could have trained another supervised classification model, but that would require a massive, manually labeled dataset of specific complaints, which we didn't have.

Instead, we used unsupervised **KMeans clustering** on the embeddings of the negative reviews:
- We clustered 68,000 negative reviews into 15 centroids. 
- We manually inspected the clusters and assigned human-readable names.
- We saved these coordinates into a tiny **`issue_centroids.npy`** file (~22KB).

At inference time, if a review is classified as negative, we calculate the cosine distance between its embedding and the 15 cluster centroids using pure `numpy`. If it's close enough (distance < 0.70), it gets tagged with the issue. 
This gave us a highly effective issue detection system with **zero labeled training data required** and practically **zero extra overhead** at inference time (since we reuse the exact same embedding from the sentiment pass).

---

## Architecture

```text
Review Text
     │
     ▼
Text Preprocessing
(HTML unescape, URL removal, punctuation normalization)
     │
     ▼
Quantized ONNX Encoder (BGE-small-en-v1.5, INT8, ~35MB)
     │
     ▼
384-dimensional embedding
     │
     ├──► Pure Numpy MLP Classifier + Asymmetric Threshold ──► Sentiment (neg / neu / pos)
     │
     └──► Pure Numpy Nearest-Centroid Lookup ───────────────► Issue Tag (only on neg reviews)
              (Distance threshold=0.70)
```

**One embedding, two outputs, one Lambda function.**

## Stack
- **Embedding:** BGE-small-en-v1.5 (frozen)
- **Inference Format:** ONNX Runtime (INT8 quantized) + pure Numpy
- **Deployment:** AWS Lambda + API Gateway + DynamoDB
- **Backend:** FastAPI (Python)
- **Frontend:** React + TypeScript + Tailwind CSS v4 + Recharts (Vite)

---

## ML Pipeline Retraining & Validation (CI/CD)

The project includes an automated closed-loop ML retraining pipeline using GitHub Actions (`ml_retrain.yml`). When users make sentiment corrections in the dashboard, these corrections are stored in DynamoDB and can be used to continuously improve the model.

### How to Retrain and Validate
1. **Navigate to GitHub Actions** and select the **ML Retrain** workflow.
2. **Trigger the workflow manually**. You can configure the `min_corrections` threshold (default is 10) to avoid retraining on statistically insignificant data.
3. **Automated Export & Retrain**: The pipeline exports human corrections from DynamoDB and merges them with the core training dataset. New texts are automatically embedded using the same ONNX encoder, and the MLP is retrained.
4. **Dual Quality Gates**: The newly trained model is validated against two strict gates:
   - **Gate 1 (Standard Test Set)**: Negative recall must remain `>= 0.7915`.
   - **Gate 2 (Frozen Eval Slice)**: Neutral recall must remain `>= 0.2967` (evaluated against a hand-verified 300-row difficult-neutral set).
5. **Conditional Deployment**:
   - If **both gates pass**, the new `mlp_weights.npz` artifact is uploaded to S3, and the main CI/CD deployment pipeline is triggered automatically. The new model goes live with zero manual intervention.
   - If **either gate fails**, the pipeline halts successfully (preventing a degradation in production) and the artifacts remain unchanged.

### Required S3 Artifacts Setup
The CI/CD pipeline deploys automatically, but the **ML Retraining pipeline requires the original training data** to exist in S3. 

While the inference artifacts (`model_quantized.onnx`, `mlp_weights.npz`, `issue_centroids.npy`) are already stored in S3 for deployment under `s3://<bucket>/ml-artifacts/lambda_deploy_artifacts/`, you must **newly upload the following training splits** into `s3://<bucket>/ml-artifacts/training_data/` before running the retrain pipeline for the first time:

- `bge_clean_embeddings.npy` (full embedding matrix)
- `bge_clean_metadata.parquet` (metadata + corrected labels)
- `clean_train_idx_v4.npy` (train split indices, leak-free)
- `clean_test_idx_v4.npy` (test split indices, DO NOT modify)
- `difficult_neutral_eval_FROZEN.csv` (300-row frozen eval, DO NOT modify)

