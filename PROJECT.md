# AI Customer Review Analytics Platform

> Production-grade NLP pipeline that transforms raw Amazon product reviews into
> structured business intelligence — sentiment trends, issue detection, and
> actionable category breakdowns — deployable on AWS Lambda within a 250MB budget.

---

## What This Actually Is

Most sentiment analysis projects stop at "positive / negative / neutral" and call it done.
This one doesn't.

The platform answers the questions a business owner actually needs answered on a Monday morning:

- **Which of my product categories had a spike in negative reviews this week?**
- **When customers complain, what are they complaining about specifically?**
- **Is the neutral sentiment on my new product launch genuinely mixed, or just weak positive?**

Sentiment classification is one module. The full system is a shared embedding
pipeline that feeds three downstream tasks simultaneously from a single inference pass.

---

## Architecture

```
Review Text
     │
     ▼
Text Preprocessing
(HTML unescape, URL removal, punctuation normalization,
 Unicode normalization, elongation collapse)
     │
     ▼
Quantized ONNX Encoder (BGE-small-en-v1.5, INT8, ~35MB)
     │
     ▼
384-dimensional embedding
     │
     ├──► MLP Classifier + Asymmetric Threshold ──► Sentiment (neg / neutral / pos)
     │
     └──► Nearest-Centroid Lookup ──► Issue Tag (only on negative reviews)
              (KMeans, K=15, distance threshold=0.70)
```

**One embedding, two outputs, one Lambda function.**

---

## The Sentiment Model

### What we actually built

| Stage | What happened |
|---|---|
| Data | Amazon Reviews 2023, 34 categories, ~203k reviews, balanced 3-class labels |
| Labels | Derived from star ratings (1-2★ = negative, 3★ = neutral, 4-5★ = positive), then audited |
| Audit | 9,990 rows manually reviewed across two targeted rounds using confidence-margin sampling |
| Label taxonomy | mislabel / genuinely_mixed / model_wrong (with 6-way error subtype) |
| Frozen eval | 300-row permanently-held-out difficult-neutral slice, never trained on |
| Final model | BGE-small-en-v1.5 (frozen) → 2-layer MLP (256→128→3) + asymmetric confidence threshold |
| Deployment | Quantized ONNX encoder + raw numpy MLP forward pass, no torch/sklearn in Lambda |

### Why not DistilBERT / finetuned encoder

Tested. Multiple times. Results:

| Model | Neg recall | Neutral recall | Frozen recall |
|---|---|---|---|
| LR baseline (frozen BGE) | 0.769 | 0.713 | 0.293 |
| MLP + asymmetric threshold ✓ | **0.795** | **0.743** | **0.297** |
| DistilBERT finetune (v4) | 0.772 | 0.729 | 0.213 |
| Finetuned BGE (w/ early stopping) | 0.786 | 0.753 | 0.270 |

MLP on frozen embeddings beats or ties full transformer finetuning on every
metric that matters, at a fraction of the deployment cost. The result held across
13 independent experiments. This is the documented, evidence-backed reason this
architecture was chosen — not a default.

### The asymmetric threshold

Standard argmax treats every prediction equally. This one doesn't.

When the model predicts positive but its margin over the second-place class is
below 0.30, it defaults to neutral instead. Negative predictions are never
softened — because missing a genuinely negative review was identified as the
costlier error for the target use case (business owner dashboard).

Result: negative recall protected at ~0.795 while neutral recall improves by
~3 points over plain argmax. Measurable, deliberate, documented tradeoff.

---

## The Issue Detection System

Unsupervised. No labeled training data required.

KMeans clustering on BGE embeddings of negative reviews (68,967 rows), K=15,
distance threshold 0.70 (calibrated at 95th percentile of distance-to-centroid
distribution). Reviews farther than 0.70 from every centroid return "other"
rather than a forced weak match.

**Named clusters (human-verified):**

| Cluster | Name | Size |
|---|---|---|
| 0 | sizing_and_fit | 4,834 |
| 1 | audio_and_music_quality | 3,594 |
| 2 | food_taste_and_pet | 3,849 |
| 3 | software_and_app_issues | 2,814 |
| 4 | content_quality | 3,443 |
| 5 | color_and_appearance | 5,159 |
| 7 | product_malfunction | 5,027 |
| 8 | general_dissatisfaction | 7,123 |
| 9 | value_and_price | 7,719 |
| 10 | durability_and_build | 4,963 |
| 12 | scent_and_smell | 4,479 |
| 13 | order_and_fulfillment | 4,750 |
| 14 | breakage_and_damage | 6,202 |
| 6, 11 | other (noise clusters) | — |

---

## Lambda Deployment

**Package contents:**
```
lambda_deploy_artifacts/
  handler.py
  bge_onnx_quantized/
    model_quantized.onnx    (~35MB)
    tokenizer.json
    tokenizer_config.json
    vocab.txt
  mlp_weights.npz           (~0.8MB)
  issue_centroids.npy       (~22KB)
  config.json               (<1KB)
```

**Dependencies (Lambda layer):**
```
onnxruntime
numpy
tokenizers
```

No PyTorch. No transformers. No sklearn. Total package well within 250MB.

**API contract:**
```json
POST /analyze
{
  "texts": ["review text 1", "review text 2"]
}

Response:
{
  "results": [
    {
      "text": "...",
      "sentiment": "negative",
      "sentiment_confidence_margin": 0.979,
      "sentiment_probabilities": {"negative": 0.989, "neutral": 0.010, "positive": 0.0001},
      "issue_tag": "value_and_price",
      "issue_distance": 0.661
    }
  ]
}
```

---

## Dashboard Features (Planned)

### Core (build first)
- **Sentiment trend chart** — weekly/monthly positive/neutral/negative ratios per category
- **Category breakdown** — which product lines have the worst/best sentiment this period
- **Issue tag distribution** — for negative reviews, what are customers complaining about
- **Review feed** — filterable by sentiment, category, issue tag, date range

### Filters
- Date range
- Product category (all 34 or subset)
- Sentiment label
- Issue tag
- Confidence margin (exclude low-confidence predictions)

### Metrics shown per category
- Sentiment score (weighted positive rate)
- Negative trend (delta vs prior period)
- Top issue tag this week
- Review volume

### Later (not yet built, not promised)
- Topic modeling (BERTopic on the shared embedding)
- Semantic search (vector DB + embedding lookup)
- Recommendation engine
- Knowledge graph

These are legitimate next steps that reuse the same encoder without retraining.
They are not in scope for the current version and are not listed as features
in the deployed dashboard.

---

## What Makes This Different From Other Sentiment Projects

**Most portfolio sentiment projects:**
- Take a Kaggle dataset
- Fine-tune BERT
- Report accuracy on a random split
- Stop

**This one:**
1. **Found and fixed a test-set leakage bug mid-project** (the 83% vs 78% discovery — documented with receipts)
2. **Built a manual audit taxonomy** across 9,990 hand-reviewed rows using confidence-margin sampling, not random sampling
3. **Separated irreducible label ambiguity from model error** — discovered 43% of the model's "confident mistakes" were correctly-labeled reviews the model was miscalibrated on, not mislabels to fix
4. **Made a deliberate, documented deployment tradeoff** — asymmetric threshold chosen because "missing a negative review" was identified as the costlier business error, not because it maximized the headline metric
5. **Systematically ruled out 13 alternative approaches** (data volume scaling, 4 model architectures, correction strategies, finetuned encoder) with real controlled experiments before settling on the final architecture
6. **Has a permanently frozen difficult-neutral evaluation slice** (300 rows, never trained on) that tracks hard-case performance separately from the standard test set

---

## Repository Structure

```
├── ml/
│   ├── preprocessing/
│   │   └── text_preprocessing.py
│   ├── training/
│   │   ├── save_final_mlp.py
│   │   └── issue_clustering.py
│   ├── evaluation/
│   │   ├── confidence_threshold_test.py
│   │   └── evaluate_v4_and_frozen.py
│   └── export/
│       ├── export_for_lambda.py
│       └── export_mlp_and_clusters.py
├── lambda/
│   └── handler.py
├── dashboard/          (next phase)
├── PROJECT.md
└── README.md
```

---

## Stack

| Layer | Technology |
|---|---|
| Embedding | BGE-small-en-v1.5 (BAAI), frozen |
| Inference format | ONNX Runtime (INT8 quantized) |
| Sentiment classifier | 2-layer MLP (pure numpy at inference) |
| Issue detection | KMeans (sklearn, offline only) |
| Deployment | AWS Lambda + API Gateway |
| Database | DynamoDB (serverless, pay-per-request) |
| Backend | FastAPI (Python), uv package manager |
| Frontend | React + TypeScript |
| UI components | shadcn/ui (toasts, alerts, tables, dialogs) |
| Charts | Recharts |
| Styling | Tailwind CSS v4.1 |
| Frontend deploy | Vercel |
| File storage | S3 |
| Data | Amazon Reviews 2023 (McAuley Lab) |

---

## Known Limitations and Fix Plans

### Issue Detection: Cross-Category Signal Contamination

**The problem:** KMeans clustering on general-purpose sentence embeddings picks up
product-domain signal alongside complaint-type signal. A "music" cluster forms not
because those reviews share a complaint type, but because they share vocabulary
(songs, album, sound). This means some clusters are product-category artifacts,
not genuine issue taxonomies.

**Evidence from the actual run:** Clusters 1 (audio/music), 4 and 11 (books/movies)
were clearly domain-grouped, not complaint-grouped. Cluster 6 was a genuine grab-bag
with no coherent theme. These were labeled "other" rather than forced into misleading names.

**The fix (designed, not yet built):**
- Cluster within each product category separately for categories with 500+ negative reviews
- Use cross-category clustering as fallback for smaller categories and reviews with no category metadata
- Track which path each review took via a `cluster_source` field (`per_category` vs `cross_category_fallback`)
- This prevents a dashboard showing "breakage_and_damage" for a music review that happened to sit near the wrong centroid

**Why it's not built yet:** The per-category approach multiplies the manual cluster-naming
work by the number of categories. 34 categories × naming exercise = real effort.
The cross-category version is live, works for the majority of categories, and is
honestly labeled in the output. The per-category version is the documented next step,
not a hidden gap.

### Neutral Class Ceiling

The neutral class has a structural ambiguity problem: star-rating-derived labels
(3★ = neutral) don't reliably map to "neutral sentiment in the text." A 3-star
review that opens with "great product" and closes with "but the shipping was
terrible" is genuinely ambiguous. No amount of model tuning resolves an ambiguous
label. Neutral F1 peaks around 0.75 with the asymmetric threshold, compared to
0.95 for binary positive/negative classification. This ceiling is documented and
understood, not a bug awaiting a fix.

### Out-of-Distribution Text

This model was trained on Amazon product reviews in English across 34 categories.
It is not a general-purpose sentiment engine. Different domains (Twitter, support
tickets, app reviews) have different vocabulary, length distributions, and sentiment
expression patterns. Performance on out-of-domain text is untested and should not
be assumed without validation.

- [x] Data pipeline and preprocessing
- [x] Embedding generation (BGE-small, 203k reviews)
- [x] Sentiment model training and validation (13 experiments, documented)
- [x] Manual label audit (9,990 rows, 2 rounds, confidence-based sampling)
- [x] Issue detection clustering (K=15, human-named, calibrated threshold)
- [x] Lambda handler (sentiment + issue, single embedding pass)
- [x] ONNX export and INT8 quantization
- [x] End-to-end local test passing
- [ ] AWS Lambda deployment
- [ ] API Gateway setup
- [ ] Dashboard (React frontend)
- [ ] Per-category clustering (volume-gated)
