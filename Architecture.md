# Architecture

## App Flow

```
User uploads CSV of reviews
         │
         ▼
Frontend (React)
validates CSV columns, shows preview
         │
         ▼
POST /api/upload
Backend (FastAPI / Node)
stores raw reviews in DB,
queues batch job
         │
         ▼
Batch processor
calls Lambda for each batch of reviews (50-100 at a time)
         │
         ▼
AWS Lambda (250MB, cold start <3s)
│
├── Text preprocessing
│     (HTML unescape, URL removal, Unicode normalize,
│      punctuation normalize, truncate long reviews)
│
├── ONNX Runtime (BGE-small INT8)
│     produces 384-dim embedding
│
├── MLP forward pass (pure numpy)
│     + asymmetric threshold (pos_margin=0.30)
│     → sentiment: negative / neutral / positive
│
└── Nearest-centroid lookup (numpy, KMeans centroids)
      only runs when sentiment == negative
      → issue_tag + issue_distance
         │
         ▼
Results written back to DB
(sentiment, issue_tag, cluster_source, confidence_margin, probabilities)
         │
         ▼
Dashboard queries aggregated endpoints
(trends, category breakdown, issue distribution, review feed)
```

---

## Folder and File Structure

```
sentiment-analytics/
│
├── ml/                             # All ML work, already done
│   ├── data/
│   │   ├── text_preprocessing.py
│   │   ├── text_preprocessing_strict.py
│   │   ├── movie_review_filter.py
│   │   ├── label_correction_loop.py
│   │   ├── merge_confidence_audit.py
│   │   ├── reaudit_merge_and_freeze_eval.py
│   │   ├── rebuild_split_v3.py
│   │   └── generate_splits.py
│   ├── training/
│   │   ├── colab_embed.py
│   │   ├── save_final_mlp.py
│   │   └── issue_clustering.py
│   ├── evaluation/
│   │   ├── error_review.py
│   │   ├── model_swap_test.py
│   │   ├── confidence_threshold_test.py
│   │   ├── asymmetric_threshold_test.py
│   │   ├── neutral_confidence_check.py
│   │   ├── evaluate_v4_and_frozen.py
│   │   ├── final_three_model_comparison.py
│   │   └── [other experiment scripts]
│   └── export/
│       ├── export_for_lambda.py
│       └── export_mlp_and_clusters.py
│
├── lambda/                         # Production inference, already working
│   ├── handler.py                  # lambda_handler_final.py
│   └── artifacts/
│       ├── bge_onnx_quantized/     # ONLY this ships to Lambda - NOT bge_onnx_fp32
│       │   ├── model_quantized.onnx
│       │   ├── tokenizer.json
│       │   └── [tokenizer files]
│       ├── mlp_weights.npz
│       ├── issue_centroids.npy
│       └── config.json
│
├── backend/                        # To build in Phase 2
│   ├── main.py                     # FastAPI entry point
│   ├── routers/
│   │   ├── upload.py               # POST /api/upload
│   │   ├── reviews.py              # GET /api/reviews
│   │   ├── trends.py               # GET /api/trends
│   │   ├── categories.py           # GET /api/categories/summary
│   │   └── issues.py               # GET /api/issues/distribution
│   ├── models/
│   │   ├── review.py               # DB model
│   │   └── batch.py                # Upload batch model
│   ├── services/
│   │   ├── lambda_client.py        # calls AWS Lambda in batches
│   │   ├── aggregation.py          # trend/summary queries
│   │   └── preprocessing.py        # lightweight text clean before Lambda
│   ├── db/
│   │   ├── database.py             # connection + session
│   │   └── migrations/
│   └── requirements.txt
│
├── frontend/                       # To build in Phase 3
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard/
│   │   │   │   ├── SentimentTrendChart.tsx
│   │   │   │   ├── CategoryBreakdownTable.tsx
│   │   │   │   ├── IssueDistributionChart.tsx
│   │   │   │   ├── SummaryCards.tsx
│   │   │   │   └── AlertStrip.tsx
│   │   │   ├── ReviewFeed/
│   │   │   │   ├── ReviewFeed.tsx
│   │   │   │   ├── ReviewCard.tsx
│   │   │   │   └── FilterPanel.tsx
│   │   │   └── Upload/
│   │   │       ├── UploadCSV.tsx
│   │   │       └── UploadPreview.tsx
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Reviews.tsx
│   │   │   └── Upload.tsx
│   │   ├── hooks/
│   │   │   ├── useTrends.ts
│   │   │   ├── useCategories.ts
│   │   │   └── useReviews.ts
│   │   ├── api/
│   │   │   └── client.ts           # axios/fetch wrapper
│   │   └── types/
│   │       └── index.ts
│   ├── package.json
│   └── tsconfig.json
│
├── PROJECT.md
├── ProjectRequirement.md
├── Architecture.md
├── Rules.md
└── Phases.md
```

---

## Database Schema (DynamoDB)

No GROUP BY in DynamoDB — swap loses that for free. `Aggregates` table added below to carry the load the old SQL indexes used to do, updated incrementally via DynamoDB Streams instead of computed at query time.

```
Table: Reviews
  PK: review_id (S, UUID)

  Attributes:
    batch_id            S
    external_id         S    -- original ID from uploaded CSV
    category             S
    text                  S
    clean_text            S
    rating                N    -- original star rating if available
    review_date           S    -- ISO date from CSV
    processed_at           S

    sentiment              S    -- negative / neutral / positive
    confidence_margin      N
    prob_negative           N
    prob_neutral             N
    prob_positive             N

    issue_tag                 S    -- absent if sentiment != negative
    issue_distance             N
    cluster_source              S    -- per_category / cross_category_fallback

  GSI1 (batch-sentiment-index):
    PK: batch_id   SK: sentiment
    -- batch-scoped review feed filtered by sentiment

  GSI2 (category-date-index):
    PK: category   SK: review_date
    -- category + date-range queries, feeds trends/category endpoints

  GSI3 (issue-date-index):
    PK: issue_tag   SK: review_date
    -- sparse index: only items with issue_tag set (negative reviews)
    -- feeds issue distribution endpoint

Table: Batches
  PK: batch_id (S, UUID)

  Attributes:
    uploaded_at        S
    filename            S
    total_reviews        N
    processed_count       N
    status                  S    -- pending / processing / done / failed

Table: Aggregates
  PK: agg_key (S)   -- e.g. "TREND#Electronics#2025-W03"
  SK: metric (S)    -- "negative" / "neutral" / "positive" / "count"

  Attributes:
    value        N
    updated_at    S

  -- Populated incrementally: DynamoDB Stream on Reviews -> small Lambda
  -- increments the relevant agg_key/metric counter on every write.
  -- This is what /api/trends, /api/categories/summary, and
  -- /api/issues/distribution actually read from — never a table scan.
```

---

## API Endpoints

```
POST   /api/upload                  Upload CSV, returns batch_id
GET    /api/batches/:id/status      Check processing progress

GET    /api/trends                  ?from=&to=&category=
GET    /api/categories/summary      ?from=&to=
GET    /api/issues/distribution     ?from=&to=&category=
GET    /api/reviews                 ?sentiment=&category=&issue_tag=&from=&to=&page=&limit=
GET    /api/reviews/:id             Single review detail
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| ML inference | ONNX Runtime + numpy | No torch in Lambda, 250MB budget |
| Serverless inference | AWS Lambda | Scales to zero, fits budget |
| API Gateway | AWS API Gateway | Routes HTTP to Lambda |
| Backend API | FastAPI (Python) | Same language as ML code, fast to build |
| Backend package mgmt | `uv` + `uv venv` | Fast installs, one lockfile, no pip/venv juggling |
| Database | DynamoDB | Native AWS, serverless-aligned, pay-per-request — no separate DB server to run alongside Lambda |
| Local AWS emulation | floci | Run DynamoDB + Lambda locally before touching real AWS |
| Frontend | React + TypeScript | Standard, component model suits dashboard |
| Charts | Recharts | Lightweight, React-native, no D3 overhead |
| Styling | Tailwind CSS | Fast, consistent, no custom CSS sprawl |
| Deployment (backend) | Lambda | Simple, cheap |
| Deployment (frontend) | Vercel | Free tier, instant deploys |
| File uploads | S3 | Don't store CSVs in the DB |

---

## Data Flow for Dashboard Queries

```
Frontend                Backend                   DynamoDB
   │                       │                          │
   │  GET /api/trends       │                          │
   │  ?from=2025-01-01      │                          │
   │  &to=2025-01-31        │                          │
   │  &category=Electronics │                          │
   ├──────────────────────► │                          │
   │                        │  Query Aggregates table  │
   │                        │  PK: TREND#Electronics#* │
   │                        │  SK between week range   │
   │                        ├─────────────────────────►│
   │                        │◄─────────────────────────┤
   │◄───────────────────────┤                          │
   │  { weeks: [...],        │                          │
   │    negative: [...],     │                          │
   │    neutral: [...],      │                          │
   │    positive: [...] }    │                          │
```
