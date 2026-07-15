# Project Requirements

## What We Are Building

**AI Customer Review Analytics Platform**

A business intelligence dashboard that ingests raw Amazon product reviews,
runs them through a production NLP pipeline (sentiment classification +
issue detection), and surfaces actionable insights to business owners —
trend charts, category breakdowns, issue tag distributions, and a filterable
review feed. Not a toy demo. Not a Jupyter notebook. A real, deployable product.

The ML backend is already built and tested:
- Sentiment: BGE-small → MLP + asymmetric threshold (79% macro F1, deployed on Lambda)
- Issue detection: KMeans clustering on negative reviews (15 clusters, named, calibrated)
- Deployment: quantized ONNX + pure numpy, no torch/sklearn in Lambda, <250MB

What remains: the product layer on top of that working backend.

---

## Targeted Users

### Primary: Small-to-mid e-commerce business owners
- Sell products on Amazon or similar marketplaces
- Receive hundreds to thousands of reviews per month
- Do NOT have a data science team
- Currently read reviews manually or ignore them entirely
- Need: "what are my customers unhappy about this week, and is it getting worse?"

### Secondary: E-commerce managers / analysts at mid-size brands
- Responsible for product quality, customer experience, or marketing
- Have some data literacy but are not engineers
- Need: exportable reports, date-range comparisons, per-category breakdowns

### Not targeted (explicitly out of scope for v1):
- Enterprise companies with existing BI tools
- Developers wanting API access only (no dashboard UI)
- Non-English review sources

---

## Features

### v1 (Build This First)

#### Review Ingestion
- Upload CSV of reviews (columns: text, category, date, optional rating)
- Batch processing through Lambda pipeline (sentiment + issue tag per review)
- Results stored in database, associated with the upload batch

#### Sentiment Dashboard
- Sentiment trend chart: weekly positive/neutral/negative % over selected date range
- Category breakdown table: each category ranked by sentiment score, with delta vs prior period
- Summary cards: total reviews, % negative, most complained-about issue this week

#### Issue Detection Panel
- Bar chart: issue tag distribution for negative reviews in selected period
- Drill-down: click an issue tag → see the actual reviews behind it
- Per-category filter: "show issue tags for Electronics only"

#### Review Feed
- Paginated list of individual reviews with sentiment label, issue tag, confidence margin
- Filters: sentiment, category, issue tag, date range, confidence level
- Sort by: date, confidence margin (most certain predictions first/last)

#### Alerts (simple, rule-based)
- "Negative reviews up X% vs last week in [category]"
- "New issue tag [breakage_and_damage] appearing more frequently"
- Shown as a strip at top of dashboard, not email (v1)

### v2 (After v1 Ships)

#### Per-Category Issue Clustering
- Cluster negative reviews within each category separately (not global)
- Volume gate: only for categories with 500+ negative reviews
- Fallback to global clusters for smaller categories
- `cluster_source` field in every review record (per_category vs cross_category_fallback)

#### Topic Modeling
- BERTopic on the shared BGE embedding (same encoder, no retraining)
- Complements issue tags with discovered topics rather than predefined clusters
- Shown as "what customers talk about" word clouds per category

#### Trend Alerts via Email
- Weekly digest: biggest sentiment movers, new issue spikes
- User configures threshold (e.g. alert if negative rate > 30%)

### v3 (Later, not promised)
- Semantic search: "find all reviews mentioning packaging damage"
- Recommendation signals from positive review embeddings
- Knowledge graph: product attributes extracted from review text
- Multi-source ingestion (Shopify, Google reviews, app stores)
