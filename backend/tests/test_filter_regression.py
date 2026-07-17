"""
Regression tests for filter wiring — ensures query params actually filter data.

The root cause was:
  (a) Frontend: date range Select was static, never wired to state/API calls.
  (c) Backend: categories.py accepted from/to params but used CAT# keys (no date dimension),
      so date filtering was silently ignored.

These tests verify the backend returns filtered data when query params are provided.
"""

BATCH = "filter-test-batch"


def _seed_all(aws_mock):
    """Seed Aggregates + Reviews with data spanning multiple weeks and categories."""
    agg = aws_mock.Table("Aggregates")
    rev = aws_mock.Table("Reviews")

    # TREND data — 2 categories, 3 weeks
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "TREND#Electronics#2025-W03", "positive": 10, "negative": 3, "neutral": 5})
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "TREND#Electronics#2025-W04", "positive": 8, "negative": 2, "neutral": 4})
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "TREND#Electronics#2025-W05", "positive": 12, "negative": 1, "neutral": 3})
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "TREND#Books#2025-W03", "positive": 7, "negative": 1, "neutral": 2})
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "TREND#Books#2025-W04", "positive": 5, "negative": 0, "neutral": 3})

    # CAT data (no date dimension)
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "CAT#Electronics", "positive": 30, "negative": 6, "neutral": 12})
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "CAT#Books", "positive": 12, "negative": 1, "neutral": 5})

    # ISSUE data
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "ISSUE#battery_drain#2025-W03", "count": 5})
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "ISSUE#battery_drain#2025-W05", "count": 3})
    agg.put_item(Item={"batch_id": BATCH, "agg_type": "ISSUE#sizing_fit#2025-W03", "count": 2})

    # Review items — span two weeks
    for i, (date, sent, cat) in enumerate([
        ("2025-01-15", "positive", "Electronics"),  # W03
        ("2025-01-16", "negative", "Electronics"),   # W03
        ("2025-01-22", "positive", "Electronics"),   # W04
        ("2025-01-28", "neutral", "Books"),           # W05
    ]):
        rev.put_item(Item={
            "review_id": f"rev-{i}",
            "batch_id": BATCH,
            "text": f"Review {i}",
            "category": cat,
            "review_date": date,
            "sentiment": sent,
            "confidence_margin": "0.5",
            "prob_negative": "0.1",
            "prob_neutral": "0.2",
            "prob_positive": "0.7",
        })


# ── /api/trends — date range filtering ──────────────────────────────────────

def test_trends_date_filter_narrows_results(client, aws_mock):
    """Trends with from/to should return only weeks in that range, not the full dataset."""
    _seed_all(aws_mock)
    # W03 = 2025-01-13..19, W04 = 2025-01-20..26, W05 = 2025-01-27..
    # Query only W03 range
    resp = client.get(f"/api/trends?batch_id={BATCH}&from=2025-01-13&to=2025-01-19")
    body = resp.json()
    assert body["success"] is True
    weeks = body["data"]["weeks"]
    assert all(w["week"] == "2025-W03" for w in weeks), f"Expected only W03, got {weeks}"


def test_trends_no_date_filter_returns_all(client, aws_mock):
    """Trends without from/to should return all weeks."""
    _seed_all(aws_mock)
    resp = client.get(f"/api/trends?batch_id={BATCH}")
    body = resp.json()
    week_names = {w["week"] for w in body["data"]["weeks"]}
    assert "2025-W03" in week_names
    assert "2025-W04" in week_names
    assert "2025-W05" in week_names


# ── /api/categories/summary — date range filtering ──────────────────────────

def test_categories_date_filter_narrows_results(client, aws_mock):
    """Categories with from/to should use TREND# keys and return only matching data."""
    _seed_all(aws_mock)
    # Only W03 range — should include Electronics (10+3+5=18) and Books (7+1+2=10)
    resp = client.get(f"/api/categories/summary?batch_id={BATCH}&from=2025-01-13&to=2025-01-19")
    body = resp.json()
    assert body["success"] is True
    cats = {c["category"]: c for c in body["data"]["categories"]}
    assert "Electronics" in cats
    # W03 Electronics: pos=10, neg=3, neu=5 → total=18
    assert cats["Electronics"]["total"] == 18
    # Should NOT include W04/W05 data
    if "Electronics" in cats:
        assert cats["Electronics"]["positive"] == 10  # only W03


def test_categories_no_date_filter_uses_cat_keys(client, aws_mock):
    """Categories without from/to should use fast CAT# path and return full totals."""
    _seed_all(aws_mock)
    resp = client.get(f"/api/categories/summary?batch_id={BATCH}")
    body = resp.json()
    cats = {c["category"]: c for c in body["data"]["categories"]}
    # CAT# has total positive=30 for Electronics
    assert cats["Electronics"]["positive"] == 30


# ── /api/issues/distribution — date range filtering ──────────────────────────

def test_issues_date_filter_narrows_results(client, aws_mock):
    """Issues with from/to should return only matching weeks."""
    _seed_all(aws_mock)
    # Only W03 range
    resp = client.get(f"/api/issues/distribution?batch_id={BATCH}&from=2025-01-13&to=2025-01-19")
    body = resp.json()
    issues = {i["issue_tag"]: i["count"] for i in body["data"]["issues"]}
    assert issues.get("battery_drain") == 5  # only W03 count
    assert issues.get("sizing_fit") == 2


def test_issues_no_date_returns_all(client, aws_mock):
    """Issues without from/to should aggregate across all weeks."""
    _seed_all(aws_mock)
    resp = client.get(f"/api/issues/distribution?batch_id={BATCH}")
    body = resp.json()
    issues = {i["issue_tag"]: i["count"] for i in body["data"]["issues"]}
    assert issues.get("battery_drain") == 8  # 5 + 3


# ── /api/reviews — date range + sentiment filtering ──────────────────────────

def test_reviews_date_filter(client, aws_mock):
    """Reviews with from/to should return only reviews in that date range."""
    _seed_all(aws_mock)
    resp = client.get(f"/api/reviews?batch_id={BATCH}&from=2025-01-15&to=2025-01-16")
    body = resp.json()
    assert body["success"] is True
    reviews = body["data"]["reviews"]
    dates = {r["review_date"] for r in reviews}
    assert all(d >= "2025-01-15" and d <= "2025-01-16" for d in dates)
    assert len(reviews) == 2  # only the 2 reviews on Jan 15-16


def test_reviews_sentiment_filter(client, aws_mock):
    """Reviews with sentiment filter should return only matching sentiment."""
    _seed_all(aws_mock)
    resp = client.get(f"/api/reviews?batch_id={BATCH}&sentiment=negative")
    body = resp.json()
    reviews = body["data"]["reviews"]
    assert all(r["sentiment"] == "negative" for r in reviews)
    assert len(reviews) == 1


def test_reviews_category_filter(client, aws_mock):
    """Reviews with category filter should return only that category."""
    _seed_all(aws_mock)
    resp = client.get(f"/api/reviews?batch_id={BATCH}&category=Books")
    body = resp.json()
    reviews = body["data"]["reviews"]
    assert all(r["category"] == "Books" for r in reviews)
    assert len(reviews) == 1
