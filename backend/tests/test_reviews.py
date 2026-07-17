"""
Tests for GET /api/reviews and GET /api/reviews/:id.

Purpose: Verify review list (filters, pagination, sort) and single review detail.
         All list queries require batch_id.
"""

from unittest.mock import patch

BATCH = "batch-1"


def _seed_reviews(aws_mock, count=5):
    """Seed Reviews table with test data including batch-scoped GSI sort keys."""
    table = aws_mock.Table("Reviews")
    reviews = []
    for i in range(count):
        sentiment = "negative" if i % 3 == 0 else ("neutral" if i % 3 == 1 else "positive")
        category = "Electronics" if i % 2 == 0 else "Books"
        review_date = f"2025-01-{15 + i:02d}"
        item = {
            "review_id": f"rev-{i}",
            "batch_id": BATCH,
            "text": f"Review text {i}",
            "category": category,
            "review_date": review_date,
            "processed_at": f"2025-01-20T00:00:{i:02d}Z",
            "sentiment": sentiment,
            "confidence_margin": str(0.5 + i * 0.1),
            "prob_negative": "0.3",
            "prob_neutral": "0.3",
            "prob_positive": "0.4",
            "batch_cat_sort": f"{category}#{review_date}",
        }
        if sentiment == "negative":
            item["issue_tag"] = "general_dissatisfaction"
            item["issue_distance"] = "0.3"
            item["cluster_source"] = "cross_category_fallback"
            item["batch_issue_sort"] = f"general_dissatisfaction#{review_date}"
        if i == 0:
            item["extra_columns"] = {"rating": "4", "helpful_votes": "10"}
        table.put_item(Item=item)
        reviews.append(item)
    return reviews


def test_reviews_no_filters(client, aws_mock):
    _seed_reviews(aws_mock)
    with patch("routers.reviews.cache_get", return_value=None), patch("routers.reviews.cache_set"):
        resp = client.get(f"/api/reviews?batch_id={BATCH}")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total"] == 5
    assert len(body["data"]["reviews"]) == 5


def test_reviews_sentiment_filter(client, aws_mock):
    _seed_reviews(aws_mock)
    with patch("routers.reviews.cache_get", return_value=None), patch("routers.reviews.cache_set"):
        resp = client.get(f"/api/reviews?batch_id={BATCH}&sentiment=negative")
    body = resp.json()
    assert body["success"] is True
    for r in body["data"]["reviews"]:
        assert r["sentiment"] == "negative"


def test_reviews_category_filter(client, aws_mock):
    _seed_reviews(aws_mock)
    with patch("routers.reviews.cache_get", return_value=None), patch("routers.reviews.cache_set"):
        resp = client.get(f"/api/reviews?batch_id={BATCH}&category=Electronics")
    body = resp.json()
    for r in body["data"]["reviews"]:
        assert r["category"] == "Electronics"


def test_reviews_pagination(client, aws_mock):
    _seed_reviews(aws_mock, count=10)
    with patch("routers.reviews.cache_get", return_value=None), patch("routers.reviews.cache_set"):
        resp = client.get(f"/api/reviews?batch_id={BATCH}&page=1&limit=3")
    body = resp.json()
    assert len(body["data"]["reviews"]) == 3
    assert body["data"]["total"] == 10
    assert body["data"]["total_pages"] == 4

    with patch("routers.reviews.cache_get", return_value=None), patch("routers.reviews.cache_set"):
        resp2 = client.get(f"/api/reviews?batch_id={BATCH}&page=4&limit=3")
    body2 = resp2.json()
    assert len(body2["data"]["reviews"]) == 1


def test_reviews_sort_order(client, aws_mock):
    _seed_reviews(aws_mock)
    with patch("routers.reviews.cache_get", return_value=None), patch("routers.reviews.cache_set"):
        resp = client.get(f"/api/reviews?batch_id={BATCH}")
    reviews = resp.json()["data"]["reviews"]
    dates = [r["review_date"] for r in reviews]
    assert dates == sorted(dates, reverse=True)


def test_reviews_missing_batch_id(client, aws_mock):
    resp = client.get("/api/reviews?sentiment=negative")
    assert resp.status_code == 422


def test_reviews_empty_result(client, aws_mock):
    with patch("routers.reviews.cache_get", return_value=None), patch("routers.reviews.cache_set"):
        resp = client.get(f"/api/reviews?batch_id={BATCH}&sentiment=negative")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["reviews"] == []
    assert body["data"]["total"] == 0


def test_reviews_combined_filters(client, aws_mock):
    _seed_reviews(aws_mock)
    with patch("routers.reviews.cache_get", return_value=None), patch("routers.reviews.cache_set"):
        resp = client.get(f"/api/reviews?batch_id={BATCH}&sentiment=negative&category=Electronics")
    body = resp.json()
    for r in body["data"]["reviews"]:
        assert r["sentiment"] == "negative"
        assert r["category"] == "Electronics"


def test_reviews_includes_extra_columns(client, aws_mock):
    _seed_reviews(aws_mock)
    with patch("routers.reviews.cache_get", return_value=None), patch("routers.reviews.cache_set"):
        resp = client.get(f"/api/reviews?batch_id={BATCH}&sentiment=negative")
    body = resp.json()
    # rev-0 is negative and has extra_columns
    rev0 = [r for r in body["data"]["reviews"] if r["review_id"] == "rev-0"]
    assert len(rev0) == 1
    assert rev0[0]["rating"] == "4"
    assert rev0[0]["helpful_votes"] == "10"


def test_review_detail_valid_id(client, aws_mock):
    _seed_reviews(aws_mock)
    resp = client.get("/api/reviews/rev-0")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["review_id"] == "rev-0"
    assert body["data"]["text"] == "Review text 0"
    # Extra columns should be flattened
    assert body["data"]["rating"] == "4"


def test_review_detail_unknown_id(client, aws_mock):
    resp = client.get("/api/reviews/nonexistent")
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "NOT_FOUND"
