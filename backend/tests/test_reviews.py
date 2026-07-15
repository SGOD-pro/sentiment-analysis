"""
Tests for GET /api/reviews and GET /api/reviews/:id.

Purpose: Verify review list (filters, pagination, sort) and single review detail.
"""

from decimal import Decimal


def _seed_reviews(aws_mock, count=5):
    """Seed Reviews table with test data."""
    table = aws_mock.Table("Reviews")
    reviews = []
    for i in range(count):
        sentiment = "negative" if i % 3 == 0 else ("neutral" if i % 3 == 1 else "positive")
        item = {
            "review_id": f"rev-{i}",
            "batch_id": "batch-1",
            "text": f"Review text {i}",
            "category": "Electronics" if i % 2 == 0 else "Books",
            "review_date": f"2025-01-{15 + i:02d}",
            "processed_at": f"2025-01-20T00:00:{i:02d}Z",
            "sentiment": sentiment,
            "confidence_margin": str(0.5 + i * 0.1),
            "prob_negative": "0.3",
            "prob_neutral": "0.3",
            "prob_positive": "0.4",
        }
        if sentiment == "negative":
            item["issue_tag"] = "general_dissatisfaction"
            item["issue_distance"] = "0.3"
            item["cluster_source"] = "cross_category_fallback"
        if i == 0:
            item["extra_columns"] = {"rating": "4", "helpful_votes": "10"}
        table.put_item(Item=item)
        reviews.append(item)
    return reviews


def test_reviews_no_filters(client, aws_mock):
    _seed_reviews(aws_mock)
    resp = client.get("/api/reviews")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["total"] == 5
    assert len(body["data"]["reviews"]) == 5


def test_reviews_sentiment_filter(client, aws_mock):
    _seed_reviews(aws_mock)
    resp = client.get("/api/reviews?sentiment=negative")
    body = resp.json()
    assert body["success"] is True
    for r in body["data"]["reviews"]:
        assert r["sentiment"] == "negative"


def test_reviews_category_filter(client, aws_mock):
    _seed_reviews(aws_mock)
    resp = client.get("/api/reviews?category=Electronics")
    body = resp.json()
    for r in body["data"]["reviews"]:
        assert r["category"] == "Electronics"


def test_reviews_pagination(client, aws_mock):
    _seed_reviews(aws_mock, count=10)
    resp = client.get("/api/reviews?page=1&limit=3")
    body = resp.json()
    assert len(body["data"]["reviews"]) == 3
    assert body["data"]["total"] == 10
    assert body["data"]["total_pages"] == 4

    resp2 = client.get("/api/reviews?page=4&limit=3")
    body2 = resp2.json()
    assert len(body2["data"]["reviews"]) == 1


def test_reviews_sort_order(client, aws_mock):
    _seed_reviews(aws_mock)
    resp = client.get("/api/reviews")
    reviews = resp.json()["data"]["reviews"]
    dates = [r["review_date"] for r in reviews]
    assert dates == sorted(dates, reverse=True)


def test_reviews_empty_result(client, aws_mock):
    resp = client.get("/api/reviews?sentiment=negative")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["reviews"] == []
    assert body["data"]["total"] == 0


def test_reviews_combined_filters(client, aws_mock):
    _seed_reviews(aws_mock)
    resp = client.get("/api/reviews?sentiment=negative&category=Electronics")
    body = resp.json()
    for r in body["data"]["reviews"]:
        assert r["sentiment"] == "negative"
        assert r["category"] == "Electronics"


def test_reviews_includes_extra_columns(client, aws_mock):
    _seed_reviews(aws_mock)
    resp = client.get("/api/reviews?sentiment=negative")
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
