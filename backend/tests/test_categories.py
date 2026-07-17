"""
Tests for GET /api/categories/summary.

Purpose: Verify categories endpoint returns ranked list and handles empty periods.
"""

BATCH = "test-batch-1"


def _seed_categories(aws_mock):
    table = aws_mock.Table("Aggregates")
    # Electronics: mostly positive
    table.put_item(Item={"batch_id": BATCH, "agg_type": "CAT#Electronics", "positive": 20, "negative": 5, "neutral": 5})
    # Books: mostly negative
    table.put_item(Item={"batch_id": BATCH, "agg_type": "CAT#Books", "positive": 3, "negative": 15, "neutral": 2})


def test_categories_ranked_list(client, aws_mock):
    _seed_categories(aws_mock)
    resp = client.get(f"/api/categories/summary?batch_id={BATCH}")
    body = resp.json()
    assert body["success"] is True
    cats = body["data"]["categories"]
    assert len(cats) == 2
    # Electronics should rank higher (better sentiment score)
    assert cats[0]["category"] == "Electronics"
    assert cats[1]["category"] == "Books"
    assert cats[0]["sentiment_score"] > cats[1]["sentiment_score"]


def test_categories_missing_batch_id(client, aws_mock):
    resp = client.get("/api/categories/summary")
    assert resp.status_code == 422


def test_categories_empty_period(client, aws_mock):
    resp = client.get(f"/api/categories/summary?batch_id={BATCH}")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["categories"] == []
