"""
Tests for GET /api/trends.

Purpose: Verify trends endpoint with batch_id, date ranges, categories, and edge cases.
"""

BATCH = "test-batch-1"


def _seed_aggregates(aws_mock):
    table = aws_mock.Table("Aggregates")
    # Week 3, Electronics
    table.put_item(Item={"batch_id": BATCH, "agg_type": "TREND#Electronics#2025-W03", "positive": 10, "negative": 3, "neutral": 5})
    # Week 4, Electronics
    table.put_item(Item={"batch_id": BATCH, "agg_type": "TREND#Electronics#2025-W04", "positive": 15, "negative": 2})
    # Week 3, Books
    table.put_item(Item={"batch_id": BATCH, "agg_type": "TREND#Books#2025-W03", "positive": 7, "negative": 1})


def test_trends_valid_range(client, aws_mock):
    _seed_aggregates(aws_mock)
    resp = client.get(f"/api/trends?batch_id={BATCH}&from=2025-01-13&to=2025-01-26")
    body = resp.json()
    assert body["success"] is True
    assert len(body["data"]["weeks"]) >= 1


def test_trends_category_filter(client, aws_mock):
    _seed_aggregates(aws_mock)
    resp = client.get(f"/api/trends?batch_id={BATCH}&from=2025-01-13&to=2025-01-26&category=Electronics")
    body = resp.json()
    weeks = body["data"]["weeks"]
    assert all(w["positive"] > 0 or w["negative"] > 0 for w in weeks)


def test_trends_missing_batch_id(client, aws_mock):
    resp = client.get("/api/trends?from=2025-01-13")
    assert resp.status_code == 422  # FastAPI validation error for missing required param


def test_trends_invalid_date(client, aws_mock):
    resp = client.get(f"/api/trends?batch_id={BATCH}&from=not-a-date")
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "INVALID_DATE"


def test_trends_empty_result(client, aws_mock):
    resp = client.get(f"/api/trends?batch_id={BATCH}&from=2030-01-01&to=2030-12-31")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["weeks"] == []
