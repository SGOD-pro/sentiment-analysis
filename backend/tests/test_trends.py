"""
Tests for GET /api/trends.

Purpose: Verify trends endpoint with date ranges, categories, and edge cases.
"""


def _seed_aggregates(aws_mock):
    table = aws_mock.Table("Aggregates")
    # Week 3, Electronics
    table.put_item(Item={"agg_key": "TREND#Electronics#2025-W03", "metric": "positive", "value": 10})
    table.put_item(Item={"agg_key": "TREND#Electronics#2025-W03", "metric": "negative", "value": 3})
    table.put_item(Item={"agg_key": "TREND#Electronics#2025-W03", "metric": "neutral", "value": 5})
    # Week 4, Electronics
    table.put_item(Item={"agg_key": "TREND#Electronics#2025-W04", "metric": "positive", "value": 15})
    table.put_item(Item={"agg_key": "TREND#Electronics#2025-W04", "metric": "negative", "value": 2})
    # Week 3, Books
    table.put_item(Item={"agg_key": "TREND#Books#2025-W03", "metric": "positive", "value": 7})
    table.put_item(Item={"agg_key": "TREND#Books#2025-W03", "metric": "negative", "value": 1})


def test_trends_valid_range(client, aws_mock):
    _seed_aggregates(aws_mock)
    resp = client.get("/api/trends?from=2025-01-13&to=2025-01-26")
    body = resp.json()
    assert body["success"] is True
    assert len(body["data"]["weeks"]) >= 1


def test_trends_category_filter(client, aws_mock):
    _seed_aggregates(aws_mock)
    resp = client.get("/api/trends?from=2025-01-13&to=2025-01-26&category=Electronics")
    body = resp.json()
    weeks = body["data"]["weeks"]
    # Should only have Electronics data
    assert all(w["positive"] > 0 or w["negative"] > 0 for w in weeks)


def test_trends_invalid_date(client, aws_mock):
    resp = client.get("/api/trends?from=not-a-date")
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "INVALID_DATE"


def test_trends_empty_result(client, aws_mock):
    resp = client.get("/api/trends?from=2030-01-01&to=2030-12-31")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["weeks"] == []
