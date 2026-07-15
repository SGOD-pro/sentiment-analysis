"""
Tests for GET /api/categories/summary.

Purpose: Verify categories endpoint returns ranked list and handles empty periods.
"""


def _seed_categories(aws_mock):
    table = aws_mock.Table("Aggregates")
    # Electronics: mostly positive
    table.put_item(Item={"agg_key": "CAT#Electronics", "metric": "positive", "value": 20})
    table.put_item(Item={"agg_key": "CAT#Electronics", "metric": "negative", "value": 5})
    table.put_item(Item={"agg_key": "CAT#Electronics", "metric": "neutral", "value": 5})
    # Books: mostly negative
    table.put_item(Item={"agg_key": "CAT#Books", "metric": "positive", "value": 3})
    table.put_item(Item={"agg_key": "CAT#Books", "metric": "negative", "value": 15})
    table.put_item(Item={"agg_key": "CAT#Books", "metric": "neutral", "value": 2})


def test_categories_ranked_list(client, aws_mock):
    _seed_categories(aws_mock)
    resp = client.get("/api/categories/summary")
    body = resp.json()
    assert body["success"] is True
    cats = body["data"]["categories"]
    assert len(cats) == 2
    # Electronics should rank higher (better sentiment score)
    assert cats[0]["category"] == "Electronics"
    assert cats[1]["category"] == "Books"
    assert cats[0]["sentiment_score"] > cats[1]["sentiment_score"]


def test_categories_empty_period(client, aws_mock):
    resp = client.get("/api/categories/summary")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["categories"] == []
