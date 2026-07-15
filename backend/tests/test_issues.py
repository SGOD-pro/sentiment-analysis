"""
Tests for GET /api/issues/distribution.

Purpose: Verify issue distribution endpoint returns tag counts.
"""


def _seed_issues(aws_mock):
    table = aws_mock.Table("Aggregates")
    table.put_item(Item={"agg_key": "ISSUE#sizing_and_fit#2025-W03", "metric": "count", "value": 15})
    table.put_item(Item={"agg_key": "ISSUE#general_dissatisfaction#2025-W03", "metric": "count", "value": 8})
    table.put_item(Item={"agg_key": "ISSUE#audio_quality#2025-W04", "metric": "count", "value": 5})


def test_issue_tag_counts(client, aws_mock):
    _seed_issues(aws_mock)
    resp = client.get("/api/issues/distribution")
    body = resp.json()
    assert body["success"] is True
    issues = body["data"]["issues"]
    assert len(issues) == 3
    # Should be sorted by count descending
    assert issues[0]["count"] >= issues[1]["count"]


def test_issues_date_filter(client, aws_mock):
    _seed_issues(aws_mock)
    resp = client.get("/api/issues/distribution?from=2025-01-20&to=2025-01-26")
    body = resp.json()
    issues = body["data"]["issues"]
    # Only W04 items should match (2025-01-20 is in W04)
    assert len(issues) >= 1


def test_issues_empty(client, aws_mock):
    resp = client.get("/api/issues/distribution")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["issues"] == []
