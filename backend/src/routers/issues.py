"""
Issues distribution router.

Purpose: Return issue tag distribution from Aggregates table, scoped to a batch.
Input: Query params: batch_id (required), from, to (ISO dates), category (optional).
Output: Issue tag counts for negative reviews.
Dependencies: database, models
Example:
    GET /api/issues/distribution?batch_id=abc&from=2025-01-01&to=2025-03-31
    → {"success": true, "data": {"issues": [{"issue_tag": "sizing_and_fit", "count": 42}]}}
"""

from datetime import datetime

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter, Query

from database import get_tables
from models import ApiResponse

router = APIRouter(prefix="/api")


def _iso_to_week(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str)
    return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"


@router.get("/issues/distribution", response_model=ApiResponse)
def issues_distribution(
    batch_id: str = Query(...),
    date_from: str = Query(alias="from", default=""),
    date_to: str = Query(alias="to", default=""),
    category: str = Query(default=""),
):
    """Return issue tag counts for negative reviews, scoped to batch_id."""
    if not batch_id:
        return ApiResponse(success=False, error_code="MISSING_PARAM", message="batch_id is required")

    tables = get_tables()

    week_from = _iso_to_week(date_from) if date_from else ""
    week_to = _iso_to_week(date_to) if date_to else ""

    resp = tables.aggregates.query(
        KeyConditionExpression=Key("batch_id").eq(batch_id) & Key("agg_type").begins_with("ISSUE#"),
    )
    items = resp.get("Items", [])

    issue_counts = {}
    for item in items:
        agg_type = item["agg_type"]
        parts = agg_type.split("#")  # ISSUE#tag#week
        if len(parts) != 3:
            continue

        tag, week = parts[1], parts[2]

        if week_from and week < week_from:
            continue
        if week_to and week > week_to:
            continue

        count = int(item.get("count", 0))
        issue_counts[tag] = issue_counts.get(tag, 0) + count

    # ponytail: category filtering for issues requires scanning Reviews GSI
    # For v1, skip category filter on issues (Aggregates don't store per-category issue data)
    # Upgrade: add ISSUE#tag#category#week aggregate keys

    issues = [{"issue_tag": tag, "count": count} for tag, count in issue_counts.items()]
    issues.sort(key=lambda i: i["count"], reverse=True)

    return ApiResponse(success=True, data={"issues": issues})
