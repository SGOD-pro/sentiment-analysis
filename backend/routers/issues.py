"""
Issues distribution router.

Purpose: Return issue tag distribution from Aggregates table.
Input: Query params: from, to (ISO dates), category (optional).
Output: Issue tag counts for negative reviews.
Dependencies: database, models
Example:
    GET /api/issues/distribution?from=2025-01-01&to=2025-03-31
    → {"success": true, "data": {"issues": [{"issue_tag": "sizing_and_fit", "count": 42}]}}
"""

from datetime import datetime

from fastapi import APIRouter, Query

from database import get_tables
from models import ApiResponse

router = APIRouter(prefix="/api")


def _iso_to_week(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str)
    return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"


@router.get("/issues/distribution", response_model=ApiResponse)
def issues_distribution(
    date_from: str = Query(alias="from", default=""),
    date_to: str = Query(alias="to", default=""),
    category: str = Query(default=""),
):
    """Return issue tag counts for negative reviews."""
    tables = get_tables()

    week_from = _iso_to_week(date_from) if date_from else ""
    week_to = _iso_to_week(date_to) if date_to else ""

    # ponytail: scan on small Aggregates table
    resp = tables.aggregates.scan()
    items = resp.get("Items", [])

    issue_counts = {}
    for item in items:
        key = item["agg_key"]
        if not key.startswith("ISSUE#"):
            continue

        parts = key.split("#")  # ISSUE#tag#week
        if len(parts) != 3:
            continue

        tag, week = parts[1], parts[2]

        if week_from and week < week_from:
            continue
        if week_to and week > week_to:
            continue

        count = int(item.get("value", 0))
        issue_counts[tag] = issue_counts.get(tag, 0) + count

    # If category filter, we need to cross-reference with Reviews table
    # ponytail: category filtering for issues requires scanning Reviews GSI
    # For v1, skip category filter on issues (Aggregates don't store per-category issue data)
    # Upgrade: add ISSUE#tag#category#week aggregate keys

    issues = [{"issue_tag": tag, "count": count} for tag, count in issue_counts.items()]
    issues.sort(key=lambda i: i["count"], reverse=True)

    return ApiResponse(success=True, data={"issues": issues})
