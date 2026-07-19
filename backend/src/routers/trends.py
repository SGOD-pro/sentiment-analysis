"""
Trends router — weekly sentiment counts.

Purpose: Return weekly sentiment breakdown from Aggregates table, scoped to a batch.
Input: Query params: batch_id (required), from, to (ISO dates), category (optional).
Output: List of weekly sentiment counts.
Dependencies: database, models, datetime
Example:
    GET /api/trends?batch_id=abc&from=2025-01-01&to=2025-03-31&category=Electronics
    → {"success": true, "data": {"weeks": [{"week": "2025-W03", "positive": 10, ...}]}}
"""

from datetime import datetime

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter, Query

from database import get_tables
from logger import get_logger
from models import ApiResponse

router = APIRouter(prefix="/api")
log = get_logger(__name__)


def _iso_to_week(date_str: str) -> str:
    """Convert ISO date to ISO week string."""
    dt = datetime.fromisoformat(date_str)
    return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"


def _validate_date(date_str: str) -> bool:
    """Check if string is a valid ISO date."""
    try:
        datetime.fromisoformat(date_str)
        return True
    except (ValueError, TypeError):
        return False


@router.get("/trends", response_model=ApiResponse)
def get_trends(
    batch_id: str = Query(...),
    date_from: str = Query(alias="from", default=""),
    date_to: str = Query(alias="to", default=""),
    category: str = Query(default=""),
):
    """Return weekly sentiment counts from Aggregates table, scoped to batch_id."""
    if not batch_id:
        return ApiResponse(success=False, error_code="MISSING_PARAM", message="batch_id is required")
    if date_from and not _validate_date(date_from):
        return ApiResponse(success=False, error_code="INVALID_DATE", message=f"Invalid 'from' date: {date_from}")
    if date_to and not _validate_date(date_to):
        return ApiResponse(success=False, error_code="INVALID_DATE", message=f"Invalid 'to' date: {date_to}")

    tables = get_tables()

    # Query aggregates by batch_id, filter for TREND# prefix
    resp = tables.aggregates.query(
        KeyConditionExpression=Key("batch_id").eq(batch_id) & Key("agg_type").begins_with("TREND#"),
    )
    items = resp.get("Items", [])

    # Filter to matching category and date range
    week_from = _iso_to_week(date_from) if date_from else ""
    week_to = _iso_to_week(date_to) if date_to else ""

    weekly = {}
    for item in items:
        agg_type = item["agg_type"]
        parts = agg_type.split("#")  # TREND#category#week
        if len(parts) != 3:
            continue

        item_cat, item_week = parts[1], parts[2]

        if category and item_cat != category:
            continue
        if week_from and item_week < week_from:
            continue
        if week_to and item_week > week_to:
            continue

        if item_week not in weekly:
            weekly[item_week] = {"week": item_week, "positive": 0, "neutral": 0, "negative": 0}

        for sentiment in ("positive", "neutral", "negative"):
            weekly[item_week][sentiment] += int(item.get(sentiment, 0))

    weeks_sorted = sorted(weekly.values(), key=lambda w: w["week"])

    return ApiResponse(success=True, data={"weeks": weeks_sorted})
