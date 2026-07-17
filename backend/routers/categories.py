"""
Categories summary router.

Purpose: Return per-category sentiment summary from Aggregates table, scoped to a batch.
Input: Query params: batch_id (required), from, to (ISO dates).
Output: Ranked list of categories by sentiment score.
Dependencies: database, models
Example:
    GET /api/categories/summary?batch_id=abc&from=2025-01-01&to=2025-03-31
    → {"success": true, "data": {"categories": [{"category": "Electronics", ...}]}}
"""

from datetime import datetime

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter, Query

from database import get_tables
from models import ApiResponse

router = APIRouter(prefix="/api")


def _iso_to_week(date_str: str) -> str:
    """Convert ISO date to ISO week string."""
    dt = datetime.fromisoformat(date_str)
    return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"


@router.get("/categories/summary", response_model=ApiResponse)
def categories_summary(
    batch_id: str = Query(...),
    date_from: str = Query(alias="from", default=""),
    date_to: str = Query(alias="to", default=""),
):
    """Return per-category sentiment summary ranked by sentiment score."""
    if not batch_id:
        return ApiResponse(success=False, error_code="MISSING_PARAM", message="batch_id is required")

    tables = get_tables()
    use_date_filter = bool(date_from or date_to)

    if use_date_filter:
        # TREND# keys have format TREND#{category}#{week} — use them for date filtering
        resp = tables.aggregates.query(
            KeyConditionExpression=Key("batch_id").eq(batch_id) & Key("agg_type").begins_with("TREND#"),
        )
        items = resp.get("Items", [])

        week_from = _iso_to_week(date_from) if date_from else ""
        week_to = _iso_to_week(date_to) if date_to else ""

        categories = {}
        for item in items:
            parts = item["agg_type"].split("#")  # TREND#category#week
            if len(parts) != 3:
                continue
            cat, week = parts[1], parts[2]
            if week_from and week < week_from:
                continue
            if week_to and week > week_to:
                continue
            if cat not in categories:
                categories[cat] = {"category": cat, "positive": 0, "neutral": 0, "negative": 0}
            for sentiment in ("positive", "neutral", "negative"):
                categories[cat][sentiment] += int(item.get(sentiment, 0))
    else:
        # No date filter — fast path using CAT# keys
        resp = tables.aggregates.query(
            KeyConditionExpression=Key("batch_id").eq(batch_id) & Key("agg_type").begins_with("CAT#"),
        )
        items = resp.get("Items", [])

        categories = {}
        for item in items:
            cat = item["agg_type"].split("#", 1)[1]
            if cat not in categories:
                categories[cat] = {"category": cat, "positive": 0, "neutral": 0, "negative": 0}
            for sentiment in ("positive", "neutral", "negative"):
                categories[cat][sentiment] += int(item.get(sentiment, 0))

    # Compute sentiment score: (positive - negative) / total, handle zero division
    result = []
    for cat_data in categories.values():
        total = cat_data["positive"] + cat_data["neutral"] + cat_data["negative"]
        if total > 0:
            cat_data["total"] = total
            cat_data["sentiment_score"] = round(
                (cat_data["positive"] - cat_data["negative"]) / total, 4
            )
        else:
            cat_data["total"] = 0
            cat_data["sentiment_score"] = 0.0
        result.append(cat_data)

    result.sort(key=lambda c: c["sentiment_score"], reverse=True)

    return ApiResponse(success=True, data={"categories": result})
