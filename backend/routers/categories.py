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

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter, Query

from database import get_tables
from models import ApiResponse

router = APIRouter(prefix="/api")


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
