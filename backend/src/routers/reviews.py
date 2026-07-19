"""
Reviews router — paginated, filterable review feed + single review detail.

Purpose: Return paginated reviews with filters, scoped to a batch, and single review lookup.
Input: Query params for list: batch_id (required), sentiment, category, issue_tag, from, to, page, limit.
       Path param for detail: review_id.
Output: Paginated review list or single review.
Dependencies: database, models, cache
Example:
    GET /api/reviews?batch_id=abc&sentiment=negative&category=Electronics&page=1&limit=20
    GET /api/reviews/abc-123
"""

import json

from boto3.dynamodb.conditions import Attr, Key
from fastapi import APIRouter, Query

from cache import cache_get, cache_set
from database import get_tables
from models import ApiResponse

router = APIRouter(prefix="/api")


def _build_cache_key(batch_id: str, filters: dict) -> str:
    """Build a deterministic cache key from request params."""
    sorted_filters = json.dumps(filters, sort_keys=True, default=str)
    return f"reviews:{batch_id}:{sorted_filters}"


@router.get("/reviews", response_model=ApiResponse)
def list_reviews(
    batch_id: str = Query(...),
    sentiment: str = Query(default=""),
    category: str = Query(default=""),
    issue_tag: str = Query(default=""),
    date_from: str = Query(alias="from", default=""),
    date_to: str = Query(alias="to", default=""),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    Return paginated, filtered reviews scoped to batch_id.

    Uses batch-sentiment-index GSI for batch scoping + optional sentiment filter.
    Results cached in Redis when available.
    """
    if not batch_id:
        return ApiResponse(success=False, error_code="MISSING_PARAM", message="batch_id is required")

    # Check Redis cache
    filter_params = {
        "sentiment": sentiment, "category": category, "issue_tag": issue_tag,
        "from": date_from, "to": date_to, "page": page, "limit": limit,
    }
    cache_key = _build_cache_key(batch_id, filter_params)
    cached = cache_get(cache_key)
    if cached is not None:
        return ApiResponse(success=True, data=cached)

    tables = get_tables()

    # Use batch-sentiment-index for batch scoping
    # If sentiment filter is set, use it as the sort key condition
    if sentiment:
        query_kwargs = {
            "IndexName": "batch-sentiment-index",
            "KeyConditionExpression": Key("batch_id").eq(batch_id) & Key("sentiment").eq(sentiment),
        }
    else:
        query_kwargs = {
            "IndexName": "batch-sentiment-index",
            "KeyConditionExpression": Key("batch_id").eq(batch_id),
        }

    # Build additional filter expression for non-key attributes
    filter_parts = []
    if category:
        filter_parts.append(Attr("category").eq(category))
    if issue_tag:
        filter_parts.append(Attr("issue_tag").eq(issue_tag))
    if date_from:
        filter_parts.append(Attr("review_date").gte(date_from))
    if date_to:
        filter_parts.append(Attr("review_date").lte(date_to))

    filter_expr = None
    for part in filter_parts:
        filter_expr = part if filter_expr is None else filter_expr & part

    if filter_expr is not None:
        query_kwargs["FilterExpression"] = filter_expr

    # Collect all matching items (DynamoDB query pagination)
    all_items = []
    while True:
        resp = tables.reviews.query(**query_kwargs)
        all_items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        query_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    # Sort by review_date descending
    all_items.sort(key=lambda r: r.get("review_date", ""), reverse=True)

    # Paginate
    total = len(all_items)
    start = (page - 1) * limit
    page_items = all_items[start : start + limit]

    # Flatten extra_columns into each review
    reviews = []
    for item in page_items:
        review = {k: v for k, v in item.items() if k != "extra_columns"}
        if "extra_columns" in item:
            review.update(item["extra_columns"])
        reviews.append(review)

    result_data = {
        "reviews": reviews,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit if total > 0 else 0,
    }

    # Cache the result
    cache_set(cache_key, result_data)

    return ApiResponse(success=True, data=result_data)


@router.get("/reviews/{review_id}", response_model=ApiResponse)
def get_review(review_id: str, batch_id: str):
    """Return a single review by ID within the requested batch."""
    tables = get_tables()
    resp = tables.reviews.get_item(Key={"review_id": review_id})
    item = resp.get("Item")

    if not item or item.get("batch_id") != batch_id:
        return ApiResponse(success=False, error_code="NOT_FOUND", message="Review not found")

    review = {k: v for k, v in item.items() if k != "extra_columns"}
    if "extra_columns" in item:
        review.update(item["extra_columns"])

    return ApiResponse(success=True, data=review)
