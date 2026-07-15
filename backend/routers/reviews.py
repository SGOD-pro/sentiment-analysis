"""
Reviews router — paginated, filterable review feed + single review detail.

Purpose: Return paginated reviews with filters, and single review lookup.
Input: Query params for list: sentiment, category, issue_tag, from, to, page, limit.
       Path param for detail: review_id.
Output: Paginated review list or single review.
Dependencies: database, models
Example:
    GET /api/reviews?sentiment=negative&category=Electronics&page=1&limit=20
    GET /api/reviews/abc-123
"""

from boto3.dynamodb.conditions import Attr, Key
from fastapi import APIRouter, Query

from database import get_tables
from models import ApiResponse

router = APIRouter(prefix="/api")


@router.get("/reviews", response_model=ApiResponse)
def list_reviews(
    sentiment: str = Query(default=""),
    category: str = Query(default=""),
    issue_tag: str = Query(default=""),
    date_from: str = Query(alias="from", default=""),
    date_to: str = Query(alias="to", default=""),
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=100),
):
    """
    Return paginated, filtered reviews.

    Uses GSIs when possible, falls back to scan with filters.
    All original CSV columns are included via extra_columns.
    """
    tables = get_tables()

    # Build filter expression
    filter_parts = []
    if sentiment:
        filter_parts.append(Attr("sentiment").eq(sentiment))
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

    # ponytail: scan with filter — acceptable for v1 with moderate data volumes.
    # Upgrade path: use GSI queries (category-date-index, batch-sentiment-index)
    # based on which filters are present.
    scan_kwargs = {}
    if filter_expr is not None:
        scan_kwargs["FilterExpression"] = filter_expr

    # Collect all matching items (DynamoDB scan pagination)
    all_items = []
    while True:
        resp = tables.reviews.scan(**scan_kwargs)
        all_items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

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

    return ApiResponse(
        success=True,
        data={
            "reviews": reviews,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit if total > 0 else 0,
        },
    )


@router.get("/reviews/{review_id}", response_model=ApiResponse)
def get_review(review_id: str):
    """Return a single review by ID."""
    tables = get_tables()
    resp = tables.reviews.get_item(Key={"review_id": review_id})
    item = resp.get("Item")

    if not item:
        return ApiResponse(success=False, error_code="NOT_FOUND", message="Review not found")

    review = {k: v for k, v in item.items() if k != "extra_columns"}
    if "extra_columns" in item:
        review.update(item["extra_columns"])

    return ApiResponse(success=True, data=review)
