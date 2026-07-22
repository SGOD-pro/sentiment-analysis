"""
Corrections router — PATCH /api/reviews/{review_id}/correct

Purpose: Let users flag incorrect model predictions. Stores the human-supplied
         label in the Corrections table for later retraining.
Input:   Path param: review_id.  Body: { "manual_label": "positive"|"neutral"|"negative" }
Output:  Saved correction record.
Dependencies: database, cache, models
Example:
    PATCH /api/reviews/abc-123/correct
    { "manual_label": "positive" }
"""

import uuid
from datetime import datetime, timezone

from boto3.dynamodb.conditions import Key
from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from cache import cache_delete_prefix
from database import get_tables
from models import ApiResponse

router = APIRouter(prefix="/api")

VALID_LABELS = {"positive", "neutral", "negative"}


class CorrectionRequest(BaseModel):
    manual_label: str

    @field_validator("manual_label")
    @classmethod
    def must_be_valid(cls, v: str) -> str:
        if v not in VALID_LABELS:
            raise ValueError(f"manual_label must be one of {VALID_LABELS}")
        return v


@router.patch("/reviews/{review_id}/correct", response_model=ApiResponse)
def correct_review(review_id: str, body: CorrectionRequest):
    """
    Upsert a human correction for a review.

    Looks up the original review, rejects no-ops (manual == original),
    then writes/overwrites a single Corrections row keyed by review_id.
    Invalidates the Redis cache entries for this batch so the next page
    load reflects the correction flag.
    """
    tables = get_tables()

    # Fetch original review to get text, batch_id, and original label
    resp = tables.reviews.get_item(Key={"review_id": review_id})
    review = resp.get("Item")
    if not review:
        return ApiResponse(success=False, error_code="NOT_FOUND", message="Review not found")

    original_label = review["sentiment"]
    if body.manual_label == original_label:
        return ApiResponse(success=False, error_code="NO_OP", message="manual_label matches current label — nothing to correct")

    # Check if a correction already exists (upsert: overwrite it)
    existing = tables.corrections.query(
        IndexName="review-corrections-index",
        KeyConditionExpression=Key("review_id").eq(review_id),
        Limit=1,
    )
    existing_items = existing.get("Items", [])

    correction_id = existing_items[0]["correction_id"] if existing_items else str(uuid.uuid4())

    correction = {
        "correction_id": correction_id,
        "review_id": review_id,
        "batch_id": review["batch_id"],
        "text": review.get("text", ""),
        "label": original_label,
        "manual_label": body.manual_label,
        "date": datetime.now(timezone.utc).isoformat(),
    }
    tables.corrections.put_item(Item=correction)

    # Invalidate all cached review pages for this batch so correction flag
    # shows on next load without stale data
    cache_delete_prefix(f"reviews:{review['batch_id']}:")

    return ApiResponse(success=True, data=correction)
