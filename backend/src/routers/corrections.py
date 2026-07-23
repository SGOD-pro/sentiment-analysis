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
    session_id: str = ""  # batchId from frontend — identifies the upload session

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

    # Check if a correction already exists for this review AND session (upsert per session)
    existing = tables.corrections.query(
        IndexName="review-corrections-index",
        KeyConditionExpression=Key("review_id").eq(review_id),
    )
    existing_items = existing.get("Items", [])

    matching_item = None
    if body.session_id:
        for item in existing_items:
            if item.get("correction_source_session_id") == body.session_id:
                matching_item = item
                break
    elif existing_items:
        matching_item = existing_items[0]

    correction_id = matching_item["correction_id"] if matching_item else str(uuid.uuid4())

    # Note: confidence_margin is taken from the original review object (Reviews table)
    # and is NEVER overwritten or modified by human corrections.
    correction = {
        "correction_id": correction_id,
        "review_id": review_id,
        "batch_id": review["batch_id"],
        "text": review.get("text", ""),
        "label": original_label,
        "manual_label": body.manual_label,
        "date": datetime.now(timezone.utc).isoformat(),
        "correction_source_session_id": body.session_id,
        "confidence_margin": review.get("confidence_margin", "0"),
    }
    tables.corrections.put_item(Item=correction)

    # Invalidate all cached review pages for this batch so correction flag
    # shows on next load without stale data
    cache_delete_prefix(f"reviews:{review['batch_id']}:")

    return ApiResponse(success=True, data=correction)


@router.get("/admin/corrections")
def get_admin_corrections(format: str | None = None):
    """
    Admin endpoint to fetch all human corrections.
    TODO(auth): This endpoint has NO authentication in v1. Add a token/session guard
                before any public or multi-tenant deployment.
    """
    tables = get_tables()

    # Full scan of corrections
    response = tables.corrections.scan()
    items = response.get("Items", [])
    while "LastEvaluatedKey" in response:
        response = tables.corrections.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        items.extend(response.get("Items", []))

    # Fetch confidence_margin and category from Reviews table in chunks of 100
    review_ids = list({item["review_id"] for item in items})
    reviews_map = {}

    if review_ids:
        # We need the low-level dynamodb client for batch_get_item
        import boto3
        dynamodb = boto3.resource("dynamodb", region_name="ap-south-1")
        
        for i in range(0, len(review_ids), 100):
            batch_keys = [{"review_id": r_id} for r_id in review_ids[i:i+100]]
            batch_response = dynamodb.batch_get_item(
                RequestItems={
                    tables.reviews.name: {
                        "Keys": batch_keys,
                        "ProjectionExpression": "review_id, confidence_margin, category"
                    }
                }
            )
            batch_reviews = batch_response.get("Responses", {}).get(tables.reviews.name, [])
            for br in batch_reviews:
                reviews_map[br["review_id"]] = br

    for item in items:
        r_info = reviews_map.get(item["review_id"], {})
        if "confidence_margin" in r_info:
            item["confidence_margin"] = str(r_info["confidence_margin"])
        if "category" in r_info:
            item["category"] = r_info["category"]

    if format == "csv":
        import csv
        from io import StringIO
        from fastapi.responses import StreamingResponse
        
        output = StringIO()
        writer = csv.writer(output)
        # Must match export_corrections.py and retrain_with_corrections.py exactly
        writer.writerow(["text", "label", "manual_label", "date", "review_id", "batch_id", "correction_source_session_id", "confidence_margin"])
        for item in items:
            writer.writerow([
                item.get("text", ""),
                item.get("label", ""),
                item.get("manual_label", ""),
                item.get("date", ""),
                item.get("review_id", ""),
                item.get("batch_id", ""),
                item.get("correction_source_session_id", ""),
                item.get("confidence_margin", ""),
            ])
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=corrections_export.csv"}
        )

    batch_count = len(set(item.get("batch_id") for item in items if item.get("batch_id")))
    
    return ApiResponse(success=True, data={
        "corrections": items,
        "total": len(items),
        "batch_count": batch_count
    })


class AdminAuthRequest(BaseModel):
    password: str


@router.post("/admin/auth", response_model=ApiResponse)
def admin_auth(body: AdminAuthRequest):
    """
    Validate admin password for the corrections panel.
    """
    if body.password == "sentrixadmin":
        return ApiResponse(success=True, data={"authenticated": True})
    return ApiResponse(success=False, error_code="UNAUTHORIZED", message="Incorrect password")

