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

import hashlib
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel, field_validator

from cache import cache_delete_prefix
from database import get_dynamodb_resource, get_tables
from logger import get_logger
from models import ApiResponse

log = get_logger(__name__)

router = APIRouter(prefix="/api")

VALID_LABELS = {"positive", "neutral", "negative"}


def compute_correction_hash(text: str, original_label: str, manual_label: str) -> str:
    """
    Generate deterministic SHA-256 primary key for a human correction:
    1. Trim whitespace on text, original_label, and manual_label.
    2. Lowercase all three values.
    3. Concatenate with colon delimiter.
    4. Compute SHA-256 hex digest.

    Acts as the primary key ('correction_id') in DynamoDB to eliminate duplicate
    corrections for identical review text + sentiment transition.
    """
    clean_text = (text or "").strip().lower()
    clean_orig = (original_label or "").strip().lower()
    clean_manual = (manual_label or "").strip().lower()
    merged = f"{clean_text}:{clean_orig}:{clean_manual}".encode("utf-8")
    return hashlib.sha256(merged).hexdigest()


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
    generates a deterministic SHA-256 primary key from (text, original_label, manual_label),
    and writes/overwrites the Corrections row keyed by that SHA-256 hash.
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
    if body.manual_label.strip().lower() == original_label.strip().lower():
        return ApiResponse(success=False, error_code="NO_OP", message="manual_label matches current label — nothing to correct")

    raw_text = review.get("text", "")
    # Deterministic SHA-256 primary key: guarantees absolute deduplication
    correction_id = compute_correction_hash(raw_text, original_label, body.manual_label)

    # Note: confidence_margin is taken from the original review object (Reviews table)
    # and is NEVER overwritten or modified by human corrections.
    correction = {
        "correction_id": correction_id,
        "review_id": review_id,
        "batch_id": review.get("batch_id", ""),
        "text": raw_text,
        "label": original_label,
        "manual_label": body.manual_label,
        "date": datetime.now(timezone.utc).isoformat(),
        "correction_source_session_id": body.session_id or review.get("batch_id", ""),
        "confidence_margin": str(review.get("confidence_margin", "0")),
    }
    # Direct put_item with SHA-256 primary key updates/deduplicates in-place in DynamoDB
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
    raw_items = response.get("Items", [])
    while "LastEvaluatedKey" in response:
        response = tables.corrections.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
        raw_items.extend(response.get("Items", []))

    # Deduplicate records by SHA-256 hash of (text, label, manual_label) keeping the newest record
    deduped = {}
    for item in raw_items:
        c_hash = compute_correction_hash(
            item.get("text", ""),
            item.get("label", ""),
            item.get("manual_label", ""),
        )
        item["correction_id"] = c_hash
        if c_hash not in deduped or (item.get("date", "") > deduped[c_hash].get("date", "")):
            deduped[c_hash] = item

    items = sorted(deduped.values(), key=lambda x: x.get("date", ""), reverse=True)

    # Fetch confidence_margin and category from Reviews table in chunks of 100
    review_ids = list({item["review_id"] for item in items})
    reviews_map = {}

    if review_ids:
        try:
            ddb = get_dynamodb_resource()
            for i in range(0, len(review_ids), 100):
                batch_keys = [{"review_id": r_id} for r_id in review_ids[i:i+100]]
                batch_response = ddb.batch_get_item(
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
        except Exception as e:
            # ponytail: do not fail entire corrections panel if review metadata lookup fails
            log.warning("Failed to fetch review metadata for corrections: %s", e)

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

