"""
Batch status router.

Purpose: Return processing status for a given batch.
Input: batch_id path parameter.
Output: Batch status, total_reviews, processed_count.
Dependencies: database, models
Example:
    GET /api/batches/abc-123/status
    → {"success": true, "data": {"status": "processing", "total_reviews": 500, "processed_count": 250}}
"""

from fastapi import APIRouter

from database import get_tables
from models import ApiResponse

router = APIRouter(prefix="/api")


@router.get("/batches/{batch_id}/status", response_model=ApiResponse)
def batch_status(batch_id: str):
    """Return processing status for a batch."""
    tables = get_tables()
    resp = tables.batches.get_item(Key={"batch_id": batch_id})
    item = resp.get("Item")

    if not item:
        return ApiResponse(success=False, error_code="NOT_FOUND", message="Batch not found")

    return ApiResponse(
        success=True,
        data={
            "batch_id": item["batch_id"],
            "status": item["status"],
            "total_reviews": item.get("total_reviews", 0),
            "processed_count": item.get("processed_count", 0),
            "filename": item.get("filename", ""),
            "uploaded_at": item.get("uploaded_at", ""),
        },
    )
