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

from config import get_settings
from database import get_tables
from logger import get_logger
from models import ApiResponse

router = APIRouter(prefix="/api")
log = get_logger(__name__)


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
            "processing_duration_seconds": item.get("processing_duration_seconds"),
            "batch_size": get_settings().lambda_batch_size,
        },
    )


@router.get("/batches/stats", response_model=ApiResponse)
def batch_stats():
    """Return aggregate batch processing stats for the Reports page."""
    tables = get_tables()
    resp = tables.batches.scan()
    items = resp.get("Items", [])

    done = [i for i in items if i.get("status") == "done"]
    durations = [float(i["processing_duration_seconds"]) for i in done if i.get("processing_duration_seconds")]
    avg_time = round(sum(durations) / len(durations), 2) if durations else None

    return ApiResponse(
        success=True,
        data={
            "batches_processed": len(done),
            "avg_processing_time_seconds": avg_time,
        },
    )


def _wipe_table(table, key_schema) -> int:
    """Delete every item in a DynamoDB table. Returns count deleted."""
    key_names = [k["AttributeName"] for k in key_schema]
    deleted = 0
    scan_kwargs = {"ProjectionExpression": ", ".join(key_names)}
    while True:
        resp = table.scan(**scan_kwargs)
        with table.batch_writer() as batch:
            for item in resp.get("Items", []):
                batch.delete_item(Key={k: item[k] for k in key_names})
                deleted += 1
        if "LastEvaluatedKey" not in resp:
            break
        scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return deleted


@router.delete("/data/reset", response_model=ApiResponse)
def reset_all_data():
    """Wipe all Reviews, Batches, and Aggregates data. Only works when DEBUG=true."""
    settings = get_settings()
    if not settings.debug:
        return ApiResponse(success=False, error_code="FORBIDDEN", message="Only available in debug mode")

    tables = get_tables()
    counts = {}
    counts["reviews"] = _wipe_table(tables.reviews, [{"AttributeName": "review_id"}])
    counts["batches"] = _wipe_table(tables.batches, [{"AttributeName": "batch_id"}])
    counts["aggregates"] = _wipe_table(tables.aggregates, [{"AttributeName": "batch_id"}, {"AttributeName": "agg_type"}])

    log.info("data reset", extra={"deleted": counts})
    return ApiResponse(success=True, data={"deleted": counts})
