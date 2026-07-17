"""
Batch processor — reads CSV from S3, invokes Lambda, writes results to DynamoDB.

Purpose: Process a pending batch end-to-end: read CSV, chunk reviews,
         call Lambda for inference, write results to Reviews table,
         increment Aggregates counters, update Batches status.
Input: batch_id (str) — must exist in Batches table with status=pending.
Output: None (side effects: DynamoDB writes, status updates).
Dependencies: boto3, csv, config, database, services.lambda_client, logger
Example:
    process_batch("abc-123")  # processes all reviews in batch abc-123
"""

import csv
import io
import json
import uuid
from datetime import datetime, timezone

from config import get_settings
from database import get_s3_client, get_tables
from logger import get_logger
from services.lambda_client import invoke_lambda

log = get_logger(__name__)


def _week_key(date_str: str) -> str:
    """Convert ISO date string to ISO week string like '2025-W03'."""
    try:
        dt = datetime.fromisoformat(date_str)
        return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
    except (ValueError, TypeError):
        return "unknown"


def process_batch(batch_id: str) -> None:
    """
    Process a batch: read CSV from S3, run inference, store results.

    Updates Batches.status through pending → processing → done/failed.
    On partial failure (some Lambda chunks fail), continues with remaining
    chunks and marks batch as failed only if zero reviews succeed.
    """
    settings = get_settings()
    tables = get_tables()
    s3 = get_s3_client()

    # Get batch metadata
    batch_resp = tables.batches.get_item(Key={"batch_id": batch_id})
    batch = batch_resp.get("Item")
    if not batch:
        log.error("batch not found", extra={"batch_id": batch_id})
        return

    # Mark as processing
    tables.batches.update_item(
        Key={"batch_id": batch_id},
        UpdateExpression="SET #s = :s",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "processing"},
    )

    # Read CSV from S3
    obj = s3.get_object(Bucket=settings.s3_bucket, Key=f"uploads/{batch_id}/original.csv")
    csv_text = obj["Body"].read().decode("utf-8")

    col_map = batch["column_mapping"]
    text_col = col_map["text_col"]
    category_col = col_map.get("category_col")
    date_col = col_map.get("date_col")

    # Store column mapping as JSON alongside CSV
    s3.put_object(
        Bucket=settings.s3_bucket,
        Key=f"uploads/{batch_id}/column_mapping.json",
        Body=json.dumps(col_map).encode(),
    )

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    # Collect aggregates in memory, write once per batch
    # Key: agg_type string, Value: {positive, neutral, negative} or {count}
    agg_accum: dict[str, dict[str, int]] = {}

    # Process in chunks
    chunk_size = settings.lambda_batch_size
    processed = 0
    failed_chunks = 0

    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        texts = [row[text_col] for row in chunk]

        try:
            results = invoke_lambda(texts)
        except Exception:
            failed_chunks += 1
            log.exception(
                "lambda chunk failed",
                extra={"batch_id": batch_id, "chunk_start": i, "chunk_size": len(chunk)},
            )
            continue

        # Write results to Reviews table and accumulate aggregates
        for row, result in zip(chunk, results):
            review_id = str(uuid.uuid4())
            category = row.get(category_col, "") if category_col else ""
            review_date = row.get(date_col, "") if date_col else ""
            sentiment = result.get("sentiment", "unknown")
            week = _week_key(review_date) if review_date else "unknown"

            # Build review item with all original CSV columns
            item = {
                "review_id": review_id,
                "batch_id": batch_id,
                "text": row[text_col],
                "category": category,
                "review_date": review_date,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "sentiment": sentiment,
                "confidence_margin": str(result.get("sentiment_confidence_margin", 0)),
                "prob_negative": str(result.get("sentiment_probabilities", {}).get("negative", 0)),
                "prob_neutral": str(result.get("sentiment_probabilities", {}).get("neutral", 0)),
                "prob_positive": str(result.get("sentiment_probabilities", {}).get("positive", 0)),
                # Composite sort keys for batch-scoped GSIs
                "batch_cat_sort": f"{category}#{review_date}",
            }

            # Store all original CSV columns as extra_columns
            extra = {k: v for k, v in row.items() if k not in (text_col, category_col, date_col)}
            if extra:
                item["extra_columns"] = extra

            if result.get("issue_tag"):
                item["issue_tag"] = result["issue_tag"]
                item["issue_distance"] = str(result.get("issue_distance", 0))
                item["cluster_source"] = result.get("cluster_source", "cross_category_fallback")
                # Sparse GSI sort key — only on negative reviews with issue tags
                item["batch_issue_sort"] = f"{result['issue_tag']}#{review_date}"

            tables.reviews.put_item(Item=item)

            # Accumulate aggregates
            _accum_aggregate(agg_accum, f"TREND#{category}#{week}", sentiment)
            _accum_aggregate(agg_accum, f"CAT#{category}", sentiment)
            if result.get("issue_tag"):
                _accum_aggregate(agg_accum, f"ISSUE#{result['issue_tag']}#{week}", "count")

            processed += 1

        # Update processed count
        tables.batches.update_item(
            Key={"batch_id": batch_id},
            UpdateExpression="SET processed_count = :c",
            ExpressionAttributeValues={":c": processed},
        )

    # Flush accumulated aggregates to DynamoDB
    _flush_aggregates(tables, batch_id, agg_accum)

    # Final status
    final_status = "done" if processed > 0 else "failed"
    tables.batches.update_item(
        Key={"batch_id": batch_id},
        UpdateExpression="SET #s = :s, processed_count = :c",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": final_status, ":c": processed},
    )

    log.info(
        "batch processing complete",
        extra={
            "batch_id": batch_id,
            "processed": processed,
            "failed_chunks": failed_chunks,
            "final_status": final_status,
        },
    )


def _accum_aggregate(accum: dict, agg_type: str, metric: str) -> None:
    """Accumulate a count in-memory before flushing."""
    if agg_type not in accum:
        accum[agg_type] = {}
    accum[agg_type][metric] = accum[agg_type].get(metric, 0) + 1


def _flush_aggregates(tables, batch_id: str, accum: dict) -> None:
    """Write all accumulated aggregates to DynamoDB in batch."""
    ts = datetime.now(timezone.utc).isoformat()
    for agg_type, metrics in accum.items():
        item = {"batch_id": batch_id, "agg_type": agg_type, "updated_at": ts}
        for metric, value in metrics.items():
            item[metric] = value
        tables.aggregates.put_item(Item=item)
