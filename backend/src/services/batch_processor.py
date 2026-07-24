"""
Batch processor — reads CSV from S3, invokes Lambda, writes results to DynamoDB.

Purpose: Process a pending batch end-to-end: read CSV, chunk reviews,
         call Lambda for inference (concurrently), write results to Reviews table
         (batch_write_item), increment Aggregates counters, update Batches status.
Input: batch_id (str) — must exist in Batches table with status=pending.
Output: None (side effects: DynamoDB writes, status updates).
Dependencies: boto3, csv, config, database, services.lambda_client, logger
Example:
    process_batch("abc-123")  # processes all reviews in batch abc-123
"""

import csv
import io
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from config import get_settings
from database import get_s3_client, get_tables
from logger import get_logger
from services.lambda_client import invoke_lambda
from services.text_preprocessing import text_preprocessing

log = get_logger(__name__)


def _week_key(date_str: str) -> str:
    """Convert ISO date string to ISO week string like '2025-W03'."""
    try:
        dt = datetime.fromisoformat(date_str)
        return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
    except (ValueError, TypeError):
        return "unknown"


def _batch_write_reviews(table, items: list[dict]) -> None:
    """Write review items to DynamoDB using boto3 batch_writer."""
    with table.batch_writer() as writer:
        for item in items:
            writer.put_item(Item=item)


def process_batch(batch_id: str) -> None:
    """
    Process a batch: read CSV from S3, run inference, store results.

    Updates Batches.status through pending → processing → done/failed.
    On partial failure (some Lambda chunks fail), continues with remaining
    chunks and marks batch as failed only if zero reviews succeed.
    """
    t_start = time.monotonic()
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
    t_s3 = time.monotonic()
    obj = s3.get_object(Bucket=settings.s3_bucket, Key=f"uploads/{batch_id}/original.csv")
    csv_text = obj["Body"].read().decode("utf-8")
    log.info("s3 csv read", extra={"batch_id": batch_id, "duration_ms": round((time.monotonic() - t_s3) * 1000)})

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

    # Build chunks
    chunk_size = settings.lambda_batch_size
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

    # Invoke Lambda concurrently
    t_lambda = time.monotonic()
    chunk_results: list[tuple[list[dict], list[dict]]] = []  # (chunk_rows, lambda_results)
    failed_chunks = 0

    def _invoke_chunk(chunk: list[dict]) -> list[dict]:
        texts = [text_preprocessing(row[text_col]) or row[text_col] for row in chunk]
        categories = [row.get(category_col, "") if category_col else "" for row in chunk]
        return invoke_lambda(texts, categories)

    with ThreadPoolExecutor(max_workers=min(len(chunks), 6)) as pool:
        futures = {pool.submit(_invoke_chunk, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                results = future.result()
                chunk_results.append((chunk, results))
                
                # Increment processed_count in DynamoDB for realtime progress bar
                tables.batches.update_item(
                    Key={"batch_id": batch_id},
                    UpdateExpression="ADD processed_count :c",
                    ExpressionAttributeValues={":c": len(results)}
                )
            except Exception:
                failed_chunks += 1
                log.exception(
                    "lambda chunk failed",
                    extra={"batch_id": batch_id, "chunk_size": len(chunk)},
                )

    log.info(
        "lambda invocations complete",
        extra={
            "batch_id": batch_id,
            "total_chunks": len(chunks),
            "failed_chunks": failed_chunks,
            "duration_ms": round((time.monotonic() - t_lambda) * 1000),
        },
    )

    # Build review items and accumulate aggregates
    review_items: list[dict] = []
    processed = 0

    for chunk, results in chunk_results:
        for row, result in zip(chunk, results):
            review_id = str(uuid.uuid4())
            category = row.get(category_col, "") if category_col else ""
            review_date = row.get(date_col, "") if date_col else ""
            sentiment = result.get("sentiment", "unknown")
            week = _week_key(review_date) if review_date else "unknown"

            # Build review item with all original CSV columns
            item: dict = {
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

            review_items.append(item)

            # Accumulate aggregates
            _accum_aggregate(agg_accum, f"TREND#{category}#{week}", sentiment)
            _accum_aggregate(agg_accum, f"CAT#{category}", sentiment)
            if result.get("issue_tag"):
                source = result.get("cluster_source", "cross_category_fallback")
                # Updated format: ISSUE#{tag}#{source}#{week}
                # Handles backwards compatibility natively when querying via split()
                _accum_aggregate(agg_accum, f"ISSUE#{result['issue_tag']}#{source}#{week}", "count")

            processed += 1

    # Batch write reviews to DynamoDB
    t_db = time.monotonic()
    _batch_write_reviews(tables.reviews, review_items)
    log.info("dynamodb reviews write", extra={"batch_id": batch_id, "count": len(review_items), "duration_ms": round((time.monotonic() - t_db) * 1000)})

    # Flush accumulated aggregates to DynamoDB
    _flush_aggregates(tables, batch_id, agg_accum)

    # Compute total duration
    duration_seconds = round(time.monotonic() - t_start, 2)

    # Final status
    final_status = "done" if processed > 0 else "failed"
    tables.batches.update_item(
        Key={"batch_id": batch_id},
        UpdateExpression="SET #s = :s, processed_count = :c, processing_duration_seconds = :d",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": final_status, ":c": processed, ":d": str(duration_seconds)},
    )

    log.info(
        "batch processing complete",
        extra={
            "batch_id": batch_id,
            "processed": processed,
            "failed_chunks": failed_chunks,
            "final_status": final_status,
            "total_duration_seconds": duration_seconds,
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
    with tables.aggregates.batch_writer() as writer:
        for agg_type, metrics in accum.items():
            item = {"batch_id": batch_id, "agg_type": agg_type, "updated_at": ts}
            for metric, value in metrics.items():
                item[metric] = value
            writer.put_item(Item=item)
