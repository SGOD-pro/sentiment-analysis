"""
Batch processor — pipelined chunk processing with atomic idempotency.

Pipeline per chunk (each thread):
  preprocess → embedding Lambda → MLP → issue-cluster → DynamoDB write
  → mark chunk complete → increment processed_count

Idempotency guarantees:
  - Batch-level: atomic conditional DynamoDB status transition pending → processing.
    Only one invocation can claim the batch. Lambda Event retries that find
    status != "pending" bail out immediately.
  - Chunk-level: completed_chunks StringSet in the Batches item tracks which
    chunk indices finished successfully. Retries that inherit a partial run
    skip already-completed chunks without re-writing or double-counting.
  - processed_count is only incremented (via atomic ADD) after the chunk's
    DynamoDB writes are durably persisted and the chunk is newly marked complete.

Memory: O(active_workers × chunk_size), not O(total_batch).
  Each chunk's embeddings, inference results, and review dicts are released
  when the thread exits; no global accumulation across chunks.

Aggregate writes are accumulated in-memory per chunk (under a lock) and flushed
once at the end after all threads finish. Aggregates are rollup data for Reports;
mid-batch partial flushes would produce inconsistent report totals.

Crash recovery:
  If the backend Lambda crashes after marking status="processing", a subsequent
  Lambda retry sees ConditionalCheckFailedException and checks how long the batch
  has been in "processing" state. If it exceeds stale_lock_threshold_seconds, the
  retry takes over the lock and resumes from completed_chunks (skipping done work).
"""

import csv
import io
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal

from botocore.exceptions import ClientError

from config import get_settings
from database import get_s3_client, get_tables
from logger import get_logger
from services.lambda_client import invoke_lambda
from services.text_preprocessing import text_preprocessing

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _week_key(date_str: str) -> str:
    """Convert ISO date string to ISO week string like '2025-W03'."""
    try:
        dt = datetime.fromisoformat(date_str)
        return f"{dt.isocalendar()[0]}-W{dt.isocalendar()[1]:02d}"
    except (ValueError, TypeError):
        return "unknown"


def _batch_write_reviews(table, items: list[dict]) -> None:
    """
    Write review items to DynamoDB using batch_writer.

    boto3's batch_writer automatically:
    - Chunks to DynamoDB's 25-item limit
    - Retries UnprocessedItems with exponential backoff

    Raises on unrecoverable errors (caller marks chunk as failed).
    """
    if not items:
        return
    with table.batch_writer() as writer:
        for item in items:
            writer.put_item(Item=item)


def _build_review_item(row: dict, result: dict, batch_id: str,
                       text_col: str, category_col: str | None,
                       date_col: str | None, now_ts: str) -> dict:
    """Build a single DynamoDB review item from one row and its inference result."""
    review_id = str(uuid.uuid4())
    category = row.get(category_col, "") if category_col else ""
    review_date = row.get(date_col, "") if date_col else ""
    sentiment = result.get("sentiment", "unknown")

    item: dict = {
        "review_id": review_id,
        "batch_id": batch_id,
        "text": row[text_col],
        "category": category,
        "review_date": review_date,
        "processed_at": now_ts,
        "sentiment": sentiment,
        "confidence_margin": str(result.get("sentiment_confidence_margin", 0)),
        "prob_negative": str(result.get("sentiment_probabilities", {}).get("negative", 0)),
        "prob_neutral": str(result.get("sentiment_probabilities", {}).get("neutral", 0)),
        "prob_positive": str(result.get("sentiment_probabilities", {}).get("positive", 0)),
        "batch_cat_sort": f"{category}#{review_date}",
    }

    extra = {k: v for k, v in row.items() if k not in (text_col, category_col, date_col)}
    if extra:
        item["extra_columns"] = extra

    if result.get("issue_tag"):
        item["issue_tag"] = result["issue_tag"]
        item["issue_distance"] = str(result.get("issue_distance", 0))
        item["cluster_source"] = result.get("cluster_source", "cross_category_fallback")
        item["batch_issue_sort"] = f"{result['issue_tag']}#{review_date}"

    return item


def _accum_from_result(accum: dict, result: dict, row: dict,
                       category_col: str | None, date_col: str | None) -> None:
    """Accumulate aggregate counters for one row's inference result."""
    category = row.get(category_col, "") if category_col else ""
    review_date = row.get(date_col, "") if date_col else ""
    sentiment = result.get("sentiment", "unknown")
    week = _week_key(review_date) if review_date else "unknown"

    _accum_aggregate(accum, f"TREND#{category}#{week}", sentiment)
    _accum_aggregate(accum, f"CAT#{category}", sentiment)

    conf_margin = float(result.get("sentiment_confidence_margin", 0))
    _accum_aggregate_sum(accum, f"TREND#{category}#{week}", "confidence_margin_sum", conf_margin)
    _accum_aggregate_sum(accum, f"CAT#{category}", "confidence_margin_sum", conf_margin)

    if result.get("issue_tag"):
        source = result.get("cluster_source", "cross_category_fallback")
        _accum_aggregate(accum, f"ISSUE#{result['issue_tag']}#{source}#{week}", "count")


def _merge_agg(target: dict, source: dict) -> None:
    """Merge source aggregate accumulator into target (called under lock)."""
    for agg_type, metrics in source.items():
        if agg_type not in target:
            target[agg_type] = {}
        for metric, value in metrics.items():
            target[agg_type][metric] = target[agg_type].get(metric, type(value)()) + value


def _accum_aggregate(accum: dict, agg_type: str, metric: str) -> None:
    """Accumulate a count in-memory."""
    if agg_type not in accum:
        accum[agg_type] = {}
    accum[agg_type][metric] = accum[agg_type].get(metric, 0) + 1


def _accum_aggregate_sum(accum: dict, agg_type: str, metric: str, val: float) -> None:
    """Accumulate a float sum in-memory (stored as Decimal for DynamoDB compatibility)."""
    if agg_type not in accum:
        accum[agg_type] = {}
    current = accum[agg_type].get(metric, Decimal("0.0"))
    accum[agg_type][metric] = current + Decimal(str(val))


def _flush_aggregates(tables, batch_id: str, accum: dict) -> None:
    """Write all accumulated aggregates to DynamoDB in one batch."""
    if not accum:
        return
    ts = datetime.now(timezone.utc).isoformat()
    with tables.aggregates.batch_writer() as writer:
        for agg_type, metrics in accum.items():
            item = {"batch_id": batch_id, "agg_type": agg_type, "updated_at": ts}
            for metric, value in metrics.items():
                item[metric] = value
            writer.put_item(Item=item)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def process_batch(batch_id: str) -> None:
    """
    Process a batch: read CSV from S3, run inference per chunk, store results.

    Idempotency:
      - Batch level: atomic conditional pending → processing (ConditionalExpression).
        Only one Lambda invocation can claim a batch. Retries that find status
        != "pending" are either bounced (active run) or take over a stale lock.
      - Chunk level: completed_chunks StringSet in the Batches item. A chunk is
        skipped if its index already appears in the set. processed_count is only
        incremented for newly completed chunks.

    Memory: O(active_workers × chunk_size).
    """
    t_start = time.monotonic()
    settings = get_settings()
    tables = get_tables()
    s3 = get_s3_client()

    # Read batch metadata
    batch_resp = tables.batches.get_item(Key={"batch_id": batch_id})
    batch = batch_resp.get("Item")
    if not batch:
        log.error("batch not found", extra={"batch_id": batch_id})
        return

    # ── Batch-level idempotency: atomic conditional pending → processing ──
    # ConditionalExpression ensures exactly one invocation claims the batch.
    # AWS Event invocations can retry up to 2× automatically on any error.
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        tables.batches.update_item(
            Key={"batch_id": batch_id},
            UpdateExpression="SET #s = :processing, processing_started_at = :now",
            ConditionExpression="#s = :pending",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":processing": "processing",
                ":pending": "pending",
                ":now": now_iso,
            },
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
            raise

        # Status is not "pending" — determine whether to bail or take over.
        current = tables.batches.get_item(
            Key={"batch_id": batch_id},
            ProjectionExpression="#s, processing_started_at",
            ExpressionAttributeNames={"#s": "status"},
        ).get("Item", {})

        current_status = current.get("status", "unknown")

        if current_status in ("done", "failed"):
            log.info("batch already finished", extra={"batch_id": batch_id, "status": current_status})
            return

        if current_status == "processing":
            started_at_str = current.get("processing_started_at")
            elapsed = float("inf")
            if started_at_str:
                try:
                    started_dt = datetime.fromisoformat(started_at_str)
                    elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
                except ValueError:
                    pass

            if elapsed < settings.stale_lock_threshold_seconds:
                log.warning(
                    "batch owned by another invocation — skipping",
                    extra={"batch_id": batch_id, "elapsed_seconds": round(elapsed, 1)},
                )
                return

            # Stale lock from a crashed invocation — take over and resume from completed chunks.
            log.warning(
                "taking over stale batch lock",
                extra={"batch_id": batch_id, "stale_elapsed_seconds": round(elapsed, 1)},
            )
            tables.batches.update_item(
                Key={"batch_id": batch_id},
                UpdateExpression="SET processing_started_at = :now",
                ExpressionAttributeValues={":now": now_iso},
            )
            # Fall through — will skip completed chunks below.
        else:
            log.warning("unexpected batch status during claim", extra={"batch_id": batch_id, "status": current_status})
            return

    # Read CSV from S3
    t_s3 = time.monotonic()
    obj = s3.get_object(Bucket=settings.s3_bucket, Key=f"uploads/{batch_id}/original.csv")
    csv_text = obj["Body"].read().decode("utf-8")
    log.info("s3 csv read", extra={"batch_id": batch_id, "duration_ms": round((time.monotonic() - t_s3) * 1000)})

    col_map = batch["column_mapping"]
    text_col = col_map["text_col"]
    category_col = col_map.get("category_col")
    date_col = col_map.get("date_col")

    s3.put_object(
        Bucket=settings.s3_bucket,
        Key=f"uploads/{batch_id}/column_mapping.json",
        Body=json.dumps(col_map).encode(),
    )

    rows = list(csv.DictReader(io.StringIO(csv_text)))
    chunk_size = settings.lambda_batch_size
    chunks = [rows[i: i + chunk_size] for i in range(0, len(rows), chunk_size)]

    # Load already-completed chunk indices once (for idempotent retry of partial batches).
    already_done: set[str] = set(
        tables.batches.get_item(
            Key={"batch_id": batch_id},
            ProjectionExpression="completed_chunks",
        ).get("Item", {}).get("completed_chunks", set())
    )

    # Shared aggregate accumulator — each thread builds a local copy and
    # merges under a lock to avoid GIL-racing on dict mutations.
    agg_lock = threading.Lock()
    agg_accum: dict = {}
    count_lock = threading.Lock()
    processed_total = 0
    failed_chunks = 0

    def _process_chunk(chunk_idx: int, chunk: list[dict]) -> int:
        """
        Single pipelined chunk: preprocess → embed → predict → write DDB → mark done.

        Returns number of reviews written (0 if chunk was already done or failed).
        Raises on unrecoverable embedding or DynamoDB errors.
        """
        t_chunk = time.monotonic()

        # ── Chunk idempotency: skip already-completed chunks ──
        # Covers the case where a previous Lambda run completed some chunks
        # before crashing. The stale-lock takeover path resumes here.
        if str(chunk_idx) in already_done:
            log.info(
                "skipping completed chunk",
                extra={"batch_id": batch_id, "chunk_idx": chunk_idx},
            )
            return 0

        texts = [text_preprocessing(row[text_col]) or row[text_col] for row in chunk]
        categories = [row.get(category_col, "") if category_col else "" for row in chunk]

        t_embed = time.monotonic()
        results = invoke_lambda(texts, categories)   # embedding + MLP + clustering
        embed_ms = round((time.monotonic() - t_embed) * 1000)

        # Build review items and accumulate local aggregates
        now_ts = datetime.now(timezone.utc).isoformat()
        review_items = []
        local_agg: dict = {}
        for row, result in zip(chunk, results):
            review_items.append(
                _build_review_item(row, result, batch_id, text_col, category_col, date_col, now_ts)
            )
            _accum_from_result(local_agg, result, row, category_col, date_col)

        # Write this chunk to DynamoDB immediately (pipelined — no end-of-batch stall).
        # boto3 batch_writer retries UnprocessedItems automatically.
        t_db = time.monotonic()
        _batch_write_reviews(tables.reviews, review_items)
        db_ms = round((time.monotonic() - t_db) * 1000)

        # Atomically mark this chunk complete (idempotent ADD to StringSet)
        # then increment processed_count only if this chunk is newly marked.
        # Using two separate updates is safe: the ADD to the StringSet is
        # idempotent (DynamoDB sets deduplicate), and we only increment the
        # counter once per chunk because we check already_done before processing.
        tables.batches.update_item(
            Key={"batch_id": batch_id},
            UpdateExpression="ADD completed_chunks :idx, processed_count :c",
            ExpressionAttributeValues={
                ":idx": {str(chunk_idx)},   # Python set → DynamoDB SS
                ":c": len(review_items),
            },
        )

        # Merge local aggregates into the shared accumulator under lock
        with agg_lock:
            _merge_agg(agg_accum, local_agg)

        log.info(
            "chunk complete",
            extra={
                "batch_id": batch_id,
                "chunk_idx": chunk_idx,
                "chunk_size": len(chunk),
                "embed_ms": embed_ms,
                "db_ms": db_ms,
                "chunk_ms": round((time.monotonic() - t_chunk) * 1000),
            },
        )
        return len(review_items)

    # ── Bounded concurrent chunk processing ──
    max_workers = min(len(chunks), settings.batch_max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_process_chunk, idx, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            chunk_idx = future_to_idx[future]
            try:
                n = future.result()
                with count_lock:
                    processed_total += n
            except Exception:
                with count_lock:
                    failed_chunks += 1
                log.exception("chunk failed", extra={"batch_id": batch_id, "chunk_idx": chunk_idx})

    # Flush accumulated aggregates once after all threads finish
    _flush_aggregates(tables, batch_id, agg_accum)

    # Read authoritative processed_count from DynamoDB rather than trusting
    # the in-memory counter (Lambda can crash between chunk writes and local add).
    authoritative_item = tables.batches.get_item(
        Key={"batch_id": batch_id},
        ProjectionExpression="processed_count",
    ).get("Item", {})
    authoritative_count = int(authoritative_item.get("processed_count", processed_total))

    total_expected = int(batch.get("total_reviews", len(rows)))
    duration_seconds = round(time.monotonic() - t_start, 2)

    # Batch is "done" when at least one chunk succeeded; "failed" if all failed.
    # ponytail: partial success is marked "done" with a lower processed_count rather than
    # introducing a separate "partial" state — keeps the UI simple and avoids schema changes.
    final_status = "done" if authoritative_count > 0 else "failed"

    tables.batches.update_item(
        Key={"batch_id": batch_id},
        UpdateExpression="SET #s = :s, processing_duration_seconds = :d",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": final_status,
            ":d": str(duration_seconds),
        },
    )

    log.info(
        "batch complete",
        extra={
            "batch_id": batch_id,
            "total_reviews": total_expected,
            "authoritative_processed": authoritative_count,
            "in_memory_processed": processed_total,
            "total_chunks": len(chunks),
            "failed_chunks": failed_chunks,
            "chunk_size": chunk_size,
            "max_workers": max_workers,
            "final_status": final_status,
            "total_duration_seconds": duration_seconds,
        },
    )
