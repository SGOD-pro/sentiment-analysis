"""
Tests for batch processor.

Purpose: Verify batch processing pipeline — Lambda invocation, Reviews/Aggregates writes,
         error handling for Lambda failures, partial batch recovery.
"""

import csv
import io
import json
from unittest.mock import patch

import boto3


def _seed_batch(aws_mock, batch_id="test-batch-1", csv_content=None, column_mapping=None):
    """Helper: upload CSV to S3 and create a Batches record."""
    if csv_content is None:
        csv_content = "text,category,date\nGreat product,Electronics,2025-01-15\nTerrible,Books,2025-01-16\n"
    if column_mapping is None:
        column_mapping = {"text_col": "text", "category_col": "category", "date_col": "date"}

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(Bucket="test-bucket", Key=f"uploads/{batch_id}/original.csv", Body=csv_content.encode())

    table = aws_mock.Table("Batches")
    reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(reader)
    table.put_item(Item={
        "batch_id": batch_id,
        "uploaded_at": "2025-01-15T00:00:00Z",
        "filename": "test.csv",
        "total_reviews": len(rows),
        "processed_count": 0,
        "status": "pending",
        "column_mapping": column_mapping,
        "csv_columns": list(csv.DictReader(io.StringIO(csv_content)).fieldnames),
    })
    return batch_id


def _fake_invoke(texts, *args, **kwargs):
    """Build fake Lambda results for a list of texts."""
    results = []
    for t in texts:
        is_neg = "terrible" in t.lower() or "bad" in t.lower()
        r = {
            "sentiment": "negative" if is_neg else "positive",
            "confidence_margin": 0.85,
            "prob_negative": 0.9 if is_neg else 0.05,
            "prob_neutral": 0.05,
            "prob_positive": 0.05 if is_neg else 0.9,
        }
        if is_neg:
            r["issue_tag"] = "general_dissatisfaction"
            r["issue_distance"] = 0.3
            r["cluster_source"] = "cross_category_fallback"
        results.append(r)
    return results


def test_batch_completes_successfully(aws_mock):
    """Full batch with 2 reviews should process both and mark done."""
    batch_id = _seed_batch(aws_mock)

    with patch("services.batch_processor.invoke_lambda", side_effect=_fake_invoke):
        from services.batch_processor import process_batch
        process_batch(batch_id)

    batch = aws_mock.Table("Batches").get_item(Key={"batch_id": batch_id})["Item"]
    assert batch["status"] == "done"
    assert batch["processed_count"] == 2

    reviews_table = aws_mock.Table("Reviews")
    scan = reviews_table.scan()
    assert scan["Count"] == 2

    sentiments = {r["sentiment"] for r in scan["Items"]}
    assert "positive" in sentiments
    assert "negative" in sentiments

    neg_review = [r for r in scan["Items"] if r["sentiment"] == "negative"][0]
    assert neg_review["issue_tag"] == "general_dissatisfaction"


def test_batch_updates_aggregates(aws_mock):
    """Aggregates table should have trend, category, and issue counters keyed by batch_id."""
    batch_id = _seed_batch(aws_mock)

    with patch("services.batch_processor.invoke_lambda", side_effect=_fake_invoke):
        from services.batch_processor import process_batch
        process_batch(batch_id)

    agg_table = aws_mock.Table("Aggregates")
    scan = agg_table.scan()
    agg_types = {item["agg_type"] for item in scan["Items"]}
    batch_ids = {item["batch_id"] for item in scan["Items"]}

    # All aggregates should be scoped to this batch
    assert batch_ids == {batch_id}
    assert any(k.startswith("TREND#") for k in agg_types)
    assert any(k.startswith("CAT#") for k in agg_types)
    assert any(k.startswith("ISSUE#") for k in agg_types)


def test_lambda_timeout_partial_recovery(aws_mock):
    """If one Lambda chunk fails, others should still process."""
    csv_content = "text,category,date\nGood,A,2025-01-15\nFine,A,2025-01-15\nBad item,B,2025-01-16\n"
    batch_id = _seed_batch(aws_mock, csv_content=csv_content)

    call_count = 0

    def _invoke_with_failure(texts, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError("Lambda timeout")
        return _fake_invoke(texts)

    with patch("services.batch_processor.invoke_lambda", side_effect=_invoke_with_failure):
        with patch("services.batch_processor.get_settings") as mock_gs:
            from config import Settings
            settings = Settings(
                _env_file=None,
                s3_bucket="test-bucket",
                lambda_function_name="test-lambda",
                lambda_batch_size=2,
            )
            mock_gs.return_value = settings

            from services.batch_processor import process_batch
            process_batch(batch_id)

    batch = aws_mock.Table("Batches").get_item(Key={"batch_id": batch_id})["Item"]
    assert batch["status"] == "done"
    assert batch["processed_count"] == 1

    reviews_table = aws_mock.Table("Reviews")
    assert reviews_table.scan()["Count"] == 1


def test_batch_all_chunks_fail(aws_mock):
    """If all Lambda calls fail, batch should be marked failed."""
    batch_id = _seed_batch(aws_mock)

    with patch("services.batch_processor.invoke_lambda", side_effect=TimeoutError("Lambda timeout")):
        from services.batch_processor import process_batch
        process_batch(batch_id)

    batch = aws_mock.Table("Batches").get_item(Key={"batch_id": batch_id})["Item"]
    assert batch["status"] == "failed"
    assert batch["processed_count"] == 0


def test_batch_nonexistent_id(aws_mock):
    """Processing a non-existent batch should log error and return."""
    from services.batch_processor import process_batch
    process_batch("nonexistent-batch-id")


def test_batch_idempotency_duplicate_trigger(aws_mock):
    """Calling process_batch twice on the same batch must be idempotent (no double-counting)."""
    batch_id = _seed_batch(aws_mock)

    invoke_count = 0

    def _counting_invoke(texts, *args, **kwargs):
        nonlocal invoke_count
        invoke_count += 1
        return _fake_invoke(texts)

    with patch("services.batch_processor.invoke_lambda", side_effect=_counting_invoke):
        from services.batch_processor import process_batch
        # First execution: claims pending -> processing -> done
        process_batch(batch_id)
        first_invoke_count = invoke_count

        # Second execution (e.g. AWS Lambda Event retry or duplicate trigger)
        process_batch(batch_id)
        # Should not have run again!
        assert invoke_count == first_invoke_count

    batch = aws_mock.Table("Batches").get_item(Key={"batch_id": batch_id})["Item"]
    assert batch["status"] == "done"
    # Never 4 / 2! Must strictly remain 2
    assert batch["processed_count"] == 2

    # Reviews table must contain exactly 2 items, not duplicated
    reviews_count = aws_mock.Table("Reviews").scan()["Count"]
    assert reviews_count == 2


def test_batch_active_lock_skips_concurrent_caller(aws_mock):
    """If another worker is currently processing the batch, a second caller must back off."""
    from datetime import datetime, timezone
    batch_id = _seed_batch(aws_mock)

    # Set status to processing with a timestamp just 5 seconds ago
    aws_mock.Table("Batches").update_item(
        Key={"batch_id": batch_id},
        UpdateExpression="SET #s = :s, processing_started_at = :ts",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "processing",
            ":ts": datetime.now(timezone.utc).isoformat(),
        },
    )

    with patch("services.batch_processor.invoke_lambda") as mock_invoke:
        from services.batch_processor import process_batch
        process_batch(batch_id)
        # Must not invoke lambda since another active worker owns it
        assert mock_invoke.call_count == 0


def test_batch_stale_lock_recovery(aws_mock):
    """If a worker crashed and lock is older than threshold, a new invocation takes over."""
    batch_id = _seed_batch(aws_mock)

    # Simulate worker crash 20 minutes ago (stale lock)
    stale_ts = "2020-01-01T00:00:00Z"
    aws_mock.Table("Batches").update_item(
        Key={"batch_id": batch_id},
        UpdateExpression="SET #s = :s, processing_started_at = :ts",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "processing",
            ":ts": stale_ts,
        },
    )

    with patch("services.batch_processor.invoke_lambda", side_effect=_fake_invoke):
        from services.batch_processor import process_batch
        process_batch(batch_id)

    batch = aws_mock.Table("Batches").get_item(Key={"batch_id": batch_id})["Item"]
    assert batch["status"] == "done"
    assert batch["processed_count"] == 2


def test_chunk_idempotency_skips_already_completed_chunks(aws_mock):
    """If partial execution already completed chunk 0, retry skips chunk 0 and processes chunk 1."""
    csv_content = "text,category,date\nReview 1,A,2025-01-15\nReview 2,A,2025-01-15\nReview 3,B,2025-01-16\nReview 4,B,2025-01-16\n"
    batch_id = _seed_batch(aws_mock, csv_content=csv_content)

    # Pre-mark chunk 0 as completed with 2 reviews processed
    aws_mock.Table("Batches").update_item(
        Key={"batch_id": batch_id},
        UpdateExpression="SET completed_chunks = :cc, processed_count = :pc",
        ExpressionAttributeValues={
            ":cc": {"0"},
            ":pc": 2,
        },
    )

    invoked_texts = []

    def _tracking_invoke(texts, *args, **kwargs):
        invoked_texts.extend(texts)
        return _fake_invoke(texts)

    with patch("services.batch_processor.invoke_lambda", side_effect=_tracking_invoke):
        with patch("services.batch_processor.get_settings") as mock_gs:
            from config import Settings
            settings = Settings(
                _env_file=None,
                s3_bucket="test-bucket",
                lambda_function_name="test-lambda",
                lambda_batch_size=2,  # 4 rows -> chunk 0 (rows 0-1), chunk 1 (rows 2-3)
            )
            mock_gs.return_value = settings

            from services.batch_processor import process_batch
            process_batch(batch_id)

    # Chunk 0 was skipped! Only Chunk 1 ("Review 3", "Review 4") was invoked
    assert "Review 1" not in invoked_texts
    assert "Review 2" not in invoked_texts
    assert "Review 3" in invoked_texts
    assert "Review 4" in invoked_texts

    batch = aws_mock.Table("Batches").get_item(Key={"batch_id": batch_id})["Item"]
    assert batch["status"] == "done"
    # Total processed count must be 2 (initial) + 2 (chunk 1) = 4, never exceeding total_reviews!
    assert batch["processed_count"] == 4
    assert batch["completed_chunks"] == {"0", "1"}

