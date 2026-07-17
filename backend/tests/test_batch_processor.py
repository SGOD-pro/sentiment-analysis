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


def _fake_invoke(texts):
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

    def _invoke_with_failure(texts):
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
