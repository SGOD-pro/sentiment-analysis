"""
Tests for GET /api/batches/:id/status.

Purpose: Verify batch status endpoint for various states.
"""


def _create_batch(aws_mock, batch_id, status, total=100, processed=0):
    table = aws_mock.Table("Batches")
    table.put_item(Item={
        "batch_id": batch_id,
        "status": status,
        "total_reviews": total,
        "processed_count": processed,
        "filename": "test.csv",
        "uploaded_at": "2025-01-15T00:00:00Z",
    })


def test_pending_batch(client, aws_mock):
    _create_batch(aws_mock, "b1", "pending", total=100)
    resp = client.get("/api/batches/b1/status")
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "pending"
    assert int(body["data"]["total_reviews"]) == 100
    assert int(body["data"]["processed_count"]) == 0


def test_processing_batch(client, aws_mock):
    _create_batch(aws_mock, "b2", "processing", total=100, processed=50)
    resp = client.get("/api/batches/b2/status")
    body = resp.json()
    assert body["data"]["status"] == "processing"
    assert int(body["data"]["processed_count"]) == 50


def test_completed_batch(client, aws_mock):
    _create_batch(aws_mock, "b3", "done", total=100, processed=100)
    resp = client.get("/api/batches/b3/status")
    body = resp.json()
    assert body["data"]["status"] == "done"
    assert int(body["data"]["processed_count"]) == 100


def test_unknown_batch_id(client, aws_mock):
    resp = client.get("/api/batches/nonexistent/status")
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "NOT_FOUND"


def test_reset_data_requires_explicit_flag(client, aws_mock):
    resp = client.delete("/api/data/reset")
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "FORBIDDEN"
