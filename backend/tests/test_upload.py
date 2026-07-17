"""
Tests for POST /api/upload.

Purpose: Verify CSV upload validation, S3 storage, and batch creation.
"""

import io


def _make_csv(content: str) -> tuple[str, io.BytesIO, str]:
    """Helper to create a CSV file tuple for upload."""
    return ("file", (("test.csv", io.BytesIO(content.encode("utf-8")), "text/csv")))


def test_upload_valid_csv(client):
    content = "review_text,category,date\nGreat product,Electronics,2025-01-01\nBad quality,Books,2025-01-02\n"
    resp = client.post(
        "/api/upload",
        files={"file": ("test.csv", io.BytesIO(content.encode()), "text/csv")},
        data={"text_col": "review_text", "category_col": "category", "date_col": "date"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "batch_id" in body["data"]


def test_upload_valid_csv_text_col_only(client):
    """Minimal upload — only text_col required, category and date optional."""
    content = "my_text\nSome review\n"
    resp = client.post(
        "/api/upload",
        files={"file": ("reviews.csv", io.BytesIO(content.encode()), "text/csv")},
        data={"text_col": "my_text"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_upload_missing_text_column(client):
    content = "category,date\nElectronics,2025-01-01\n"
    resp = client.post(
        "/api/upload",
        files={"file": ("test.csv", io.BytesIO(content.encode()), "text/csv")},
        data={"text_col": "review_text"},
    )
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "MISSING_COLUMN"


def test_upload_empty_csv(client):
    resp = client.post(
        "/api/upload",
        files={"file": ("test.csv", io.BytesIO(b""), "text/csv")},
        data={"text_col": "review_text"},
    )
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "EMPTY_FILE"


def test_upload_csv_headers_only(client):
    content = "review_text,category\n"
    resp = client.post(
        "/api/upload",
        files={"file": ("test.csv", io.BytesIO(content.encode()), "text/csv")},
        data={"text_col": "review_text"},
    )
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "EMPTY_CSV"


def test_upload_non_csv_file(client):
    resp = client.post(
        "/api/upload",
        files={"file": ("data.json", io.BytesIO(b'{"key": "val"}'), "application/json")},
        data={"text_col": "review_text"},
    )
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "INVALID_FILE_TYPE"


def test_upload_stores_batch_in_dynamo(client, aws_mock):
    content = "text,cat\nHello,A\nWorld,B\n"
    resp = client.post(
        "/api/upload",
        files={"file": ("test.csv", io.BytesIO(content.encode()), "text/csv")},
        data={"text_col": "text", "category_col": "cat"},
    )
    batch_id = resp.json()["data"]["batch_id"]

    # Verify batch record exists
    table = aws_mock.Table("Batches")
    item = table.get_item(Key={"batch_id": batch_id})["Item"]
    # Background task may have changed status from pending → processing/failed,
    # so verify immutable fields instead
    assert item["batch_id"] == batch_id
    assert int(item["total_reviews"]) == 2
    assert item["column_mapping"]["text_col"] == "text"
    assert item["column_mapping"]["category_col"] == "cat"


def test_upload_stores_csv_in_s3(client):
    import boto3

    content = "text\nHello\n"
    resp = client.post(
        "/api/upload",
        files={"file": ("test.csv", io.BytesIO(content.encode()), "text/csv")},
        data={"text_col": "text"},
    )
    batch_id = resp.json()["data"]["batch_id"]

    s3 = boto3.client("s3", region_name="us-east-1")
    obj = s3.get_object(Bucket="test-bucket", Key=f"uploads/{batch_id}/original.csv")
    assert obj["Body"].read().decode() == content
