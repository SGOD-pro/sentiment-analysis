"""
Tests for Human Label Corrections & SHA-256 Deduplication.

Purpose:
1. Verify deterministic SHA-256 primary key hashing (trim whitespace, lowercase, merge).
2. Verify PATCH /api/reviews/:id/correct deduplicates identical text + original + corrected labels.
3. Verify GET /api/admin/corrections returns deduplicated records.
"""

import hashlib
from routers.corrections import compute_correction_hash


def test_compute_correction_hash_properties():
    """Verify whitespace trimming, lowercasing, and deterministic SHA-256 hashing."""
    h1 = compute_correction_hash("  Great phone!  ", "POSITIVE", "Negative  ")
    h2 = compute_correction_hash("great phone!", "positive", "negative")
    assert h1 == h2, "Hash must be insensitive to case and outer whitespace"

    expected = hashlib.sha256("great phone!:positive:negative".encode("utf-8")).hexdigest()
    assert h1 == expected
    assert len(h1) == 64


def test_correct_review_deduplicates_in_dynamo(client, aws_mock):
    """Submitting the same correction twice updates the record with the same SHA-256 primary key."""
    reviews_table = aws_mock.Table("Reviews")
    reviews_table.put_item(Item={
        "review_id": "rev-100",
        "batch_id": "batch-1",
        "text": "Battery drains too fast.",
        "sentiment": "positive",
        "category": "electronics",
        "confidence_margin": "0.85",
    })

    # First correction
    resp1 = client.patch(
        "/api/reviews/rev-100/correct",
        json={"manual_label": "negative", "session_id": "session-1"},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()["data"]
    expected_hash = compute_correction_hash("Battery drains too fast.", "positive", "negative")
    assert data1["correction_id"] == expected_hash

    # Second correction for identical text and label flip (from another session)
    resp2 = client.patch(
        "/api/reviews/rev-100/correct",
        json={"manual_label": "negative", "session_id": "session-2"},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()["data"]
    assert data2["correction_id"] == expected_hash

    # Query admin corrections endpoint: must contain exactly 1 deduplicated record
    admin_resp = client.get("/api/admin/corrections")
    admin_data = admin_resp.json()["data"]
    assert admin_data["total"] == 1
    assert admin_data["corrections"][0]["correction_id"] == expected_hash
    assert admin_data["corrections"][0]["correction_source_session_id"] == "session-2"


def test_correct_review_noop_rejected(client, aws_mock):
    """Reject correction when manual_label equals original label."""
    reviews_table = aws_mock.Table("Reviews")
    reviews_table.put_item(Item={
        "review_id": "rev-200",
        "batch_id": "batch-1",
        "text": "Solid laptop.",
        "sentiment": "positive",
    })

    resp = client.patch(
        "/api/reviews/rev-200/correct",
        json={"manual_label": "positive"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == "NO_OP"
