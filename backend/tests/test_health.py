"""
Tests for health endpoint and config loading.

Purpose: Verify /health returns 200 and config loads from env correctly.
"""


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["status"] == "healthy"


def test_health_response_has_standard_envelope(client):
    body = client.get("/health").json()
    assert "success" in body
    assert "data" in body


def test_config_loads_from_env():
    """Config should read from environment variables set in conftest."""
    from config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    assert settings.dynamodb_reviews_table == "Reviews"
    assert settings.dynamodb_batches_table == "Batches"
    assert settings.dynamodb_aggregates_table == "Aggregates"
    assert settings.s3_bucket == "test-bucket"
    assert settings.lambda_function_name == "test-lambda"
    get_settings.cache_clear()


def test_config_defaults():
    """Defaults should be sensible when env vars aren't set."""
    from config import Settings

    # Construct directly without env to check defaults
    s = Settings(
        _env_file=None,
        s3_bucket="x",
        lambda_function_name="x",
    )
    assert s.aws_region == "us-east-1"
    assert s.lambda_batch_size == 50
    assert s.max_upload_size_mb == 50
