"""
Tests for services.lambda_client.

Verifies get_embeddings and invoke_lambda with embedding Lambda invocation.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from services.lambda_client import get_embeddings, invoke_lambda


def test_get_embeddings_empty_input():
    """Empty input should return empty array of shape (0, 384)."""
    emb = get_embeddings([])
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (0, 384)


def test_invoke_lambda_empty_input():
    """Empty input should return empty list."""
    res = invoke_lambda([])
    assert res == []


def test_get_embeddings_with_mocked_lambda():
    """Verify get_embeddings parses Lambda response payload correctly."""
    fake_embeddings = np.random.randn(2, 384).astype(np.float32).tolist()

    mock_client = MagicMock()
    mock_payload = MagicMock()
    mock_payload.read.return_value = (
        b'{"statusCode": 200, "body": "{\\"embeddings\\": '
        + str(fake_embeddings).replace("'", '"').encode()
        + b'}"}'
    )
    mock_client.invoke.return_value = {"Payload": mock_payload}

    with patch("boto3.client", return_value=mock_client):
        with patch("services.lambda_client.get_settings") as mock_settings:
            settings_obj = MagicMock()
            settings_obj.aws_endpoint_url = None
            settings_obj.aws_region = "ap-south-1"
            settings_obj.lambda_function_name = "test-embed-lambda"
            mock_settings.return_value = settings_obj

            emb = get_embeddings(["Review one", "Review two"])
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (2, 384)


def test_invoke_lambda_end_to_end_with_mock_embedding():
    """Verify invoke_lambda embeds via Lambda and completes prediction in backend."""
    fake_embeddings = np.random.randn(2, 384).astype(np.float32)

    with patch("services.lambda_client.get_embeddings", return_value=fake_embeddings):
        results = invoke_lambda(["Great service", "Terrible quality"], ["Service", "Product"])
        assert len(results) == 2
        for r in results:
            assert "sentiment" in r
            assert "sentiment_probabilities" in r
            assert "sentiment_confidence_margin" in r
