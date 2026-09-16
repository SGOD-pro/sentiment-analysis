"""
Lambda inference client.

Purpose:
- Obtain 384-dimensional text embeddings from the dedicated embedding Lambda (or local bypass).
- Execute sentiment prediction (MLP) and issue clustering (KMeans) inside the main backend API.

Input: List of review text strings, optional categories list.
Output: Parsed inference results or raw embeddings.
Dependencies: boto3, config, logger, numpy, services.ml_inference
"""

import json
import os
import sys
import time

import boto3
import numpy as np

from config import get_settings
from logger import get_logger
from services.ml_inference import analyze_reviews

log = get_logger(__name__)

# Module-level Lambda client — reused across warm invocations.
# Created lazily on first use; reset when settings change (tests).
# boto3 Lambda client is thread-safe for concurrent invoke() calls.
_lambda_client = None


def _get_lambda_client():
    """Return a module-level boto3 Lambda client, creating it on first call."""
    global _lambda_client
    if _lambda_client is None:
        settings = get_settings()
        kwargs = {"region_name": settings.aws_region}
        if settings.aws_endpoint_url and "localhost" not in settings.aws_endpoint_url:
            kwargs["endpoint_url"] = settings.aws_endpoint_url
        _lambda_client = boto3.client("lambda", **kwargs)
    return _lambda_client


def reset_lambda_client() -> None:
    """Force recreation of the Lambda client (used by tests when mocking changes)."""
    global _lambda_client
    _lambda_client = None


def get_embeddings(texts: list[str]) -> np.ndarray:
    """
    Call the embedding Lambda function (or local bypass) to generate BGE embeddings.

    Returns np.ndarray of shape (len(texts), 384).
    Logs invoke latency for benchmarking: lambda_invoke_ms, embedding_compute_ms.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    settings = get_settings()
    event = {"texts": texts}

    t_invoke = time.monotonic()

    if settings.aws_endpoint_url and "localhost" in settings.aws_endpoint_url:
        # Local bypass: run the embedding ONNX model directly (skips LocalStack Lambda mock)
        lambda_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "lambda")
        )
        if lambda_path not in sys.path:
            sys.path.append(lambda_path)

        import handler  # noqa: PLC0415

        response = handler.lambda_handler(event, None)
        response_payload = {"body": response["body"]}
    else:
        client = _get_lambda_client()
        payload = json.dumps(event)
        response = client.invoke(
            FunctionName=settings.lambda_function_name,
            InvocationType="RequestResponse",
            Payload=payload.encode(),
        )
        response_payload = json.loads(response["Payload"].read())

    invoke_ms = round((time.monotonic() - t_invoke) * 1000)
    log.debug(
        "embedding lambda invoked",
        extra={"texts": len(texts), "lambda_invoke_ms": invoke_ms},
    )

    if "errorMessage" in response_payload:
        log.error(
            "embedding lambda invocation failed",
            extra={"error": response_payload["errorMessage"]},
        )
        raise RuntimeError(f"Lambda error: {response_payload['errorMessage']}")

    if "body" in response_payload:
        body = (
            json.loads(response_payload["body"])
            if isinstance(response_payload["body"], str)
            else response_payload["body"]
        )
        embeddings_list = body["embeddings"]
    else:
        embeddings_list = response_payload["embeddings"]

    return np.array(embeddings_list, dtype=np.float32)


def invoke_lambda(texts: list[str], categories: list[str] | None = None) -> list[dict]:
    """
    Run review analysis:
    1. Fetch embeddings from the embedding Lambda.
    2. Run MLP sentiment prediction & issue clustering inside the backend.
    """
    if not texts:
        return []

    embeddings = get_embeddings(texts)
    return analyze_reviews(embeddings, texts, categories)
