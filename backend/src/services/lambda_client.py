"""
Lambda inference client.

Purpose: Invoke the sentiment inference Lambda with a batch of review texts.
Input: List of review text strings.
Output: List of dicts with sentiment, probabilities, issue_tag, etc.
Dependencies: boto3, config, logger
Example:
    results = invoke_lambda(["Great product", "Terrible quality"])
    # [{"sentiment": "positive", ...}, {"sentiment": "negative", ...}]
"""

import json

import boto3

from config import get_settings
from logger import get_logger

log = get_logger(__name__)


def invoke_lambda(texts: list[str]) -> list[dict]:
    """
    Call the inference Lambda with a batch of texts.

    Returns parsed results list. Raises on Lambda error.
    """
    settings = get_settings()
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url

    if settings.aws_endpoint_url and "localhost" in settings.aws_endpoint_url:
        # Local bypass: Run the actual ML model directly instead of the LocalStack mock
        import os
        import sys
        
        lambda_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "lambda"))
        # NOTE: three ".." — file now lives at backend/src/services/lambda_client.py
        # Resolves to: backend/src/services/ → backend/src/ → backend/ → repo-root/ → lambda/
        if lambda_path not in sys.path:
            sys.path.append(lambda_path)
            
        import handler
        event = {"texts": texts}
        response = handler.lambda_handler(event, None)
        response_payload = {"body": response["body"]}
    else:
        client = boto3.client("lambda", **kwargs)
        payload = json.dumps({"texts": texts})
        response = client.invoke(
            FunctionName=settings.lambda_function_name,
            InvocationType="RequestResponse",
            Payload=payload.encode(),
        )
        response_payload = json.loads(response["Payload"].read())

    if "errorMessage" in response_payload:
        log.error(
            "lambda invocation failed",
            extra={"error": response_payload["errorMessage"]},
        )
        raise RuntimeError(f"Lambda error: {response_payload['errorMessage']}")

    # Lambda returns an API Gateway format (statusCode + stringified body)
    if "body" in response_payload:
        body = json.loads(response_payload["body"])
        return body["results"]

    return response_payload["results"]
