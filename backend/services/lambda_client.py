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
        
        ml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "ml"))
        if ml_path not in sys.path:
            sys.path.append(ml_path)
            
        # Set artifact path before importing the handler
        os.environ["ARTIFACT_DIR"] = os.path.abspath(os.path.join(ml_path, "..", "lambda", "artifacts"))
        
        import lambda_handler_final
        event = {"texts": texts}
        response = lambda_handler_final.lambda_handler(event, None)
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
