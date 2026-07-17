"""
AWS startup diagnostic.

Purpose: On server start, verify DynamoDB tables and S3 bucket exist.
         Log each resource status (already exists / created / unreachable).
         Does NOT block startup on failure — logs error and continues.
Input: Settings from config.py.
Output: Structured log lines at INFO/WARNING/ERROR level.
Dependencies: boto3, config, logger
"""

import io
import json
import zipfile

import boto3
from botocore.exceptions import ClientError, EndpointResolutionError, NoRegionError

from config import get_settings
from database import get_dynamodb_resource, get_s3_client
from logger import get_logger

log = get_logger(__name__)

# Schema required to create tables if missing.
_TABLE_SCHEMAS = {
    "reviews": {
        "KeySchema": [{"AttributeName": "review_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "review_id", "AttributeType": "S"},
            {"AttributeName": "batch_id", "AttributeType": "S"},
            {"AttributeName": "sentiment", "AttributeType": "S"},
            {"AttributeName": "batch_cat_sort", "AttributeType": "S"},
            {"AttributeName": "batch_issue_sort", "AttributeType": "S"},
        ],
        "GlobalSecondaryIndexes": [
            {
                "IndexName": "batch-sentiment-index",
                "KeySchema": [
                    {"AttributeName": "batch_id", "KeyType": "HASH"},
                    {"AttributeName": "sentiment", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "batch-category-index",
                "KeySchema": [
                    {"AttributeName": "batch_id", "KeyType": "HASH"},
                    {"AttributeName": "batch_cat_sort", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
            {
                "IndexName": "batch-issue-index",
                "KeySchema": [
                    {"AttributeName": "batch_id", "KeyType": "HASH"},
                    {"AttributeName": "batch_issue_sort", "KeyType": "RANGE"},
                ],
                "Projection": {"ProjectionType": "ALL"},
            },
        ],
        "BillingMode": "PAY_PER_REQUEST",
    },
    "batches": {
        "KeySchema": [{"AttributeName": "batch_id", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "batch_id", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
    },
    "aggregates": {
        "KeySchema": [
            {"AttributeName": "batch_id", "KeyType": "HASH"},
            {"AttributeName": "agg_type", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "batch_id", "AttributeType": "S"},
            {"AttributeName": "agg_type", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
    },
}


def _check_dynamodb(settings) -> None:
    """Check each DynamoDB table; create if missing."""
    table_map = {
        "reviews": settings.dynamodb_reviews_table,
        "batches": settings.dynamodb_batches_table,
        "aggregates": settings.dynamodb_aggregates_table,
    }

    try:
        ddb = get_dynamodb_resource()
        ddb_client = ddb.meta.client
        existing = set(ddb_client.list_tables()["TableNames"])
        log.info("DynamoDB reachable", extra={"existing_tables": sorted(existing)})
    except Exception as exc:
        log.error(
            "DynamoDB unreachable — skipping table check",
            extra={"error": str(exc)},
            exc_info=True,
        )
        return

    for key, table_name in table_map.items():
        if table_name in existing:
            log.info("DynamoDB table exists", extra={"table": table_name})
        else:
            log.warning(
                "DynamoDB table missing — creating",
                extra={"table": table_name},
            )
            try:
                schema = _TABLE_SCHEMAS[key]
                ddb_client.create_table(TableName=table_name, **schema)
                ddb_client.get_waiter("table_exists").wait(TableName=table_name)
                log.info("DynamoDB table created", extra={"table": table_name})
            except ClientError as exc:
                log.error(
                    "DynamoDB table creation failed",
                    extra={"table": table_name, "error": str(exc)},
                    exc_info=True,
                )


def _check_s3(settings) -> None:
    """Check S3 bucket; create if missing."""
    bucket = settings.s3_bucket
    try:
        s3 = get_s3_client()
        s3.head_bucket(Bucket=bucket)
        log.info("S3 bucket exists", extra={"bucket": bucket})
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            log.warning("S3 bucket missing — creating", extra={"bucket": bucket})
            try:
                region = settings.aws_region
                # us-east-1 does NOT accept LocationConstraint — all others do.
                if region == "us-east-1":
                    s3.create_bucket(Bucket=bucket)
                else:
                    s3.create_bucket(
                        Bucket=bucket,
                        CreateBucketConfiguration={"LocationConstraint": region},
                    )
                log.info("S3 bucket created", extra={"bucket": bucket})
            except ClientError as create_exc:
                log.error(
                    "S3 bucket creation failed",
                    extra={"bucket": bucket, "error": str(create_exc)},
                    exc_info=True,
                )
        else:
            log.error(
                "S3 unreachable — skipping bucket check",
                extra={"bucket": bucket, "error": str(exc)},
                exc_info=True,
            )


def _check_lambda(settings) -> None:
    """Check Lambda function; create a mock one if missing (for local dev)."""
    func_name = settings.lambda_function_name
    try:
        kwargs = {"region_name": settings.aws_region}
        if settings.aws_endpoint_url:
            kwargs["endpoint_url"] = settings.aws_endpoint_url
        client = boto3.client("lambda", **kwargs)
        
        client.get_function(FunctionName=func_name)
        log.info("Lambda function exists", extra={"function": func_name})
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "ResourceNotFoundException":
            log.warning("Lambda function missing — creating mock for local dev", extra={"function": func_name})
            
            mock_code = """
import json
def handler(event, context):
    texts = event.get('texts', [])
    results = []
    for t in texts:
        results.append({
            "text": t,
            "sentiment": "neutral",
            "sentiment_confidence_margin": 0.99,
            "sentiment_probabilities": {"positive": 0.0, "neutral": 1.0, "negative": 0.0},
            "issue_tag": None,
            "issue_distance": None
        })
    return {"statusCode": 200, "body": json.dumps({"results": results})}
"""
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('index.py', mock_code)
            
            try:
                client.create_function(
                    FunctionName=func_name,
                    Runtime='python3.9',
                    Role='arn:aws:iam::000000000000:role/dummy-role',
                    Handler='index.handler',
                    Code={'ZipFile': zip_buffer.getvalue()}
                )
                log.info("Mock Lambda function created", extra={"function": func_name})
            except ClientError as create_exc:
                log.error(
                    "Mock Lambda creation failed",
                    extra={"function": func_name, "error": str(create_exc)},
                    exc_info=True,
                )
        else:
            log.error(
                "Lambda unreachable — skipping check",
                extra={"function": func_name, "error": str(exc)},
                exc_info=True,
            )
    except Exception as exc:
        log.error(
            "Lambda unreachable — skipping check",
            extra={"function": func_name, "error": str(exc)},
            exc_info=True,
        )


def run_aws_startup_checks() -> None:
    """
    Run all AWS connectivity and resource checks.
    Called once at server startup. Never raises — logs and continues.
    """
    settings = get_settings()
    log.info(
        "AWS startup check — begin",
        extra={
            "region": settings.aws_region,
            "endpoint": settings.aws_endpoint_url or "AWS (public)",
            "dynamodb_tables": [
                settings.dynamodb_reviews_table,
                settings.dynamodb_batches_table,
                settings.dynamodb_aggregates_table,
            ],
            "s3_bucket": settings.s3_bucket,
        },
    )
    _check_dynamodb(settings)
    _check_s3(settings)
    _check_lambda(settings)
    log.info("AWS startup check — done")
