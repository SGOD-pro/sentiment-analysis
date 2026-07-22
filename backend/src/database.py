"""
DynamoDB connection and table references.

Purpose: Initialize boto3 DynamoDB resource and provide table references.
Input: Settings from config.py (region, endpoint_url, table names).
Output: Table objects for Reviews, Batches, Aggregates.
Dependencies: boto3, config
Example:
    tables = get_tables()
    tables.batches.put_item(Item={"batch_id": "abc"})
"""

from dataclasses import dataclass
from typing import Any

import boto3

from config import get_settings
from logger import get_logger

log = get_logger(__name__)


@dataclass
class Tables:
    reviews: Any
    batches: Any
    aggregates: Any
    corrections: Any


_tables: Tables | None = None


def get_dynamodb_resource():
    """Create boto3 DynamoDB resource from config."""
    settings = get_settings()
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
    return boto3.resource("dynamodb", **kwargs)


def get_tables() -> Tables:
    """Return cached table references."""
    global _tables
    if _tables is None:
        settings = get_settings()
        ddb = get_dynamodb_resource()
        _tables = Tables(
            reviews=ddb.Table(settings.dynamodb_reviews_table),
            batches=ddb.Table(settings.dynamodb_batches_table),
            aggregates=ddb.Table(settings.dynamodb_aggregates_table),
            corrections=ddb.Table(settings.dynamodb_corrections_table),
        )
        log.info("DynamoDB tables initialized")
    return _tables


def reset_tables() -> None:
    """Reset cached tables — used in tests when mocking."""
    global _tables
    _tables = None


def get_s3_client():
    """Create boto3 S3 client from config."""
    settings = get_settings()
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
    return boto3.client("s3", **kwargs)
