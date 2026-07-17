"""
Shared test fixtures.

Purpose: Provide mocked AWS resources and FastAPI test client for all tests.
Dependencies: pytest, moto, httpx, boto3
"""

import os

import boto3
import pytest
from moto import mock_aws

# Set env vars BEFORE importing app modules so config picks them up
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_ENDPOINT_URL"] = ""  # clear so moto intercepts (not LocalStack)
os.environ["DYNAMODB_REVIEWS_TABLE"] = "Reviews"
os.environ["DYNAMODB_BATCHES_TABLE"] = "Batches"
os.environ["DYNAMODB_AGGREGATES_TABLE"] = "Aggregates"
os.environ["S3_BUCKET"] = "test-bucket"
os.environ["LAMBDA_FUNCTION_NAME"] = "test-lambda"
os.environ["DEBUG"] = "true"


@pytest.fixture
def aws_mock():
    """Mock all AWS services via moto."""
    with mock_aws():
        # Create DynamoDB tables
        ddb = boto3.resource("dynamodb", region_name="us-east-1")

        ddb.create_table(
            TableName="Reviews",
            KeySchema=[{"AttributeName": "review_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "review_id", "AttributeType": "S"},
                {"AttributeName": "batch_id", "AttributeType": "S"},
                {"AttributeName": "sentiment", "AttributeType": "S"},
                {"AttributeName": "batch_cat_sort", "AttributeType": "S"},
                {"AttributeName": "batch_issue_sort", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
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
            BillingMode="PAY_PER_REQUEST",
        )

        ddb.create_table(
            TableName="Batches",
            KeySchema=[{"AttributeName": "batch_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "batch_id", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        ddb.create_table(
            TableName="Aggregates",
            KeySchema=[
                {"AttributeName": "batch_id", "KeyType": "HASH"},
                {"AttributeName": "agg_type", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "batch_id", "AttributeType": "S"},
                {"AttributeName": "agg_type", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        # Create S3 bucket
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        # Reset cached settings + table refs so they pick up mocked resources
        from config import get_settings
        from database import reset_tables

        get_settings.cache_clear()
        reset_tables()

        yield ddb

        reset_tables()
        get_settings.cache_clear()


@pytest.fixture
def client(aws_mock):
    """FastAPI test client with mocked AWS."""
    from config import get_settings

    get_settings.cache_clear()

    from main import app
    from starlette.testclient import TestClient

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
