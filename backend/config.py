"""
Configuration loaded from environment variables.

Purpose: Central config for the backend. All AWS IDs, table names, bucket names,
         Lambda function names, and thresholds come from env — never hardcoded.
Input: Environment variables (or .env file via pydantic-settings).
Output: Singleton Settings instance accessible via get_settings().
Dependencies: pydantic-settings
Example:
    settings = get_settings()
    print(settings.dynamodb_reviews_table)  # "Reviews"
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configuration from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # AWS
    aws_region: str = "us-east-1"
    aws_endpoint_url: str | None = None  # for local DynamoDB/S3

    # DynamoDB table names
    dynamodb_reviews_table: str = "Reviews"
    dynamodb_batches_table: str = "Batches"
    dynamodb_aggregates_table: str = "Aggregates"

    # Redis (optional — app works without it)
    redis_url: str = "redis://localhost:6379/0"

    # S3
    s3_bucket: str = "review-uploads"

    # Lambda
    lambda_function_name: str = "sentiment-inference"
    lambda_batch_size: int = 50

    # Upload limits
    max_upload_size_mb: int = 50

    # App
    app_name: str = "Review Analytics API"
    debug: bool = False
    reset_data_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
