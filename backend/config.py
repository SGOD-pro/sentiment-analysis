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
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    aws_region: str = Field(default="ap-south-1", validation_alias="AWS_REGION")
    aws_endpoint_url: str | None = Field(default=None, validation_alias="AWS_ENDPOINT_URL")

    dynamodb_reviews_table: str = Field(
        default="Reviews",
        validation_alias="DYNAMODB_REVIEWS_TABLE",
    )

    dynamodb_batches_table: str = Field(
        default="Batches",
        validation_alias="DYNAMODB_BATCHES_TABLE",
    )

    dynamodb_aggregates_table: str = Field(
        default="Aggregates",
        validation_alias="DYNAMODB_AGGREGATES_TABLE",
    )

    s3_bucket: str = Field(
        default="sentimetric-prod-storage",
        validation_alias="S3_BUCKET_NAME",
    )

    lambda_function_name: str = Field(
        default="sentimetric-ml-inference",
        validation_alias="ML_INFERENCE_FUNCTION_NAME",
    )
    lambda_batch_size: int = 50

    # Upload limits
    max_upload_size_mb: int = 50

    # Redis (optional — app works without it)
    redis_url: str = "redis://localhost:6379/0"

    # App
    app_name: str = "Review Analytics API"
    debug: bool = False
    reset_data_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
