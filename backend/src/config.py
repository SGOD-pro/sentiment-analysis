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
# agy --conversation=c92ce513-24e3-41ee-9b88-9c36c9a75e41

from functools import lru_cache

from pydantic import AliasChoices, Field
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

    dynamodb_corrections_table: str = Field(
        default="Corrections",
        validation_alias="DYNAMODB_CORRECTIONS_TABLE",
    )

    s3_bucket: str = Field(
        default="sentimetric-prod-storage",
        validation_alias="S3_BUCKET_NAME",
    )

    lambda_function_name: str = Field(
        default="bge-text-embeder",
        validation_alias=AliasChoices(
            "ML_INFERENCE_FUNCTION_NAME",
            "BGE_TEXT_EMBEDDER_FUNCTION_NAME",
            "LAMBDA_FUNCTION_NAME",
        ),
    )
    lambda_batch_size: int = 20

    # Upload limits
    max_upload_size_mb: int = 50

    # Redis (optional — app works without it)
    redis_url: str = "redis://localhost:6379/0"

    # App
    app_name: str = "SWYRA Review Analytics API"
    environment: str = Field(default="production", validation_alias="ENVIRONMENT")
    frontend_url: str = Field(
        default="http://localhost:5173",
        validation_alias=AliasChoices("FRONTEND_URL", "FRONTEND_ORIGIN"),
    )
    debug: bool = False
    reset_data_enabled: bool = False

    @property
    def allowed_origins(self) -> list[str]:
        if self.environment == "production":
            # In production: strictly allow only configured frontend URL from env (no localhost)
            origins = []
            if self.frontend_url:
                for url in self.frontend_url.split(","):
                    clean = url.strip().rstrip("/")
                    if clean and clean not in origins:
                        origins.append(clean)
            return origins if origins else ["https://sentiment-analysis-peach-eta.vercel.app"]
        else:
            # In development: allow local dev servers
            origins = ["http://localhost:5173", "http://localhost:5174"]
            if self.frontend_url:
                for url in self.frontend_url.split(","):
                    clean = url.strip().rstrip("/")
                    if clean and clean not in origins:
                        origins.append(clean)
            return origins


@lru_cache
def get_settings() -> Settings:
    return Settings()
