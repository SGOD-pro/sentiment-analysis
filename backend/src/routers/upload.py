"""
Upload router — CSV upload and batch creation.

Purpose: Accept CSV file upload, validate columns, store in S3, create batch record.
Input: Multipart CSV file + form fields (text_col, category_col, date_col).
Output: batch_id on success, structured error on failure.
Dependencies: fastapi, boto3, csv, config, database, logger
Example:
    POST /api/upload
    Form: file=reviews.csv, text_col=review_text, category_col=category
    Response: {"success": true, "data": {"batch_id": "abc-123"}}
"""

import csv
import io
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, File, Form, UploadFile

from config import get_settings
from database import get_s3_client, get_tables
from logger import get_logger
from models import ApiResponse
from services.batch_processor import process_batch

router = APIRouter(prefix="/api")
log = get_logger(__name__)


@router.post("/upload", response_model=ApiResponse)
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    text_col: str = Form(...),
    category_col: str = Form(default=""),
    date_col: str = Form(default=""),
):
    """
    Upload a CSV of reviews for batch processing.

    Validates that the CSV is non-empty and contains the required text column.
    Stores raw CSV in S3, creates a Batches record with column mapping.
    """
    settings = get_settings()

    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".csv"):
        return ApiResponse(
            success=False, error_code="INVALID_FILE_TYPE", message="File must be a CSV"
        )

    # Read and validate content
    try:
        raw = await file.read()
        if not raw.strip():
            return ApiResponse(
                success=False, error_code="EMPTY_FILE", message="CSV file is empty"
            )
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return ApiResponse(
            success=False,
            error_code="INVALID_ENCODING",
            message="File must be UTF-8 encoded",
        )

    # Parse header and validate columns
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return ApiResponse(
            success=False,
            error_code="INVALID_CSV",
            message="CSV has no header row",
        )

    if text_col not in reader.fieldnames:
        return ApiResponse(
            success=False,
            error_code="MISSING_COLUMN",
            message=f"Column '{text_col}' not found. Available: {list(reader.fieldnames)}",
        )

    if category_col and category_col not in reader.fieldnames:
        return ApiResponse(
            success=False,
            error_code="MISSING_COLUMN",
            message=f"Column '{category_col}' not found. Available: {list(reader.fieldnames)}",
        )

    if date_col and date_col not in reader.fieldnames:
        return ApiResponse(
            success=False,
            error_code="MISSING_COLUMN",
            message=f"Column '{date_col}' not found. Available: {list(reader.fieldnames)}",
        )

    # Count rows
    rows = list(reader)
    if not rows:
        return ApiResponse(
            success=False,
            error_code="EMPTY_CSV",
            message="CSV has headers but no data rows",
        )

    # Upload to S3
    batch_id = str(uuid.uuid4())
    s3 = get_s3_client()
    s3.put_object(
        Bucket=settings.s3_bucket,
        Key=f"uploads/{batch_id}/original.csv",
        Body=raw,
    )

    # Create batch record
    column_mapping = {"text_col": text_col}
    if category_col:
        column_mapping["category_col"] = category_col
    if date_col:
        column_mapping["date_col"] = date_col

    tables = get_tables()
    tables.batches.put_item(
        Item={
            "batch_id": batch_id,
            "filename": file.filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "total_reviews": len(rows),
            "processed_count": 0,
            "status": "pending",
            "column_mapping": column_mapping,
            "csv_columns": list(reader.fieldnames),
        }
    )

    log.info(
        "batch created",
        extra={"batch_id": batch_id, "total_reviews": len(rows), "upload_filename": file.filename},
    )

    import os
    import json
    import boto3
    
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if function_name:
        log.info("triggering async lambda invocation", extra={"batch_id": batch_id, "function_name": function_name})
        lambda_client = boto3.client("lambda", region_name=settings.aws_region)
        lambda_client.invoke(
            FunctionName=function_name,
            InvocationType="Event",
            Payload=json.dumps({"action": "process_batch", "batch_id": batch_id}).encode(),
        )
    else:
        log.info("triggering background task (local dev)", extra={"batch_id": batch_id})
        background_tasks.add_task(process_batch, batch_id)

    return ApiResponse(success=True, data={"batch_id": batch_id})
