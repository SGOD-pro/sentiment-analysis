"""
Pydantic models for API requests and responses.

Purpose: Shared response/request models for all endpoints.
Input: N/A (type definitions).
Output: Pydantic models used by routers.
Dependencies: pydantic
Example:
    return ApiResponse(success=True, data={"status": "ok"})
"""

from typing import Any

from pydantic import BaseModel


class ApiResponse(BaseModel):
    """Standard API response envelope per Rules.md."""

    success: bool
    data: Any | None = None
    error_code: str | None = None
    message: str | None = None
