"""
AWS Lambda entry point for the FastAPI backend.

Purpose: Wrap the FastAPI app with Mangum so API Gateway HTTP API (v2)
events can be translated into ASGI requests this app understands.
Input: API Gateway HTTP API v2 event + Lambda context.
Output: API Gateway-compatible response dict.
Dependencies: mangum, main
Example:
    (invoked automatically by AWS Lambda runtime, not called directly)
"""

from mangum import Mangum

from main import app

handler = Mangum(app)
