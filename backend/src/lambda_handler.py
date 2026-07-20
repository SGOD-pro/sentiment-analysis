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
from services.batch_processor import process_batch
from logger import get_logger

log = get_logger(__name__)
_mangum_handler = Mangum(app)

def handler(event, context):
    """
    Handle incoming Lambda events.
    If the event is an internal process_batch trigger, process it synchronously.
    Otherwise, route it to FastAPI via Mangum.
    """
    if isinstance(event, dict) and event.get("action") == "process_batch":
        batch_id = event.get("batch_id")
        log.info("received internal process_batch event", extra={"batch_id": batch_id})
        if batch_id:
            process_batch(batch_id)
        return {"success": True}
        
    return _mangum_handler(event, context)
