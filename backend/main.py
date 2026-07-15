"""
FastAPI application entry point.

Purpose: Create and configure the FastAPI app with CORS, routers, and health check.
Input: N/A (ASGI app).
Output: FastAPI app instance.
Dependencies: fastapi, config, logger
Example:
    uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from logger import get_logger
from models import ApiResponse
from routers.batches import router as batches_router
from routers.categories import router as categories_router
from routers.issues import router as issues_router
from routers.reviews import router as reviews_router
from routers.trends import router as trends_router
from routers.upload import router as upload_router

log = get_logger(__name__)
settings = get_settings()

app = FastAPI(title=settings.app_name, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ponytail: tighten to frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(batches_router)
app.include_router(trends_router)
app.include_router(categories_router)
app.include_router(issues_router)
app.include_router(reviews_router)


@app.get("/health", response_model=ApiResponse)
def health():
    """Health check — returns 200 if the service is running."""
    log.info("health check")
    return ApiResponse(success=True, data={"status": "healthy"})
