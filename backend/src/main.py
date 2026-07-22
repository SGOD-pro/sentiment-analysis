"""
FastAPI application entry point.

Purpose: Create and configure the FastAPI app with CORS, routers, and health check.
Input: N/A (ASGI app).
Output: FastAPI app instance.
Dependencies: fastapi, config, logger, startup_check
Example:
    uvicorn main:app --reload
"""

from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add src/ to sys.path so that absolute imports work when running `uvicorn src.main:app` from the backend directory
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()  # populate os.environ before boto3 clients are created

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from logger import get_logger
from startup_check import run_aws_startup_checks
from models import ApiResponse
from routers.batches import router as batches_router
from routers.categories import router as categories_router
from routers.corrections import router as corrections_router
from routers.issues import router as issues_router
from routers.reviews import router as reviews_router
from routers.trends import router as trends_router
from routers.upload import router as upload_router

log = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup checks before the server starts accepting requests."""
    run_aws_startup_checks()
    yield


app = FastAPI(title=settings.app_name, debug=settings.debug, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # ponytail: tighten to frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(batches_router)
app.include_router(trends_router)
app.include_router(categories_router)
app.include_router(issues_router)
app.include_router(reviews_router)
app.include_router(corrections_router)


@app.get("/health", response_model=ApiResponse)
def health():
    """Health check — returns 200 if the service is running."""
    log.info("health check")
    return ApiResponse(success=True, data={"status": "healthy"})
