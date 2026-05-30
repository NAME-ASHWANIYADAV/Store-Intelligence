"""
Store Intelligence System — FastAPI Application
Main entrypoint with lifespan management, CORS, structured logging, and all routers.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import init_db, close_db
from app.middleware import RequestLoggingMiddleware, setup_logging
from app.routers import ingest, metrics, funnel, heatmap, anomalies, health

import structlog

settings = get_settings()
logger = structlog.get_logger("store_intelligence")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize DB on startup, close on shutdown."""
    setup_logging()
    logger.info("starting_up", version="1.0.0", debug=settings.debug)

    # Initialize database tables
    await init_db()
    logger.info("database_initialized")

    yield

    # Cleanup
    await close_db()
    logger.info("shutdown_complete")


app = FastAPI(
    title="Store Intelligence System",
    description=(
        "Real-time retail analytics API powered by CCTV-based detection pipeline. "
        "Processes visitor events to compute metrics, conversion funnels, zone heatmaps, "
        "and anomaly alerts for physical retail stores."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Streamlit dashboard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Structured Logging Middleware =====
app.add_middleware(RequestLoggingMiddleware)

# ===== Register Routers =====
app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(metrics.router)
app.include_router(funnel.router)
app.include_router(heatmap.router)
app.include_router(anomalies.router)


# ===== Global Exception Handlers =====
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler.
    Never expose raw stack traces — return structured error response.
    """
    logger.error(
        "unhandled_exception",
        endpoint=request.url.path,
        method=request.method,
        error=str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred. Please try again later.",
            "detail": str(exc) if settings.debug else None,
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "not_found",
            "message": f"Endpoint {request.url.path} not found.",
        },
    )


# ===== Root Redirect =====
@app.get("/", tags=["root"])
async def root():
    """API root — redirect to docs."""
    return {
        "service": "Store Intelligence System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers,
    )
