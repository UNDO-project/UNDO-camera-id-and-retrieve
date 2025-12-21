"""FastAPI application for camera identification service."""

from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.config import settings
from src.api.routes import health, identification, catalog
from src.api.exceptions import generic_exception_handler

# Create FastAPI app
app = FastAPI(
    title="cIDaR - Camera Identification and Research API",
    description="API for identifying CCTV cameras in images using YOLOv8 and CLIP",
    version="0.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(Exception, generic_exception_handler)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(identification.router, prefix="/api/v1", tags=["identification"])
app.include_router(catalog.router, prefix="/api/v1/catalog", tags=["catalog"])


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint.

    Returns:
        Dict[str, Any]: Welcome message and API information
    """
    return {
        "message": "cIDaR - Camera Identification and Research API",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.on_event("startup")
async def startup_event() -> None:
    """Run on application startup."""
    logger.info("Starting cIDaR API server...")
    logger.info(f"CORS origins: {settings.cors_origins}")
    logger.info("API documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Run on application shutdown."""
    logger.info("Shutting down cIDaR API server...")


def run() -> None:
    """Run the API server.

    This function is called by the console script entry point.
    """
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
