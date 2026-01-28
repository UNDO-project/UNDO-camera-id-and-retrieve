"""FastAPI application for camera identification service."""

from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.config import settings
from src.api.routes import health, identification, catalog, video_stream
from src.api.exceptions import generic_exception_handler
from src.config import paths


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown.

    This replaces the deprecated @app.on_event("startup") and
    @app.on_event("shutdown") decorators.
    """
    # Startup
    logger.info("Starting cIDaR API server...")
    logger.info(f"CORS origins: {settings.cors_origins}")
    logger.info("API documentation available at /docs")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down cIDaR API server...")

    # Import here to avoid circular dependency
    from src.api.dependencies import state

    # Shutdown connection manager if it was initialized
    if state._connection_manager is not None:
        await state._connection_manager.shutdown()

    logger.info("Shutdown complete")


class CORSMiddlewareStaticFiles(StaticFiles):
    """StaticFiles middleware with CORS headers for cross-origin image access.

    This custom StaticFiles implementation adds CORS headers to all static file
    responses, allowing frontend applications to access images and datasheets
    from different origins.
    """

    async def __call__(self, scope, receive, send):
        """Serve static files with CORS headers.

        Args:
            scope: ASGI scope dict containing request information
            receive: ASGI receive callable
            send: ASGI send callable
        """
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return

        path = scope.get("path", "")

        # Check if request is for a static file path that needs CORS
        needs_cors = any(path.startswith(p) for p in settings.static_file_paths)

        if needs_cors:

            async def send_wrapper(message):
                """Wrapper to inject CORS headers into response."""
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"access-control-allow-origin", b"*"))
                    headers.append((b"access-control-allow-methods", b"GET, OPTIONS"))
                    headers.append((b"access-control-allow-headers", b"*"))
                    message = {**message, "headers": headers}
                await send(message)

            await super().__call__(scope, receive, send_wrapper)
        else:
            await super().__call__(scope, receive, send)


# Create FastAPI app with lifespan handler
app = FastAPI(
    title="cIDaR - Camera Identification and Research API",
    description="API for identifying CCTV cameras in images using YOLOv8 and CLIP",
    version="0.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS for API endpoints
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static file directories with CORS support
# CORS headers are added by CORSMiddlewareStaticFiles class
images_dir = paths.images_dir
if images_dir.exists():
    app.mount(
        "/api/v1/images",
        CORSMiddlewareStaticFiles(directory=str(images_dir)),
        name="images",
    )
    logger.info(f"Images served from {images_dir}")
else:
    logger.warning(f"Images directory not found: {images_dir}")

# Mount datasheets directory with CORS support
pdfs_dir = paths.pdfs_dir
if pdfs_dir.exists():
    app.mount(
        "/api/v1/datasheets",
        CORSMiddlewareStaticFiles(directory=str(pdfs_dir)),
        name="datasheets",
    )
    logger.info(f"Datasheets served from {pdfs_dir}")
else:
    logger.warning(f"Datasheets directory not found: {pdfs_dir}")

# Register exception handlers
app.add_exception_handler(Exception, generic_exception_handler)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(identification.router, prefix="/api/v1", tags=["identification"])
app.include_router(catalog.router, prefix="/api/v1/catalog", tags=["catalog"])
app.include_router(video_stream.router, prefix="/api/v1/ws", tags=["video-streaming"])


@app.get("/")
async def root() -> Dict[str, Any]:
    r"""Root endpoint.

    :return: Welcome message and API information
    """
    return {
        "message": "cIDaR - Camera Identification and Research API",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


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
