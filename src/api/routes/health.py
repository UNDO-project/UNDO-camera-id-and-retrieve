"""Health check endpoints."""

from fastapi import APIRouter

from src.api.models.responses import HealthResponse, ReadinessResponse
from src.api.dependencies import state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint.

    Returns:
        HealthResponse: Service health status.

    Example:
        ```bash
        curl http://localhost:8000/api/v1/health
        ```
    """
    return HealthResponse(status="healthy", version="0.5.1")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Readiness check endpoint.

    Checks if the service is fully initialized and ready to handle requests.
    This includes:
    - Catalog loaded
    - Embeddings available
    - Detector model loaded

    Returns:
        ReadinessResponse: Detailed readiness status.

    Example:
        ```bash
        curl http://localhost:8000/api/v1/health/ready
        ```
    """
    # Accessing state.service triggers lazy initialization
    try:
        service = state.service
    except Exception:
        # If service initialization fails, return not ready
        return ReadinessResponse(
            ready=False,
            catalog_loaded=False,
            embeddings_ready=False,
            detector_ready=False,
        )

    # Check if catalog is loaded (has catalog index)
    catalog_loaded = hasattr(service, "index") and service.index is not None

    # Check if embeddings are ready
    embeddings_ready = (
        catalog_loaded
        and hasattr(service.index, "embeddings")
        and service.index.embeddings is not None
    )

    # Check if detector is loaded
    detector_ready = hasattr(service, "detector") and service.detector is not None

    ready = catalog_loaded and embeddings_ready and detector_ready

    return ReadinessResponse(
        ready=ready,
        catalog_loaded=catalog_loaded,
        embeddings_ready=embeddings_ready,
        detector_ready=detector_ready,
    )
