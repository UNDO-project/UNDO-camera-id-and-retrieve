"""Health check endpoints."""

from fastapi import APIRouter, HTTPException

from src.api.models.responses import (
    HealthResponse,
    ReadinessResponse,
    StreamStatsResponse,
)
from src.api.dependencies import state

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    r"""Basic health check endpoint.

    :return: Service health status

    Example:
        ```bash
        curl http://localhost:8000/api/v1/health
        ```
    """
    return HealthResponse(status="healthy", version="0.6.0")


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    r"""Readiness check endpoint.

    Checks if the service is fully initialized and ready to handle requests.
    This includes:
    - Catalog loaded
    - Embeddings available
    - Detector model loaded

    :return: Detailed readiness status

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

    # Include WebSocket connection stats if connection manager is initialized
    connection_stats = None
    if state._connection_manager is not None:
        connection_stats = state._connection_manager.get_connection_stats()

    return ReadinessResponse(
        ready=ready,
        catalog_loaded=catalog_loaded,
        embeddings_ready=embeddings_ready,
        detector_ready=detector_ready,
        websocket_connections=connection_stats,
    )


@router.get("/stream/stats", response_model=StreamStatsResponse)
async def stream_stats() -> StreamStatsResponse:
    r"""Get video streaming statistics.

    Returns statistics about active WebSocket connections, including
    connection counts, frame processing metrics, and per-connection details.

    :return: Stream statistics
    :raises HTTPException: If connection manager not initialized

    Example:
        ```bash
        curl http://localhost:8000/api/v1/stream/stats
        ```
    """
    if state._connection_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Stream manager not initialized. No streams have connected yet.",
        )

    stats = state._connection_manager.get_connection_stats()

    return StreamStatsResponse(
        active_connections=stats["active_connections"],
        max_connections=stats["max_connections"],
        total_connections_served=stats["total_connections_served"],
        total_frames_received=stats["total_frames_received"],
        total_frames_processed=stats["total_frames_processed"],
        total_frames_dropped=stats["total_frames_dropped"],
        average_fps=stats["average_fps"],
        average_latency_ms=stats["average_latency_ms"],
        connections=stats["connections"],
    )
