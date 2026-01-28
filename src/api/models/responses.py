"""API response models."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from src.models.identification import RetrievalResult


class IdentifyResponse(BaseModel):
    """Response for camera identification request."""

    success: bool = Field(..., description="Whether the request succeeded")
    detections_count: int = Field(..., description="Number of cameras detected")
    results: List[RetrievalResult] = Field(
        ..., description="Detection results with matches"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


class CatalogStatsResponse(BaseModel):
    """Response for catalog statistics."""

    total_cameras: int = Field(..., description="Total cameras in catalog")
    vendor_count: int = Field(..., description="Number of unique vendors in catalog")
    embeddings_loaded: bool = Field(..., description="Whether embeddings are loaded")
    embedding_count: int = Field(..., description="Number of embeddings available")
    catalog_path: str = Field(..., description="Path to catalog parquet file")


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")


class ReadinessResponse(BaseModel):
    """Response for readiness check."""

    ready: bool = Field(..., description="Whether the service is ready")
    catalog_loaded: bool = Field(..., description="Whether catalog is loaded")
    embeddings_ready: bool = Field(..., description="Whether embeddings are ready")
    detector_ready: bool = Field(..., description="Whether detector model is loaded")
    websocket_connections: Optional[Dict[str, Any]] = Field(
        None, description="WebSocket connection statistics (if initialized)"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
