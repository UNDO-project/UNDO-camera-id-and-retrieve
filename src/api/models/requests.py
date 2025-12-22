"""API request models."""

from pydantic import BaseModel, Field, HttpUrl


class IdentifyURLRequest(BaseModel):
    """Request for identifying cameras from a URL."""

    image_url: HttpUrl
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of top matches to return"
    )
    min_similarity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold",
    )
    save_crops: bool = Field(
        default=False, description="Whether to save detected camera crops"
    )


class CatalogSearchRequest(BaseModel):
    """Request for searching the camera catalog."""

    query: str = Field(..., min_length=1, description="Search query string")
    limit: int = Field(
        default=10, ge=1, le=100, description="Maximum number of results"
    )
