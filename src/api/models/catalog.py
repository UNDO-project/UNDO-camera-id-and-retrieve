"""Catalog API request and response models."""

from pydantic import BaseModel, Field
from typing import Optional, List


class CameraSpecsSummary(BaseModel):
    """Flattened camera specifications for summary view."""

    max_resolution: Optional[str] = Field(None, description="Maximum resolution")
    lens: Optional[str] = Field(None, description="Lens type")
    horizontal_fov: Optional[str] = Field(None, description="Horizontal field of view")


class CameraSummaryResponse(BaseModel):
    """Camera data for list view."""

    camera_id: str = Field(..., description="Unique camera identifier")
    model_name: str = Field(..., description="Official model name")
    display_name: str = Field(..., description="Human-readable name")
    source: str = Field(..., description="Vendor/source name")
    category: str = Field(..., description="Camera category")
    product_series: str = Field(..., description="Product series name")
    thumbnail_url: str = Field(..., description="URL to thumbnail image")
    specs: Optional[CameraSpecsSummary] = Field(
        None, description="Flattened specifications"
    )


class CameraDetailResponse(BaseModel):
    """Full camera data for detail view."""

    camera_id: str = Field(..., description="Unique camera identifier")
    model_name: str = Field(..., description="Official model name")
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Camera description")
    source: str = Field(..., description="Vendor/source name")
    category: str = Field(..., description="Camera category")
    product_series: str = Field(..., description="Product series name")
    image_urls: List[str] = Field(
        default_factory=list, description="Original image URLs"
    )
    image_files: List[str] = Field(
        default_factory=list, description="Local image paths"
    )
    datasheet_url: Optional[str] = Field(None, description="Datasheet PDF path")
    specs: Optional[CameraSpecsSummary] = Field(
        None, description="Flattened specifications"
    )


class PaginationInfo(BaseModel):
    """Pagination information."""

    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Items per page")
    total: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")


class CatalogListResponse(BaseModel):
    """Response for camera list endpoint."""

    data: List[CameraSummaryResponse] = Field(..., description="List of cameras")
    pagination: PaginationInfo = Field(..., description="Pagination info")
    facets: dict = Field(
        default_factory=dict, description="Available filters with counts"
    )


class FacetItem(BaseModel):
    """Single facet item with count."""

    value: str = Field(..., description="Facet value")
    count: int = Field(..., description="Number of items with this value")


class CatalogFacetsResponse(BaseModel):
    """Response for catalog facets endpoint."""

    vendors: List[FacetItem] = Field(default_factory=list, description="Vendor facets")
    categories: List[FacetItem] = Field(
        default_factory=list, description="Category facets"
    )
    series: List[FacetItem] = Field(default_factory=list, description="Series facets")


class CatalogCamerasRequest(BaseModel):
    """Query parameters for catalog list endpoint."""

    page: int = Field(default=1, ge=1, description="Page number")
    limit: int = Field(default=20, ge=1, le=100, description="Items per page")
    vendor: Optional[str] = Field(None, description="Filter by vendor")
    category: Optional[str] = Field(None, description="Filter by category")
    series: Optional[str] = Field(None, description="Filter by series")
    search: Optional[str] = Field(None, description="Search query")
