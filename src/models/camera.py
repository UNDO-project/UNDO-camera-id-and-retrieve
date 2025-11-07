"""Camera data model."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class CategoryLink(BaseModel):
    r"""
    Represents a category link to camera collection.

    :ivar name: The display name of the category
    :ivar href: The relative URL path to the category page
    :ivar node_id: The Drupal node ID for the category
    """

    name: str = Field(..., description="Category name")
    href: str = Field(..., description="Relative URL path")
    node_id: Optional[str] = Field(None, description="Drupal node ID")


class CameraRecord(BaseModel):
    r"""
    Represents a single CCTV camera with its metadata and images.

    :ivar camera_id: Unique identifier for the camera
    :ivar model_name: Official model name/number
    :ivar display_name: Human-readable display name
    :ivar description: Camera description/tagline
    :ivar specifications: Camera technical specifications as key-value pairs
    :ivar image_url: URL to primary camera image
    :ivar image_data: Raw image bytes (populated after download)
    :ivar images: List of URLs to all carousel images
    :ivar datasheet_url: URL to PDF datasheet
    :ivar datasheet_pdf: Raw PDF bytes (populated after download)
    :ivar specifications_html: Technical specs parsed from HTML tables
    :ivar source: Which website/vendor this came from
    :ivar category: Camera category (e.g., "dome", "box", "bullet")
    :ivar product_category: Top-level category (e.g., "DOME CAMERAS")
    :ivar product_series: Product series name (e.g., "AXIS M30 Dome Camera Series")
    """

    camera_id: str = Field(..., description="Unique camera identifier")
    model_name: str = Field(..., description="Official model name/number")
    display_name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Camera description")
    specifications: dict[str, Any] = Field(
        default_factory=dict, description="Technical specifications"
    )
    image_url: Optional[str] = Field(None, description="URL to primary camera image")
    image_data: Optional[bytes] = Field(None, description="Raw image bytes")
    images: list[str] = Field(
        default_factory=list, description="URLs to all carousel images"
    )
    datasheet_url: Optional[str] = Field(None, description="URL to PDF datasheet")
    datasheet_pdf: Optional[bytes] = Field(None, description="Raw PDF bytes")
    specifications_html: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Technical specs from HTML tables"
    )
    source: str = Field(..., description="Data source (e.g., 'Axis')")
    category: str = Field(..., description="Camera category")
    product_category: str = Field(..., description="Top-level product category")
    product_series: str = Field(..., description="Product series name")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
