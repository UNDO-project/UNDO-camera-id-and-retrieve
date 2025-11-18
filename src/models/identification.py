"""Identification and retrieval data models."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.models.camera import CameraRecord


class BoundingBox(BaseModel):
    r"""Axis-aligned bounding box for a detection.

    :ivar x_min: Left coordinate of the bounding box (inclusive)
    :ivar y_min: Top coordinate of the bounding box (inclusive)
    :ivar x_max: Right coordinate of the bounding box (exclusive)
    :ivar y_max: Bottom coordinate of the bounding box (exclusive)
    """

    x_min: int = Field(
        ..., description="Left coordinate of the bounding box (inclusive)"
    )
    y_min: int = Field(
        ..., description="Top coordinate of the bounding box (inclusive)"
    )
    x_max: int = Field(
        ..., description="Right coordinate of the bounding box (exclusive)"
    )
    y_max: int = Field(
        ..., description="Bottom coordinate of the bounding box (exclusive)"
    )


class CameraDetection(BaseModel):
    r"""Single camera detection in an image.

    :ivar image_path: Path to the original input image
    :ivar crop_path: Optional path to the cropped detection patch
    :ivar bbox: Bounding box for the detection
    :ivar confidence: Detection confidence score (0.0-1.0)
    :ivar label: Detected class label (e.g., "camera")
    :ivar class_id: Optional numeric class identifier from the detector
    """

    image_path: Path = Field(..., description="Path to the original input image")
    crop_path: Optional[Path] = Field(
        None, description="Optional path to the cropped detection patch"
    )
    bbox: BoundingBox = Field(..., description="Bounding box for the detection")
    confidence: float = Field(..., description="Detection confidence score (0.0-1.0)")
    label: str = Field("camera", description="Detected class label")
    class_id: Optional[int] = Field(
        None, description="Optional numeric class identifier from the detector"
    )


class CameraMatch(BaseModel):
    r"""Match between a detection and a catalog camera entry.

    :ivar camera_id: Identifier of the matched catalog camera
    :ivar score: Similarity score between query and catalog image
    :ivar catalog_image_path: Path to the catalog image used for matching
    :ivar source: Data source of the matched camera (e.g., vendor name)
    :ivar record: Optional full camera record for the matched item
    """

    camera_id: str = Field(..., description="Identifier of the matched catalog camera")
    score: float = Field(
        ..., description="Similarity score between query and catalog image"
    )
    catalog_image_path: Optional[Path] = Field(
        None, description="Path to the catalog image used for matching"
    )
    source: str = Field(..., description="Data source of the matched camera")
    record: Optional[CameraRecord] = Field(
        None, description="Optional full camera record for the matched item"
    )


class RetrievalResult(BaseModel):
    r"""Complete identification result for a single detection.

    :ivar detection: The original camera detection
    :ivar matches: List of candidate catalog matches ordered by score
    """

    detection: CameraDetection = Field(..., description="The original camera detection")
    matches: list[CameraMatch] = Field(
        default_factory=list,
        description="List of candidate catalog matches ordered by score",
    )
