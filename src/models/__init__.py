"""Data models for CCTV camera information."""

from src.models.camera import CategoryLink, CameraRecord
from src.models.identification import (
    BoundingBox,
    CameraDetection,
    CameraMatch,
    RetrievalResult,
)

__all__ = [
    "CategoryLink",
    "CameraRecord",
    "BoundingBox",
    "CameraDetection",
    "CameraMatch",
    "RetrievalResult",
]
