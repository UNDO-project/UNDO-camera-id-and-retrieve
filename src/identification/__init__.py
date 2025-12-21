"""Camera identification and retrieval package.

This package contains the components required to detect cameras in
images and retrieve matching camera records from the catalog dataset.

The implementation is organized into the following modules:

- ``detector``: Model wrapper for camera detection.
- ``embeddings``: Image embedding utilities.
- ``index``: Catalog indexing and nearest-neighbor search.
- ``service``: High-level identification pipeline orchestration.
- ``cli``: Command-line interface entry point.
"""

from src.identification.detector import Detector
from src.identification.index import CatalogIndex
from src.identification.service import IdentificationService

__all__ = [
    "Detector",
    "CatalogIndex",
    "IdentificationService",
]
