"""Integration-style tests for the IdentificationService.

These tests exercise the orchestration logic of the identification
pipeline (detection -> cropping -> embedding -> catalog search) while
mocking out the heavy YOLO and embedding/index components so that they
run quickly and deterministically.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from src.identification import service as service_module
from src.models.camera import CameraRecord
from src.models.identification import BoundingBox, CameraDetection, CameraMatch


class FakeDetector:
    """Fake detector that returns a single fixed detection."""

    def __init__(
        self, model_path: Path | str | None = None, conf_threshold: float = 0.25
    ) -> None:  # noqa: D401
        """Accept any arguments without using a real model."""
        self.model_path = Path(model_path) if model_path is not None else None
        self.conf_threshold = conf_threshold

    def detect_from_path(self, image_path: Path | str) -> List[CameraDetection]:
        """Return one detection in the centre of the image.

        The exact bounding box is not important; it only needs to be
        valid so that cropping works.
        """
        image_path = Path(image_path)
        bbox = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220)
        detection = CameraDetection(
            image_path=image_path,
            crop_path=None,
            bbox=bbox,
            confidence=0.9,
            label="camera",
            class_id=0,
        )
        return [detection]


class FakeIndex:
    """Fake catalog index that ignores the query vector.

    Always returns two CameraMatch objects with different scores.
    """

    def __init__(self, embeddings_path: Path | str | None = None) -> None:  # noqa: D401
        """Accept any embeddings path without reading from disk."""
        self.embeddings_path = (
            Path(embeddings_path) if embeddings_path is not None else None
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[CameraMatch]:
        """Return a fixed list of matches irrespective of the query."""
        matches = [
            CameraMatch(
                camera_id="cam-1",
                score=0.9,
                catalog_image_path=Path("/dummy/path/cam-1.png"),
                source="Axis Communications",
                record=None,
            ),
            CameraMatch(
                camera_id="cam-2",
                score=0.5,
                catalog_image_path=Path("/dummy/path/cam-2.png"),
                source="Axis Communications",
                record=None,
            ),
        ]
        return matches[:top_k]


def _make_fake_catalog() -> Dict[str, CameraRecord]:
    """Create a small in-memory catalog for testing."""
    catalog: Dict[str, CameraRecord] = {}

    catalog["cam-1"] = CameraRecord(
        camera_id="cam-1",
        model_name="Test Camera 1",
        display_name="Test Cam 1",
        description="First test camera",
        specifications={},
        image_url=None,
        images=[],
        image_files=[],
        datasheet_url=None,
        datasheet_file=None,
        specifications_html={},
        source="Axis Communications",
        category="Network Camera",
        product_category="Test Category",
        product_series="Test Series",
    )

    catalog["cam-2"] = CameraRecord(
        camera_id="cam-2",
        model_name="Test Camera 2",
        display_name="Test Cam 2",
        description="Second test camera",
        specifications={},
        image_url=None,
        images=[],
        image_files=[],
        datasheet_url=None,
        datasheet_file=None,
        specifications_html={},
        source="Axis Communications",
        category="Network Camera",
        product_category="Test Category",
        product_series="Test Series",
    )

    return catalog


def test_identification_service_integration_with_mocks(
    tmp_path: Path, monkeypatch: Any
) -> None:
    r"""IdentificationService should orchestrate detection and retrieval.

    This test uses:

    - FakeDetector: avoids YOLOv8 weights and GPU.
    - FakeIndex: avoids real embeddings and nearest-neighbor search.
    - In-memory catalog: avoids reading parquet.

    It asserts that the service returns a RetrievalResult with matches
    enriched by the catalog records.
    """

    # Patch heavy components in the service module before creating the service
    monkeypatch.setattr(service_module, "Detector", FakeDetector)
    monkeypatch.setattr(service_module, "CatalogIndex", FakeIndex)
    monkeypatch.setattr(service_module, "load_catalog", lambda _: _make_fake_catalog())

    # Create a dummy input image on disk
    from PIL import Image

    image_path = tmp_path / "test_scene.jpg"
    img = Image.new("RGB", (320, 240), color=(128, 128, 128))
    img.save(image_path)

    from src.identification.service import IdentificationService

    service = IdentificationService(
        model_path=tmp_path / "dummy.pt",  # not used by FakeDetector
        parquet_path=tmp_path / "dummy.parquet",  # not used by patched load_catalog
        embeddings_path=tmp_path / "dummy_embeddings.npz",  # not used by FakeIndex
        min_similarity=0.4,
        save_crops=False,
    )

    results = service.identify_from_image(image_path, top_k=3)

    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]

    # One detection with confidence propagated from FakeDetector
    assert result.detection.confidence == pytest.approx(0.9, rel=1e-6)

    # Matches should be filtered by min_similarity (0.4), so both cam-1 and cam-2 remain
    assert len(result.matches) == 2

    match_ids = {m.camera_id for m in result.matches}
    assert match_ids == {"cam-1", "cam-2"}

    # Records should be attached from the fake catalog
    for m in result.matches:
        assert m.record is not None
        assert m.record.camera_id == m.camera_id
