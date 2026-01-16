"""Integration tests for API routes.

Tests all API endpoints using FastAPI TestClient with mocked services.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.models.camera import CameraRecord
from src.models.identification import (
    BoundingBox,
    CameraDetection,
    CameraMatch,
    RetrievalResult,
)


class FakeDetector:
    """Fake detector for testing."""

    def __init__(
        self, model_path: Path | str | None = None, conf_threshold: float = 0.25
    ):
        self.model_path = Path(model_path) if model_path is not None else None
        self.conf_threshold = conf_threshold

    def detect_from_path(self, image_path: Path | str) -> List[CameraDetection]:
        """Return one fake detection."""
        image_path = Path(image_path)
        bbox = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220)
        return [
            CameraDetection(
                image_path=image_path,
                crop_path=None,
                bbox=bbox,
                confidence=0.9,
                label="camera",
                class_id=0,
            )
        ]


class FakeCatalogIndex:
    """Fake catalog index for testing."""

    def __init__(self, embeddings_path: Path | str | None = None):
        self.embeddings_path = (
            Path(embeddings_path) if embeddings_path is not None else None
        )
        self.embeddings = np.random.rand(10, 512)  # Fake embeddings

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[CameraMatch]:
        """Return fake matches."""
        return [
            CameraMatch(
                camera_id="test-cam-1",
                score=0.85,
                catalog_image_path=Path("/fake/path/cam1.jpg"),
                source="Axis Communications",
                record=None,
            ),
            CameraMatch(
                camera_id="test-cam-2",
                score=0.72,
                catalog_image_path=Path("/fake/path/cam2.jpg"),
                source="HikVision",
                record=None,
            ),
        ][:top_k]


class FakeIdentificationService:
    """Fake identification service for testing."""

    def __init__(self, *args, **kwargs):
        self.detector = FakeDetector()
        self.index = FakeCatalogIndex()
        self.catalog = _make_fake_catalog()
        self.catalog_index = self.index  # Alias for health check compatibility

    def identify_from_image(
        self, image_path: Path | str, top_k: int = 5
    ) -> List[RetrievalResult]:
        """Return fake identification results."""
        image_path = Path(image_path)
        detections = self.detector.detect_from_path(image_path)

        results = []
        for detection in detections:
            # Get matches from index
            query_vector = np.random.rand(512)  # Fake embedding
            matches = self.index.search(query_vector, top_k=top_k)

            # Enrich matches with catalog records
            for match in matches:
                if match.camera_id in self.catalog:
                    match.record = self.catalog[match.camera_id]

            results.append(
                RetrievalResult(
                    detection=detection,
                    matches=matches,
                )
            )

        return results


def _make_fake_catalog() -> Dict[str, CameraRecord]:
    """Create a small fake catalog for testing."""
    return {
        "test-cam-1": CameraRecord(
            camera_id="test-cam-1",
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
        ),
        "test-cam-2": CameraRecord(
            camera_id="test-cam-2",
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
            source="HikVision",
            category="PTZ Camera",
            product_category="Test Category",
            product_series="Test Series",
        ),
    }


@pytest.fixture
def test_client(monkeypatch: Any) -> TestClient:
    """Create a test client with mocked services."""
    # Patch the IdentificationService before importing the app
    from src import identification

    monkeypatch.setattr(
        identification.service, "IdentificationService", FakeIdentificationService
    )
    monkeypatch.setattr(identification.service, "Detector", FakeDetector)
    monkeypatch.setattr(identification.service, "CatalogIndex", FakeCatalogIndex)
    monkeypatch.setattr(
        identification.service, "load_catalog", lambda _: _make_fake_catalog()
    )

    # Import app after patching
    from src.api.main import app

    return TestClient(app)


def test_health_endpoint(test_client: TestClient) -> None:
    """Test GET /api/v1/health endpoint."""
    response = test_client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_readiness_endpoint_service_not_ready(test_client: TestClient) -> None:
    """Test GET /api/v1/health/ready when service is not initialized."""
    # The service might not be ready on first call
    response = test_client.get("/api/v1/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "catalog_loaded" in data
    assert "embeddings_ready" in data
    assert "detector_ready" in data


def test_catalog_stats_endpoint(test_client: TestClient, monkeypatch: Any) -> None:
    """Test GET /api/v1/catalog/stats endpoint."""
    # Ensure service is initialized by accessing it
    from src.api.dependencies import state

    if not state.is_ready():
        _ = state.service  # Accessing the property triggers initialization

    response = test_client.get("/api/v1/catalog/stats")

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "total_cameras" in data
    assert "vendor_count" in data
    assert "embeddings_loaded" in data
    assert "embedding_count" in data
    assert "catalog_path" in data


def test_catalog_reload_endpoint(test_client: TestClient) -> None:
    """Test POST /api/v1/catalog/reload endpoint."""
    response = test_client.post("/api/v1/catalog/reload")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "message" in data


def test_identify_endpoint_with_valid_image(
    test_client: TestClient, tmp_path: Path
) -> None:
    """Test POST /api/v1/identify with a valid image."""
    # Create a test image
    image = Image.new("RGB", (320, 240), color=(100, 150, 200))
    image_path = tmp_path / "test_image.jpg"
    image.save(image_path)

    # Read image as bytes for upload
    with open(image_path, "rb") as f:
        files = {"image": ("test_image.jpg", f, "image/jpeg")}
        response = test_client.post(
            "/api/v1/identify", files=files, params={"top_k": 3, "min_similarity": 0.3}
        )

    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert data["success"] is True
    assert "detections_count" in data
    assert "results" in data
    assert "processing_time_ms" in data

    # Check results structure
    if data["detections_count"] > 0:
        result = data["results"][0]
        assert "detection" in result
        assert "matches" in result

        # Check detection structure
        detection = result["detection"]
        assert "bbox" in detection
        assert "confidence" in detection

        # Check matches structure
        if len(result["matches"]) > 0:
            match = result["matches"][0]
            assert "camera_id" in match
            assert "score" in match
            assert "source" in match


def test_identify_endpoint_with_parameters(
    test_client: TestClient, tmp_path: Path
) -> None:
    """Test POST /api/v1/identify with custom parameters."""
    # Create a test image
    image = Image.new("RGB", (640, 480), color=(50, 100, 150))
    image_path = tmp_path / "test_scene.jpg"
    image.save(image_path)

    with open(image_path, "rb") as f:
        files = {"image": ("test_scene.jpg", f, "image/jpeg")}
        response = test_client.post(
            "/api/v1/identify", files=files, params={"top_k": 2, "min_similarity": 0.5}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True


def test_identify_endpoint_missing_image(test_client: TestClient) -> None:
    """Test POST /api/v1/identify without uploading an image."""
    response = test_client.post("/api/v1/identify")

    # Should return 422 Unprocessable Entity (missing required field)
    assert response.status_code == 422


def test_identify_endpoint_invalid_image_format(
    test_client: TestClient, tmp_path: Path
) -> None:
    """Test POST /api/v1/identify with an invalid image file."""
    # Create a fake non-image file
    fake_file = tmp_path / "not_an_image.txt"
    fake_file.write_text("This is not an image")

    with open(fake_file, "rb") as f:
        files = {"image": ("not_an_image.txt", f, "text/plain")}
        response = test_client.post("/api/v1/identify", files=files)

    # Should return an error (either 400 or 500 depending on error handling)
    assert response.status_code in [400, 422, 500]


def test_root_endpoint(test_client: TestClient) -> None:
    """Test GET / endpoint (API info)."""
    response = test_client.get("/")

    assert response.status_code == 200
    data = response.json()

    # Should contain API information
    assert "name" in data or "message" in data or "version" in data


def test_docs_endpoint(test_client: TestClient) -> None:
    """Test that Swagger UI documentation is accessible."""
    response = test_client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_redoc_endpoint(test_client: TestClient) -> None:
    """Test that ReDoc documentation is accessible."""
    response = test_client.get("/redoc")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
