"""Tests for identification CLI."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.models.identification import (
    BoundingBox,
    CameraDetection,
    CameraMatch,
    RetrievalResult,
)
from src.models.camera import CameraRecord


@pytest.fixture
def mock_service():
    """Mock IdentificationService."""
    service = MagicMock()

    # Create fake results
    detection = CameraDetection(
        image_path=Path("/fake/image.jpg"),
        crop_path=Path("/fake/crop.jpg"),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        confidence=0.92,
        label="camera",
        class_id=0,
    )

    match = CameraMatch(
        camera_id="test-cam-1",
        score=0.85,
        catalog_image_path=Path("/fake/catalog.jpg"),
        source="Axis Communications",
        record=CameraRecord(
            camera_id="test-cam-1",
            model_name="AXIS M3027-PVE",
            display_name="AXIS M3027-PVE",
            description="Test camera",
            specifications={},
            image_url=None,
            images=[],
            image_files=[],
            datasheet_url=None,
            datasheet_file=None,
            specifications_html={},
            source="Axis Communications",
            category="Network Camera",
            product_category="Network Cameras",
            product_series="AXIS M3000",
        ),
    )

    result = RetrievalResult(detection=detection, matches=[match])
    service.identify_from_image.return_value = [result]

    return service


def test_cli_identify_basic(tmp_path: Path, mock_service: Any, capsys: Any) -> None:
    """Test basic identification CLI with required arguments."""
    # Create a fake image file
    image_path = tmp_path / "test_image.jpg"
    image_path.write_text("fake image")

    # Mock sys.argv
    test_args = ["cidar-identify", "--image", str(image_path)]

    with patch("sys.argv", test_args):
        with patch(
            "src.identification.cli.IdentificationService", return_value=mock_service
        ):
            from src.identification.cli import main

            main()

    # Verify service was called correctly
    mock_service.identify_from_image.assert_called_once()
    call_args = mock_service.identify_from_image.call_args
    assert call_args[0][0] == image_path
    assert call_args[1]["top_k"] == 5  # default value

    # Check output
    captured = capsys.readouterr()
    assert "Detection 1:" in captured.out
    assert "Confidence: 0.920" in captured.out
    assert "test-cam-1" in captured.out
    assert "score=0.850" in captured.out


def test_cli_identify_with_parameters(tmp_path: Path, mock_service: Any) -> None:
    """Test CLI with custom parameters."""
    image_path = tmp_path / "test_scene.jpg"
    image_path.write_text("fake image")

    parquet_path = tmp_path / "custom_products.parquet"
    embeddings_path = tmp_path / "custom_embeddings.npz"

    test_args = [
        "cidar-identify",
        "--image",
        str(image_path),
        "--top-k",
        "3",
        "--min-similarity",
        "0.5",
        "--no-save-crops",
        "--parquet",
        str(parquet_path),
        "--embeddings",
        str(embeddings_path),
    ]

    with patch("sys.argv", test_args):
        with patch(
            "src.identification.cli.IdentificationService", return_value=mock_service
        ) as mock_init:
            from src.identification.cli import main

            main()

    # Verify IdentificationService was initialized with correct params
    mock_init.assert_called_once()
    init_kwargs = mock_init.call_args[1]
    assert init_kwargs["parquet_path"] == parquet_path
    assert init_kwargs["embeddings_path"] == embeddings_path
    assert init_kwargs["min_similarity"] == 0.5
    assert init_kwargs["save_crops"] is False

    # Verify identify_from_image was called with correct top_k
    call_args = mock_service.identify_from_image.call_args
    assert call_args[1]["top_k"] == 3


def test_cli_identify_missing_image() -> None:
    """Test CLI with non-existent image file."""
    test_args = ["cidar-identify", "--image", "/nonexistent/image.jpg"]

    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as exc_info:
            from src.identification.cli import main

            main()

    # Should exit with error message
    assert "not found" in str(exc_info.value)


def test_cli_identify_no_detections(tmp_path: Path, capsys: Any) -> None:
    """Test CLI when no cameras are detected."""
    image_path = tmp_path / "empty_scene.jpg"
    image_path.write_text("fake image")

    # Mock service that returns no detections
    mock_service = MagicMock()
    mock_service.identify_from_image.return_value = []

    test_args = ["cidar-identify", "--image", str(image_path)]

    with patch("sys.argv", test_args):
        with patch(
            "src.identification.cli.IdentificationService", return_value=mock_service
        ):
            from src.identification.cli import main

            main()

    captured = capsys.readouterr()
    assert "No cameras detected" in captured.out


def test_cli_identify_no_matches(tmp_path: Path, capsys: Any) -> None:
    """Test CLI when camera is detected but no catalog matches found."""
    image_path = tmp_path / "unknown_camera.jpg"
    image_path.write_text("fake image")

    # Mock service with detection but no matches
    mock_service = MagicMock()

    detection = CameraDetection(
        image_path=image_path,
        crop_path=None,
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        confidence=0.88,
        label="camera",
        class_id=0,
    )

    result = RetrievalResult(detection=detection, matches=[])
    mock_service.identify_from_image.return_value = [result]

    test_args = ["cidar-identify", "--image", str(image_path)]

    with patch("sys.argv", test_args):
        with patch(
            "src.identification.cli.IdentificationService", return_value=mock_service
        ):
            from src.identification.cli import main

            main()

    captured = capsys.readouterr()
    assert "Detection 1:" in captured.out
    assert "No matches above similarity threshold" in captured.out


def test_cli_identify_multiple_detections(tmp_path: Path, capsys: Any) -> None:
    """Test CLI with multiple camera detections."""
    image_path = tmp_path / "multi_camera.jpg"
    image_path.write_text("fake image")

    # Mock service with multiple detections
    mock_service = MagicMock()

    detection1 = CameraDetection(
        image_path=image_path,
        crop_path=Path("/fake/crop1.jpg"),
        bbox=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        confidence=0.95,
        label="camera",
        class_id=0,
    )

    detection2 = CameraDetection(
        image_path=image_path,
        crop_path=Path("/fake/crop2.jpg"),
        bbox=BoundingBox(x_min=200, y_min=100, x_max=300, y_max=200),
        confidence=0.87,
        label="camera",
        class_id=0,
    )

    match1 = CameraMatch(
        camera_id="cam-1",
        score=0.82,
        catalog_image_path=None,
        source="Axis Communications",
        record=None,
    )

    match2 = CameraMatch(
        camera_id="cam-2",
        score=0.75,
        catalog_image_path=None,
        source="HikVision",
        record=None,
    )

    result1 = RetrievalResult(detection=detection1, matches=[match1])
    result2 = RetrievalResult(detection=detection2, matches=[match2])

    mock_service.identify_from_image.return_value = [result1, result2]

    test_args = ["cidar-identify", "--image", str(image_path)]

    with patch("sys.argv", test_args):
        with patch(
            "src.identification.cli.IdentificationService", return_value=mock_service
        ):
            from src.identification.cli import main

            main()

    captured = capsys.readouterr()
    assert "Detection 1:" in captured.out
    assert "Detection 2:" in captured.out
    assert "Confidence: 0.950" in captured.out
    assert "Confidence: 0.870" in captured.out
    assert "cam-1" in captured.out
    assert "cam-2" in captured.out
