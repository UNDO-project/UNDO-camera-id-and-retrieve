"""Unit tests for IdentificationService extracted methods."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.identification.service import CropInfo, IdentificationService
from src.models.identification import BoundingBox, CameraDetection, CameraMatch


class TestCropInfo:
    """Tests for the CropInfo dataclass."""

    def test_crop_info_creation(self):
        """Test creating a CropInfo instance."""
        detection = MagicMock(spec=CameraDetection)
        crop = MagicMock(spec=Image.Image)
        bbox = MagicMock(spec=BoundingBox)

        crop_info = CropInfo(
            detection=detection,
            crop=crop,
            crop_path=Path("/tmp/test.png"),
            clamped_bbox=bbox,
        )

        assert crop_info.detection == detection
        assert crop_info.crop == crop
        assert crop_info.crop_path == Path("/tmp/test.png")
        assert crop_info.clamped_bbox == bbox


class TestClampBbox:
    """Tests for the _clamp_bbox method."""

    def test_clamp_bbox_within_bounds(self):
        """Test clamping a bbox that is already within image bounds."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200)

        result = service._clamp_bbox(bbox, width=500, height=500)

        assert result.x_min == 10
        assert result.y_min == 20
        assert result.x_max == 100
        assert result.y_max == 200

    def test_clamp_bbox_negative_values(self):
        """Test clamping a bbox with negative coordinates."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=-10, y_min=-20, x_max=100, y_max=200)

        result = service._clamp_bbox(bbox, width=500, height=500)

        assert result.x_min == 0
        assert result.y_min == 0
        assert result.x_max == 100
        assert result.y_max == 200

    def test_clamp_bbox_exceeds_bounds(self):
        """Test clamping a bbox that exceeds image bounds."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=10, y_min=20, x_max=600, y_max=700)

        result = service._clamp_bbox(bbox, width=500, height=500)

        assert result.x_min == 10
        assert result.y_min == 20
        assert result.x_max == 500
        assert result.y_max == 500

    def test_clamp_bbox_completely_outside(self):
        """Test clamping a bbox completely outside image bounds."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=-100, y_min=-100, x_max=-10, y_max=-10)

        result = service._clamp_bbox(bbox, width=500, height=500)

        # When bbox is completely outside, clamping still applies min/max
        # Note: min(width, negative) returns the negative value
        assert result.x_min == 0
        assert result.y_min == 0
        assert result.x_max == -10  # min(500, -10) = -10
        assert result.y_max == -10  # min(500, -10) = -10
        # The bbox will be invalid (is_valid_bbox returns False)


class TestIsValidBbox:
    """Tests for the _is_valid_bbox method."""

    def test_valid_bbox(self):
        """Test a valid bbox with positive area."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=200)

        assert service._is_valid_bbox(bbox) is True

    def test_invalid_bbox_zero_width(self):
        """Test an invalid bbox with zero width."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=10, y_min=20, x_max=10, y_max=200)

        assert service._is_valid_bbox(bbox) is False

    def test_invalid_bbox_zero_height(self):
        """Test an invalid bbox with zero height."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=10, y_min=20, x_max=100, y_max=20)

        assert service._is_valid_bbox(bbox) is False

    def test_invalid_bbox_negative_area(self):
        """Test an invalid bbox with negative area."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=100, y_min=200, x_max=10, y_max=20)

        assert service._is_valid_bbox(bbox) is False


class TestFilterAndEnrichMatches:
    """Tests for the _filter_and_enrich_matches method."""

    @pytest.fixture
    def service(self):
        service = IdentificationService.__new__(IdentificationService)
        service.min_similarity = 0.5
        # Create proper CameraRecord objects instead of MagicMocks
        from src.models.camera import CameraRecord

        service.catalog = {
            "cam-1": CameraRecord(
                camera_id="cam-1",
                model_name="Test Camera 1",
                display_name="Test Cam 1",
                description="",
                specifications={},
                image_url=None,
                images=[],
                image_files=[],
                datasheet_url=None,
                datasheet_file=None,
                specifications_html={},
                source="Test",
                category="Test",
                product_category="Test",
                product_series="Test",
            ),
        }
        return service

    def test_filter_matches_above_threshold(self, service):
        """Test filtering matches above similarity threshold."""
        matches = [
            CameraMatch(
                camera_id="cam-1",
                score=0.8,
                catalog_image_path=Path("/dummy/1.png"),
                source="Test",
            ),
            CameraMatch(
                camera_id="cam-2",
                score=0.3,
                catalog_image_path=Path("/dummy/2.png"),
                source="Test",
            ),
        ]

        result = service._filter_and_enrich_matches(matches, 0.5)

        assert len(result) == 1
        assert result[0].camera_id == "cam-1"
        assert result[0].record is not None

    def test_filter_matches_missing_from_catalog(self, service):
        """Test filtering matches not in catalog."""
        matches = [
            CameraMatch(
                camera_id="cam-3",  # Not in catalog
                score=0.8,
                catalog_image_path=Path("/dummy/3.png"),
                source="Test",
            ),
        ]

        result = service._filter_and_enrich_matches(matches, 0.5)

        assert len(result) == 1
        assert result[0].record is None  # Not enriched


class TestValidateImagePath:
    """Tests for the _validate_image_path method."""

    def test_validate_existing_image(self, tmp_path: Path):
        """Test validation with existing image file."""
        service = IdentificationService.__new__(IdentificationService)
        image_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100))
        img.save(image_path)

        # Should not raise
        service._validate_image_path(image_path)

    def test_validate_nonexistent_image(self, tmp_path: Path):
        """Test validation with non-existent image file raises error."""
        service = IdentificationService.__new__(IdentificationService)
        image_path = tmp_path / "nonexistent.jpg"

        with pytest.raises(FileNotFoundError):
            service._validate_image_path(image_path)
