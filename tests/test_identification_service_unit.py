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


def _make_detection(x_min: int, y_min: int, x_max: int, y_max: int) -> CameraDetection:
    """Build a minimal CameraDetection for crop tests."""
    return CameraDetection(
        image_path=None,
        crop_path=None,
        bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        confidence=0.9,
        label="camera",
        class_id=0,
    )


class TestExpandBbox:
    """Tests for _expand_bbox."""

    def test_zero_margin_returns_bbox_unchanged(self):
        """A margin of 0.0 must leave the bbox exactly as-is."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=120)

        result = service._expand_bbox(bbox, 0.0)

        assert result is bbox

    def test_margin_expands_symmetrically(self):
        """Known bbox + margin produces the expected expanded box."""
        service = IdentificationService.__new__(IdentificationService)
        # 100x100 box, 10% margin -> 10 px on each side
        bbox = BoundingBox(x_min=50, y_min=60, x_max=150, y_max=160)

        result = service._expand_bbox(bbox, 0.1)

        assert result.x_min == 40
        assert result.y_min == 50
        assert result.x_max == 160
        assert result.y_max == 170

    def test_margin_scales_per_axis(self):
        """Margins use each axis's own side length."""
        service = IdentificationService.__new__(IdentificationService)
        # 200-wide, 100-tall box, 5% margin -> 10 px x, 5 px y
        bbox = BoundingBox(x_min=0, y_min=0, x_max=200, y_max=100)

        result = service._expand_bbox(bbox, 0.05)

        assert result.x_min == -10
        assert result.y_min == -5
        assert result.x_max == 210
        assert result.y_max == 105

    def test_expansion_is_not_clamped(self):
        """Expansion may go negative; clamping happens later."""
        service = IdentificationService.__new__(IdentificationService)
        bbox = BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100)

        result = service._expand_bbox(bbox, 0.2)

        assert result.x_min == -20
        assert result.y_min == -20


class TestExtractSingleCrop:
    """Tests for _extract_single_crop."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> IdentificationService:
        service = IdentificationService.__new__(IdentificationService)
        service.save_crops = False
        service.crop_dir = tmp_path / "crops"
        service.crop_margin = 0.0
        return service

    def test_returns_crop_info_for_valid_bbox(self, service, tmp_path):
        image = Image.new("RGB", (200, 200), color="red")
        detection = _make_detection(10, 20, 110, 120)

        crop_info = service._extract_single_crop(
            image, detection, idx=0, image_path=tmp_path / "src.jpg"
        )

        assert crop_info is not None
        assert crop_info.detection is detection
        assert crop_info.crop.size == (100, 100)
        assert crop_info.clamped_bbox.x_min == 10
        assert crop_info.clamped_bbox.x_max == 110
        assert crop_info.crop_path is None  # save_crops=False

    def test_clamps_oversized_bbox(self, service, tmp_path):
        image = Image.new("RGB", (50, 50))
        # bbox extends past image boundaries; should be clamped to (0..50, 0..50)
        detection = _make_detection(-10, -10, 100, 100)

        crop_info = service._extract_single_crop(
            image, detection, idx=0, image_path=tmp_path / "src.jpg"
        )

        assert crop_info is not None
        assert crop_info.clamped_bbox.x_min == 0
        assert crop_info.clamped_bbox.y_min == 0
        assert crop_info.clamped_bbox.x_max == 50
        assert crop_info.clamped_bbox.y_max == 50

    def test_returns_none_for_degenerate_bbox(self, service, tmp_path):
        image = Image.new("RGB", (200, 200))
        # zero-width bbox
        detection = _make_detection(50, 50, 50, 100)

        crop_info = service._extract_single_crop(
            image, detection, idx=0, image_path=tmp_path / "src.jpg"
        )

        assert crop_info is None

    def test_zero_margin_reproduces_exact_bbox_crop(self, service, tmp_path):
        """With crop_margin=0.0 the crop is pixel-identical to the raw bbox crop."""
        import numpy as np

        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")
        detection = _make_detection(30, 40, 130, 140)

        crop_info = service._extract_single_crop(
            image, detection, idx=0, image_path=tmp_path / "src.jpg"
        )

        expected = image.crop((30, 40, 130, 140))
        assert crop_info is not None
        assert np.array_equal(np.array(crop_info.crop), np.array(expected))

    def test_margin_expands_crop(self, service, tmp_path):
        """With crop_margin>0 the crop is symmetrically larger than the bbox."""
        service.crop_margin = 0.1
        image = Image.new("RGB", (200, 200), color="green")
        detection = _make_detection(50, 50, 150, 150)

        crop_info = service._extract_single_crop(
            image, detection, idx=0, image_path=tmp_path / "src.jpg"
        )

        assert crop_info is not None
        assert crop_info.clamped_bbox.x_min == 40
        assert crop_info.clamped_bbox.y_min == 40
        assert crop_info.clamped_bbox.x_max == 160
        assert crop_info.clamped_bbox.y_max == 160
        assert crop_info.crop.size == (120, 120)

    def test_margin_never_exceeds_image_bounds(self, service, tmp_path):
        """Expanded crops are clamped to the image edges."""
        service.crop_margin = 0.2
        image = Image.new("RGB", (100, 100))
        detection = _make_detection(0, 0, 100, 100)

        crop_info = service._extract_single_crop(
            image, detection, idx=0, image_path=tmp_path / "src.jpg"
        )

        assert crop_info is not None
        assert crop_info.clamped_bbox.x_min == 0
        assert crop_info.clamped_bbox.y_min == 0
        assert crop_info.clamped_bbox.x_max == 100
        assert crop_info.clamped_bbox.y_max == 100
        assert crop_info.crop.size == (100, 100)

    def test_saves_crop_when_enabled(self, tmp_path):
        service = IdentificationService.__new__(IdentificationService)
        service.save_crops = True
        service.crop_dir = tmp_path / "crops"
        service.crop_dir.mkdir()
        service.crop_margin = 0.0

        image = Image.new("RGB", (200, 200))
        detection = _make_detection(10, 20, 110, 120)
        source_path = tmp_path / "frame.jpg"

        crop_info = service._extract_single_crop(
            image, detection, idx=3, image_path=source_path
        )

        assert crop_info is not None
        assert crop_info.crop_path == service.crop_dir / "frame_det3.png"
        assert crop_info.crop_path.exists()


class TestExtractDetectionCrops:
    """Tests for _extract_detection_crops."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> IdentificationService:
        service = IdentificationService.__new__(IdentificationService)
        service.save_crops = False
        service.crop_dir = tmp_path / "crops"
        service.crop_margin = 0.0
        return service

    @pytest.fixture
    def image_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "scene.jpg"
        Image.new("RGB", (200, 200), color="blue").save(path)
        return path

    def test_returns_crops_for_each_valid_detection(self, service, image_path):
        detections = [
            _make_detection(10, 10, 50, 50),
            _make_detection(60, 60, 100, 100),
        ]

        crops = service._extract_detection_crops(image_path, detections)

        assert len(crops) == 2

    def test_skips_degenerate_detections(self, service, image_path):
        detections = [
            _make_detection(10, 10, 50, 50),
            _make_detection(60, 60, 60, 100),  # zero-width — skipped
            _make_detection(100, 100, 150, 150),
        ]

        crops = service._extract_detection_crops(image_path, detections)

        assert len(crops) == 2

    def test_returns_empty_for_empty_detections(self, service, image_path):
        crops = service._extract_detection_crops(image_path, [])
        assert crops == []
