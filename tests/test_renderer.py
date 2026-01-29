"""Tests for frame annotation renderer."""

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.identification.renderer import FrameRenderer, RenderConfig
from src.models.identification import BoundingBox, CameraDetection, CameraMatch


@pytest.fixture
def test_frame():
    """Create a test frame (numpy array)."""
    # Create 640x480 RGB image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some color
    frame[:, :] = (100, 150, 200)  # RGB
    return frame


@pytest.fixture
def test_detection():
    """Create a sample detection."""
    return CameraDetection(
        image_path=None,
        crop_path=None,
        bbox=BoundingBox(x_min=100, y_min=100, x_max=300, y_max=250),
        confidence=0.95,
        label="camera",
        class_id=0,
    )


@pytest.fixture
def test_matches():
    """Create sample matches."""
    return [
        [
            CameraMatch(
                camera_id="axis-m3057",
                score=0.87,
                catalog_image_path=Path("data/images/axis/m3057.jpg"),
                source="Axis Communications",
                record=None,
            )
        ]
    ]


@pytest.fixture
def renderer():
    """Create default renderer."""
    return FrameRenderer()


@pytest.fixture
def minimal_renderer():
    """Create renderer with minimal style."""
    config = RenderConfig(style="minimal")
    return FrameRenderer(config)


class TestRenderConfig:
    """Test suite for RenderConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RenderConfig()

        assert config.box_color == (0, 255, 0)  # Green
        assert config.box_thickness == 2
        assert config.font_scale == 0.6
        assert config.show_confidence is True
        assert config.show_manufacturer is True
        assert config.style == "detailed"

    def test_custom_config(self):
        """Test custom configuration."""
        config = RenderConfig(
            box_color=(255, 0, 0),  # Red
            box_thickness=3,
            style="minimal",
            show_confidence=False,
        )

        assert config.box_color == (255, 0, 0)
        assert config.box_thickness == 3
        assert config.style == "minimal"
        assert config.show_confidence is False


class TestFrameRenderer:
    """Test suite for FrameRenderer."""

    def test_initialization_default(self):
        """Test renderer initializes with default config."""
        renderer = FrameRenderer()

        assert renderer.config is not None
        assert renderer.config.style == "detailed"

    def test_initialization_custom_config(self):
        """Test renderer initializes with custom config."""
        config = RenderConfig(style="minimal")
        renderer = FrameRenderer(config)

        assert renderer.config.style == "minimal"

    def test_annotate_frame_empty_detections(self, renderer, test_frame):
        """Test annotating frame with no detections."""
        annotated = renderer.annotate_frame(test_frame, [])

        # Should return a frame without errors
        assert annotated.shape == test_frame.shape
        assert isinstance(annotated, np.ndarray)

    def test_annotate_frame_single_detection(
        self, renderer, test_frame, test_detection
    ):
        """Test annotating frame with single detection."""
        annotated = renderer.annotate_frame(test_frame, [test_detection])

        assert annotated.shape == test_frame.shape
        # Frame should be modified (different from input)
        assert not np.array_equal(annotated, test_frame)

    def test_annotate_frame_multiple_detections(self, renderer, test_frame):
        """Test annotating frame with multiple detections."""
        detections = [
            CameraDetection(
                image_path=None,
                crop_path=None,
                bbox=BoundingBox(x_min=50, y_min=50, x_max=150, y_max=150),
                confidence=0.9,
                label="camera",
                class_id=0,
            ),
            CameraDetection(
                image_path=None,
                crop_path=None,
                bbox=BoundingBox(x_min=200, y_min=200, x_max=400, y_max=350),
                confidence=0.85,
                label="camera",
                class_id=0,
            ),
        ]

        annotated = renderer.annotate_frame(test_frame, detections)

        assert annotated.shape == test_frame.shape
        assert not np.array_equal(annotated, test_frame)

    def test_annotate_frame_with_matches(
        self, renderer, test_frame, test_detection, test_matches
    ):
        """Test annotating with matches (full mode)."""
        annotated = renderer.annotate_frame(
            test_frame, [test_detection], matches=test_matches
        )

        assert annotated.shape == test_frame.shape
        # Should have manufacturer label in detailed mode
        assert not np.array_equal(annotated, test_frame)

    def test_annotate_frame_minimal_style(
        self, minimal_renderer, test_frame, test_detection
    ):
        """Test minimal style rendering."""
        annotated = minimal_renderer.annotate_frame(test_frame, [test_detection])

        assert annotated.shape == test_frame.shape

    def test_annotate_frame_no_confidence(self, test_frame, test_detection):
        """Test rendering without confidence labels."""
        config = RenderConfig(show_confidence=False)
        renderer = FrameRenderer(config)

        annotated = renderer.annotate_frame(test_frame, [test_detection])

        assert annotated.shape == test_frame.shape

    def test_annotate_frame_no_manufacturer(
        self, test_frame, test_detection, test_matches
    ):
        """Test rendering without manufacturer labels."""
        config = RenderConfig(show_manufacturer=False)
        renderer = FrameRenderer(config)

        annotated = renderer.annotate_frame(
            test_frame, [test_detection], matches=test_matches
        )

        assert annotated.shape == test_frame.shape

    def test_annotate_frame_edge_cases(self, renderer, test_frame):
        """Test edge case bounding boxes."""
        # Box at frame edges
        edge_detection = CameraDetection(
            image_path=None,
            crop_path=None,
            bbox=BoundingBox(x_min=0, y_min=0, x_max=50, y_max=50),
            confidence=0.8,
            label="camera",
            class_id=0,
        )

        # Should handle without errors
        annotated = renderer.annotate_frame(test_frame, [edge_detection])
        assert annotated.shape == test_frame.shape

    def test_encode_jpeg_default_quality(self, renderer, test_frame):
        """Test JPEG encoding with default quality."""
        jpeg_bytes = renderer.encode_jpeg(test_frame)

        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0
        # Should have JPEG magic bytes
        assert jpeg_bytes[:2] == b"\xff\xd8"

    def test_encode_jpeg_custom_quality(self, renderer, test_frame):
        """Test JPEG encoding with custom quality."""
        high_quality = renderer.encode_jpeg(test_frame, quality=95)
        low_quality = renderer.encode_jpeg(test_frame, quality=50)

        # High quality should produce larger file
        assert len(high_quality) > len(low_quality)

    def test_encode_jpeg_quality_bounds(self, renderer, test_frame):
        """Test JPEG quality is clamped to valid range."""
        # Should clamp to 1-100 range
        jpeg_low = renderer.encode_jpeg(test_frame, quality=0)  # Clamped to 1
        jpeg_high = renderer.encode_jpeg(test_frame, quality=150)  # Clamped to 100

        assert isinstance(jpeg_low, bytes)
        assert isinstance(jpeg_high, bytes)

    def test_encode_jpeg_roundtrip(self, renderer, test_frame):
        """Test encoding and decoding JPEG."""
        jpeg_bytes = renderer.encode_jpeg(test_frame, quality=85)

        # Decode with PIL
        import io

        decoded = Image.open(io.BytesIO(jpeg_bytes))
        decoded_array = np.array(decoded)

        # Should have same dimensions
        assert decoded_array.shape[:2] == test_frame.shape[:2]

    def test_annotate_and_encode(self, renderer, test_frame, test_detection):
        """Test convenience method for annotate + encode."""
        jpeg_bytes = renderer.annotate_and_encode(test_frame, [test_detection])

        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0
        assert jpeg_bytes[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_annotate_and_encode_with_matches(
        self, renderer, test_frame, test_detection, test_matches
    ):
        """Test annotate_and_encode with matches."""
        jpeg_bytes = renderer.annotate_and_encode(
            test_frame, [test_detection], matches=test_matches, quality=90
        )

        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 0


class TestPerformance:
    """Performance tests for renderer."""

    def test_annotate_performance_single_detection(
        self, renderer, test_frame, test_detection
    ):
        """Test annotation performance (should be <5ms)."""
        # Warm up
        renderer.annotate_frame(test_frame, [test_detection])

        # Measure
        start = time.time()
        for _ in range(10):
            renderer.annotate_frame(test_frame, [test_detection])
        elapsed_ms = ((time.time() - start) / 10) * 1000

        # Average should be well under 5ms
        assert elapsed_ms < 5

    def test_annotate_performance_multiple_detections(self, renderer, test_frame):
        """Test annotation performance with 5 detections."""
        detections = [
            CameraDetection(
                image_path=None,
                crop_path=None,
                bbox=BoundingBox(
                    x_min=i * 100, y_min=i * 50, x_max=i * 100 + 100, y_max=i * 50 + 100
                ),
                confidence=0.9,
                label="camera",
                class_id=0,
            )
            for i in range(5)
        ]

        # Warm up
        renderer.annotate_frame(test_frame, detections)

        # Measure
        start = time.time()
        for _ in range(10):
            renderer.annotate_frame(test_frame, detections)
        elapsed_ms = ((time.time() - start) / 10) * 1000

        # Should still be fast with multiple detections
        assert elapsed_ms < 10  # Generous threshold for 5 detections

    def test_encode_performance(self, renderer, test_frame):
        """Test JPEG encoding performance."""
        # Warm up
        renderer.encode_jpeg(test_frame)

        # Measure
        start = time.time()
        for _ in range(10):
            renderer.encode_jpeg(test_frame)
        elapsed_ms = ((time.time() - start) / 10) * 1000

        # Encoding should be fast
        assert elapsed_ms < 10


class TestColorConversions:
    """Test color space handling."""

    def test_rgb_input(self, renderer):
        """Test with RGB input (common from PIL)."""
        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb_frame[:, :] = (255, 0, 0)  # Red in RGB

        detection = CameraDetection(
            image_path=None,
            crop_path=None,
            bbox=BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200),
            confidence=0.9,
            label="camera",
            class_id=0,
        )

        annotated = renderer.annotate_frame(rgb_frame, [detection])

        # Should return RGB frame
        assert annotated.shape == rgb_frame.shape
        # Red channel should still be highest (after annotation)
        assert np.mean(annotated[:, :, 0]) > np.mean(annotated[:, :, 1])

    def test_grayscale_handling(self, renderer):
        """Test that renderer works with color frames only."""
        # Renderer expects 3-channel images
        gray_frame = np.zeros((480, 640), dtype=np.uint8)
        detection = CameraDetection(
            image_path=None,
            crop_path=None,
            bbox=BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200),
            confidence=0.9,
            label="camera",
            class_id=0,
        )

        # Should handle gracefully (may not annotate perfectly, but shouldn't crash)
        # In production, frames should always be RGB
        try:
            renderer.annotate_frame(gray_frame, [detection])
        except Exception:
            # Expected - renderer is designed for RGB frames
            pass
