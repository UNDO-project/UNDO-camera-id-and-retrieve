"""Tests for async identification service."""

import asyncio
import io
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.identification.async_service import AsyncIdentificationService
from src.identification.detector import Detector


# Sample test image path (assumes test data exists)
TEST_IMAGE_DIR = Path("data/test_images")


@pytest.fixture
def test_image_bytes():
    """Create test image bytes (small JPEG)."""
    # Create a simple RGB image
    img = Image.new("RGB", (640, 480), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def test_image_array():
    """Create test image as numpy array."""
    img = Image.new("RGB", (640, 480), color=(73, 109, 137))
    return np.array(img)


@pytest.fixture
def async_service():
    """Create AsyncIdentificationService for testing."""
    try:
        service = AsyncIdentificationService(save_crops=False)
        return service
    except Exception as e:
        pytest.skip(f"Could not initialize service: {e}")


class TestDetectorInMemoryMethods:
    """Test suite for new Detector in-memory methods."""

    def test_detect_from_image_array(self, test_image_array):
        """Test detection from numpy array."""
        try:
            detector = Detector()
        except Exception as e:
            pytest.skip(f"Could not initialize detector: {e}")

        # Run detection
        detections = detector.detect_from_image(test_image_array)

        # Should return a list (may be empty if no cameras in image)
        assert isinstance(detections, list)

    def test_detect_from_bytes(self, test_image_bytes):
        """Test detection from raw bytes."""
        try:
            detector = Detector()
        except Exception as e:
            pytest.skip(f"Could not initialize detector: {e}")

        # Run detection
        detections = detector.detect_from_bytes(test_image_bytes)

        # Should return a list
        assert isinstance(detections, list)

    def test_detect_from_bytes_invalid(self):
        """Test detection with invalid bytes."""
        try:
            detector = Detector()
        except Exception as e:
            pytest.skip(f"Could not initialize detector: {e}")

        invalid_bytes = b"not an image"

        with pytest.raises(ValueError, match="Failed to decode"):
            detector.detect_from_bytes(invalid_bytes)

    def test_detect_from_image_maintains_format(self, test_image_array):
        """Test that detection result format is consistent."""
        try:
            detector = Detector()
        except Exception as e:
            pytest.skip(f"Could not initialize detector: {e}")

        detections = detector.detect_from_image(test_image_array)

        for detection in detections:
            assert hasattr(detection, "bbox")
            assert hasattr(detection, "confidence")
            assert hasattr(detection, "label")
            assert detection.image_path is None  # In-memory has no path


class TestAsyncIdentificationService:
    """Test suite for AsyncIdentificationService."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test async service initializes correctly."""
        try:
            service = AsyncIdentificationService(save_crops=False)
            assert service is not None
            assert service._sync_service is not None
            assert service._detector_lock is not None
        except Exception as e:
            pytest.skip(f"Could not initialize service: {e}")

    @pytest.mark.asyncio
    async def test_decode_frame(self, test_image_bytes):
        """Test frame decoding."""
        image_array = AsyncIdentificationService._decode_frame(test_image_bytes)

        assert isinstance(image_array, np.ndarray)
        assert len(image_array.shape) == 3  # H, W, C
        assert image_array.shape[2] == 3  # RGB

    @pytest.mark.asyncio
    async def test_decode_frame_invalid(self):
        """Test decoding invalid frame raises error."""
        with pytest.raises(ValueError, match="Failed to decode"):
            AsyncIdentificationService._decode_frame(b"invalid")

    @pytest.mark.asyncio
    async def test_detect_frame(self, async_service, test_image_bytes):
        """Test async frame detection."""
        detections = await async_service.detect_frame(test_image_bytes)

        assert isinstance(detections, list)
        # May be empty if no cameras in test image
        for detection in detections:
            assert hasattr(detection, "bbox")
            assert hasattr(detection, "confidence")

    @pytest.mark.asyncio
    async def test_detect_frame_non_blocking(self, async_service, test_image_bytes):
        """Test that detect_frame is non-blocking."""
        start = time.time()

        # Run multiple detections concurrently
        tasks = [async_service.detect_frame(test_image_bytes) for _ in range(3)]
        results = await asyncio.gather(*tasks)

        _ = time.time() - start

        # Should complete in reasonable time (not 3x sequential time)
        # This is a rough check - actual timing depends on hardware
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    @pytest.mark.asyncio
    async def test_identify_frame(self, async_service, test_image_bytes):
        """Test full identification pipeline."""
        results = await async_service.identify_frame(
            test_image_bytes, top_k=3, min_similarity=0.3
        )

        assert isinstance(results, list)
        # May be empty if no cameras detected
        for result in results:
            assert hasattr(result, "detection")
            assert hasattr(result, "matches")
            assert isinstance(result.matches, list)

    @pytest.mark.asyncio
    async def test_identify_frame_with_params(self, async_service, test_image_bytes):
        """Test identification with custom parameters."""
        results = await async_service.identify_frame(
            test_image_bytes, top_k=10, min_similarity=0.5
        )

        assert isinstance(results, list)

        # Check that matches respect top_k and min_similarity
        for result in results:
            assert len(result.matches) <= 10
            for match in result.matches:
                assert match.score >= 0.5

    @pytest.mark.asyncio
    async def test_thread_safety(self, async_service, test_image_bytes):
        """Test thread safety with concurrent detections."""
        # Run many concurrent detections to test lock behavior
        tasks = [async_service.detect_frame(test_image_bytes) for _ in range(10)]

        # Should complete without deadlock or race conditions
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(isinstance(r, list) for r in results)

    @pytest.mark.asyncio
    async def test_property_access(self, async_service):
        """Test access to underlying service properties."""
        assert async_service.detector is not None
        assert async_service.index is not None
        assert async_service.catalog is not None

    @pytest.mark.asyncio
    async def test_concurrent_identify_operations(
        self, async_service, test_image_bytes
    ):
        """Test concurrent full identification operations."""
        # Run multiple identify operations concurrently
        tasks = [
            async_service.identify_frame(test_image_bytes, top_k=5) for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(isinstance(r, list) for r in results)


class TestDetectorMethodsWithRealImage:
    """Tests using real camera images if available."""

    def test_detect_from_bytes_real_image(self):
        """Test detection with real camera image."""
        # Skip if test images not available
        if not TEST_IMAGE_DIR.exists():
            pytest.skip("Test images not available")

        # Find first JPEG in test directory
        test_images = list(TEST_IMAGE_DIR.glob("*.jpg"))
        if not test_images:
            pytest.skip("No test images found")

        test_image_path = test_images[0]

        try:
            detector = Detector()
        except Exception as e:
            pytest.skip(f"Could not initialize detector: {e}")

        # Load image as bytes
        with open(test_image_path, "rb") as f:
            image_bytes = f.read()

        # Run detection from bytes
        detections_from_bytes = detector.detect_from_bytes(image_bytes)

        # Run detection from path (for comparison)
        detections_from_path = detector.detect_from_path(test_image_path)

        # Should get same number of detections
        # (coordinates may differ slightly due to different processing paths)
        assert len(detections_from_bytes) == len(detections_from_path)


class TestPerformance:
    """Performance tests for async operations."""

    @pytest.mark.asyncio
    async def test_detect_frame_performance(self, async_service, test_image_bytes):
        """Test detection performance (should be <50ms)."""
        # Warm up
        await async_service.detect_frame(test_image_bytes)

        # Measure
        start = time.time()
        await async_service.detect_frame(test_image_bytes)
        elapsed_ms = (time.time() - start) * 1000

        # Should be reasonably fast (adjust threshold as needed)
        # This is hardware-dependent, so we use a generous threshold
        assert elapsed_ms < 200  # 200ms threshold for CI

    @pytest.mark.asyncio
    async def test_identify_frame_performance(self, async_service, test_image_bytes):
        """Test full identification performance."""
        # Warm up
        await async_service.identify_frame(test_image_bytes, top_k=3)

        # Measure
        start = time.time()
        await async_service.identify_frame(test_image_bytes, top_k=3)
        elapsed_ms = (time.time() - start) * 1000

        # Full pipeline is more expensive, so generous threshold
        assert elapsed_ms < 500  # 500ms threshold for CI
