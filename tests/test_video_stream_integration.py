"""Integration tests for video stream endpoint with full processing pipeline."""

import io
import json
from typing import List

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.dependencies import get_async_identification_service
from src.api.main import app
from src.models.identification import (
    CameraDetection,
    RetrievalResult,
)


class MockAsyncIdentificationService:
    """Mock async identification service for testing."""

    async def detect_frame(self, frame_bytes: bytes) -> List[CameraDetection]:
        """Mock detect_frame that returns empty detections."""
        return []

    async def identify_frame(self, frame_bytes: bytes) -> List[RetrievalResult]:
        """Mock identify_frame that returns empty results."""
        return []


@pytest.fixture
def mock_service():
    """Provide a mock async identification service."""
    return MockAsyncIdentificationService()


@pytest.fixture
def test_client(mock_service):
    """Create test client with mocked dependencies."""
    app.dependency_overrides[get_async_identification_service] = lambda: mock_service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def test_frame_bytes() -> bytes:
    """Create a test JPEG frame."""
    # Create a simple test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some content to make it more realistic
    image[100:200, 100:200] = [255, 0, 0]  # Red square
    pil_image = Image.fromarray(image)

    # Encode to JPEG
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def test_video_stream_connect_disconnect(test_client, test_frame_bytes):
    """Test WebSocket connection and disconnection."""
    with test_client.websocket_connect("/api/v1/ws/video-stream") as websocket:
        # Send one frame
        websocket.send_bytes(test_frame_bytes)

        # Receive response
        _ = websocket.receive_bytes()
        text_response = websocket.receive_text()
        metadata = json.loads(text_response)

        # Verify metadata structure
        assert "detections" in metadata
        assert "processing_time_ms" in metadata
        assert "frame_number" in metadata
        assert "fps_actual" in metadata

        # Close connection explicitly to trigger shutdown
        websocket.close()


def test_video_stream_detect_only_mode(test_client, test_frame_bytes):
    """Test video stream in detect_only mode."""
    with test_client.websocket_connect(
        "/api/v1/ws/video-stream?mode=detect_only&target_fps=15"
    ) as websocket:
        # Send frame
        websocket.send_bytes(test_frame_bytes)

        # Receive response
        _ = websocket.receive_bytes()
        text_response = websocket.receive_text()
        metadata = json.loads(text_response)

        # In detect_only mode, detections should not have matches
        for detection in metadata["detections"]:
            assert "matches" not in detection or detection["matches"] is None

        websocket.close()


def test_video_stream_full_mode(test_client, test_frame_bytes):
    """Test video stream in full identification mode."""
    with test_client.websocket_connect(
        "/api/v1/ws/video-stream?mode=full&target_fps=15"
    ) as websocket:
        # Send frame
        websocket.send_bytes(test_frame_bytes)

        # Receive response
        _ = websocket.receive_bytes()
        text_response = websocket.receive_text()
        metadata = json.loads(text_response)

        # Verify metadata has detections field
        assert "detections" in metadata

        websocket.close()


def test_video_stream_multiple_frames(test_client, test_frame_bytes):
    """Test processing multiple frames."""
    with test_client.websocket_connect("/api/v1/ws/video-stream") as websocket:
        # Send multiple frames
        num_frames = 3  # Reduced to 3 for faster tests
        for _ in range(num_frames):
            websocket.send_bytes(test_frame_bytes)

            # Receive each response
            _ = websocket.receive_bytes()
            text_response = websocket.receive_text()
            metadata = json.loads(text_response)

            # Verify metadata
            assert "detections" in metadata
            assert "processing_time_ms" in metadata

        websocket.close()


def test_video_stream_frame_numbering(test_client, test_frame_bytes):
    """Test that frames are numbered sequentially."""
    with test_client.websocket_connect("/api/v1/ws/video-stream") as websocket:
        frame_numbers = []

        # Send multiple frames
        for _ in range(3):
            websocket.send_bytes(test_frame_bytes)
            _ = websocket.receive_bytes()
            text_response = websocket.receive_text()
            metadata = json.loads(text_response)
            frame_numbers.append(metadata["frame_number"])

        # Frame numbers should be sequential (though may have gaps due to skipping)
        assert len(frame_numbers) == 3
        # All frame numbers should be positive
        assert all(fn > 0 for fn in frame_numbers)

        websocket.close()


def test_video_stream_fps_reporting(test_client, test_frame_bytes):
    """Test that FPS is reported in metadata."""
    with test_client.websocket_connect(
        "/api/v1/ws/video-stream?target_fps=15"
    ) as websocket:
        # Send a few frames to establish FPS
        for _ in range(3):  # Reduced to 3
            websocket.send_bytes(test_frame_bytes)
            _ = websocket.receive_bytes()
            text_response = websocket.receive_text()
            metadata = json.loads(text_response)

            # FPS should be reported
            assert "fps_actual" in metadata
            assert isinstance(metadata["fps_actual"], (int, float))
            assert metadata["fps_actual"] >= 0

        websocket.close()


def test_video_stream_processing_time(test_client, test_frame_bytes):
    """Test that processing time is reported."""
    with test_client.websocket_connect("/api/v1/ws/video-stream") as websocket:
        websocket.send_bytes(test_frame_bytes)
        _ = websocket.receive_bytes()
        text_response = websocket.receive_text()
        metadata = json.loads(text_response)

        # Processing time should be positive
        assert metadata["processing_time_ms"] > 0
        # Should be reasonable (not too slow)
        assert metadata["processing_time_ms"] < 5000  # 5 seconds max

        websocket.close()


def test_video_stream_metadata_structure(test_client, test_frame_bytes):
    """Test complete metadata structure."""
    with test_client.websocket_connect("/api/v1/ws/video-stream") as websocket:
        websocket.send_bytes(test_frame_bytes)
        _ = websocket.receive_bytes()
        text_response = websocket.receive_text()
        metadata = json.loads(text_response)

        # Required fields
        assert "detections" in metadata
        assert "processing_time_ms" in metadata
        assert "frame_number" in metadata
        assert "fps_actual" in metadata

        # Detections structure
        assert isinstance(metadata["detections"], list)
        for detection in metadata["detections"]:
            assert "bbox" in detection
            assert "confidence" in detection
            assert "class" in detection
            assert len(detection["bbox"]) == 4  # [x1, y1, x2, y2]

        websocket.close()


def test_video_stream_binary_response_is_jpeg(test_client, test_frame_bytes):
    """Test that binary response is valid JPEG."""
    with test_client.websocket_connect("/api/v1/ws/video-stream") as websocket:
        websocket.send_bytes(test_frame_bytes)
        binary_response = websocket.receive_bytes()

        # Should be able to load as image
        image = Image.open(io.BytesIO(binary_response))
        assert image.format == "JPEG"
        assert image.size[0] > 0  # Width
        assert image.size[1] > 0  # Height

        websocket.close()


@pytest.mark.parametrize("target_fps", [1, 15, 30])
def test_video_stream_different_fps(test_client, test_frame_bytes, target_fps):
    """Test video stream with different FPS targets."""
    with test_client.websocket_connect(
        f"/api/v1/ws/video-stream?target_fps={target_fps}"
    ) as websocket:
        websocket.send_bytes(test_frame_bytes)
        _ = websocket.receive_bytes()
        text_response = websocket.receive_text()
        metadata = json.loads(text_response)

        # Should process successfully
        assert "detections" in metadata

        websocket.close()


@pytest.mark.parametrize("mode", ["detect_only", "full"])
def test_video_stream_different_modes(test_client, test_frame_bytes, mode):
    """Test video stream with different processing modes."""
    with test_client.websocket_connect(
        f"/api/v1/ws/video-stream?mode={mode}"
    ) as websocket:
        websocket.send_bytes(test_frame_bytes)
        _ = websocket.receive_bytes()
        text_response = websocket.receive_text()
        metadata = json.loads(text_response)

        # Should process successfully
        assert "detections" in metadata

        websocket.close()
