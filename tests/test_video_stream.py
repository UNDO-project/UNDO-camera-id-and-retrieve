"""Tests for WebSocket video stream endpoint."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def test_app():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestVideoStreamEndpoint:
    """Test suite for video stream WebSocket endpoint."""

    def test_websocket_connect_default_params(self, test_app):
        """Test WebSocket connection with default parameters."""
        with test_app.websocket_connect("/api/v1/ws/video-stream") as websocket:
            # Send a test frame
            test_frame = b"fake_jpeg_data"
            websocket.send_bytes(test_frame)

            # Receive processed frame (binary)
            received_frame = websocket.receive_bytes()
            assert received_frame == test_frame  # Echo for now

            # Receive metadata (text)
            metadata_text = websocket.receive_text()
            metadata = json.loads(metadata_text)

            assert "detections" in metadata
            assert "processing_time_ms" in metadata
            assert "frame_number" in metadata
            assert "fps_actual" in metadata
            assert metadata["frame_number"] == 1
            assert isinstance(metadata["detections"], list)

    def test_websocket_connect_with_custom_params(self, test_app):
        """Test WebSocket connection with custom parameters."""
        with test_app.websocket_connect(
            "/api/v1/ws/video-stream?mode=full&target_fps=30"
        ) as websocket:
            # Send a test frame
            test_frame = b"fake_jpeg_data"
            websocket.send_bytes(test_frame)

            # Receive processed frame
            received_frame = websocket.receive_bytes()
            assert received_frame == test_frame

            # Receive metadata
            metadata_text = websocket.receive_text()
            metadata = json.loads(metadata_text)

            assert metadata["frame_number"] == 1
            assert metadata["fps_actual"] == 30

    def test_websocket_multiple_frames(self, test_app):
        """Test sending multiple frames in sequence."""
        with test_app.websocket_connect("/api/v1/ws/video-stream") as websocket:
            for i in range(3):
                # Send frame
                test_frame = f"frame_{i}".encode()
                websocket.send_bytes(test_frame)

                # Receive processed frame
                received_frame = websocket.receive_bytes()
                assert received_frame == test_frame

                # Receive metadata
                metadata_text = websocket.receive_text()
                metadata = json.loads(metadata_text)
                assert metadata["frame_number"] == i + 1

    def test_websocket_detect_only_mode(self, test_app):
        """Test detect_only mode."""
        with test_app.websocket_connect(
            "/api/v1/ws/video-stream?mode=detect_only&target_fps=15"
        ) as websocket:
            test_frame = b"test_frame"
            websocket.send_bytes(test_frame)

            # Receive responses
            received_frame = websocket.receive_bytes()
            metadata_text = websocket.receive_text()
            metadata = json.loads(metadata_text)

            assert received_frame == test_frame
            assert metadata["detections"] == []  # Placeholder for now

    def test_websocket_full_mode(self, test_app):
        """Test full identification mode."""
        with test_app.websocket_connect(
            "/api/v1/ws/video-stream?mode=full&target_fps=20"
        ) as websocket:
            test_frame = b"test_frame"
            websocket.send_bytes(test_frame)

            # Receive responses
            received_frame = websocket.receive_bytes()
            metadata_text = websocket.receive_text()
            metadata = json.loads(metadata_text)

            assert received_frame == test_frame
            assert "detections" in metadata

    def test_websocket_invalid_fps_too_low(self, test_app):
        """Test that FPS below 1 is rejected."""
        with pytest.raises(Exception):  # FastAPI will raise validation error
            with test_app.websocket_connect("/api/v1/ws/video-stream?target_fps=0"):
                pass

    def test_websocket_invalid_fps_too_high(self, test_app):
        """Test that FPS above 30 is rejected."""
        with pytest.raises(Exception):  # FastAPI will raise validation error
            with test_app.websocket_connect("/api/v1/ws/video-stream?target_fps=31"):
                pass

    def test_websocket_invalid_mode(self, test_app):
        """Test that invalid mode is rejected."""
        with pytest.raises(Exception):  # FastAPI will raise validation error
            with test_app.websocket_connect("/api/v1/ws/video-stream?mode=invalid"):
                pass

    def test_websocket_binary_protocol(self, test_app):
        """Test that binary protocol works correctly (no base64)."""
        with test_app.websocket_connect("/api/v1/ws/video-stream") as websocket:
            # Send binary data
            binary_data = bytes(range(256))  # All byte values
            websocket.send_bytes(binary_data)

            # Receive binary response
            received = websocket.receive_bytes()
            assert isinstance(received, bytes)
            assert received == binary_data  # Echo for now

            # Metadata should be text/JSON
            metadata_text = websocket.receive_text()
            assert isinstance(metadata_text, str)
            metadata = json.loads(metadata_text)
            assert isinstance(metadata, dict)

    def test_websocket_metadata_structure(self, test_app):
        """Test that metadata has correct structure."""
        with test_app.websocket_connect("/api/v1/ws/video-stream") as websocket:
            websocket.send_bytes(b"test_frame")

            # Skip binary frame
            websocket.receive_bytes()

            # Check metadata structure
            metadata_text = websocket.receive_text()
            metadata = json.loads(metadata_text)

            # Required fields
            assert "detections" in metadata
            assert "processing_time_ms" in metadata
            assert "frame_number" in metadata
            assert "fps_actual" in metadata

            # Type checks
            assert isinstance(metadata["detections"], list)
            assert isinstance(metadata["processing_time_ms"], (int, float))
            assert isinstance(metadata["frame_number"], int)
            assert isinstance(metadata["fps_actual"], int)

            # Value checks
            assert metadata["processing_time_ms"] >= 0
            assert metadata["frame_number"] > 0

    def test_websocket_connection_cleanup(self, test_app):
        """Test that connection is properly cleaned up on disconnect."""
        # Connect and immediately disconnect
        with test_app.websocket_connect("/api/v1/ws/video-stream") as websocket:
            # Send one frame to establish connection
            websocket.send_bytes(b"test")
            websocket.receive_bytes()
            websocket.receive_text()

        # Connection should be cleaned up
        # (Verified by no errors and proper shutdown)

    def test_websocket_frame_counter(self, test_app):
        """Test that frame numbers increment correctly."""
        with test_app.websocket_connect("/api/v1/ws/video-stream") as websocket:
            frame_numbers = []

            for i in range(5):
                websocket.send_bytes(f"frame_{i}".encode())
                websocket.receive_bytes()  # Skip binary
                metadata_text = websocket.receive_text()
                metadata = json.loads(metadata_text)
                frame_numbers.append(metadata["frame_number"])

            # Check sequential numbering
            assert frame_numbers == [1, 2, 3, 4, 5]

    def test_websocket_processing_time_recorded(self, test_app):
        """Test that processing time is recorded in metadata."""
        with test_app.websocket_connect("/api/v1/ws/video-stream") as websocket:
            websocket.send_bytes(b"test_frame")
            websocket.receive_bytes()

            metadata_text = websocket.receive_text()
            metadata = json.loads(metadata_text)

            # Processing time should be non-negative (can be 0 for echo)
            assert metadata["processing_time_ms"] >= 0
            # Should be reasonable (less than 1 second for echo)
            assert metadata["processing_time_ms"] < 1000

    def test_websocket_concurrent_messages(self, test_app):
        """Test receiving both binary and text messages."""
        with test_app.websocket_connect("/api/v1/ws/video-stream") as websocket:
            # Send frame
            test_frame = b"test_frame"
            websocket.send_bytes(test_frame)

            # Should receive binary first
            received_frame = websocket.receive_bytes()
            assert isinstance(received_frame, bytes)
            assert received_frame == test_frame

            # Then text metadata
            metadata_text = websocket.receive_text()
            assert isinstance(metadata_text, str)
            metadata = json.loads(metadata_text)
            assert "detections" in metadata
