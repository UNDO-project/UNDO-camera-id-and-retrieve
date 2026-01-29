"""Tests for video streaming configuration and monitoring."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.config import VideoStreamSettings, stream_settings
from src.api.main import app
from src.api.websocket.manager import ConnectionManager, ConnectionState, StreamConfig


@pytest.fixture
def test_client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


class TestVideoStreamSettings:
    """Test suite for VideoStreamSettings configuration."""

    def test_default_settings(self):
        """Test default configuration values."""
        settings = VideoStreamSettings()
        assert settings.max_connections == 5
        assert settings.max_fps == 30
        assert settings.default_fps == 15
        assert settings.frame_timeout_seconds == 5.0
        assert settings.connection_timeout_seconds == 30
        assert settings.input_buffer_size == 10
        assert settings.output_buffer_size == 10
        assert settings.jpeg_quality == 85
        assert settings.max_frame_size_mb == 5.0

    def test_env_prefix(self):
        """Test that settings use correct env prefix."""
        assert VideoStreamSettings.model_config["env_prefix"] == "CIDAR_STREAM_"

    def test_global_settings_instance(self):
        """Test that global stream_settings is available."""
        assert stream_settings is not None
        assert isinstance(stream_settings, VideoStreamSettings)


class TestConnectionManagerWithSettings:
    """Test suite for ConnectionManager with settings integration."""

    def test_default_max_connections(self):
        """Test that manager uses settings for max connections."""
        manager = ConnectionManager()
        assert manager.max_connections == stream_settings.max_connections

    def test_default_connection_timeout(self):
        """Test that manager uses settings for connection timeout."""
        manager = ConnectionManager()
        assert manager.connection_timeout == stream_settings.connection_timeout_seconds

    def test_custom_max_connections(self):
        """Test manager with custom max connections."""
        manager = ConnectionManager(max_connections=10)
        assert manager.max_connections == 10

    def test_custom_connection_timeout(self):
        """Test manager with custom connection timeout."""
        manager = ConnectionManager(connection_timeout=60)
        assert manager.connection_timeout == 60


class TestConnectionLimits:
    """Test suite for connection limit enforcement."""

    @pytest.mark.asyncio
    async def test_connection_limit_enforced(self):
        """Test that connection limit is enforced."""
        manager = ConnectionManager(max_connections=2)

        # Create mock websockets with async methods
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws2 = MagicMock()
        ws2.accept = AsyncMock()
        ws3 = MagicMock()
        ws3.close = AsyncMock()

        config = StreamConfig()

        # Connect first two clients
        await manager.connect(ws1, "client-1", config)
        await manager.connect(ws2, "client-2", config)

        # Third connection should be rejected
        with pytest.raises(RuntimeError, match="Maximum connections reached"):
            await manager.connect(ws3, "client-3", config)

        ws3.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_after_disconnect(self):
        """Test that new connection allowed after disconnect."""
        manager = ConnectionManager(max_connections=1)

        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws1.close = AsyncMock()
        ws2 = MagicMock()
        ws2.accept = AsyncMock()

        config = StreamConfig()

        # Connect first client
        await manager.connect(ws1, "client-1", config)
        assert len(manager.active_connections) == 1

        # Disconnect first client
        await manager.disconnect("client-1")
        assert len(manager.active_connections) == 0

        # Now second client should be able to connect
        await manager.connect(ws2, "client-2", config)
        assert len(manager.active_connections) == 1


class TestConnectionState:
    """Test suite for ConnectionState properties."""

    def test_average_processing_time(self):
        """Test average processing time calculation."""
        ws = MagicMock()
        state = ConnectionState(websocket=ws, client_id="test")
        state.frames_processed = 10
        state.total_processing_time_ms = 500.0

        assert state.average_processing_time_ms == 50.0

    def test_average_processing_time_zero_frames(self):
        """Test average processing time with zero frames."""
        ws = MagicMock()
        state = ConnectionState(websocket=ws, client_id="test")
        assert state.average_processing_time_ms == 0.0

    def test_current_fps(self):
        """Test current FPS calculation."""
        ws = MagicMock()
        # Set connected_at to 10 seconds ago
        past_time = datetime.now(timezone.utc)
        state = ConnectionState(websocket=ws, client_id="test", connected_at=past_time)
        state.frames_processed = 100

        # FPS should be approximately frames / duration
        fps = state.current_fps
        assert fps > 0


class TestConnectionStats:
    """Test suite for connection statistics."""

    @pytest.mark.asyncio
    async def test_stats_with_no_connections(self):
        """Test stats when no connections exist."""
        manager = ConnectionManager()
        stats = manager.get_connection_stats()

        assert stats["active_connections"] == 0
        assert stats["max_connections"] == manager.max_connections
        assert stats["total_frames_processed"] == 0
        assert stats["total_frames_dropped"] == 0
        assert stats["average_fps"] == 0.0
        assert stats["connections"] == []

    @pytest.mark.asyncio
    async def test_stats_with_active_connection(self):
        """Test stats with active connection."""
        manager = ConnectionManager()

        ws = MagicMock()
        ws.accept = AsyncMock()

        config = StreamConfig(identification_mode="detect_only", target_fps=15)
        await manager.connect(ws, "client-1", config)

        # Simulate some processing
        manager.increment_frames_received("client-1")
        manager.increment_frames_processed("client-1", processing_time_ms=50.0)

        stats = manager.get_connection_stats()

        assert stats["active_connections"] == 1
        assert stats["total_connections_served"] == 1
        assert len(stats["connections"]) == 1
        assert stats["connections"][0]["client_id"] == "client-1"
        assert stats["connections"][0]["frames_received"] == 1
        assert stats["connections"][0]["frames_processed"] == 1

    @pytest.mark.asyncio
    async def test_stats_accumulate_after_disconnect(self):
        """Test that stats accumulate after disconnect."""
        manager = ConnectionManager()

        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.close = AsyncMock()

        config = StreamConfig()
        await manager.connect(ws, "client-1", config)

        # Simulate processing
        for _ in range(10):
            manager.increment_frames_received("client-1")
            manager.increment_frames_processed("client-1", processing_time_ms=10.0)

        # Disconnect
        await manager.disconnect("client-1")

        # Stats should still show totals
        stats = manager.get_connection_stats()
        assert stats["active_connections"] == 0
        assert stats["total_frames_processed"] == 10
        assert stats["total_connections_served"] == 1


class TestStreamStatsEndpoint:
    """Test suite for /stream/stats endpoint."""

    def test_stream_stats_no_manager(self, test_client: TestClient):
        """Test stream stats when manager not initialized."""
        # Clear the connection manager
        from src.api.dependencies import state

        original_manager = state._connection_manager
        state._connection_manager = None

        try:
            response = test_client.get("/api/v1/stream/stats")
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"]
        finally:
            state._connection_manager = original_manager

    def test_stream_stats_with_manager(self, test_client: TestClient):
        """Test stream stats with initialized manager."""
        from src.api.dependencies import state

        # Ensure manager is initialized
        _ = state.connection_manager

        response = test_client.get("/api/v1/stream/stats")
        assert response.status_code == 200

        data = response.json()
        assert "active_connections" in data
        assert "max_connections" in data
        assert "total_connections_served" in data
        assert "total_frames_received" in data
        assert "total_frames_processed" in data
        assert "total_frames_dropped" in data
        assert "average_fps" in data
        assert "average_latency_ms" in data
        assert "connections" in data


class TestStreamConfigDefaults:
    """Test suite for StreamConfig defaults."""

    def test_stream_config_uses_settings_fps(self):
        """Test that StreamConfig uses settings for default FPS."""
        config = StreamConfig()
        assert config.target_fps == stream_settings.default_fps

    def test_stream_config_custom_fps(self):
        """Test StreamConfig with custom FPS."""
        config = StreamConfig(target_fps=30)
        assert config.target_fps == 30

    def test_stream_config_default_mode(self):
        """Test StreamConfig default identification mode."""
        config = StreamConfig()
        assert config.identification_mode == "detect_only"
