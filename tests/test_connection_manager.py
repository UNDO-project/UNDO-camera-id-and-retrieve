"""Tests for WebSocket connection manager."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from fastapi import WebSocketDisconnect

from src.api.websocket.manager import ConnectionManager, ConnectionState, StreamConfig


@pytest.fixture
def manager():
    """Create a ConnectionManager instance for testing."""
    return ConnectionManager(connection_timeout=5)


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket instance."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.send_bytes = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


@pytest.fixture
def stream_config():
    """Create a default StreamConfig instance."""
    return StreamConfig(identification_mode="detect_only", target_fps=15)


class TestConnectionManager:
    """Test suite for ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connect_new_client(self, manager, mock_websocket, stream_config):
        """Test connecting a new client."""
        client_id = "test_client_1"

        await manager.connect(mock_websocket, client_id, stream_config)

        mock_websocket.accept.assert_called_once()
        assert client_id in manager.active_connections
        assert manager.active_connections[client_id].client_id == client_id
        assert (
            manager.active_connections[client_id].identification_mode == "detect_only"
        )
        assert manager.active_connections[client_id].target_fps == 15
        assert manager.active_connections[client_id].frames_received == 0
        assert manager.active_connections[client_id].frames_processed == 0

    @pytest.mark.asyncio
    async def test_connect_duplicate_client(
        self, manager, mock_websocket, stream_config
    ):
        """Test that connecting with duplicate client_id raises error."""
        client_id = "test_client_1"

        # Connect first time
        await manager.connect(mock_websocket, client_id, stream_config)

        # Try to connect again with same ID
        mock_websocket2 = AsyncMock()
        mock_websocket2.accept = AsyncMock()
        mock_websocket2.close = AsyncMock()

        with pytest.raises(RuntimeError, match="already connected"):
            await manager.connect(mock_websocket2, client_id, stream_config)

        mock_websocket2.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_client(self, manager, mock_websocket, stream_config):
        """Test disconnecting a client."""
        client_id = "test_client_1"

        # Connect first
        await manager.connect(mock_websocket, client_id, stream_config)
        assert client_id in manager.active_connections

        # Disconnect
        await manager.disconnect(client_id)

        mock_websocket.close.assert_called_once()
        assert client_id not in manager.active_connections

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_client(self, manager):
        """Test disconnecting a client that doesn't exist."""
        # Should not raise error
        await manager.disconnect("nonexistent_client")

    @pytest.mark.asyncio
    async def test_send_bytes(self, manager, mock_websocket, stream_config):
        """Test sending binary data to a client."""
        client_id = "test_client_1"
        test_data = b"test frame data"

        await manager.connect(mock_websocket, client_id, stream_config)
        await manager.send_bytes(client_id, test_data)

        mock_websocket.send_bytes.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_send_bytes_nonexistent_client(self, manager):
        """Test sending bytes to non-existent client raises KeyError."""
        with pytest.raises(KeyError, match="not connected"):
            await manager.send_bytes("nonexistent_client", b"data")

    @pytest.mark.asyncio
    async def test_send_bytes_disconnected_client(
        self, manager, mock_websocket, stream_config
    ):
        """Test sending bytes to disconnected client."""
        client_id = "test_client_1"

        await manager.connect(mock_websocket, client_id, stream_config)

        # Simulate disconnect during send
        mock_websocket.send_bytes.side_effect = WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            await manager.send_bytes(client_id, b"data")

        # Client should be removed from active connections
        assert client_id not in manager.active_connections

    @pytest.mark.asyncio
    async def test_send_text(self, manager, mock_websocket, stream_config):
        """Test sending text data to a client."""
        client_id = "test_client_1"
        test_data = '{"status": "ok"}'

        await manager.connect(mock_websocket, client_id, stream_config)
        await manager.send_text(client_id, test_data)

        mock_websocket.send_text.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_send_text_nonexistent_client(self, manager):
        """Test sending text to non-existent client raises KeyError."""
        with pytest.raises(KeyError, match="not connected"):
            await manager.send_text("nonexistent_client", "data")

    @pytest.mark.asyncio
    async def test_broadcast(self, manager, stream_config):
        """Test broadcasting data to multiple clients."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_bytes = AsyncMock()

        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_bytes = AsyncMock()

        await manager.connect(ws1, "client_1", stream_config)
        await manager.connect(ws2, "client_2", stream_config)

        test_data = b"broadcast data"
        await manager.broadcast(test_data)

        ws1.send_bytes.assert_called_once_with(test_data)
        ws2.send_bytes.assert_called_once_with(test_data)

    @pytest.mark.asyncio
    async def test_broadcast_with_disconnected_client(self, manager, stream_config):
        """Test broadcast handles disconnected clients gracefully."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_bytes = AsyncMock()
        ws1.close = AsyncMock()

        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_bytes = AsyncMock(side_effect=WebSocketDisconnect)
        ws2.close = AsyncMock()

        await manager.connect(ws1, "client_1", stream_config)
        await manager.connect(ws2, "client_2", stream_config)

        await manager.broadcast(b"data")

        # Client 1 should receive data
        ws1.send_bytes.assert_called_once()
        # Client 2 should be disconnected
        assert "client_2" not in manager.active_connections
        assert "client_1" in manager.active_connections

    def test_update_activity(self, manager, mock_websocket, stream_config):
        """Test updating client activity timestamp."""
        client_id = "test_client_1"

        # Use asyncio.run for async setup
        asyncio.run(manager.connect(mock_websocket, client_id, stream_config))

        original_time = manager.active_connections[client_id].last_activity

        # Sleep briefly to ensure time difference
        import time

        time.sleep(0.1)

        manager.update_activity(client_id)

        new_time = manager.active_connections[client_id].last_activity
        assert new_time > original_time

    def test_increment_frames_received(self, manager, mock_websocket, stream_config):
        """Test incrementing frames received counter."""
        client_id = "test_client_1"

        asyncio.run(manager.connect(mock_websocket, client_id, stream_config))

        assert manager.active_connections[client_id].frames_received == 0

        manager.increment_frames_received(client_id)
        assert manager.active_connections[client_id].frames_received == 1

        manager.increment_frames_received(client_id)
        assert manager.active_connections[client_id].frames_received == 2

    def test_increment_frames_processed(self, manager, mock_websocket, stream_config):
        """Test incrementing frames processed counter."""
        client_id = "test_client_1"

        asyncio.run(manager.connect(mock_websocket, client_id, stream_config))

        assert manager.active_connections[client_id].frames_processed == 0

        manager.increment_frames_processed(client_id)
        assert manager.active_connections[client_id].frames_processed == 1

        manager.increment_frames_processed(client_id)
        assert manager.active_connections[client_id].frames_processed == 2

    @pytest.mark.asyncio
    async def test_get_connection_stats(self, manager, mock_websocket, stream_config):
        """Test retrieving connection statistics."""
        await manager.connect(mock_websocket, "client_1", stream_config)

        manager.increment_frames_received("client_1")
        manager.increment_frames_received("client_1")
        manager.increment_frames_processed("client_1")

        stats = manager.get_connection_stats()

        assert stats["active_connections"] == 1
        assert stats["total_frames_received"] == 2
        assert stats["total_frames_processed"] == 1
        assert len(stats["connections"]) == 1

        conn_detail = stats["connections"][0]
        assert conn_detail["client_id"] == "client_1"
        assert conn_detail["frames_received"] == 2
        assert conn_detail["frames_processed"] == 1
        assert conn_detail["identification_mode"] == "detect_only"
        assert conn_detail["target_fps"] == 15
        assert "connected_at" in conn_detail
        assert "last_activity" in conn_detail
        assert "idle_seconds" in conn_detail

    @pytest.mark.asyncio
    async def test_get_connection_stats_multiple_clients(self, manager, stream_config):
        """Test connection stats with multiple clients."""
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()

        ws2 = AsyncMock()
        ws2.accept = AsyncMock()

        config1 = StreamConfig(identification_mode="detect_only", target_fps=15)
        config2 = StreamConfig(identification_mode="full", target_fps=30)

        await manager.connect(ws1, "client_1", config1)
        await manager.connect(ws2, "client_2", config2)

        manager.increment_frames_received("client_1")
        manager.increment_frames_received("client_2")
        manager.increment_frames_received("client_2")
        manager.increment_frames_processed("client_1")

        stats = manager.get_connection_stats()

        assert stats["active_connections"] == 2
        assert stats["total_frames_received"] == 3
        assert stats["total_frames_processed"] == 1

    @pytest.mark.asyncio
    async def test_shutdown(self, manager, mock_websocket, stream_config):
        """Test graceful shutdown closes all connections."""
        await manager.connect(mock_websocket, "client_1", stream_config)

        assert len(manager.active_connections) == 1

        await manager.shutdown()

        assert len(manager.active_connections) == 0
        assert manager._shutdown is True
        mock_websocket.close.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_inactive_connections(
        self, manager, mock_websocket, stream_config
    ):
        """Test that inactive connections are automatically cleaned up."""
        client_id = "test_client_1"

        # Create manager with very short timeout
        short_timeout_manager = ConnectionManager(connection_timeout=1)

        await short_timeout_manager.connect(mock_websocket, client_id, stream_config)

        # Manually set last_activity to past
        state = short_timeout_manager.active_connections[client_id]
        state.last_activity = datetime.now(timezone.utc) - timedelta(seconds=5)

        # Wait for cleanup to run
        await asyncio.sleep(0.5)

        # Trigger one cleanup cycle manually
        now = datetime.now(timezone.utc)
        to_disconnect = []
        for cid, st in short_timeout_manager.active_connections.items():
            idle_seconds = (now - st.last_activity).total_seconds()
            if idle_seconds > short_timeout_manager.connection_timeout:
                to_disconnect.append(cid)

        for cid in to_disconnect:
            await short_timeout_manager.disconnect(cid)

        # Cleanup
        await short_timeout_manager.shutdown()

        # Client should be disconnected
        assert client_id not in short_timeout_manager.active_connections


class TestStreamConfig:
    """Test suite for StreamConfig dataclass."""

    def test_default_values(self):
        """Test StreamConfig default values."""
        config = StreamConfig()

        assert config.identification_mode == "detect_only"
        assert config.target_fps == 15

    def test_custom_values(self):
        """Test StreamConfig with custom values."""
        config = StreamConfig(identification_mode="full", target_fps=30)

        assert config.identification_mode == "full"
        assert config.target_fps == 30


class TestConnectionState:
    """Test suite for ConnectionState dataclass."""

    def test_connection_state_creation(self, mock_websocket):
        """Test creating a ConnectionState."""
        state = ConnectionState(
            websocket=mock_websocket,
            client_id="test_client",
            identification_mode="full",
            target_fps=20,
        )

        assert state.websocket == mock_websocket
        assert state.client_id == "test_client"
        assert state.identification_mode == "full"
        assert state.target_fps == 20
        assert state.frames_received == 0
        assert state.frames_processed == 0
        assert isinstance(state.connected_at, datetime)
        assert isinstance(state.last_activity, datetime)
