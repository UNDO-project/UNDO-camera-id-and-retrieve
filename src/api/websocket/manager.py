"""WebSocket connection manager for video streaming."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


@dataclass
class StreamConfig:
    """Configuration for a video stream connection."""

    identification_mode: Literal["detect_only", "full"] = "detect_only"
    target_fps: int = 15


@dataclass
class ConnectionState:
    """State tracking for an active WebSocket connection."""

    websocket: WebSocket
    client_id: str
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    frames_received: int = 0
    frames_processed: int = 0
    identification_mode: Literal["detect_only", "full"] = "detect_only"
    target_fps: int = 15


class ConnectionManager:
    """Manages WebSocket connections for video streaming.

    Handles connection lifecycle (connect, disconnect), tracks connection state,
    implements timeouts, and provides connection statistics.
    """

    def __init__(self, connection_timeout: int = 30):
        r"""Initialize the connection manager.

        :param connection_timeout: Seconds of inactivity before closing connection
        """
        self.active_connections: Dict[str, ConnectionState] = {}
        self.connection_timeout = connection_timeout
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def connect(
        self, websocket: WebSocket, client_id: str, config: StreamConfig
    ) -> None:
        r"""Accept and register a new WebSocket connection.

        :param websocket: The WebSocket connection to accept
        :param client_id: Unique identifier for this client
        :param config: Stream configuration settings
        :raises RuntimeError: If client_id already exists
        """
        if client_id in self.active_connections:
            logger.warning(f"Client {client_id} already connected, rejecting duplicate")
            await websocket.close(code=1008, reason="Client ID already connected")
            raise RuntimeError(f"Client {client_id} already connected")

        await websocket.accept()

        state = ConnectionState(
            websocket=websocket,
            client_id=client_id,
            identification_mode=config.identification_mode,
            target_fps=config.target_fps,
        )

        self.active_connections[client_id] = state
        logger.info(
            f"Client {client_id} connected (mode={config.identification_mode}, "
            f"fps={config.target_fps})"
        )

        # Start cleanup task if not already running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(
                self._cleanup_inactive_connections()
            )

    async def disconnect(self, client_id: str) -> None:
        r"""Disconnect and remove a client connection.

        :param client_id: The client to disconnect
        """
        if client_id in self.active_connections:
            state = self.active_connections[client_id]
            try:
                await state.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket for {client_id}: {e}")

            del self.active_connections[client_id]
            logger.info(
                f"Client {client_id} disconnected "
                f"(received={state.frames_received}, processed={state.frames_processed})"
            )

    async def send_bytes(self, client_id: str, data: bytes) -> None:
        r"""Send binary data to a specific client.

        :param client_id: The client to send to
        :param data: Binary data to send
        :raises KeyError: If client_id not found
        :raises WebSocketDisconnect: If client has disconnected
        """
        if client_id not in self.active_connections:
            raise KeyError(f"Client {client_id} not connected")

        state = self.active_connections[client_id]
        try:
            await state.websocket.send_bytes(data)
        except WebSocketDisconnect:
            logger.warning(f"Client {client_id} disconnected during send")
            await self.disconnect(client_id)
            raise

    async def send_text(self, client_id: str, data: str) -> None:
        r"""Send text data to a specific client.

        :param client_id: The client to send to
        :param data: Text data to send
        :raises KeyError: If client_id not found
        :raises WebSocketDisconnect: If client has disconnected
        """
        if client_id not in self.active_connections:
            raise KeyError(f"Client {client_id} not connected")

        state = self.active_connections[client_id]
        try:
            await state.websocket.send_text(data)
        except WebSocketDisconnect:
            logger.warning(f"Client {client_id} disconnected during send")
            await self.disconnect(client_id)
            raise

    async def broadcast(self, data: bytes) -> None:
        r"""Broadcast binary data to all connected clients.

        :param data: Binary data to broadcast
        """
        disconnected = []
        for client_id, state in self.active_connections.items():
            try:
                await state.websocket.send_bytes(data)
            except WebSocketDisconnect:
                logger.warning(f"Client {client_id} disconnected during broadcast")
                disconnected.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected:
            await self.disconnect(client_id)

    def update_activity(self, client_id: str) -> None:
        r"""Update the last activity timestamp for a client.

        :param client_id: The client to update
        """
        if client_id in self.active_connections:
            self.active_connections[client_id].last_activity = datetime.now(
                timezone.utc
            )

    def increment_frames_received(self, client_id: str) -> None:
        r"""Increment the frames received counter for a client.

        :param client_id: The client to update
        """
        if client_id in self.active_connections:
            self.active_connections[client_id].frames_received += 1
            self.update_activity(client_id)

    def increment_frames_processed(self, client_id: str) -> None:
        r"""Increment the frames processed counter for a client.

        :param client_id: The client to update
        """
        if client_id in self.active_connections:
            self.active_connections[client_id].frames_processed += 1

    def get_connection_stats(self) -> dict:
        r"""Get statistics about all active connections.

        :return: Connection statistics dict with active count and per-connection details
        """
        total_received = sum(
            s.frames_received for s in self.active_connections.values()
        )
        total_processed = sum(
            s.frames_processed for s in self.active_connections.values()
        )

        connection_details = [
            {
                "client_id": client_id,
                "connected_at": state.connected_at.isoformat(),
                "last_activity": state.last_activity.isoformat(),
                "frames_received": state.frames_received,
                "frames_processed": state.frames_processed,
                "identification_mode": state.identification_mode,
                "target_fps": state.target_fps,
                "idle_seconds": (
                    datetime.now(timezone.utc) - state.last_activity
                ).total_seconds(),
            }
            for client_id, state in self.active_connections.items()
        ]

        return {
            "active_connections": len(self.active_connections),
            "total_frames_received": total_received,
            "total_frames_processed": total_processed,
            "connections": connection_details,
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown all connections.

        Closes all active WebSocket connections and stops background tasks.
        """
        self._shutdown = True
        logger.info(
            f"Shutting down connection manager ({len(self.active_connections)} active)"
        )

        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Disconnect all clients
        client_ids = list(self.active_connections.keys())
        for client_id in client_ids:
            await self.disconnect(client_id)

        logger.info("Connection manager shutdown complete")

    async def _cleanup_inactive_connections(self) -> None:
        """Background task to clean up inactive connections.

        Runs continuously, checking for connections that have exceeded
        the inactivity timeout and closing them.
        """
        while not self._shutdown:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                now = datetime.now(timezone.utc)
                to_disconnect = []

                for client_id, state in self.active_connections.items():
                    idle_seconds = (now - state.last_activity).total_seconds()
                    if idle_seconds > self.connection_timeout:
                        logger.warning(
                            f"Client {client_id} idle for {idle_seconds:.1f}s, "
                            "disconnecting"
                        )
                        to_disconnect.append(client_id)

                # Disconnect idle clients
                for client_id in to_disconnect:
                    await self.disconnect(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
