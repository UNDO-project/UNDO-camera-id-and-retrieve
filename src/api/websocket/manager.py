"""WebSocket connection manager for video streaming."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Literal, Optional

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from src.api.config import stream_settings


@dataclass
class StreamConfig:
    """Configuration for a video stream connection."""

    identification_mode: Literal["detect_only", "full"] = "detect_only"
    target_fps: int = field(default_factory=lambda: stream_settings.default_fps)


@dataclass
class ConnectionState:
    """State tracking for an active WebSocket connection."""

    websocket: WebSocket
    client_id: str
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    total_processing_time_ms: float = 0.0
    identification_mode: Literal["detect_only", "full"] = "detect_only"
    target_fps: int = field(default_factory=lambda: stream_settings.default_fps)

    @property
    def average_processing_time_ms(self) -> float:
        r"""Calculate average processing time per frame.

        :return: Average processing time in milliseconds
        """
        if self.frames_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.frames_processed

    @property
    def current_fps(self) -> float:
        r"""Calculate current FPS based on frames processed and connection duration.

        :return: Frames per second
        """
        duration = (datetime.now(timezone.utc) - self.connected_at).total_seconds()
        if duration == 0:
            return 0.0
        return self.frames_processed / duration


class ConnectionManager:
    """Manages WebSocket connections for video streaming.

    Handles connection lifecycle (connect, disconnect), tracks connection state,
    implements timeouts, enforces connection limits, and provides statistics.
    """

    def __init__(
        self,
        max_connections: Optional[int] = None,
        connection_timeout: Optional[int] = None,
    ):
        r"""Initialize the connection manager.

        :param max_connections: Maximum concurrent connections (uses settings if None)
        :param connection_timeout: Seconds of inactivity before closing (uses settings if None)
        """
        self.active_connections: Dict[str, ConnectionState] = {}
        self.max_connections = max_connections or stream_settings.max_connections
        self.connection_timeout = (
            connection_timeout or stream_settings.connection_timeout_seconds
        )
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        self._total_connections_served: int = 0
        self._total_frames_received: int = 0
        self._total_frames_processed: int = 0
        self._total_frames_dropped: int = 0
        self._total_processing_time_ms: float = 0.0

    async def connect(
        self, websocket: WebSocket, client_id: str, config: StreamConfig
    ) -> None:
        r"""Accept and register a new WebSocket connection.

        :param websocket: The WebSocket connection to accept
        :param client_id: Unique identifier for this client
        :param config: Stream configuration settings
        :raises RuntimeError: If client_id already exists or max connections reached
        """
        # Check connection limit
        if len(self.active_connections) >= self.max_connections:
            logger.warning(
                f"Connection limit reached ({self.max_connections}), "
                f"rejecting client {client_id}"
            )
            await websocket.close(
                code=1013, reason="Maximum connections reached, try again later"
            )
            raise RuntimeError("Maximum connections reached")

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
        self._total_connections_served += 1
        logger.info(
            f"Client {client_id} connected (mode={config.identification_mode}, "
            f"fps={config.target_fps}, active={len(self.active_connections)}/{self.max_connections})"
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

            # Update global counters
            self._total_frames_received += state.frames_received
            self._total_frames_processed += state.frames_processed
            self._total_frames_dropped += state.frames_dropped
            self._total_processing_time_ms += state.total_processing_time_ms

            try:
                await state.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket for {client_id}: {e}")

            del self.active_connections[client_id]
            logger.info(
                f"Client {client_id} disconnected "
                f"(received={state.frames_received}, processed={state.frames_processed}, "
                f"dropped={state.frames_dropped})"
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

    def increment_frames_processed(
        self, client_id: str, processing_time_ms: float = 0.0
    ) -> None:
        r"""Increment the frames processed counter for a client.

        :param client_id: The client to update
        :param processing_time_ms: Processing time for this frame in milliseconds
        """
        if client_id in self.active_connections:
            self.active_connections[client_id].frames_processed += 1
            self.active_connections[
                client_id
            ].total_processing_time_ms += processing_time_ms

    def increment_frames_dropped(self, client_id: str) -> None:
        r"""Increment the frames dropped counter for a client.

        :param client_id: The client to update
        """
        if client_id in self.active_connections:
            self.active_connections[client_id].frames_dropped += 1

    def get_connection_stats(self) -> dict:
        r"""Get statistics about all active connections.

        :return: Connection statistics dict with active count and per-connection details
        """
        # Current session stats
        current_received = sum(
            s.frames_received for s in self.active_connections.values()
        )
        current_processed = sum(
            s.frames_processed for s in self.active_connections.values()
        )
        current_dropped = sum(
            s.frames_dropped for s in self.active_connections.values()
        )
        total_processing_time = sum(
            s.total_processing_time_ms for s in self.active_connections.values()
        )

        # Calculate average FPS across active connections
        avg_fps = 0.0
        if self.active_connections:
            fps_values = [s.current_fps for s in self.active_connections.values()]
            avg_fps = sum(fps_values) / len(fps_values)

        connection_details = [
            {
                "client_id": client_id,
                "connected_at": state.connected_at.isoformat(),
                "last_activity": state.last_activity.isoformat(),
                "frames_received": state.frames_received,
                "frames_processed": state.frames_processed,
                "frames_dropped": state.frames_dropped,
                "current_fps": round(state.current_fps, 1),
                "avg_processing_time_ms": round(state.average_processing_time_ms, 2),
                "identification_mode": state.identification_mode,
                "target_fps": state.target_fps,
                "idle_seconds": round(
                    (datetime.now(timezone.utc) - state.last_activity).total_seconds(),
                    1,
                ),
            }
            for client_id, state in self.active_connections.items()
        ]

        # Calculate global average latency
        total_all_processed = self._total_frames_processed + current_processed
        total_all_processing_time = (
            self._total_processing_time_ms + total_processing_time
        )
        global_avg_latency = (
            total_all_processing_time / total_all_processed
            if total_all_processed > 0
            else 0.0
        )

        return {
            "active_connections": len(self.active_connections),
            "max_connections": self.max_connections,
            "total_connections_served": self._total_connections_served,
            "total_frames_received": self._total_frames_received + current_received,
            "total_frames_processed": total_all_processed,
            "total_frames_dropped": self._total_frames_dropped + current_dropped,
            "average_fps": round(avg_fps, 1),
            "average_latency_ms": round(global_avg_latency, 2),
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
