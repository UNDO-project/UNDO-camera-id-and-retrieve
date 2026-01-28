"""WebSocket video stream endpoint."""

import json
import time
import uuid
from typing import Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from loguru import logger

from src.api.dependencies import get_connection_manager
from src.api.websocket.manager import ConnectionManager, StreamConfig

router = APIRouter()


@router.websocket("/video-stream")
async def video_stream(
    websocket: WebSocket,
    mode: Literal["detect_only", "full"] = Query(
        default="detect_only",
        description="Processing mode: detect_only or full identification",
    ),
    target_fps: int = Query(
        default=15,
        ge=1,
        le=30,
        description="Target frames per second for processing",
    ),
) -> None:
    r"""WebSocket endpoint for real-time video frame processing.

    Accepts a WebSocket connection and processes video frames in real-time,
    returning annotated frames and detection metadata.

    Binary Message Protocol:
      - Client sends: Raw JPEG frame bytes
      - Server sends: Annotated JPEG bytes (binary) + JSON metadata (text)

    :param websocket: The WebSocket connection
    :param mode: Processing mode (detect_only or full)
    :param target_fps: Target frames per second (1-30)

    Example:
        ```javascript
        const ws = new WebSocket('ws://localhost:8000/api/v1/ws/video-stream?mode=detect_only&target_fps=15');
        ws.binaryType = 'arraybuffer';

        // Send frame
        ws.send(frameData);

        // Receive results
        ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Annotated frame
                const blob = new Blob([event.data], {type: 'image/jpeg'});
            } else {
                // JSON metadata
                const metadata = JSON.parse(event.data);
            }
        };
        ```
    """
    client_id = str(uuid.uuid4())
    manager: ConnectionManager = get_connection_manager()

    # Create stream configuration
    config = StreamConfig(identification_mode=mode, target_fps=target_fps)

    # Connect client
    try:
        await manager.connect(websocket, client_id, config)
        logger.info(
            f"Video stream connected: {client_id} (mode={mode}, fps={target_fps})"
        )
    except RuntimeError as e:
        logger.error(f"Failed to connect client {client_id}: {e}")
        return

    frame_number = 0

    try:
        while True:
            # Receive frame from client
            frame_bytes = await websocket.receive_bytes()
            frame_number += 1

            # Update frame received counter
            manager.increment_frames_received(client_id)

            start_time = time.time()

            # TODO (Issue #4): Replace with actual async identification service
            # For now, echo back the original frame
            processed_frame = frame_bytes

            # TODO (Issue #4): Replace with actual detection results
            # Placeholder metadata
            metadata = {
                "detections": [],
                "processing_time_ms": (time.time() - start_time) * 1000,
                "frame_number": frame_number,
                "fps_actual": target_fps,  # TODO: Calculate actual FPS
            }

            # Send processed frame (binary)
            await manager.send_bytes(client_id, processed_frame)

            # Send metadata (text)
            await manager.send_text(client_id, json.dumps(metadata))

            # Update frame processed counter
            manager.increment_frames_processed(client_id)

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error processing frame for client {client_id}: {e}")
    finally:
        # Cleanup connection
        await manager.disconnect(client_id)
        logger.info(f"Video stream closed: {client_id}")
