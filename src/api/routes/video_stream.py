"""WebSocket video stream endpoint."""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from loguru import logger
from PIL import Image

from src.api.dependencies import (
    get_connection_manager,
    get_async_identification_service,
)
from src.api.websocket.frame_buffer import FrameBuffer, FrameData
from src.api.websocket.fps_tracker import FPSTracker
from src.api.websocket.manager import ConnectionManager, StreamConfig
from src.identification.async_service import AsyncIdentificationService
from src.identification.renderer import FrameRenderer
from src.models.identification import CameraDetection, CameraMatch

router = APIRouter()


def _build_metadata(
    detections: List[CameraDetection],
    matches: Optional[List[List[CameraMatch]]],
    processing_time_ms: float,
    frame_number: int,
    fps_actual: float,
) -> Dict[str, Any]:
    r"""Build JSON metadata response for a processed frame.

    :param detections: List of camera detections in the frame
    :param matches: Optional list of matches per detection (full mode only)
    :param processing_time_ms: Time taken to process frame in milliseconds
    :param frame_number: Sequential frame number
    :param fps_actual: Actual processing FPS
    :return: Dictionary containing frame metadata
    """
    detection_dicts: List[Dict[str, Any]] = []

    for idx, detection in enumerate(detections):
        detection_dict: Dict[str, Any] = {
            "bbox": [
                detection.bbox.x_min,
                detection.bbox.y_min,
                detection.bbox.x_max,
                detection.bbox.y_max,
            ],
            "confidence": detection.confidence,
            "class": detection.label,
        }

        # Add matches if in full mode
        if matches and idx < len(matches):
            match_dicts: List[Dict[str, Any]] = []
            for match in matches[idx]:
                match_dict: Dict[str, Any] = {
                    "camera_id": match.camera_id,
                    "similarity": match.score,
                    "model_name": match.record.model_name if match.record else None,
                    "manufacturer": match.record.source if match.record else None,
                }
                match_dicts.append(match_dict)
            detection_dict["matches"] = match_dicts

        detection_dicts.append(detection_dict)

    return {
        "detections": detection_dicts,
        "processing_time_ms": round(processing_time_ms, 2),
        "frame_number": frame_number,
        "fps_actual": round(fps_actual, 1),
    }


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
    service: AsyncIdentificationService = Depends(get_async_identification_service),
    manager: ConnectionManager = Depends(get_connection_manager),
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
    :param service: Injected async identification service
    :param manager: Injected connection manager

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

    # Initialize processing components (service injected via dependencies)
    renderer = FrameRenderer()
    buffer = FrameBuffer(max_input=10, max_output=10)
    fps_tracker = FPSTracker(window_size=30)

    # Shutdown flag for coordinating receiver and processor tasks
    shutdown = asyncio.Event()

    async def receiver() -> None:
        r"""Receive frames from client and add to buffer.

        Runs continuously until disconnect or error. Frames are added to
        the input buffer for processing by the processor task.
        """
        try:
            while not shutdown.is_set():
                frame_bytes = await websocket.receive_bytes()
                timestamp = time.time()

                # Add frame to buffer (will drop oldest if full)
                # Frame number is 1-indexed
                dropped = not await buffer.put_frame(
                    frame=frame_bytes,
                    timestamp=timestamp,
                    frame_number=buffer.metrics.frames_received + 1,
                )

                # Update connection state
                manager.increment_frames_received(client_id)

                if dropped:
                    logger.debug(
                        f"Frame {buffer.metrics.frames_received} dropped (buffer full)"
                    )

        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected (receiver)")
            shutdown.set()
        except Exception as e:
            logger.error(f"Error in receiver for {client_id}: {e}")
            shutdown.set()

    async def processor() -> None:
        r"""Process frames from buffer and send results.

        Runs continuously until shutdown. Gets frames from buffer, processes
        them with optional frame skipping based on FPS target, and sends
        results back to client.
        """
        try:
            while not shutdown.is_set():
                # Get next frame from buffer
                frame_data: Optional[FrameData] = await buffer.get_frame()
                if frame_data is None:
                    # No frames available, use wait_for with shutdown check
                    try:
                        await asyncio.wait_for(shutdown.wait(), timeout=0.01)
                        # Shutdown was set, exit
                        break
                    except asyncio.TimeoutError:
                        # Continue waiting for frames
                        continue

                # Check if we should skip this frame based on FPS target
                if buffer.should_skip_frame(fps_tracker.current, target_fps):
                    logger.debug(
                        f"Skipping frame {frame_data.frame_number} "
                        f"(current FPS: {fps_tracker.current:.1f} > target: {target_fps})"
                    )
                    continue

                start_time = time.time()

                try:
                    # Process frame with timeout
                    async with asyncio.timeout(5.0):
                        if mode == "detect_only":
                            # Detection only mode
                            detections = await service.detect_frame(
                                frame_data.frame_bytes
                            )
                            matches = None
                        else:
                            # Full identification mode
                            results = await service.identify_frame(
                                frame_data.frame_bytes
                            )
                            detections = [r.detection for r in results]
                            matches = [r.matches for r in results]

                        # Decode frame for annotation
                        image = Image.open(frame_data.as_bytes_io()).convert("RGB")
                        frame_array = np.array(image)

                        # Annotate frame
                        annotated = renderer.annotate_frame(
                            frame_array, detections, matches
                        )

                        # Encode to JPEG
                        processed_bytes = renderer.encode_jpeg(annotated, quality=85)

                        # Build metadata
                        processing_time_ms = (time.time() - start_time) * 1000
                        fps_tracker.record_frame()

                        metadata = _build_metadata(
                            detections=detections,
                            matches=matches,
                            processing_time_ms=processing_time_ms,
                            frame_number=frame_data.frame_number,
                            fps_actual=fps_tracker.current,
                        )

                        # Send results
                        await manager.send_bytes(client_id, processed_bytes)
                        await manager.send_text(client_id, json.dumps(metadata))

                        # Update connection state
                        manager.increment_frames_processed(client_id)

                        logger.debug(
                            f"Processed frame {frame_data.frame_number} in {processing_time_ms:.1f}ms "
                            f"({len(detections)} detections, FPS: {fps_tracker.current:.1f})"
                        )

                except asyncio.TimeoutError:
                    logger.warning(
                        f"Frame {frame_data.frame_number} processing timeout (>5s)"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Error processing frame {frame_data.frame_number}: {e}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Error in processor for {client_id}: {e}")
            shutdown.set()

    # Run receiver and processor concurrently
    try:
        await asyncio.gather(receiver(), processor())
    except Exception as e:
        logger.error(f"Error in video stream tasks for {client_id}: {e}")
    finally:
        # Ensure shutdown is set
        shutdown.set()
        # Cleanup connection
        await manager.disconnect(client_id)
        logger.info(
            f"Video stream closed: {client_id} "
            f"({fps_tracker.total_frames} frames processed, "
            f"{buffer.metrics.frames_dropped} dropped)"
        )
