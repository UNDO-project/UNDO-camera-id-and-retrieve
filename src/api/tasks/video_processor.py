"""Background video processor for camera detection and identification.

This module provides the VideoProcessorTask class that processes uploaded videos
frame-by-frame, applying camera detection and optional identification, and writes
annotated output videos.
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import cv2
import numpy as np
from loguru import logger

from src.api.tasks.store import task_store
from src.config import paths
from src.identification.async_service import AsyncIdentificationService
from src.identification.renderer import FrameRenderer, RenderConfig
from src.models.identification import CameraDetection, CameraMatch


@dataclass
class VideoProcessingConfig:
    r"""Configuration for video processing task.

    :param target_fps: Target frames per second for processing
    :param identification_mode: Processing mode (detect_only or full)
    :param output_format: Output video format
    :param jpeg_quality: Quality for intermediate frame encoding
    :param top_k: Number of top matches to return per detection (full mode)
    :param min_similarity: Minimum similarity threshold for matches
    """

    target_fps: int = 15
    identification_mode: Literal["detect_only", "full"] = "detect_only"
    output_format: str = "mp4"
    jpeg_quality: int = 85
    top_k: int = 5
    min_similarity: float = 0.3


class VideoProcessorTask:
    r"""Background task for processing video files with camera detection.

    Reads video frame-by-frame, applies detection/identification, annotates
    frames with bounding boxes and labels, and writes output video.

    Thread Safety:
        This class uses AsyncIdentificationService which handles thread safety
        internally. Multiple VideoProcessorTask instances can run concurrently.

    Example:
        ```python
        config = VideoProcessingConfig(target_fps=15, identification_mode="detect_only")
        task = VideoProcessorTask("task-123", config)
        await task.process(input_path, output_path)
        ```
    """

    def __init__(
        self,
        task_id: str,
        config: VideoProcessingConfig,
        service: Optional[AsyncIdentificationService] = None,
    ) -> None:
        r"""Initialize video processor task.

        :param task_id: Unique identifier for this task
        :param config: Processing configuration
        :param service: Optional pre-initialized identification service
        """
        self.task_id = task_id
        self.config = config
        self._service = service
        self._renderer = FrameRenderer(RenderConfig(style="detailed"))
        self._cancelled = False

    @property
    def service(self) -> AsyncIdentificationService:
        r"""Get or initialize the identification service.

        :return: AsyncIdentificationService instance
        """
        if self._service is None:
            logger.info(f"Task {self.task_id}: Initializing identification service...")
            self._service = AsyncIdentificationService()
        return self._service

    def cancel(self) -> None:
        r"""Cancel the processing task."""
        self._cancelled = True
        logger.info(f"Task {self.task_id}: Cancellation requested")

    async def process(self, input_path: Path, output_path: Path) -> None:
        r"""Process video file and write annotated output.

        :param input_path: Path to input video file
        :param output_path: Path where output video will be written
        :raises RuntimeError: If video processing fails
        """
        progress = task_store.get(self.task_id)
        if progress is None:
            raise RuntimeError(f"Task not found: {self.task_id}")

        # Mark task as processing
        progress.start()
        task_store.update(self.task_id, progress)

        logger.info(
            f"Task {self.task_id}: Starting video processing "
            f"(mode={self.config.identification_mode}, target_fps={self.config.target_fps})"
        )

        cap = None
        writer = None

        try:
            # Open video capture
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {input_path}")

            # Get video properties
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Update progress with accurate frame count
            progress.frames_total = total_frames
            task_store.update(self.task_id, progress)

            # Calculate frame interval for target FPS
            # E.g., if source is 30fps and target is 15fps, process every 2nd frame
            frame_interval = max(1, int(source_fps / self.config.target_fps))

            logger.info(
                f"Task {self.task_id}: Video properties - "
                f"{width}x{height} @ {source_fps:.1f}fps, {total_frames} frames, "
                f"frame_interval={frame_interval}"
            )

            # Create output video writer
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_fps = min(self.config.target_fps, source_fps)
            writer = cv2.VideoWriter(
                str(output_path), fourcc, output_fps, (width, height)
            )

            if not writer.isOpened():
                raise RuntimeError(f"Failed to create output video: {output_path}")

            # Process frames
            frame_num = 0
            processed_count = 0
            last_progress_update = time.time()

            while True:
                if self._cancelled:
                    logger.warning(f"Task {self.task_id}: Processing cancelled")
                    progress.fail("Processing cancelled by user")
                    task_store.update(self.task_id, progress)
                    return

                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1

                # Process this frame if it matches our interval
                if (frame_num - 1) % frame_interval == 0:
                    try:
                        annotated_frame = await self._process_frame(frame)
                        writer.write(annotated_frame)
                        processed_count += 1
                    except Exception as e:
                        logger.warning(
                            f"Task {self.task_id}: Error processing frame {frame_num}: {e}"
                        )
                        # Write original frame on error
                        writer.write(frame)
                        processed_count += 1

                # Update progress periodically (every 0.5 seconds)
                current_time = time.time()
                if current_time - last_progress_update >= 0.5:
                    progress.update(frame_num, total_frames)
                    task_store.update(self.task_id, progress)
                    last_progress_update = current_time

                    if frame_num % 100 == 0:
                        logger.debug(
                            f"Task {self.task_id}: Processed {frame_num}/{total_frames} frames "
                            f"({progress.progress_percent:.1f}%), "
                            f"FPS: {progress.current_fps:.1f}"
                        )

                # Yield control to event loop periodically
                if frame_num % 10 == 0:
                    await asyncio.sleep(0)

            # Processing complete
            progress.complete(output_path)
            task_store.update(self.task_id, progress)

            logger.success(
                f"Task {self.task_id}: Processing complete - "
                f"{processed_count} frames processed in {progress.elapsed_seconds:.1f}s "
                f"(avg {progress.current_fps:.1f} fps)"
            )

        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"Task {self.task_id}: {error_msg}")
            progress.fail(error_msg)
            task_store.update(self.task_id, progress)
            raise RuntimeError(error_msg) from e

        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()

    async def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        r"""Process a single frame with detection/identification.

        :param frame: Input frame in BGR format (OpenCV default)
        :return: Annotated frame in BGR format
        """
        # Convert BGR to JPEG bytes for identification service
        _, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality]
        )
        frame_bytes = encoded.tobytes()

        # Run detection/identification
        detections: List[CameraDetection] = []
        matches: Optional[List[List[CameraMatch]]] = None

        if self.config.identification_mode == "detect_only":
            detections = await self.service.detect_frame(frame_bytes)
        else:
            results = await self.service.identify_frame(
                frame_bytes,
                top_k=self.config.top_k,
                min_similarity=self.config.min_similarity,
            )
            detections = [r.detection for r in results]
            matches = [r.matches for r in results]

        # Convert BGR to RGB for renderer
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Annotate frame
        annotated_rgb = self._renderer.annotate_frame(frame_rgb, detections, matches)

        # Convert back to BGR for video writer
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        return annotated_bgr


async def run_video_processing_task(task_id: str) -> None:
    r"""Run video processing as a background task.

    This function is designed to be called from FastAPI BackgroundTasks.
    It retrieves task information from the store and processes the video.

    :param task_id: Unique identifier of the task to process
    """
    progress = task_store.get(task_id)
    if progress is None:
        logger.error(f"Task not found: {task_id}")
        return

    if progress.input_path is None:
        logger.error(f"Task {task_id}: No input path specified")
        progress.fail("No input file specified")
        task_store.update(task_id, progress)
        return

    # Build output path
    paths.video_outputs_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{task_id}_processed.{progress.output_format}"
    output_path = paths.video_outputs_dir / output_filename

    # Create processor config
    config = VideoProcessingConfig(
        target_fps=progress.target_fps,
        identification_mode=progress.identification_mode,
        output_format=progress.output_format,
    )

    # Run processing
    processor = VideoProcessorTask(task_id, config)

    try:
        await processor.process(progress.input_path, output_path)
    except Exception as e:
        logger.error(f"Task {task_id}: Processing failed - {e}")
        # Error already recorded in progress by processor
