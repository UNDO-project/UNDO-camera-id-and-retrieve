"""Frame buffer queue system for video streaming."""

import asyncio
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional


@dataclass
class FrameData:
    """Raw frame data with metadata."""

    frame_bytes: bytes
    timestamp: float
    frame_number: int

    def as_bytes_io(self) -> BytesIO:
        r"""Convert frame bytes to BytesIO for PIL Image loading.

        :return: BytesIO object containing frame data
        """
        return BytesIO(self.frame_bytes)


@dataclass
class ProcessedFrame:
    """Processed frame with detection results."""

    frame_bytes: bytes
    metadata: dict
    frame_number: int
    processing_time_ms: float


@dataclass
class BufferMetrics:
    """Metrics for frame buffer performance tracking."""

    frames_received: int = 0
    frames_dropped: int = 0
    frames_processed: int = 0
    avg_queue_depth: float = 0.0
    avg_processing_time_ms: float = 0.0
    _queue_depth_samples: list = field(default_factory=list, repr=False)
    _processing_time_samples: list = field(default_factory=list, repr=False)

    def update_queue_depth(self, depth: int) -> None:
        r"""Update average queue depth with new sample.

        :param depth: Current queue depth
        """
        self._queue_depth_samples.append(depth)
        # Keep last 100 samples for rolling average
        if len(self._queue_depth_samples) > 100:
            self._queue_depth_samples.pop(0)
        self.avg_queue_depth = sum(self._queue_depth_samples) / len(
            self._queue_depth_samples
        )

    def update_processing_time(self, time_ms: float) -> None:
        r"""Update average processing time with new sample.

        :param time_ms: Processing time in milliseconds
        """
        self._processing_time_samples.append(time_ms)
        # Keep last 100 samples for rolling average
        if len(self._processing_time_samples) > 100:
            self._processing_time_samples.pop(0)
        self.avg_processing_time_ms = sum(self._processing_time_samples) / len(
            self._processing_time_samples
        )


class FrameBuffer:
    """Bounded frame buffer with input/output queues and metrics tracking.

    Handles burst traffic by buffering frames and dropping oldest frames
    when capacity is reached. Tracks performance metrics and supports
    adaptive frame skipping.
    """

    def __init__(self, max_input: int = 10, max_output: int = 10):
        r"""Initialize frame buffer with bounded queues.

        :param max_input: Maximum input queue size
        :param max_output: Maximum output queue size
        """
        self.max_input = max_input
        self.max_output = max_output
        self.input_queue: asyncio.Queue[FrameData] = asyncio.Queue(maxsize=max_input)
        self.output_queue: asyncio.Queue[ProcessedFrame] = asyncio.Queue(
            maxsize=max_output
        )
        self.metrics = BufferMetrics()
        self._last_fps_check = time.time()
        self._frames_since_last_check = 0

    async def put_frame(
        self, frame: bytes, timestamp: float, frame_number: int
    ) -> bool:
        r"""Add frame to input queue, dropping oldest if full.

        When the input queue is full, the oldest frame is dropped to make
        room for the new frame. This ensures we always process the most
        recent frames.

        :param frame: Raw frame bytes
        :param timestamp: Frame timestamp
        :param frame_number: Sequential frame number
        :return: True if frame was added, False if dropped
        """
        self.metrics.frames_received += 1
        self.metrics.update_queue_depth(self.input_queue.qsize())

        frame_data = FrameData(
            frame_bytes=frame, timestamp=timestamp, frame_number=frame_number
        )

        # If queue is full, drop oldest frame
        if self.input_queue.full():
            try:
                # Remove oldest frame (non-blocking)
                _ = self.input_queue.get_nowait()
                self.metrics.frames_dropped += 1
            except asyncio.QueueEmpty:
                pass  # Queue emptied between check and get

        # Add new frame (should never block due to above logic)
        try:
            self.input_queue.put_nowait(frame_data)
            return True
        except asyncio.QueueFull:
            # Edge case: queue filled between operations
            self.metrics.frames_dropped += 1
            return False

    async def get_frame(self) -> Optional[FrameData]:
        r"""Get next frame from input queue for processing.

        :return: FrameData if available, None if queue is empty
        """
        try:
            frame_data = self.input_queue.get_nowait()
            return frame_data
        except asyncio.QueueEmpty:
            return None

    async def put_result(self, result: ProcessedFrame) -> bool:
        r"""Add processed frame to output queue.

        If output queue is full, drops the oldest result to make room.

        :param result: Processed frame with metadata
        :return: True if result was added, False if dropped
        """
        self.metrics.frames_processed += 1
        self.metrics.update_processing_time(result.processing_time_ms)

        # If queue is full, drop oldest result
        if self.output_queue.full():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        # Add new result
        try:
            self.output_queue.put_nowait(result)
            return True
        except asyncio.QueueFull:
            return False

    async def get_result(self) -> Optional[ProcessedFrame]:
        r"""Get next processed frame from output queue.

        :return: ProcessedFrame if available, None if queue is empty
        """
        try:
            result = self.output_queue.get_nowait()
            return result
        except asyncio.QueueEmpty:
            return None

    def should_skip_frame(self, current_fps: float, target_fps: float) -> bool:
        r"""Determine if frame should be skipped based on processing load.

        Implements adaptive frame skipping:
        - If processing faster than target, don't skip
        - If processing slower, skip frames to catch up
        - Takes queue depth into account

        :param current_fps: Current processing rate
        :param target_fps: Target processing rate
        :return: True if frame should be skipped
        """
        # Never skip if we haven't processed any frames yet
        if current_fps == 0.0:
            return False

        # Never skip if processing faster than target
        if current_fps >= target_fps:
            return False

        # Skip if significantly behind target
        fps_ratio = current_fps / target_fps if target_fps > 0 else 1.0
        if fps_ratio < 0.7:  # Processing at less than 70% of target
            return True

        # Skip if input queue is backing up
        queue_utilization = self.input_queue.qsize() / self.max_input
        if queue_utilization > 0.8:  # Queue more than 80% full
            return True

        return False

    def calculate_current_fps(self) -> float:
        r"""Calculate current processing FPS based on recent frames.

        :return: Current frames per second
        """
        self._frames_since_last_check += 1
        elapsed = time.time() - self._last_fps_check

        if elapsed >= 1.0:  # Update every second
            fps = self._frames_since_last_check / elapsed
            self._frames_since_last_check = 0
            self._last_fps_check = time.time()
            return fps

        return 0.0  # Not enough time elapsed

    def get_metrics(self) -> dict:
        r"""Get current buffer metrics.

        :return: Dictionary with buffer statistics
        """
        return {
            "frames_received": self.metrics.frames_received,
            "frames_dropped": self.metrics.frames_dropped,
            "frames_processed": self.metrics.frames_processed,
            "avg_queue_depth": round(self.metrics.avg_queue_depth, 2),
            "avg_processing_time_ms": round(self.metrics.avg_processing_time_ms, 2),
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "input_queue_utilization": round(
                self.input_queue.qsize() / self.max_input, 2
            ),
            "output_queue_utilization": round(
                self.output_queue.qsize() / self.max_output, 2
            ),
        }

    def clear(self) -> None:
        """Clear all queues and reset metrics."""
        # Clear input queue
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clear output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Reset metrics
        self.metrics = BufferMetrics()
        self._last_fps_check = time.time()
        self._frames_since_last_check = 0
