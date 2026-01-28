"""Tests for frame buffer queue system."""

import asyncio
import time

import pytest

from src.api.websocket.frame_buffer import (
    BufferMetrics,
    FrameBuffer,
    FrameData,
    ProcessedFrame,
)


@pytest.fixture
def frame_buffer():
    """Create a FrameBuffer instance for testing."""
    return FrameBuffer(max_input=10, max_output=10)


@pytest.fixture
def small_buffer():
    """Create a small buffer for testing overflow behavior."""
    return FrameBuffer(max_input=3, max_output=3)


class TestFrameData:
    """Test suite for FrameData dataclass."""

    def test_frame_data_creation(self):
        """Test creating FrameData."""
        frame = FrameData(
            frame_bytes=b"test_frame", timestamp=time.time(), frame_number=1
        )

        assert frame.frame_bytes == b"test_frame"
        assert isinstance(frame.timestamp, float)
        assert frame.frame_number == 1


class TestProcessedFrame:
    """Test suite for ProcessedFrame dataclass."""

    def test_processed_frame_creation(self):
        """Test creating ProcessedFrame."""
        frame = ProcessedFrame(
            frame_bytes=b"processed",
            metadata={"detections": []},
            frame_number=1,
            processing_time_ms=45.5,
        )

        assert frame.frame_bytes == b"processed"
        assert frame.metadata == {"detections": []}
        assert frame.frame_number == 1
        assert frame.processing_time_ms == 45.5


class TestBufferMetrics:
    """Test suite for BufferMetrics."""

    def test_metrics_initialization(self):
        """Test metrics are initialized to zero."""
        metrics = BufferMetrics()

        assert metrics.frames_received == 0
        assert metrics.frames_dropped == 0
        assert metrics.frames_processed == 0
        assert metrics.avg_queue_depth == 0.0
        assert metrics.avg_processing_time_ms == 0.0

    def test_update_queue_depth(self):
        """Test updating queue depth average."""
        metrics = BufferMetrics()

        metrics.update_queue_depth(5)
        assert metrics.avg_queue_depth == 5.0

        metrics.update_queue_depth(3)
        assert metrics.avg_queue_depth == 4.0  # Average of 5 and 3

        metrics.update_queue_depth(7)
        assert metrics.avg_queue_depth == 5.0  # Average of 5, 3, 7

    def test_update_processing_time(self):
        """Test updating processing time average."""
        metrics = BufferMetrics()

        metrics.update_processing_time(100.0)
        assert metrics.avg_processing_time_ms == 100.0

        metrics.update_processing_time(200.0)
        assert metrics.avg_processing_time_ms == 150.0  # Average of 100 and 200

    def test_rolling_average_limit(self):
        """Test that rolling averages keep last 100 samples."""
        metrics = BufferMetrics()

        # Add 150 samples
        for i in range(150):
            metrics.update_queue_depth(i)

        # Should only keep last 100
        assert len(metrics._queue_depth_samples) == 100
        # First 50 should be dropped
        assert metrics._queue_depth_samples[0] == 50


class TestFrameBuffer:
    """Test suite for FrameBuffer."""

    @pytest.mark.asyncio
    async def test_buffer_initialization(self, frame_buffer):
        """Test buffer is initialized correctly."""
        assert frame_buffer.max_input == 10
        assert frame_buffer.max_output == 10
        assert frame_buffer.input_queue.qsize() == 0
        assert frame_buffer.output_queue.qsize() == 0
        assert frame_buffer.metrics.frames_received == 0

    @pytest.mark.asyncio
    async def test_put_and_get_frame(self, frame_buffer):
        """Test adding and retrieving frames."""
        # Add a frame
        result = await frame_buffer.put_frame(b"frame1", time.time(), 1)
        assert result is True
        assert frame_buffer.input_queue.qsize() == 1
        assert frame_buffer.metrics.frames_received == 1

        # Get the frame
        frame = await frame_buffer.get_frame()
        assert frame is not None
        assert frame.frame_bytes == b"frame1"
        assert frame.frame_number == 1
        assert frame_buffer.input_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_get_frame_empty_queue(self, frame_buffer):
        """Test getting frame from empty queue returns None."""
        frame = await frame_buffer.get_frame()
        assert frame is None

    @pytest.mark.asyncio
    async def test_multiple_frames(self, frame_buffer):
        """Test adding multiple frames."""
        # Add 5 frames
        for i in range(5):
            await frame_buffer.put_frame(f"frame{i}".encode(), time.time(), i)

        assert frame_buffer.input_queue.qsize() == 5
        assert frame_buffer.metrics.frames_received == 5

        # Retrieve all frames
        for i in range(5):
            frame = await frame_buffer.get_frame()
            assert frame is not None
            assert frame.frame_bytes == f"frame{i}".encode()
            assert frame.frame_number == i

    @pytest.mark.asyncio
    async def test_queue_full_drops_oldest(self, small_buffer):
        """Test that oldest frame is dropped when queue is full."""
        # Fill buffer (max 3)
        for i in range(3):
            await small_buffer.put_frame(f"frame{i}".encode(), time.time(), i)

        assert small_buffer.input_queue.qsize() == 3
        assert small_buffer.metrics.frames_dropped == 0

        # Add 4th frame - should drop frame0
        await small_buffer.put_frame(b"frame3", time.time(), 3)

        assert small_buffer.input_queue.qsize() == 3  # Still at max
        assert small_buffer.metrics.frames_dropped == 1
        assert small_buffer.metrics.frames_received == 4

        # Verify frame0 was dropped (frame1 should be first)
        frame = await small_buffer.get_frame()
        assert frame.frame_bytes == b"frame1"
        assert frame.frame_number == 1

    @pytest.mark.asyncio
    async def test_multiple_drops(self, small_buffer):
        """Test multiple frames get dropped correctly."""
        # Add 10 frames to buffer with max 3
        for i in range(10):
            await small_buffer.put_frame(f"frame{i}".encode(), time.time(), i)

        # Should have dropped 7 oldest frames
        assert small_buffer.metrics.frames_dropped == 7
        assert small_buffer.metrics.frames_received == 10
        assert small_buffer.input_queue.qsize() == 3

        # Only last 3 frames should remain (7, 8, 9)
        frame1 = await small_buffer.get_frame()
        frame2 = await small_buffer.get_frame()
        frame3 = await small_buffer.get_frame()

        assert frame1.frame_number == 7
        assert frame2.frame_number == 8
        assert frame3.frame_number == 9

    @pytest.mark.asyncio
    async def test_put_and_get_result(self, frame_buffer):
        """Test adding and retrieving processed results."""
        result = ProcessedFrame(
            frame_bytes=b"processed",
            metadata={"detections": []},
            frame_number=1,
            processing_time_ms=50.0,
        )

        # Add result
        success = await frame_buffer.put_result(result)
        assert success is True
        assert frame_buffer.output_queue.qsize() == 1
        assert frame_buffer.metrics.frames_processed == 1

        # Get result
        retrieved = await frame_buffer.get_result()
        assert retrieved is not None
        assert retrieved.frame_bytes == b"processed"
        assert retrieved.frame_number == 1

    @pytest.mark.asyncio
    async def test_get_result_empty_queue(self, frame_buffer):
        """Test getting result from empty queue returns None."""
        result = await frame_buffer.get_result()
        assert result is None

    @pytest.mark.asyncio
    async def test_output_queue_drops_oldest(self, small_buffer):
        """Test output queue drops oldest when full."""
        # Fill output queue (max 3)
        for i in range(3):
            result = ProcessedFrame(
                frame_bytes=f"result{i}".encode(),
                metadata={},
                frame_number=i,
                processing_time_ms=50.0,
            )
            await small_buffer.put_result(result)

        assert small_buffer.output_queue.qsize() == 3

        # Add 4th result - should drop result0
        result = ProcessedFrame(
            frame_bytes=b"result3", metadata={}, frame_number=3, processing_time_ms=50.0
        )
        await small_buffer.put_result(result)

        assert small_buffer.output_queue.qsize() == 3

        # Verify result0 was dropped (result1 should be first)
        retrieved = await small_buffer.get_result()
        assert retrieved.frame_bytes == b"result1"

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, frame_buffer):
        """Test that metrics are correctly tracked."""
        # Add frames
        for i in range(5):
            await frame_buffer.put_frame(f"frame{i}".encode(), time.time(), i)

        assert frame_buffer.metrics.frames_received == 5
        assert frame_buffer.metrics.avg_queue_depth > 0

        # Process results
        for i in range(3):
            result = ProcessedFrame(
                frame_bytes=b"result",
                metadata={},
                frame_number=i,
                processing_time_ms=100.0,
            )
            await frame_buffer.put_result(result)

        assert frame_buffer.metrics.frames_processed == 3
        assert frame_buffer.metrics.avg_processing_time_ms == 100.0

    def test_should_skip_frame_faster_than_target(self, frame_buffer):
        """Test no skipping when processing faster than target."""
        # Processing at 20 fps, target is 15 fps
        should_skip = frame_buffer.should_skip_frame(current_fps=20.0, target_fps=15.0)
        assert should_skip is False

    def test_should_skip_frame_significantly_behind(self, frame_buffer):
        """Test skipping when significantly behind target."""
        # Processing at 10 fps, target is 20 fps (50% ratio)
        should_skip = frame_buffer.should_skip_frame(current_fps=10.0, target_fps=20.0)
        assert should_skip is True

    @pytest.mark.asyncio
    async def test_should_skip_frame_high_queue_utilization(self, small_buffer):
        """Test skipping when queue is backing up."""
        # Fill queue to 100% (3/3)
        for i in range(3):
            await small_buffer.put_frame(f"frame{i}".encode(), time.time(), i)

        # With slightly behind FPS and high queue, should skip
        should_skip = small_buffer.should_skip_frame(current_fps=14.0, target_fps=15.0)
        assert should_skip is True

    def test_get_metrics(self, frame_buffer):
        """Test getting metrics dictionary."""
        metrics = frame_buffer.get_metrics()

        assert "frames_received" in metrics
        assert "frames_dropped" in metrics
        assert "frames_processed" in metrics
        assert "avg_queue_depth" in metrics
        assert "avg_processing_time_ms" in metrics
        assert "input_queue_size" in metrics
        assert "output_queue_size" in metrics
        assert "input_queue_utilization" in metrics
        assert "output_queue_utilization" in metrics

        # Check types
        assert isinstance(metrics["frames_received"], int)
        assert isinstance(metrics["avg_queue_depth"], float)
        assert isinstance(metrics["input_queue_utilization"], float)

    @pytest.mark.asyncio
    async def test_clear(self, frame_buffer):
        """Test clearing queues and resetting metrics."""
        # Add some frames
        for i in range(5):
            await frame_buffer.put_frame(f"frame{i}".encode(), time.time(), i)

        # Add some results
        for i in range(3):
            result = ProcessedFrame(
                frame_bytes=b"result",
                metadata={},
                frame_number=i,
                processing_time_ms=50.0,
            )
            await frame_buffer.put_result(result)

        assert frame_buffer.input_queue.qsize() == 5
        assert frame_buffer.output_queue.qsize() == 3
        assert frame_buffer.metrics.frames_received == 5

        # Clear
        frame_buffer.clear()

        assert frame_buffer.input_queue.qsize() == 0
        assert frame_buffer.output_queue.qsize() == 0
        assert frame_buffer.metrics.frames_received == 0
        assert frame_buffer.metrics.frames_processed == 0

    @pytest.mark.asyncio
    async def test_queue_utilization_calculation(self, small_buffer):
        """Test queue utilization percentage calculation."""
        # Add 2 frames to buffer with max 3
        await small_buffer.put_frame(b"frame1", time.time(), 1)
        await small_buffer.put_frame(b"frame2", time.time(), 2)

        metrics = small_buffer.get_metrics()

        # 2/3 = 0.67
        assert metrics["input_queue_utilization"] == pytest.approx(0.67, rel=0.01)
        assert metrics["input_queue_size"] == 2

    @pytest.mark.asyncio
    async def test_concurrent_put_operations(self, frame_buffer):
        """Test concurrent frame additions."""

        async def add_frames(start: int, count: int):
            for i in range(start, start + count):
                await frame_buffer.put_frame(f"frame{i}".encode(), time.time(), i)

        # Add frames concurrently
        await asyncio.gather(
            add_frames(0, 5),
            add_frames(5, 5),
            add_frames(10, 5),
        )

        assert frame_buffer.metrics.frames_received == 15
        # Due to queue limit (10), some should be dropped
        assert frame_buffer.metrics.frames_dropped >= 5
