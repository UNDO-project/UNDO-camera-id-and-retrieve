"""Tests for video processing background task."""

from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.api.tasks.store import TaskStore, task_store
from src.api.tasks.video_processor import (
    VideoProcessingConfig,
    VideoProcessorTask,
    run_video_processing_task,
)
from src.models.identification import BoundingBox, CameraDetection


@pytest.fixture
def clean_task_store() -> Generator[TaskStore, None, None]:
    """Provide a clean task store for each test."""
    for task_id in list(task_store._tasks.keys()):
        task_store.delete(task_id)
    yield task_store
    for task_id in list(task_store._tasks.keys()):
        task_store.delete(task_id)


@pytest.fixture
def temp_video_file(tmp_path: Path) -> Path:
    """Create a temporary video file for testing."""
    try:
        import cv2

        video_path = tmp_path / "test_input.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (320, 240))

        # Write 30 frames (2 seconds at 15fps)
        for i in range(30):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Add some content that changes per frame
            frame[50:100, 50 + i : 100 + i] = [255, 0, 0]
            out.write(frame)

        out.release()
        return video_path
    except ImportError:
        pytest.skip("OpenCV not available")


@pytest.fixture
def mock_detection() -> CameraDetection:
    """Create a mock camera detection."""
    return CameraDetection(
        bbox=BoundingBox(x_min=50, y_min=50, x_max=150, y_max=150),
        confidence=0.95,
        label="cctv_camera",
    )


@pytest.fixture
def mock_async_service(mock_detection: CameraDetection):
    """Create a mock AsyncIdentificationService."""
    service = MagicMock()
    service.detect_frame = AsyncMock(return_value=[mock_detection])
    service.identify_frame = AsyncMock(return_value=[])
    return service


class TestVideoProcessingConfig:
    """Test suite for VideoProcessingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = VideoProcessingConfig()
        assert config.target_fps == 15
        assert config.identification_mode == "detect_only"
        assert config.output_format == "mp4"
        assert config.jpeg_quality == 85

    def test_custom_config(self):
        """Test custom configuration values."""
        config = VideoProcessingConfig(
            target_fps=30,
            identification_mode="full",
            output_format="mp4",
            jpeg_quality=90,
        )
        assert config.target_fps == 30
        assert config.identification_mode == "full"
        assert config.jpeg_quality == 90


class TestVideoProcessorTask:
    """Test suite for VideoProcessorTask."""

    def test_init(self):
        """Test processor initialization."""
        config = VideoProcessingConfig()
        processor = VideoProcessorTask("test-task", config)
        assert processor.task_id == "test-task"
        assert processor.config == config
        assert processor._cancelled is False

    def test_cancel(self):
        """Test task cancellation."""
        config = VideoProcessingConfig()
        processor = VideoProcessorTask("test-task", config)
        processor.cancel()
        assert processor._cancelled is True

    @pytest.mark.asyncio
    async def test_process_video_success(
        self,
        clean_task_store: TaskStore,
        temp_video_file: Path,
        mock_async_service,
        tmp_path: Path,
    ):
        """Test successful video processing."""
        # Create task in store
        task_id = "process-test-1"
        progress = clean_task_store.create(task_id)
        progress.input_path = temp_video_file
        progress.frames_total = 30
        clean_task_store.update(task_id, progress)

        output_path = tmp_path / "output.mp4"

        # Create processor with mock service
        config = VideoProcessingConfig(target_fps=15)
        processor = VideoProcessorTask(task_id, config, service=mock_async_service)

        # Process video
        await processor.process(temp_video_file, output_path)

        # Verify output
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify task status
        final_progress = clean_task_store.get(task_id)
        assert final_progress.status == "completed"
        assert final_progress.output_path == output_path

    @pytest.mark.asyncio
    async def test_process_video_cancellation(
        self,
        clean_task_store: TaskStore,
        temp_video_file: Path,
        mock_async_service,
        tmp_path: Path,
    ):
        """Test video processing cancellation."""
        task_id = "cancel-test"
        progress = clean_task_store.create(task_id)
        progress.input_path = temp_video_file
        clean_task_store.update(task_id, progress)

        output_path = tmp_path / "output.mp4"
        config = VideoProcessingConfig()
        processor = VideoProcessorTask(task_id, config, service=mock_async_service)

        # Cancel immediately
        processor.cancel()

        # Process should handle cancellation
        await processor.process(temp_video_file, output_path)

        final_progress = clean_task_store.get(task_id)
        assert final_progress.status == "failed"
        assert "cancelled" in final_progress.error.lower()

    @pytest.mark.asyncio
    async def test_process_video_invalid_input(
        self,
        clean_task_store: TaskStore,
        tmp_path: Path,
        mock_async_service,
    ):
        """Test processing with invalid input file."""
        task_id = "invalid-test"
        progress = clean_task_store.create(task_id)
        progress.input_path = tmp_path / "nonexistent.mp4"
        clean_task_store.update(task_id, progress)

        output_path = tmp_path / "output.mp4"
        config = VideoProcessingConfig()
        processor = VideoProcessorTask(task_id, config, service=mock_async_service)

        with pytest.raises(RuntimeError):
            await processor.process(tmp_path / "nonexistent.mp4", output_path)

        final_progress = clean_task_store.get(task_id)
        assert final_progress.status == "failed"

    @pytest.mark.asyncio
    async def test_process_frame_detect_only(
        self,
        mock_async_service,
        mock_detection: CameraDetection,
    ):
        """Test single frame processing in detect_only mode."""
        config = VideoProcessingConfig(identification_mode="detect_only")
        processor = VideoProcessorTask("frame-test", config, service=mock_async_service)

        # Create a simple frame
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[50:100, 50:100] = [255, 0, 0]

        result = await processor._process_frame(frame)

        # Should return annotated frame
        assert result.shape == frame.shape
        mock_async_service.detect_frame.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_frame_full_mode(
        self,
        mock_async_service,
    ):
        """Test single frame processing in full mode."""
        config = VideoProcessingConfig(identification_mode="full")
        processor = VideoProcessorTask("frame-test", config, service=mock_async_service)

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = await processor._process_frame(frame)

        assert result.shape == frame.shape
        mock_async_service.identify_frame.assert_called_once()


class TestRunVideoProcessingTask:
    """Test suite for run_video_processing_task function."""

    @pytest.mark.asyncio
    async def test_run_task_not_found(self, clean_task_store: TaskStore):
        """Test running with non-existent task."""
        # Should not raise, just log error
        await run_video_processing_task("nonexistent-task")

    @pytest.mark.asyncio
    async def test_run_task_no_input(self, clean_task_store: TaskStore):
        """Test running task with no input path."""
        progress = clean_task_store.create("no-input-task")
        progress.input_path = None
        clean_task_store.update("no-input-task", progress)

        await run_video_processing_task("no-input-task")

        final_progress = clean_task_store.get("no-input-task")
        assert final_progress.status == "failed"
        assert "input" in final_progress.error.lower()

    @pytest.mark.asyncio
    async def test_run_task_success(
        self,
        clean_task_store: TaskStore,
        temp_video_file: Path,
        mock_async_service,
        tmp_path: Path,
    ):
        """Test running full task workflow."""
        task_id = "run-test"
        progress = clean_task_store.create(task_id)
        progress.input_path = temp_video_file
        progress.target_fps = 15
        progress.identification_mode = "detect_only"
        progress.output_format = "mp4"
        clean_task_store.update(task_id, progress)

        # Mock paths to use tmp_path
        with patch("src.api.tasks.video_processor.paths") as mock_paths:
            mock_paths.video_outputs_dir = tmp_path / "outputs"
            mock_paths.video_outputs_dir.mkdir(parents=True, exist_ok=True)

            # Mock the processor to use mock service
            with patch(
                "src.api.tasks.video_processor.VideoProcessorTask"
            ) as MockProcessor:
                mock_processor = MagicMock()
                mock_processor.process = AsyncMock()
                MockProcessor.return_value = mock_processor

                await run_video_processing_task(task_id)

                # Verify processor was created with correct config
                MockProcessor.assert_called_once()
                mock_processor.process.assert_called_once()


class TestVideoProcessorIntegration:
    """Integration tests for video processor (requires OpenCV)."""

    @pytest.mark.asyncio
    async def test_full_processing_pipeline(
        self,
        clean_task_store: TaskStore,
        temp_video_file: Path,
        tmp_path: Path,
    ):
        """Test complete processing pipeline with mocked detection."""
        task_id = "integration-test"
        progress = clean_task_store.create(task_id)
        progress.input_path = temp_video_file
        progress.frames_total = 30
        clean_task_store.update(task_id, progress)

        output_path = tmp_path / "integrated_output.mp4"

        # Create mock service that returns detections
        mock_service = MagicMock()
        mock_detection = CameraDetection(
            bbox=BoundingBox(x_min=60, y_min=60, x_max=100, y_max=100),
            confidence=0.9,
            label="cctv_camera",
        )
        mock_service.detect_frame = AsyncMock(return_value=[mock_detection])

        config = VideoProcessingConfig(target_fps=15)
        processor = VideoProcessorTask(task_id, config, service=mock_service)

        await processor.process(temp_video_file, output_path)

        # Verify output
        assert output_path.exists()

        # Verify output video is valid
        try:
            import cv2

            cap = cv2.VideoCapture(str(output_path))
            assert cap.isOpened()
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert frame_count > 0
            cap.release()
        except ImportError:
            pass

        # Verify final state
        final_progress = clean_task_store.get(task_id)
        assert final_progress.status == "completed"

    @pytest.mark.asyncio
    async def test_progress_tracking(
        self,
        clean_task_store: TaskStore,
        temp_video_file: Path,
        tmp_path: Path,
    ):
        """Test that progress is tracked during processing."""
        task_id = "progress-test"
        progress = clean_task_store.create(task_id)
        progress.input_path = temp_video_file
        clean_task_store.update(task_id, progress)

        output_path = tmp_path / "progress_output.mp4"

        mock_service = MagicMock()
        mock_service.detect_frame = AsyncMock(return_value=[])

        config = VideoProcessingConfig(target_fps=15)
        processor = VideoProcessorTask(task_id, config, service=mock_service)

        # Start processing
        await processor.process(temp_video_file, output_path)

        # Final progress should show completion
        final_progress = clean_task_store.get(task_id)
        assert final_progress.status == "completed"
        assert final_progress.frames_processed > 0
        assert final_progress.elapsed_seconds > 0
