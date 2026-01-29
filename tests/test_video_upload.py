"""Tests for video upload endpoint."""

import io
import json
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.tasks.store import TaskStore, task_store


@pytest.fixture
def test_client() -> TestClient:
    """Create test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def clean_task_store() -> Generator[TaskStore, None, None]:
    """Provide a clean task store for each test."""
    # Clear any existing tasks
    for task_id in list(task_store._tasks.keys()):
        task_store.delete(task_id)
    yield task_store
    # Clean up after test
    for task_id in list(task_store._tasks.keys()):
        task_store.delete(task_id)


@pytest.fixture
def temp_video_file(tmp_path: Path) -> Path:
    """Create a temporary video file for testing."""
    # Create a minimal valid MP4 file using OpenCV
    try:
        import cv2

        video_path = tmp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 15.0, (320, 240))

        # Write 30 frames (2 seconds at 15fps)
        for _ in range(30):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Add some content
            frame[50:100, 50:100] = [255, 0, 0]  # Red square
            out.write(frame)

        out.release()
        return video_path
    except ImportError:
        pytest.skip("OpenCV not available")


@pytest.fixture
def mock_video_uploads_dir(tmp_path: Path):
    """Mock the video uploads directory to use tmp_path."""
    uploads_dir = tmp_path / "video_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    with patch("src.config.paths.video_uploads_dir", uploads_dir):
        yield uploads_dir


class TestVideoUploadEndpoint:
    """Test suite for video upload endpoint."""

    def test_upload_video_success(
        self,
        test_client: TestClient,
        temp_video_file: Path,
        clean_task_store: TaskStore,
        tmp_path: Path,
    ):
        """Test successful video upload."""
        # Mock the paths to use tmp_path
        uploads_dir = tmp_path / "video_uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        with patch("src.api.routes.video_upload.paths") as mock_paths:
            mock_paths.video_uploads_dir = uploads_dir

            with open(temp_video_file, "rb") as f:
                response = test_client.post(
                    "/api/v1/process-video",
                    files={"file": ("test_video.mp4", f, "video/mp4")},
                )

        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["filename"] == "test_video.mp4"
        assert data["file_size_mb"] > 0
        assert "created_at" in data

    def test_upload_video_with_options(
        self,
        test_client: TestClient,
        temp_video_file: Path,
        clean_task_store: TaskStore,
        tmp_path: Path,
    ):
        """Test video upload with custom processing options."""
        uploads_dir = tmp_path / "video_uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        options = json.dumps(
            {
                "target_fps": 10,
                "identification_mode": "full",
                "output_format": "mp4",
            }
        )

        with patch("src.api.routes.video_upload.paths") as mock_paths:
            mock_paths.video_uploads_dir = uploads_dir

            with open(temp_video_file, "rb") as f:
                response = test_client.post(
                    "/api/v1/process-video",
                    files={"file": ("test_video.mp4", f, "video/mp4")},
                    data={"options": options},
                )

        assert response.status_code == 202
        data = response.json()

        # Verify task was created with correct options
        task_id = data["task_id"]
        progress = clean_task_store.get(task_id)
        assert progress is not None
        assert progress.target_fps == 10
        assert progress.identification_mode == "full"

    def test_upload_video_invalid_extension(
        self,
        test_client: TestClient,
    ):
        """Test rejection of invalid file extension."""
        # Create a fake txt file
        fake_video = io.BytesIO(b"not a video")

        response = test_client.post(
            "/api/v1/process-video",
            files={"file": ("test.txt", fake_video, "text/plain")},
        )

        assert response.status_code == 400
        assert "extension" in response.json()["detail"].lower()

    def test_upload_video_invalid_mime_type(
        self,
        test_client: TestClient,
    ):
        """Test rejection of invalid MIME type."""
        fake_video = io.BytesIO(b"fake video data")

        response = test_client.post(
            "/api/v1/process-video",
            files={"file": ("test.mp4", fake_video, "application/octet-stream")},
        )

        assert response.status_code == 400
        assert "content type" in response.json()["detail"].lower()

    def test_upload_video_invalid_options_json(
        self,
        test_client: TestClient,
        temp_video_file: Path,
    ):
        """Test rejection of invalid JSON in options."""
        with open(temp_video_file, "rb") as f:
            response = test_client.post(
                "/api/v1/process-video",
                files={"file": ("test_video.mp4", f, "video/mp4")},
                data={"options": "invalid json"},
            )

        assert response.status_code == 400
        assert "invalid json" in response.json()["detail"].lower()

    def test_upload_video_invalid_options_values(
        self,
        test_client: TestClient,
        temp_video_file: Path,
    ):
        """Test rejection of invalid option values."""
        options = json.dumps({"target_fps": 100})  # Too high

        with open(temp_video_file, "rb") as f:
            response = test_client.post(
                "/api/v1/process-video",
                files={"file": ("test_video.mp4", f, "video/mp4")},
                data={"options": options},
            )

        assert response.status_code == 400


class TestVideoStatusEndpoint:
    """Test suite for video status endpoint."""

    def test_get_status_queued_task(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
    ):
        """Test getting status of a queued task."""
        # Create a task directly in the store
        progress = clean_task_store.create("test-task-123")
        progress.filename = "test.mp4"
        progress.file_size_mb = 10.5
        progress.frames_total = 300
        clean_task_store.update("test-task-123", progress)

        response = test_client.get("/api/v1/video-status/test-task-123")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-task-123"
        assert data["status"] == "queued"
        assert data["filename"] == "test.mp4"
        assert data["frames_total"] == 300
        assert data["progress_percent"] == 0.0

    def test_get_status_processing_task(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
    ):
        """Test getting status of a processing task."""
        progress = clean_task_store.create("test-task-456")
        progress.start()
        progress.frames_total = 100
        progress.frames_processed = 50
        clean_task_store.update("test-task-456", progress)

        response = test_client.get("/api/v1/video-status/test-task-456")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress_percent"] == 50.0
        assert data["frames_processed"] == 50
        assert data["elapsed_seconds"] >= 0

    def test_get_status_completed_task(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
        tmp_path: Path,
    ):
        """Test getting status of a completed task."""
        output_path = tmp_path / "output.mp4"
        output_path.touch()

        progress = clean_task_store.create("test-task-789")
        progress.frames_total = 100
        progress.start()
        progress.complete(output_path)
        clean_task_store.update("test-task-789", progress)

        response = test_client.get("/api/v1/video-status/test-task-789")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress_percent"] == 100.0
        assert data["output_url"] == "/api/v1/video-download/test-task-789"

    def test_get_status_failed_task(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
    ):
        """Test getting status of a failed task."""
        progress = clean_task_store.create("test-task-fail")
        progress.start()
        progress.fail("Processing error: codec not supported")
        clean_task_store.update("test-task-fail", progress)

        response = test_client.get("/api/v1/video-status/test-task-fail")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] == "Processing error: codec not supported"

    def test_get_status_not_found(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
    ):
        """Test getting status of non-existent task."""
        response = test_client.get("/api/v1/video-status/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestVideoDownloadEndpoint:
    """Test suite for video download endpoint."""

    def test_download_completed_video(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
        tmp_path: Path,
    ):
        """Test downloading a completed video."""
        # Create a fake output file
        output_path = tmp_path / "processed.mp4"
        output_path.write_bytes(b"fake video content")

        progress = clean_task_store.create("download-test")
        progress.filename = "original.mp4"
        progress.frames_total = 100
        progress.start()
        progress.complete(output_path)
        clean_task_store.update("download-test", progress)

        response = test_client.get("/api/v1/video-download/download-test")

        assert response.status_code == 200
        assert response.content == b"fake video content"
        assert "attachment" in response.headers.get("content-disposition", "")

    def test_download_not_completed(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
    ):
        """Test downloading a video that isn't completed yet."""
        progress = clean_task_store.create("incomplete-task")
        progress.start()
        clean_task_store.update("incomplete-task", progress)

        response = test_client.get("/api/v1/video-download/incomplete-task")

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"].lower()

    def test_download_not_found(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
    ):
        """Test downloading a non-existent task."""
        response = test_client.get("/api/v1/video-download/nonexistent")

        assert response.status_code == 404


class TestVideoDeleteEndpoint:
    """Test suite for video delete endpoint."""

    def test_delete_task(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
        tmp_path: Path,
    ):
        """Test deleting a task and its files."""
        # Create input and output files
        input_path = tmp_path / "input.mp4"
        output_path = tmp_path / "output.mp4"
        input_path.write_bytes(b"input")
        output_path.write_bytes(b"output")

        progress = clean_task_store.create("delete-test")
        progress.input_path = input_path
        progress.output_path = output_path
        progress.start()
        progress.complete(output_path)
        clean_task_store.update("delete-test", progress)

        response = test_client.delete("/api/v1/video-task/delete-test")

        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()
        assert not input_path.exists()
        assert not output_path.exists()
        assert clean_task_store.get("delete-test") is None

    def test_delete_not_found(
        self,
        test_client: TestClient,
        clean_task_store: TaskStore,
    ):
        """Test deleting a non-existent task."""
        response = test_client.delete("/api/v1/video-task/nonexistent")

        assert response.status_code == 404


class TestTaskStore:
    """Test suite for TaskStore class."""

    def test_create_and_get_task(self, clean_task_store: TaskStore):
        """Test creating and retrieving a task."""
        progress = clean_task_store.create("test-1")
        assert progress.task_id == "test-1"
        assert progress.status == "queued"

        retrieved = clean_task_store.get("test-1")
        assert retrieved is not None
        assert retrieved.task_id == "test-1"

    def test_update_task(self, clean_task_store: TaskStore):
        """Test updating a task."""
        progress = clean_task_store.create("test-2")
        progress.frames_total = 100
        progress.frames_processed = 50
        clean_task_store.update("test-2", progress)

        retrieved = clean_task_store.get("test-2")
        assert retrieved.frames_total == 100
        assert retrieved.frames_processed == 50

    def test_delete_task(self, clean_task_store: TaskStore):
        """Test deleting a task."""
        clean_task_store.create("test-3")
        assert clean_task_store.get("test-3") is not None

        result = clean_task_store.delete("test-3")
        assert result is True
        assert clean_task_store.get("test-3") is None

    def test_delete_nonexistent(self, clean_task_store: TaskStore):
        """Test deleting a non-existent task."""
        result = clean_task_store.delete("nonexistent")
        assert result is False

    def test_list_tasks(self, clean_task_store: TaskStore):
        """Test listing all tasks."""
        clean_task_store.create("task-a")
        clean_task_store.create("task-b")
        clean_task_store.create("task-c")

        tasks = clean_task_store.list_tasks()
        assert len(tasks) == 3
        assert "task-a" in tasks
        assert "task-b" in tasks
        assert "task-c" in tasks

    def test_get_stats(self, clean_task_store: TaskStore):
        """Test getting task statistics."""
        p1 = clean_task_store.create("stat-1")
        p2 = clean_task_store.create("stat-2")
        p3 = clean_task_store.create("stat-3")

        p1.start()
        p2.start()
        p3.start()
        p3.fail("error")

        clean_task_store.update("stat-1", p1)
        clean_task_store.update("stat-2", p2)
        clean_task_store.update("stat-3", p3)

        stats = clean_task_store.get_stats()
        assert stats["total"] == 3
        assert stats["queued"] == 0
        assert stats["processing"] == 2
        assert stats["failed"] == 1

    def test_task_progress_properties(self, clean_task_store: TaskStore):
        """Test TaskProgress computed properties."""
        progress = clean_task_store.create("props-test")
        progress.frames_total = 200
        progress.frames_processed = 100

        assert progress.progress_percent == 50.0

        progress.start()
        assert progress.elapsed_seconds >= 0


class TestVideoValidation:
    """Test suite for video validation logic."""

    def test_validate_file_extension(self, test_client: TestClient):
        """Test that invalid extensions are rejected."""
        fake_video = io.BytesIO(b"data")

        # Test .exe
        response = test_client.post(
            "/api/v1/process-video",
            files={"file": ("malware.exe", fake_video, "video/mp4")},
        )
        assert response.status_code == 400
        assert "extension" in response.json()["detail"].lower()

    def test_validate_empty_file(self, test_client: TestClient):
        """Test that empty files are handled."""
        empty_file = io.BytesIO(b"")

        response = test_client.post(
            "/api/v1/process-video",
            files={"file": ("empty.mp4", empty_file, "video/mp4")},
        )
        # Empty file should fail validation
        assert response.status_code == 400
