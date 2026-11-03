"""Tests for zero-retention clean-up of video tasks."""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.config import VideoUploadSettings, video_settings
from src.api.tasks import cleanup
from src.api.tasks.store import TaskProgress, TaskStore, task_store


@pytest.fixture
def clean_task_store() -> Generator[TaskStore, None, None]:
    """Provide a clean task store for each test."""
    for task_id in list(task_store._tasks.keys()):
        task_store.delete(task_id)
    yield task_store
    for task_id in list(task_store._tasks.keys()):
        task_store.delete(task_id)


def _finished_task(
    store: TaskStore, task_id: str, tmp_path: Path, completed_ago: timedelta
) -> TaskProgress:
    """Create a completed task with real input/output files."""
    input_path = tmp_path / f"{task_id}.mp4"
    output_path = tmp_path / f"{task_id}_processed.mp4"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")

    progress = store.create(task_id)
    progress.input_path = input_path
    progress.start()
    progress.complete(output_path)
    progress.completed_at = datetime.now(timezone.utc) - completed_ago
    store.update(task_id, progress)
    return progress


class TestSettings:
    """Retention settings defaults."""

    def test_defaults(self):
        settings = VideoUploadSettings()
        assert settings.task_retention_minutes == 15
        assert settings.cleanup_interval_seconds == 60
        assert not hasattr(settings, "task_cleanup_hours")


class TestExpiresAt:
    """TaskProgress.expires_at."""

    def test_none_while_unfinished(self):
        progress = TaskProgress(task_id="t")
        assert progress.expires_at is None
        progress.start()
        assert progress.expires_at is None

    def test_set_after_completion(self, tmp_path: Path):
        progress = TaskProgress(task_id="t")
        progress.complete(tmp_path / "out.mp4")
        expected = progress.completed_at + timedelta(
            minutes=video_settings.task_retention_minutes
        )
        assert progress.expires_at == expected

    def test_set_after_failure(self):
        progress = TaskProgress(task_id="t")
        progress.fail("boom")
        assert progress.expires_at is not None


class TestPopExpired:
    """TaskStore.pop_expired."""

    def test_returns_only_expired_finished_tasks(
        self, clean_task_store: TaskStore, tmp_path: Path
    ):
        old = _finished_task(clean_task_store, "old", tmp_path, timedelta(hours=1))
        _finished_task(clean_task_store, "fresh", tmp_path, timedelta(seconds=1))
        processing = clean_task_store.create("processing")
        processing.start()
        clean_task_store.create("queued")

        expired = clean_task_store.pop_expired()

        assert [p.task_id for p in expired] == [old.task_id]
        assert clean_task_store.get("old") is None
        assert clean_task_store.get("fresh") is not None
        assert clean_task_store.get("processing") is not None
        assert clean_task_store.get("queued") is not None

    def test_respects_reference_time(self, clean_task_store: TaskStore, tmp_path: Path):
        _finished_task(clean_task_store, "fresh", tmp_path, timedelta(seconds=1))
        later = datetime.now(timezone.utc) + timedelta(
            minutes=video_settings.task_retention_minutes + 1
        )

        expired = clean_task_store.pop_expired(now=later)

        assert [p.task_id for p in expired] == ["fresh"]

    def test_failed_tasks_expire_too(self, clean_task_store: TaskStore):
        progress = clean_task_store.create("failed")
        progress.fail("boom")
        progress.completed_at = datetime.now(timezone.utc) - timedelta(hours=1)
        clean_task_store.update("failed", progress)

        assert [p.task_id for p in clean_task_store.pop_expired()] == ["failed"]


class TestDeleteTaskFiles:
    """cleanup.delete_task_files."""

    def test_removes_both_files(self, tmp_path: Path):
        progress = TaskProgress(task_id="t")
        progress.input_path = tmp_path / "in.mp4"
        progress.output_path = tmp_path / "out.mp4"
        progress.input_path.write_bytes(b"in")
        progress.output_path.write_bytes(b"out")

        cleanup.delete_task_files(progress)

        assert not progress.input_path.exists()
        assert not progress.output_path.exists()

    def test_tolerates_missing_and_none(self, tmp_path: Path):
        progress = TaskProgress(task_id="t")
        progress.input_path = None
        progress.output_path = tmp_path / "gone.mp4"

        cleanup.delete_task_files(progress)  # must not raise

    def test_warns_and_continues_on_os_error(self, tmp_path: Path):
        progress = TaskProgress(task_id="t")
        progress.input_path = tmp_path / "in.mp4"
        progress.output_path = tmp_path / "out.mp4"
        progress.input_path.write_bytes(b"in")
        progress.output_path.write_bytes(b"out")

        with patch.object(Path, "unlink", side_effect=OSError("locked")):
            cleanup.delete_task_files(progress)  # must not raise

        assert progress.input_path.exists()
        assert progress.output_path.exists()


class TestPurgeVideoDirs:
    """cleanup.purge_video_dirs."""

    def test_empties_both_directories(self, tmp_path: Path):
        uploads = tmp_path / "video_uploads"
        outputs = tmp_path / "video_outputs"
        uploads.mkdir()
        outputs.mkdir()
        (uploads / "a.mp4").write_bytes(b"a")
        (outputs / "b.mp4").write_bytes(b"b")
        (outputs / "c.mp4").write_bytes(b"c")

        with patch("src.api.tasks.cleanup.paths") as mock_paths:
            mock_paths.video_uploads_dir = uploads
            mock_paths.video_outputs_dir = outputs
            removed = cleanup.purge_video_dirs()

        assert removed == 3
        assert list(uploads.iterdir()) == []
        assert list(outputs.iterdir()) == []

    def test_tolerates_missing_directories(self, tmp_path: Path):
        with patch("src.api.tasks.cleanup.paths") as mock_paths:
            mock_paths.video_uploads_dir = tmp_path / "missing_uploads"
            mock_paths.video_outputs_dir = tmp_path / "missing_outputs"
            assert cleanup.purge_video_dirs() == 0


class TestSweep:
    """cleanup.sweep_once and run_sweeper."""

    def test_sweep_once_deletes_expired_files(
        self, clean_task_store: TaskStore, tmp_path: Path
    ):
        old = _finished_task(clean_task_store, "old", tmp_path, timedelta(hours=1))
        fresh = _finished_task(
            clean_task_store, "fresh", tmp_path, timedelta(seconds=1)
        )

        assert cleanup.sweep_once() == 1

        assert not old.input_path.exists()
        assert not old.output_path.exists()
        assert fresh.output_path.exists()
        assert clean_task_store.get("old") is None
        assert clean_task_store.get("fresh") is not None

    @pytest.mark.asyncio
    async def test_run_sweeper_calls_sweep_and_stops_on_cancel(self):
        with patch("src.api.tasks.cleanup.sweep_once", return_value=0) as sweep:
            task = asyncio.create_task(cleanup.run_sweeper(0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            await task  # CancelledError is swallowed inside the loop

        assert sweep.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_sweeper_survives_errors(self):
        with patch(
            "src.api.tasks.cleanup.sweep_once", side_effect=RuntimeError("boom")
        ) as sweep:
            task = asyncio.create_task(cleanup.run_sweeper(0.01))
            await asyncio.sleep(0.05)
            task.cancel()
            await task

        assert sweep.call_count >= 2


class TestLifespan:
    """Startup purge and sweeper wiring in the app lifespan."""

    def test_startup_purges_and_starts_sweeper(self, tmp_path: Path):
        from src.api.main import app

        uploads = tmp_path / "video_uploads"
        outputs = tmp_path / "video_outputs"
        uploads.mkdir()
        outputs.mkdir()
        (uploads / "orphan.mp4").write_bytes(b"x")
        (outputs / "orphan_processed.mp4").write_bytes(b"y")

        with (
            patch("src.api.tasks.cleanup.paths") as mock_paths,
            patch("src.api.main.run_sweeper") as mock_sweeper,
        ):
            mock_paths.video_uploads_dir = uploads
            mock_paths.video_outputs_dir = outputs

            async def _idle(interval: float) -> None:
                await asyncio.Event().wait()

            mock_sweeper.side_effect = _idle

            with TestClient(app):
                assert list(uploads.iterdir()) == []
                assert list(outputs.iterdir()) == []
                mock_sweeper.assert_called_once_with(
                    video_settings.cleanup_interval_seconds
                )


class TestApiNeverSavesCrops:
    """ServiceState builds the identification service without crop saving."""

    def test_service_built_with_save_crops_false(self):
        from src.api.dependencies import ServiceState

        with patch("src.api.dependencies.IdentificationService") as mock_cls:
            mock_cls.return_value = MagicMock()
            state = ServiceState()
            _ = state.service

        mock_cls.assert_called_once_with(save_crops=False)
