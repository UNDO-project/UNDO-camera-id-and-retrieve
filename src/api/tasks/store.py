"""In-memory task store for video processing tasks."""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional

from loguru import logger


@dataclass
class TaskProgress:
    r"""Progress tracking for a video processing task.

    :param task_id: Unique identifier for the task
    :param status: Current status of the task
    :param filename: Original filename of the uploaded video
    :param file_size_mb: Size of the uploaded file in megabytes
    :param input_path: Path to the uploaded video file
    :param output_path: Path where processed video will be saved
    :param created_at: Timestamp when the task was created
    :param started_at: Timestamp when processing started
    :param completed_at: Timestamp when processing completed
    :param frames_processed: Number of frames processed so far
    :param frames_total: Total number of frames in the video
    :param current_fps: Current processing frames per second
    :param target_fps: Target frames per second for processing
    :param identification_mode: Processing mode
    :param output_format: Output video format
    :param error: Error message if the task failed
    """

    task_id: str
    status: Literal["queued", "processing", "completed", "failed"] = "queued"
    filename: str = ""
    file_size_mb: float = 0.0
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    frames_processed: int = 0
    frames_total: int = 0
    current_fps: float = 0.0
    target_fps: int = 15
    identification_mode: Literal["detect_only", "full"] = "detect_only"
    output_format: str = "mp4"
    error: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        r"""Calculate progress percentage.

        :return: Percentage of frames processed (0-100)
        """
        if self.frames_total == 0:
            return 0.0
        return (self.frames_processed / self.frames_total) * 100.0

    @property
    def elapsed_seconds(self) -> float:
        r"""Calculate elapsed time since processing started.

        :return: Elapsed time in seconds
        """
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()

    def update(self, frames_processed: int, frames_total: int) -> None:
        r"""Update progress with new frame counts.

        :param frames_processed: Number of frames processed
        :param frames_total: Total number of frames
        """
        self.frames_processed = frames_processed
        self.frames_total = frames_total

        # Calculate current FPS
        if self.started_at and self.elapsed_seconds > 0:
            self.current_fps = frames_processed / self.elapsed_seconds

    def start(self) -> None:
        r"""Mark the task as started."""
        self.status = "processing"
        self.started_at = datetime.utcnow()

    def complete(self, output_path: Path) -> None:
        r"""Mark the task as completed.

        :param output_path: Path to the processed video file
        """
        self.status = "completed"
        self.output_path = output_path
        self.completed_at = datetime.utcnow()
        self.frames_processed = self.frames_total

    def fail(self, error: str) -> None:
        r"""Mark the task as failed.

        :param error: Error message describing the failure
        """
        self.status = "failed"
        self.error = error
        self.completed_at = datetime.utcnow()


class TaskStore:
    r"""Thread-safe in-memory store for video processing tasks.

    This class provides a simple in-memory storage for task progress tracking.
    It is designed to be migrated to Redis or a database in the future.

    Example:
        ```python
        store = TaskStore()
        progress = store.create("task-123")
        progress.filename = "video.mp4"
        store.update("task-123", progress)
        ```
    """

    def __init__(self) -> None:
        r"""Initialize the task store."""
        self._tasks: Dict[str, TaskProgress] = {}
        self._lock = threading.Lock()

    def create(self, task_id: str) -> TaskProgress:
        r"""Create a new task with the given ID.

        :param task_id: Unique identifier for the task
        :return: The created TaskProgress object
        """
        with self._lock:
            progress = TaskProgress(task_id=task_id)
            self._tasks[task_id] = progress
            logger.debug(f"Created task: {task_id}")
            return progress

    def get(self, task_id: str) -> Optional[TaskProgress]:
        r"""Get a task by its ID.

        :param task_id: Unique identifier for the task
        :return: The TaskProgress object if found, None otherwise
        """
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id: str, progress: TaskProgress) -> None:
        r"""Update a task with new progress information.

        :param task_id: Unique identifier for the task
        :param progress: Updated TaskProgress object
        """
        with self._lock:
            self._tasks[task_id] = progress

    def delete(self, task_id: str) -> bool:
        r"""Delete a task by its ID.

        :param task_id: Unique identifier for the task
        :return: True if the task was deleted, False if not found
        """
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.debug(f"Deleted task: {task_id}")
                return True
            return False

    def list_tasks(self) -> Dict[str, TaskProgress]:
        r"""Get all tasks.

        :return: Dictionary mapping task IDs to TaskProgress objects
        """
        with self._lock:
            return dict(self._tasks)

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        r"""Remove tasks older than the specified age.

        :param max_age_hours: Maximum age of tasks to keep in hours
        :return: Number of tasks removed
        """
        now = datetime.utcnow()
        max_age_seconds = max_age_hours * 3600
        removed = 0

        with self._lock:
            tasks_to_remove = []
            for task_id, progress in self._tasks.items():
                age = (now - progress.created_at).total_seconds()
                if age > max_age_seconds:
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self._tasks[task_id]
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} old tasks (max age: {max_age_hours}h)")

        return removed

    def get_stats(self) -> Dict[str, int]:
        r"""Get statistics about tasks in the store.

        :return: Dictionary with task counts by status
        """
        with self._lock:
            stats = {
                "total": len(self._tasks),
                "queued": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
            }
            for progress in self._tasks.values():
                stats[progress.status] += 1
            return stats


# Global task store instance
task_store = TaskStore()
