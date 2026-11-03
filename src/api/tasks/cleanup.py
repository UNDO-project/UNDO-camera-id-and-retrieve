"""Zero-retention clean-up for video tasks.

Uploaded videos are deleted as soon as processing ends (see
:func:`src.api.tasks.video_processor.run_video_processing_task`). Finished
tasks and their outputs are kept for ``video_settings.task_retention_minutes``
and then removed by the sweeper started in the application lifespan. At startup
both video directories are emptied: the task store is in memory, so any file
left from a previous run belongs to no task and could never be reached again.
"""

import asyncio

from loguru import logger

from src.api.tasks.store import TaskProgress, task_store
from src.config import paths


def delete_task_files(progress: TaskProgress) -> None:
    r"""Delete the upload and output files of a task, if they exist.

    Missing files are ignored. OS errors are logged and do not propagate, so a
    failed unlink never blocks the caller.

    :param progress: Task whose ``input_path`` and ``output_path`` are removed
    """
    for label, path in (
        ("input", progress.input_path),
        ("output", progress.output_path),
    ):
        if path is None or not path.exists():
            continue
        try:
            path.unlink()
            logger.debug(f"Task {progress.task_id}: Deleted {label} file")
        except OSError as e:
            logger.warning(
                f"Task {progress.task_id}: Failed to delete {label} file - {e}"
            )


def purge_video_dirs() -> int:
    r"""Delete every file in the video upload and output directories.

    :return: Number of files removed
    """
    removed = 0
    for directory in (paths.video_uploads_dir, paths.video_outputs_dir):
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if not entry.is_file():
                continue
            try:
                entry.unlink()
                removed += 1
            except OSError as e:
                logger.warning(f"Failed to delete {entry.name}: {e}")

    if removed:
        logger.info(f"Purged {removed} orphaned video files")
    return removed


def sweep_once() -> int:
    r"""Expire finished tasks past their retention window and delete their files.

    :return: Number of tasks removed
    """
    expired = task_store.pop_expired()
    for progress in expired:
        delete_task_files(progress)
    return len(expired)


async def run_sweeper(interval_seconds: float) -> None:
    r"""Run :func:`sweep_once` forever at a fixed interval.

    Meant to be scheduled with ``asyncio.create_task`` in the application
    lifespan and cancelled at shutdown.

    :param interval_seconds: Seconds to wait between sweeps
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            sweep_once()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in video task sweeper: {e}")
