"""Video upload and processing endpoints."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from loguru import logger

from src.api.config import video_settings
from src.api.models.video import (
    VideoProcessingOptions,
    VideoTaskProgress,
    VideoUploadResponse,
    VideoValidationResult,
)
from src.api.tasks.store import task_store
from src.api.tasks.video_processor import run_video_processing_task
from src.config import paths

router = APIRouter()


def _validate_video_file(
    file: UploadFile,
    file_size: int,
) -> VideoValidationResult:
    r"""Validate an uploaded video file.

    :param file: The uploaded file
    :param file_size: Size of the file in bytes
    :return: Validation result with file metadata
    """
    errors = []
    filename = file.filename or "unknown"
    file_size_mb = file_size / (1024 * 1024)

    # Check file size
    if file_size_mb > video_settings.max_upload_size_mb:
        errors.append(
            f"File size ({file_size_mb:.1f}MB) exceeds maximum "
            f"allowed size ({video_settings.max_upload_size_mb}MB)"
        )

    # Check file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in video_settings.allowed_extensions:
        errors.append(
            f"File extension '{file_ext}' not allowed. "
            f"Allowed extensions: {', '.join(video_settings.allowed_extensions)}"
        )

    # Check MIME type
    content_type = file.content_type or ""
    if content_type and content_type not in video_settings.allowed_mime_types:
        errors.append(
            f"Content type '{content_type}' not allowed. "
            f"Allowed types: {', '.join(video_settings.allowed_mime_types)}"
        )

    return VideoValidationResult(
        valid=len(errors) == 0,
        filename=filename,
        file_size_mb=round(file_size_mb, 2),
        errors=errors,
    )


def _validate_video_metadata(
    file_path: Path,
    validation_result: VideoValidationResult,
) -> VideoValidationResult:
    r"""Validate video file metadata using OpenCV.

    :param file_path: Path to the uploaded video file
    :param validation_result: Initial validation result to update
    :return: Updated validation result with video metadata
    """
    try:
        import cv2

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            validation_result.errors.append("Failed to open video file")
            validation_result.valid = False
            return validation_result

        # Extract video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Decode fourcc to codec name
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        cap.release()

        # Calculate duration
        duration_seconds = total_frames / fps if fps > 0 else 0
        max_duration_seconds = video_settings.max_duration_minutes * 60

        # Validate duration
        if duration_seconds > max_duration_seconds:
            validation_result.errors.append(
                f"Video duration ({duration_seconds / 60:.1f} minutes) exceeds maximum "
                f"allowed duration ({video_settings.max_duration_minutes} minutes)"
            )

        # Validate codec
        codec_lower = codec.lower().strip()
        if codec_lower and codec_lower not in [
            c.lower() for c in video_settings.allowed_codecs
        ]:
            # Log warning but don't fail - codec detection can be unreliable
            logger.warning(
                f"Video codec '{codec}' may not be in allowed list. "
                f"Attempting to process anyway."
            )

        # Update validation result with metadata
        validation_result.width = width
        validation_result.height = height
        validation_result.fps = round(fps, 2)
        validation_result.total_frames = total_frames
        validation_result.duration_seconds = round(duration_seconds, 2)
        validation_result.codec = codec.strip()
        validation_result.valid = len(validation_result.errors) == 0

    except ImportError:
        validation_result.errors.append(
            "OpenCV not available for video validation. "
            "Install opencv-python to enable video processing."
        )
        validation_result.valid = False
    except Exception as e:
        validation_result.errors.append(f"Error validating video: {str(e)}")
        validation_result.valid = False

    return validation_result


@router.post("/process-video", response_model=VideoUploadResponse, status_code=202)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file to process"),
    options: Optional[str] = Form(
        default=None,
        description="JSON string with processing options",
    ),
) -> VideoUploadResponse:
    r"""Upload a video file for processing.

    This endpoint accepts a video file and queues it for processing with
    camera detection and optional identification. The video will be processed
    in the background and the results can be retrieved using the task ID.

    :param background_tasks: FastAPI BackgroundTasks object for scheduling tasks
    :param file: Video file to process (MP4, AVI, MOV, WebM)
    :param options: JSON string with processing options
    :return: Task information with unique ID for tracking progress
    :raises HTTPException: If file validation fails

    Example:
        ```bash
        curl -X POST "http://localhost:8000/api/v1/process-video" \
          -F "file=@video.mp4" \
          -F 'options={"target_fps": 15, "identification_mode": "detect_only"}'
        ```
    """
    # Parse processing options
    processing_options = VideoProcessingOptions()
    if options:
        try:
            options_dict = json.loads(options)
            processing_options = VideoProcessingOptions(**options_dict)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON in options: {str(e)}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid processing options: {str(e)}",
            )

    # Read file content to get size
    contents = await file.read()
    file_size = len(contents)

    # Basic validation
    validation = _validate_video_file(file, file_size)
    if not validation.valid:
        raise HTTPException(
            status_code=400,
            detail=f"Video validation failed: {'; '.join(validation.errors)}",
        )

    # Generate task ID and paths
    task_id = str(uuid.uuid4())
    filename = file.filename or f"video_{task_id}.mp4"
    safe_filename = f"{task_id}_{Path(filename).name}"

    # Ensure upload directory exists
    paths.video_uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_path = paths.video_uploads_dir / safe_filename

    # Save uploaded file
    try:
        with open(upload_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved video upload: {upload_path} ({validation.file_size_mb}MB)")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    # Validate video metadata
    validation = _validate_video_metadata(upload_path, validation)
    if not validation.valid:
        # Clean up the uploaded file
        upload_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Video validation failed: {'; '.join(validation.errors)}",
        )

    # Create task in store
    progress = task_store.create(task_id)
    progress.filename = filename
    progress.file_size_mb = validation.file_size_mb
    progress.input_path = upload_path
    progress.frames_total = validation.total_frames or 0
    progress.target_fps = processing_options.target_fps
    progress.identification_mode = processing_options.identification_mode
    progress.output_format = processing_options.output_format
    task_store.update(task_id, progress)

    logger.info(
        f"Created video processing task: {task_id} "
        f"(file={filename}, frames={validation.total_frames}, "
        f"duration={validation.duration_seconds}s, mode={processing_options.identification_mode})"
    )

    # Schedule background processing task
    # Using asyncio.create_task wrapped in a sync function for BackgroundTasks
    def _run_async_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_video_processing_task(task_id))
        finally:
            loop.close()

    background_tasks.add_task(_run_async_task)
    logger.info(f"Scheduled background processing for task: {task_id}")

    return VideoUploadResponse(
        task_id=task_id,
        status="queued",
        filename=filename,
        file_size_mb=validation.file_size_mb,
        created_at=progress.created_at,
    )


@router.get("/video-status/{task_id}", response_model=VideoTaskProgress)
async def get_video_status(task_id: str) -> VideoTaskProgress:
    r"""Get the status of a video processing task.

    :param task_id: Unique identifier of the task
    :return: Current task progress and status
    :raises HTTPException: If task not found

    Example:
        ```bash
        curl "http://localhost:8000/api/v1/video-status/abc123"
        ```
    """
    progress = task_store.get(task_id)
    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {task_id}",
        )

    # Build output URL if completed
    output_url = None
    if progress.status == "completed" and progress.output_path:
        output_url = f"/api/v1/video-download/{task_id}"

    return VideoTaskProgress(
        task_id=progress.task_id,
        status=progress.status,
        progress_percent=round(progress.progress_percent, 1),
        frames_processed=progress.frames_processed,
        frames_total=progress.frames_total,
        current_fps=round(progress.current_fps, 1),
        elapsed_seconds=round(progress.elapsed_seconds, 1),
        output_url=output_url,
        error=progress.error,
        filename=progress.filename,
        created_at=progress.created_at,
        options=VideoProcessingOptions(
            target_fps=progress.target_fps,
            identification_mode=progress.identification_mode,
            output_format=progress.output_format,
        ),
    )


@router.get("/video-download/{task_id}")
async def download_video(task_id: str) -> FileResponse:
    r"""Download a processed video file.

    :param task_id: Unique identifier of the completed task
    :return: Video file stream
    :raises HTTPException: If task not found or not completed

    Example:
        ```bash
        curl -O "http://localhost:8000/api/v1/video-download/abc123"
        ```
    """
    progress = task_store.get(task_id)
    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {task_id}",
        )

    if progress.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Current status: {progress.status}",
        )

    if progress.output_path is None or not progress.output_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Processed video file not found",
        )

    # Generate output filename
    original_name = Path(progress.filename).stem
    output_filename = f"{original_name}_processed.{progress.output_format}"

    return FileResponse(
        path=progress.output_path,
        filename=output_filename,
        media_type="video/mp4",
    )


@router.delete("/video-task/{task_id}")
async def delete_video_task(task_id: str) -> dict:
    r"""Delete a video processing task and its files.

    :param task_id: Unique identifier of the task
    :return: Confirmation message
    :raises HTTPException: If task not found

    Example:
        ```bash
        curl -X DELETE "http://localhost:8000/api/v1/video-task/abc123"
        ```
    """
    progress = task_store.get(task_id)
    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {task_id}",
        )

    # Clean up files
    if progress.input_path and progress.input_path.exists():
        try:
            progress.input_path.unlink()
            logger.debug(f"Deleted input file: {progress.input_path}")
        except Exception as e:
            logger.warning(f"Failed to delete input file: {e}")

    if progress.output_path and progress.output_path.exists():
        try:
            progress.output_path.unlink()
            logger.debug(f"Deleted output file: {progress.output_path}")
        except Exception as e:
            logger.warning(f"Failed to delete output file: {e}")

    # Remove from store
    task_store.delete(task_id)

    return {"message": f"Task {task_id} deleted successfully"}
