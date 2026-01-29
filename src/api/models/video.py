"""Video processing request and response models."""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class VideoProcessingOptions(BaseModel):
    r"""Options for video processing.

    :param target_fps: Target frames per second for processing
    :param identification_mode: Processing mode (detect_only or full)
    :param output_format: Output video format
    """

    target_fps: int = Field(
        default=15,
        ge=1,
        le=30,
        description="Target frames per second for processing",
    )
    identification_mode: Literal["detect_only", "full"] = Field(
        default="detect_only",
        description="Processing mode: detect_only for fast detection, full for identification",
    )
    output_format: Literal["mp4"] = Field(
        default="mp4",
        description="Output video format",
    )


class VideoUploadResponse(BaseModel):
    r"""Response for video upload request.

    :param task_id: Unique identifier for tracking the processing task
    :param status: Current status of the task
    :param filename: Original filename of the uploaded video
    :param file_size_mb: Size of the uploaded file in megabytes
    :param created_at: Timestamp when the task was created
    """

    task_id: str = Field(..., description="Unique identifier for the processing task")
    status: Literal["queued", "processing", "completed", "failed"] = Field(
        ..., description="Current status of the task"
    )
    filename: str = Field(..., description="Original filename of the uploaded video")
    file_size_mb: float = Field(
        ..., description="Size of the uploaded file in megabytes"
    )
    created_at: datetime = Field(..., description="Timestamp when the task was created")


class VideoTaskProgress(BaseModel):
    r"""Progress information for a video processing task.

    :param task_id: Unique identifier for the task
    :param status: Current status of the task
    :param progress_percent: Percentage of frames processed
    :param frames_processed: Number of frames processed so far
    :param frames_total: Total number of frames in the video
    :param current_fps: Current processing frames per second
    :param elapsed_seconds: Time elapsed since processing started
    :param output_url: URL to download the processed video (when completed)
    :param error: Error message if the task failed
    """

    task_id: str = Field(..., description="Unique identifier for the task")
    status: Literal["queued", "processing", "completed", "failed"] = Field(
        ..., description="Current status of the task"
    )
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Percentage of frames processed"
    )
    frames_processed: int = Field(
        default=0, ge=0, description="Number of frames processed so far"
    )
    frames_total: int = Field(
        default=0, ge=0, description="Total number of frames in the video"
    )
    current_fps: float = Field(
        default=0.0, ge=0.0, description="Current processing frames per second"
    )
    elapsed_seconds: float = Field(
        default=0.0, ge=0.0, description="Time elapsed since processing started"
    )
    output_url: Optional[str] = Field(
        None, description="URL to download the processed video (when completed)"
    )
    error: Optional[str] = Field(None, description="Error message if the task failed")
    filename: str = Field(default="", description="Original filename")
    created_at: Optional[datetime] = Field(
        None, description="Timestamp when the task was created"
    )
    options: Optional[VideoProcessingOptions] = Field(
        None, description="Processing options for this task"
    )


class VideoValidationResult(BaseModel):
    r"""Result of video file validation.

    :param valid: Whether the video file is valid
    :param filename: Name of the uploaded file
    :param file_size_mb: Size of the file in megabytes
    :param duration_seconds: Duration of the video in seconds
    :param width: Video width in pixels
    :param height: Video height in pixels
    :param fps: Video frames per second
    :param codec: Video codec name
    :param total_frames: Total number of frames
    :param errors: List of validation errors if invalid
    """

    valid: bool = Field(..., description="Whether the video file is valid")
    filename: str = Field(..., description="Name of the uploaded file")
    file_size_mb: float = Field(..., description="Size of the file in megabytes")
    duration_seconds: Optional[float] = Field(
        None, description="Duration of the video in seconds"
    )
    width: Optional[int] = Field(None, description="Video width in pixels")
    height: Optional[int] = Field(None, description="Video height in pixels")
    fps: Optional[float] = Field(None, description="Video frames per second")
    codec: Optional[str] = Field(None, description="Video codec name")
    total_frames: Optional[int] = Field(None, description="Total number of frames")
    errors: List[str] = Field(
        default_factory=list, description="List of validation errors if invalid"
    )
