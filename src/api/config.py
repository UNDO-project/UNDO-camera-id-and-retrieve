"""API configuration settings."""

from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """API configuration settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="CIDAR_API_",
        case_sensitive=False,
    )

    # API Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1

    # CORS Settings
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # File Upload Settings
    max_upload_size_mb: int = 10
    allowed_extensions: List[str] = [".jpg", ".jpeg", ".png"]

    # Rate Limiting
    rate_limit_per_minute: int = 60

    # Logging
    log_level: str = "INFO"

    # Static file serving paths
    static_file_paths: List[str] = ["/api/v1/images/", "/api/v1/datasheets/"]


class VideoUploadSettings(BaseSettings):
    """Video upload configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="CIDAR_VIDEO_",
        case_sensitive=False,
    )

    # File size limits
    max_upload_size_mb: int = 500
    max_duration_minutes: int = 30

    # Allowed video formats and codecs
    allowed_extensions: List[str] = [".mp4", ".avi", ".mov", ".webm"]
    allowed_mime_types: List[str] = [
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/webm",
    ]
    allowed_codecs: List[str] = ["h264", "h265", "hevc", "vp8", "vp9", "avc1", "hvc1"]

    # Processing defaults
    default_target_fps: int = 15
    max_target_fps: int = 30
    min_target_fps: int = 1
    default_output_format: str = "mp4"

    # Task management
    task_cleanup_hours: int = 24


settings = APISettings()
video_settings = VideoUploadSettings()
