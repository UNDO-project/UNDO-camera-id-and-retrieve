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


settings = APISettings()
