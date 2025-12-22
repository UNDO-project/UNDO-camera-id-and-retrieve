"""API configuration settings."""

from typing import List

from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """API configuration settings loaded from environment variables."""

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

    class ConfigDict:
        """Pydantic config."""

        env_prefix = "CIDAR_API_"
        env_file = ".env"
        case_sensitive = False


settings = APISettings()
