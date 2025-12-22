from typing import Dict
from pydantic_settings import BaseSettings, SettingsConfigDict


class ScraperSettings(BaseSettings):
    """Configuration for web scraping behavior."""

    model_config = SettingsConfigDict(
        env_prefix="CIDAR_SCRAPER_",
        case_sensitive=False,
    )

    min_request_delay: float = 2.0
    max_request_delay: float = 5.0
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    accept: str = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    accept_language: str = "en-US,en;q=0.9"
    accept_encoding: str = "gzip, deflate, br"
    connection: str = "keep-alive"

    @property
    def default_headers(self) -> Dict[str, str]:
        """Construct headers dict from individual settings."""
        return {
            "User-Agent": self.user_agent,
            "Accept": self.accept,
            "Accept-Language": self.accept_language,
            "Accept-Encoding": self.accept_encoding,
            "DNT": "1",
            "Connection": self.connection,
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        }
