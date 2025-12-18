"""Download and URL management utilities for scrapers."""

from src.scrapers.managers.url_normalizer import (
    URLNormalizer,
    RelativeURLNormalizer,
    ProtocolRelativeURLNormalizer,
)
from src.scrapers.managers.download_manager import DownloadManager

__all__ = [
    "URLNormalizer",
    "RelativeURLNormalizer",
    "ProtocolRelativeURLNormalizer",
    "DownloadManager",
]
