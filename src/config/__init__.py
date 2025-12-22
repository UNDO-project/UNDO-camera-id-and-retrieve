"""Configuration management for cIDaR project.

This module provides Pydantic-based settings for all configurable values.
Import settings objects from here, not individual constants.

Example:
    from src.config import paths, scraper, axis, hikvision

    # Use settings objects
    print(paths.data_dir)
    delay = scraper.min_request_delay
    url = axis.products_url
"""

from dotenv import load_dotenv

from src.config.scraper import ScraperSettings
from src.config.paths import PathSettings
from src.config.vendors import AxisSettings, HikVisionSettings
from src.config.constants import (
    HIKVISION_SELECTORS,
    HIKVISION_IP_SUBCATEGORIES,
)

# Load environment variables from .env file once
load_dotenv()

# Instantiate settings (singleton pattern)
scraper = ScraperSettings()
paths = PathSettings()
axis = AxisSettings()
hikvision = HikVisionSettings()

# Ensure directories exist on first import
paths.ensure_directories_exist()

__all__ = [
    # Settings objects (primary API)
    "scraper",
    "paths",
    "axis",
    "hikvision",
    # Static constants
    "HIKVISION_SELECTORS",
    "HIKVISION_IP_SUBCATEGORIES",
]
