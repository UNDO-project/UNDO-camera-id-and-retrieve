"""Configuration for CCTV scrapers."""

import random
from pathlib import Path

# Request timing (seconds)
MIN_REQUEST_DELAY = 2.0
MAX_REQUEST_DELAY = 5.0

# Realistic headers to mimic human browsing
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Output paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Axis Communications URLs
AXIS_BASE_URL = "https://www.axis.com"
AXIS_PRODUCTS_URL = f"{AXIS_BASE_URL}/products/network-cameras"


def get_random_delay() -> float:
    r"""
    Get a random delay between MIN and MAX to mimic human behavior.

    :return: Random delay in seconds
    """
    return random.uniform(MIN_REQUEST_DELAY, MAX_REQUEST_DELAY)
