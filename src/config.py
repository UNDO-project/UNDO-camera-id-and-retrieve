"""Configuration for CCTV scrapers."""

import random
import os
from pathlib import Path
from dotenv import load_dotenv

# Request timing (seconds)
MIN_REQUEST_DELAY = 2.0
MAX_REQUEST_DELAY = 5.0

# Realistic headers to mimic human browsing
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}

# Output paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
PDFS_DIR = DATA_DIR / "pdfs"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Model paths
MODELS_DIR = PROJECT_ROOT / "model_weights"
YOLO_CAMERA_WEIGHTS_DEFAULT = MODELS_DIR / "yolov8_camera.pt"
YOLO_CAMERA_WEIGHTS_ENV_VAR = "CAMERA_DETECTOR_WEIGHTS"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)
PDFS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Axis Communications URLs
AXIS_BASE_URL = "https://www.axis.com"
AXIS_PRODUCTS_URL = f"{AXIS_BASE_URL}/products/network-cameras"

# HikVision URLs
HIKVISION_BASE_URL = "https://www.hikvision.com"
HIKVISION_REGION = "europe"
HIKVISION_PRODUCTS_BASE = (
    f"{HIKVISION_BASE_URL}/{HIKVISION_REGION}/products/IP-Products"
)
HIKVISION_IP_PRODUCTS_URL = (
    f"{HIKVISION_BASE_URL}/{HIKVISION_REGION}/products/IP-Products/"
)

# HikVision category paths (relative to base URL)
HIKVISION_CATEGORIES = {
    "Network Cameras": f"/{HIKVISION_REGION}/products/IP-Products/Network-Cameras/",
    "PTZ Cameras": f"/{HIKVISION_REGION}/products/IP-Products/PTZ-Cameras/",
    "Explosion-Proof Series": f"/{HIKVISION_REGION}/products/IP-Products/Explosion-Proof---Anti-corrosion-Series/",
}

# CSS Selectors for Playwright
HIKVISION_SELECTORS = {
    "search_list": ".search-list",
    "subcategory_dropdown": "[data-title-type='subcategory']",
    "subcategory_radio": "input[type='radio'][value='{subcategory}']",
    "product_count": ".sum-number-of-products",
    "product_grid": ".layout4-wrapper",
    "product_link": ".btn-details-link",
    "view_more_btn": ".product-view-more-btn",
    "pagination": ".pagination-section",
}

# Subcategory filter values (exact text as shown in UI)
HIKVISION_SUBCATEGORIES = {
    "Network Cameras": "Network Cameras",
    "PTZ Cameras": "PTZ Cameras",
    "Explosion-Proof Series": "Explosion-Proof and Anti-Corrosion Series",
}

# Playwright settings
HIKVISION_PAGE_LOAD_TIMEOUT = 30000  # 30 seconds
HIKVISION_PRODUCTS_PER_PAGE = 12  # Default pagination


def get_random_delay() -> float:
    r"""
    Get a random delay between MIN and MAX to mimic human behavior.

    :return: Random delay in seconds
    """
    return random.uniform(MIN_REQUEST_DELAY, MAX_REQUEST_DELAY)


def get_yolo_camera_weights_path() -> Path:
    r"""
    Resolve the path to the YOLOv8 camera detector weights.

    Resolution order:

    1. Environment variable named by :data:`YOLO_CAMERA_WEIGHTS_ENV_VAR` if set
       (typically ``CAMERA_DETECTOR_WEIGHTS`` loaded via ``python-dotenv``).
    2. Default project-relative path under :data:`MODELS_DIR`.

    :return: Path to the YOLOv8 weights file
    """
    # Load environment variables from .env if present
    load_dotenv()
    env_path = os.getenv(YOLO_CAMERA_WEIGHTS_ENV_VAR)
    if env_path:
        return Path(env_path)

    return YOLO_CAMERA_WEIGHTS_DEFAULT
