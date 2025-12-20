import os
from pathlib import Path

from dotenv import load_dotenv

from src.config import YOLO_CAMERA_WEIGHTS_ENV_VAR, YOLO_CAMERA_WEIGHTS_DEFAULT


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
