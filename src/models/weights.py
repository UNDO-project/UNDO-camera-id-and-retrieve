from pathlib import Path

from src.config import paths


def get_yolo_camera_weights_path() -> Path:
    r"""
    Resolve the path to the YOLOv8 camera detector weights.

    The path is configured via the PathSettings object, which can be
    overridden using the ``CIDAR_PATH_YOLO_CAMERA_WEIGHTS`` environment
    variable in ``.env``.

    :return: Path to the YOLOv8 weights file
    """
    return paths.yolo_camera_weights
