"""Camera detector interface using YOLOv8.

This module provides a thin wrapper around the trained YOLOv8 model
used to detect cameras in arbitrary images.
"""

import io
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from ultralytics import YOLO

from src.models.weights import get_yolo_camera_weights_path
from src.models.identification import BoundingBox, CameraDetection


class Detector:
    r"""Camera detector wrapper.

    This class loads a YOLOv8 model and exposes a simple interface for
    running detection on image files.

    :ivar model_path: Filesystem path to the YOLO model weights
    :ivar conf_threshold: Minimum confidence score to keep detections
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        conf_threshold: float = 0.25,
    ) -> None:
        r"""Initialize detector with model weights path.

        If ``model_path`` is not provided, the path is resolved using
        :func:`src.config.get_yolo_camera_weights_path` which consults
        ``.env`` and project defaults.

        :param model_path: Optional path to the YOLOv8 weights file
        :param conf_threshold: Minimum confidence score for detections
        """
        if model_path is None:
            self.model_path = get_yolo_camera_weights_path()
        else:
            self.model_path = Path(model_path)

        self.conf_threshold = conf_threshold
        self._model = YOLO(str(self.model_path))

    def detect_from_path(self, image_path: Path | str) -> List[CameraDetection]:
        r"""Run detection on a single image path.

        :param image_path: Path to the input image
        :return: List of camera detections
        :raises FileNotFoundError: If the image does not exist
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Run YOLO inference. Passing ``conf`` ensures basic filtering.
        results = self._model(str(image_path), conf=self.conf_threshold)

        detections: List[CameraDetection] = []
        if not results:
            return detections

        # For a single image call, we expect a single result object.
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or boxes.conf is None:
            return detections

        xyxy_list = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()
        cls_list = (
            boxes.cls.tolist()
            if getattr(boxes, "cls", None) is not None
            else [None] * len(conf_list)
        )

        names = getattr(self._model, "names", None)

        for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
            confidence = float(conf)
            if confidence < self.conf_threshold:
                continue

            if len(xyxy) != 4:
                continue

            x_min, y_min, x_max, y_max = [int(round(v)) for v in xyxy]

            label = "camera"
            class_id_int: Optional[int] = None
            if cls_id is not None:
                class_id_int = int(cls_id)
                if names is not None and class_id_int in names:
                    label = str(names[class_id_int])

            detections.append(
                CameraDetection(
                    image_path=image_path,
                    crop_path=None,
                    bbox=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    ),
                    confidence=confidence,
                    label=label,
                    class_id=class_id_int,
                )
            )

        return detections

    def detect_from_image(self, image: np.ndarray) -> List[CameraDetection]:
        r"""Run detection on a numpy array image.

        This method allows in-memory processing without requiring a file path.
        Useful for video frame processing where frames are already in memory.

        :param image: Image as numpy array (H, W, C) in RGB or BGR format
        :return: List of camera detections
        """
        # Run YOLO inference on numpy array
        results = self._model(image, conf=self.conf_threshold)

        detections: List[CameraDetection] = []
        if not results:
            return detections

        # For a single image call, we expect a single result object
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or boxes.conf is None:
            return detections

        xyxy_list = boxes.xyxy.tolist()
        conf_list = boxes.conf.tolist()
        cls_list = (
            boxes.cls.tolist()
            if getattr(boxes, "cls", None) is not None
            else [None] * len(conf_list)
        )

        names = getattr(self._model, "names", None)

        for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
            confidence = float(conf)
            if confidence < self.conf_threshold:
                continue

            if len(xyxy) != 4:
                continue

            x_min, y_min, x_max, y_max = [int(round(v)) for v in xyxy]

            label = "camera"
            class_id_int: Optional[int] = None
            if cls_id is not None:
                class_id_int = int(cls_id)
                if names is not None and class_id_int in names:
                    label = str(names[class_id_int])

            detections.append(
                CameraDetection(
                    image_path=None,  # No file path for in-memory images
                    crop_path=None,
                    bbox=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    ),
                    confidence=confidence,
                    label=label,
                    class_id=class_id_int,
                )
            )

        return detections

    def detect_from_bytes(self, image_bytes: bytes) -> List[CameraDetection]:
        r"""Run detection on raw image bytes (e.g., JPEG).

        This method decodes image bytes (JPEG, PNG, etc.) and runs detection.
        Useful for processing frames received over WebSocket or HTTP.

        :param image_bytes: Raw image bytes (JPEG, PNG, etc.)
        :return: List of camera detections
        :raises ValueError: If image bytes cannot be decoded
        """
        try:
            # Decode bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Convert to numpy array
            image_array = np.array(image)
            # Run detection on numpy array
            return self.detect_from_image(image_array)
        except Exception as e:
            raise ValueError(f"Failed to decode image bytes: {e}") from e
