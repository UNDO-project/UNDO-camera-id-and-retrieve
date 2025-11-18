"""Camera detector interface using YOLOv8 (skeleton).

This module will provide a thin wrapper around the trained YOLOv8 model
used to detect cameras in arbitrary images. The concrete implementation
will be added in a later step.
"""

from pathlib import Path
from typing import List


class Detector:
    r"""Camera detector wrapper.

    This class will load a YOLOv8 model and expose a simple interface for
    running detection on image files.

    :ivar model_path: Filesystem path to the YOLO model weights
    """

    def __init__(self, model_path: Path | str) -> None:
        r"""Initialize detector with model weights path.

        :param model_path: Path to the YOLOv8 weights file
        """
        self.model_path = Path(model_path)

    def detect_from_path(self, image_path: Path | str) -> List[object]:
        r"""Run detection on a single image path.

        This is a placeholder implementation and will be replaced with the
        actual YOLOv8 inference logic.

        :param image_path: Path to the input image
        :return: List of detection results
        :raises NotImplementedError: Always, until implemented
        """
        raise NotImplementedError("Detector.detect_from_path is not implemented yet")
