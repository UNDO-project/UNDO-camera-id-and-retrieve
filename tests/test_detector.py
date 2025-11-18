"""Unit tests for the YOLOv8-based camera Detector.

These tests mock the underlying YOLO model from ``ultralytics`` so that
no real model weights or GPU are required. The goal is to verify that the
Detector correctly:

- validates input paths,
- calls the underlying model,
- converts raw box outputs into ``CameraDetection`` instances.
"""

from pathlib import Path
from typing import Any, List

import pytest

from src.identification import detector as detector_module
from src.models.identification import CameraDetection


class _ListWrapper(list):
    """Minimal wrapper providing a ``tolist`` method.

    Ultralytics YOLO exposes tensors/arrays with a ``tolist`` method. This
    wrapper allows us to mimic that behaviour using plain Python lists.
    """

    def tolist(self) -> list:
        """Return the list representation."""
        return list(self)


class FakeBoxes:
    """Fake boxes object mimicking the attributes used by Detector."""

    def __init__(self) -> None:
        # Single detection with floating-point coordinates
        self.xyxy = _ListWrapper([[10.5, 20.2, 110.9, 220.7]])
        self.conf = _ListWrapper([0.9])
        self.cls = _ListWrapper([0])


class FakeResult:
    """Fake YOLO result object exposing ``boxes`` attribute."""

    def __init__(self) -> None:
        self.boxes = FakeBoxes()


class FakeYOLO:
    """Fake YOLO model used to replace ``ultralytics.YOLO`` in tests."""

    # Class-level names mapping to mimic label lookup
    names = {0: "camera"}

    def __init__(self, model_path: str) -> None:  # noqa: D401 - simple init
        """Accept any model path without touching the filesystem."""
        self.model_path = model_path

    def __call__(self, image_path: str, conf: float | None = None) -> List[FakeResult]:
        """Return a list with a single ``FakeResult`` instance."""
        # In a real model, ``conf`` would influence filtering; here we ignore it.
        return [FakeResult()]


def test_detector_raises_on_missing_image(tmp_path: Path, monkeypatch: Any) -> None:
    r"""Detector.detect_from_path should raise FileNotFoundError for missing files."""

    # Patch YOLO so that Detector initialization does not require real weights
    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)

    from src.identification.detector import Detector

    detector = Detector(model_path=tmp_path / "dummy.pt")

    missing_image = tmp_path / "does_not_exist.jpg"
    with pytest.raises(FileNotFoundError):
        detector.detect_from_path(missing_image)


def test_detector_returns_camera_detection(tmp_path: Path, monkeypatch: Any) -> None:
    r"""Detector.detect_from_path should return normalized CameraDetection objects."""

    # Patch YOLO in the detector module before creating the Detector instance
    monkeypatch.setattr(detector_module, "YOLO", FakeYOLO)

    from src.identification.detector import Detector

    # Create a dummy image file; FakeYOLO will not actually read it
    image_path = tmp_path / "test_image.jpg"
    image_path.write_bytes(b"fake image data")

    detector = Detector(model_path=tmp_path / "dummy.pt", conf_threshold=0.1)

    detections = detector.detect_from_path(image_path)
    assert isinstance(detections, list)
    assert len(detections) == 1

    det = detections[0]
    assert isinstance(det, CameraDetection)
    assert det.image_path == image_path

    # Coordinates should be rounded integers based on FakeBoxes.xyxy
    assert det.bbox.x_min == 10
    assert det.bbox.y_min == 20
    assert det.bbox.x_max == 111
    assert det.bbox.y_max == 221

    # Confidence and label propagated from the fake model
    assert det.confidence == pytest.approx(0.9, rel=1e-6)
    assert det.label == "camera"
    assert det.class_id == 0
