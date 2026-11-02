"""Frame annotation renderer for video streaming.

This module provides utilities for drawing detection bounding boxes and labels
on video frames before returning them to clients.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

import cv2
import numpy as np

from src.models.identification import CameraDetection, CameraMatch


@dataclass
class RenderConfig:
    """Configuration for frame rendering.

    Sizes (``box_thickness``, ``font_scale``, ``font_thickness``) are defined
    for a frame of ``reference_height`` pixels and scaled up proportionally
    for taller frames so labels stay legible at 1080p and above. Frames at or
    below the reference height are drawn with the values as given.
    """

    box_color: tuple[int, int, int] = (0, 255, 0)  # BGR green
    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    reference_height: int = 480
    show_confidence: bool = True
    show_manufacturer: bool = True
    style: Literal["minimal", "detailed"] = "detailed"
    label_background: bool = True
    label_bg_color: tuple[int, int, int] = (0, 0, 0)  # BGR black
    label_text_color: tuple[int, int, int] = (255, 255, 255)  # BGR white


class FrameRenderer:
    """Renders detection annotations on video frames.

    Draws bounding boxes, confidence scores, and manufacturer labels on frames
    using OpenCV. Supports configurable styles and encoding back to JPEG.
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        r"""Initialize frame renderer with configuration.

        :param config: Optional render configuration (uses defaults if None)
        """
        self.config = config or RenderConfig()

    def _scale_factor(self, frame: np.ndarray) -> float:
        r"""Compute the size multiplier for a frame.

        :param frame: Frame whose height determines the factor
        :return: ``frame_height / reference_height``, never below 1.0
        """
        return max(1.0, frame.shape[0] / self.config.reference_height)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[CameraDetection],
        matches: Optional[List[List[CameraMatch]]] = None,
    ) -> np.ndarray:
        r"""Draw annotations on frame.

        Draws bounding boxes and labels for all detections. In detailed mode,
        also shows manufacturer information from matches (if provided).

        :param frame: Input frame as numpy array (H, W, C) in RGB or BGR format
        :param detections: List of camera detections to annotate
        :param matches: Optional list of matches per detection (for full mode)
        :return: Annotated frame as numpy array
        """
        # Make a copy to avoid modifying the original
        annotated = frame.copy()

        # Convert RGB to BGR if needed (OpenCV uses BGR)
        # We'll assume input is RGB and convert
        if len(annotated.shape) == 3 and annotated.shape[2] == 3:
            # Check if it looks like RGB (common case for PIL/numpy)
            # We'll work in BGR for OpenCV
            annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        scale = self._scale_factor(annotated)
        box_thickness = max(1, round(self.config.box_thickness * scale))

        for idx, detection in enumerate(detections):
            # Get bounding box coordinates
            x1, y1 = detection.bbox.x_min, detection.bbox.y_min
            x2, y2 = detection.bbox.x_max, detection.bbox.y_max

            # Draw bounding box
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                self.config.box_color,
                box_thickness,
            )

            # Prepare labels
            labels = []

            if self.config.show_confidence:
                confidence_pct = detection.confidence * 100
                labels.append(f"{detection.label} {confidence_pct:.1f}%")

            # Add manufacturer info in detailed mode (if matches available)
            if (
                self.config.style == "detailed"
                and self.config.show_manufacturer
                and matches is not None
                and idx < len(matches)
                and matches[idx]
            ):
                # Get top match
                top_match = matches[idx][0]
                manufacturer = getattr(top_match, "source", "Unknown")
                similarity_pct = top_match.score * 100
                labels.append(f"{manufacturer} ({similarity_pct:.1f}%)")

            # Draw labels
            self._draw_labels(annotated, labels, x1, y1, x2, y2, scale)

        # Convert back to RGB for consistency
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        return annotated

    def _draw_labels(
        self,
        frame: np.ndarray,
        labels: List[str],
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        scale: float = 1.0,
    ) -> None:
        r"""Draw text labels above and below bounding box.

        :param frame: Frame to draw on (modified in place)
        :param labels: List of label strings to draw
        :param x1: Left coordinate of bounding box
        :param y1: Top coordinate of bounding box
        :param x2: Right coordinate of bounding box
        :param y2: Bottom coordinate of bounding box
        :param scale: Size multiplier from :meth:`_scale_factor`
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.config.font_scale * scale
        font_thickness = max(1, round(self.config.font_thickness * scale))
        pad = round(5 * scale)
        gap = round(10 * scale)

        for i, label in enumerate(labels):
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Position labels above box (first label) or below (subsequent labels)
            if i == 0:
                # Above box
                label_y = y1 - gap
                bg_y1 = label_y - text_height - pad
                bg_y2 = label_y + baseline + pad
            else:
                # Below box
                label_y = y2 + 2 * gap + (i - 1) * (text_height + gap)
                bg_y1 = label_y - text_height - pad
                bg_y2 = label_y + baseline + pad

            # Ensure label stays within frame bounds
            label_y = max(text_height + pad, label_y)
            label_y = min(frame.shape[0] - pad, label_y)

            label_x = x1

            # Draw background rectangle if enabled
            if self.config.label_background:
                bg_x1 = label_x
                bg_x2 = label_x + text_width + 2 * pad

                cv2.rectangle(
                    frame,
                    (bg_x1, bg_y1),
                    (bg_x2, bg_y2),
                    self.config.label_bg_color,
                    -1,  # Filled rectangle
                )

            # Draw text
            cv2.putText(
                frame,
                label,
                (label_x + pad, label_y),
                font,
                font_scale,
                self.config.label_text_color,
                font_thickness,
                cv2.LINE_AA,
            )

    @staticmethod
    def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
        r"""Encode numpy array frame to JPEG bytes.

        :param frame: Frame as numpy array (H, W, C) in RGB format
        :param quality: JPEG quality (1-100, default 85)
        :return: JPEG-encoded frame as bytes
        :raises ValueError: If encoding fails
        """
        # Ensure quality is in valid range
        quality = max(1, min(100, quality))

        # Convert RGB to BGR for OpenCV encoding
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Encode to JPEG
        success, encoded = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )

        if not success:
            raise ValueError("Failed to encode frame as JPEG")

        return encoded.tobytes()

    def annotate_and_encode(
        self,
        frame: np.ndarray,
        detections: List[CameraDetection],
        matches: Optional[List[List[CameraMatch]]] = None,
        quality: int = 85,
    ) -> bytes:
        r"""Convenience method to annotate and encode in one call.

        :param frame: Input frame as numpy array
        :param detections: List of camera detections to annotate
        :param matches: Optional list of matches per detection
        :param quality: JPEG quality (1-100)
        :return: JPEG-encoded annotated frame
        """
        annotated = self.annotate_frame(frame, detections, matches)
        return self.encode_jpeg(annotated, quality)
