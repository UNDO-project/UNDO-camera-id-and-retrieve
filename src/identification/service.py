"""High-level camera identification service.

This module orchestrates detection, optional cropping, embedding, and
catalog search to produce camera identification results for input
images.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from src.config import paths
from src.identification.catalog import load_catalog
from src.identification.detector import Detector
from src.identification.embeddings import embed_image
from src.identification.index import CatalogIndex
from src.models.identification import (
    BoundingBox,
    CameraDetection,
    CameraMatch,
    RetrievalResult,
)


@dataclass
class CropInfo:
    """Information about an extracted detection crop."""

    detection: CameraDetection
    crop: Image.Image
    crop_path: Path | None
    clamped_bbox: BoundingBox


class IdentificationService:
    r"""Camera identification service.

    Combines detector, embedding model, and catalog index into a single
    high-level interface.

    Typical usage::

        service = IdentificationService()
        results = service.identify_from_image("path/to/photo.jpg")

    :ivar detector: Underlying YOLOv8-based detector
    :ivar index: CatalogIndex used for nearest-neighbor search
    :ivar catalog: Mapping of camera IDs to CameraRecord objects
    :ivar min_similarity: Minimum cosine similarity to keep a match
    :ivar crop_dir: Directory where cropped query patches are saved
    :ivar save_crops: Whether to persist cropped patches to disk
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        parquet_path: Path | str | None = None,
        embeddings_path: Path | str | None = None,
        conf_threshold: float = 0.25,
        min_similarity: float = 0.3,
        crop_dir: Path | str | None = None,
        save_crops: bool = True,
    ) -> None:
        r"""Initialize identification service.

        All paths are optional. When omitted, the following defaults are
        used:

        - ``model_path``: resolved via
          :func:`src.config.get_yolo_camera_weights_path`.
        - ``parquet_path``: ``output/products.parquet``.
        - ``embeddings_path``: ``output/catalog_embeddings.npz``.
        - ``crop_dir``: ``output/query_patches``.

        :param model_path: Optional path to YOLOv8 model weights
        :param parquet_path: Optional path to products parquet dataset
        :param embeddings_path: Optional path to catalog embeddings artifact
        :param conf_threshold: Detection confidence threshold for YOLOv8
        :param min_similarity: Minimum cosine similarity to keep a match
        :param crop_dir: Directory where cropped patches will be stored
        :param save_crops: Whether to save cropped patches to disk
        """
        # Detector handles default model path resolution itself.
        self.detector = Detector(model_path=model_path, conf_threshold=conf_threshold)

        # Load catalog records for enriching matches with full metadata.
        self.catalog = load_catalog(parquet_path)

        # Catalog index for nearest-neighbor search over reference images.
        self.index = CatalogIndex(embeddings_path=embeddings_path)

        self.min_similarity = min_similarity
        self.save_crops = save_crops

        if crop_dir is None:
            crop_dir = paths.output_dir / "query_patches"
        self.crop_dir = Path(crop_dir)
        if self.save_crops:
            self.crop_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_image_path(image_path: Path) -> None:
        r"""
        Validate that image file exists.

        :param image_path: Path to validate
        :raises FileNotFoundError: If image does not exist
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    def _detect_cameras(self, image_path: Path) -> list[CameraDetection]:
        r"""
        Run YOLOv8 detection on image.

        :param image_path: Path to image file
        :return: List of detected cameras
        """
        logger.info(f"Running detection on image: {image_path}")
        return self.detector.detect_from_path(image_path)

    @staticmethod
    def _clamp_bbox(bbox: BoundingBox, width: int, height: int) -> BoundingBox:
        r"""
        Clamp bounding box to image boundaries.

        :param bbox: Original bounding box
        :param width: Image width
        :param height: Image height
        :return: Clamped bounding box
        """
        return BoundingBox(
            x_min=max(0, bbox.x_min),
            y_min=max(0, bbox.y_min),
            x_max=min(width, bbox.x_max),
            y_max=min(height, bbox.y_max),
        )

    @staticmethod
    def _is_valid_bbox(bbox: BoundingBox) -> bool:
        r"""
        Check if bounding box is valid (non-degenerate).

        :param bbox: Bounding box to check
        :return: True if bbox has positive area
        """
        return bbox.x_max > bbox.x_min and bbox.y_max > bbox.y_min

    def _extract_single_crop(
        self,
        image: Image.Image,
        detection: CameraDetection,
        idx: int,
        image_path: Path,
    ) -> CropInfo | None:
        r"""
        Extract single crop from image.

        :param image: PIL Image to crop from
        :param detection: Detection with bounding box
        :param idx: Detection index for naming
        :param image_path: Original image path for naming
        :return: CropInfo if valid, None if bbox is degenerate
        """
        width, height = image.size
        clamped_bbox = self._clamp_bbox(detection.bbox, width, height)

        if not self._is_valid_bbox(clamped_bbox):
            logger.debug(
                "Skipping degenerate bbox for detection %d: (%d, %d, %d, %d)",
                idx,
                clamped_bbox.x_min,
                clamped_bbox.y_min,
                clamped_bbox.x_max,
                clamped_bbox.y_max,
            )
            return None

        # Crop image
        crop = image.crop(
            (
                clamped_bbox.x_min,
                clamped_bbox.y_min,
                clamped_bbox.x_max,
                clamped_bbox.y_max,
            )
        )

        # Save crop if enabled
        crop_path: Path | None = None
        if self.save_crops:
            crop_filename = f"{image_path.stem}_det{idx}.png"
            crop_path = self.crop_dir / crop_filename
            crop.save(crop_path)

        return CropInfo(
            detection=detection,
            crop=crop,
            crop_path=crop_path,
            clamped_bbox=clamped_bbox,
        )

    def _extract_detection_crops(
        self,
        image_path: Path,
        detections: list[CameraDetection],
    ) -> list[CropInfo]:
        r"""
        Extract and optionally save crops for each detection.

        :param image_path: Path to original image
        :param detections: List of detections
        :return: List of valid crop info objects
        """
        image = Image.open(image_path).convert("RGB")
        crops: list[CropInfo] = []

        for idx, detection in enumerate(detections):
            crop_info = self._extract_single_crop(image, detection, idx, image_path)
            if crop_info is not None:
                crops.append(crop_info)

        return crops

    @staticmethod
    def _embed_crop(crop_info: CropInfo) -> np.ndarray:
        r"""
        Compute embedding for a cropped patch.

        :param crop_info: Crop information
        :return: Embedding vector
        """
        # Embed the in-memory PIL crop directly; no disk round-trip needed.
        return embed_image(crop_info.crop)

    def _filter_and_enrich_matches(
        self,
        matches: list[CameraMatch],
        similarity_threshold: float,
    ) -> list[CameraMatch]:
        r"""
        Filter matches by similarity and enrich with catalog data.

        :param matches: Raw matches from index search
        :param similarity_threshold: Minimum similarity threshold
        :return: Filtered and enriched matches
        """
        filtered_matches: list[CameraMatch] = []
        for match in matches:
            if match.score < similarity_threshold:
                continue

            record = self.catalog.get(match.camera_id)
            filtered_matches.append(
                CameraMatch(
                    camera_id=match.camera_id,
                    score=match.score,
                    catalog_image_path=match.catalog_image_path,
                    source=match.source,
                    record=record,
                )
            )
        return filtered_matches

    def _retrieve_matches_for_crop(
        self,
        crop_info: CropInfo,
        top_k: int,
        similarity_threshold: float,
    ) -> RetrievalResult:
        r"""
        Retrieve catalog matches for a single crop.

        :param crop_info: Crop information
        :param top_k: Maximum number of matches
        :param similarity_threshold: Minimum similarity threshold
        :return: Retrieval result for this detection
        """
        query_vec = self._embed_crop(crop_info)

        matches = self.index.search(query_vec, top_k=top_k)

        filtered_matches = self._filter_and_enrich_matches(
            matches, similarity_threshold
        )

        if crop_info.crop_path is not None:
            crop_info.detection.crop_path = crop_info.crop_path

        return RetrievalResult(
            detection=crop_info.detection,
            matches=filtered_matches,
        )

    def identify_from_image(
        self,
        image_path: Path | str,
        top_k: int = 5,
        min_similarity: float | None = None,
    ) -> list[RetrievalResult]:
        r"""Identify cameras in the given image.

        Orchestrates the identification pipeline:

        1. Run YOLOv8 detector to obtain camera detections.
        2. For each detection, crop the corresponding patch.
        3. Compute an embedding for the patch.
        4. Query the catalog index for nearest neighbors.
        5. Filter matches below :attr:`min_similarity`.
        6. Return :class:`RetrievalResult` objects, one per detection.

        :param image_path: Path to the input image
        :param top_k: Maximum number of matches to return per detection
        :param min_similarity: Optional minimum similarity threshold. If None, uses self.min_similarity
        :return: List of retrieval results, one per detection
        :raises FileNotFoundError: If the input image does not exist
        """
        image_path = Path(image_path)

        # Step 1: Validate input
        self._validate_image_path(image_path)

        # Step 2: Detect cameras
        detections = self._detect_cameras(image_path)
        if not detections:
            logger.info("No cameras detected in image")
            return []

        # Step 3: Extract crops
        crops = self._extract_detection_crops(image_path, detections)
        if not crops:
            logger.info("No valid detection crops extracted")
            return []

        # Use provided min_similarity or fall back to instance default
        similarity_threshold = (
            min_similarity if min_similarity is not None else self.min_similarity
        )

        # Step 4: Retrieve matches for each crop
        results: list[RetrievalResult] = []
        for crop_info in crops:
            result = self._retrieve_matches_for_crop(
                crop_info,
                top_k,
                similarity_threshold,
            )
            results.append(result)

        return results
