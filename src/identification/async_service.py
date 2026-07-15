"""Async wrapper for camera identification service.

This module provides async-compatible wrappers around the synchronous
IdentificationService, enabling non-blocking frame processing for video streaming.
"""

import asyncio
import io
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np
from loguru import logger
from PIL import Image

from src.identification.embeddings import embed_image
from src.identification.service import IdentificationService
from src.models.identification import CameraDetection, RetrievalResult


class AsyncIdentificationService:
    r"""Async wrapper for camera identification service.

    Wraps the synchronous IdentificationService with async methods that
    run blocking operations in thread pools. Includes thread safety for
    YOLO model access which is not thread-safe.

    Thread Safety:
        - YOLO model access is protected by a lock
        - CLIP embeddings can run concurrently (thread-safe)
        - Catalog searches can run concurrently (read-only)

    Typical usage::

        async_service = AsyncIdentificationService()
        detections = await async_service.detect_frame(frame_bytes)
        results = await async_service.identify_frame(frame_bytes)

    :ivar _sync_service: Underlying synchronous service
    :ivar _detector_lock: Thread lock for YOLO model access
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        parquet_path: Optional[Path | str] = None,
        embeddings_path: Optional[Path | str] = None,
        conf_threshold: float = 0.25,
        min_similarity: float = 0.3,
        save_crops: bool = False,
    ) -> None:
        r"""Initialize async identification service.

        :param model_path: Optional path to YOLOv8 model weights
        :param parquet_path: Optional path to products parquet dataset
        :param embeddings_path: Optional path to catalog embeddings
        :param conf_threshold: Detection confidence threshold
        :param min_similarity: Minimum cosine similarity for matches
        :param save_crops: Whether to save cropped patches (default: False for streaming)
        """
        # Create underlying sync service
        self._sync_service = IdentificationService(
            model_path=model_path,
            parquet_path=parquet_path,
            embeddings_path=embeddings_path,
            conf_threshold=conf_threshold,
            min_similarity=min_similarity,
            save_crops=save_crops,
        )

        # Thread lock for YOLO model (not thread-safe)
        self._detector_lock = Lock()

        logger.info("AsyncIdentificationService initialized")

    @staticmethod
    def _decode_frame(frame_bytes: bytes) -> np.ndarray:
        r"""Decode frame bytes to numpy array.

        :param frame_bytes: Raw image bytes (JPEG, PNG, etc.)
        :return: Image as numpy array (H, W, C) in RGB format
        :raises ValueError: If frame cannot be decoded
        """
        try:
            image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Failed to decode frame: {e}") from e

    @staticmethod
    def _embed_pil_image(pil_image: Image.Image) -> np.ndarray:
        r"""Compute embedding for a PIL Image object.

        Delegates to :func:`src.identification.embeddings.embed_image` so
        streaming frames and catalogue images share one preprocessing path.

        :param pil_image: PIL Image object in RGB format
        :return: Normalized embedding vector representation
        :raises RuntimeError: If embedding dependencies are missing
        """
        return embed_image(pil_image)

    async def detect_frame(self, frame_bytes: bytes) -> List[CameraDetection]:
        r"""Detect cameras in a frame (detect_only mode).

        Runs YOLO detection in a thread pool with thread safety.
        This method is non-blocking and safe for concurrent use.

        :param frame_bytes: Raw image bytes (JPEG, PNG, etc.)
        :return: List of camera detections
        :raises ValueError: If frame cannot be decoded
        """
        # Decode frame (CPU-bound, but fast)
        image = self._decode_frame(frame_bytes)

        # Run detection in thread pool with lock
        def _detect():
            with self._detector_lock:
                return self._sync_service.detector.detect_from_image(image)

        detections = await asyncio.to_thread(_detect)
        logger.debug(f"Detected {len(detections)} cameras in frame")
        return detections

    async def identify_frame(
        self,
        frame_bytes: bytes,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
    ) -> List[RetrievalResult]:
        r"""Full identification pipeline for a frame (full mode).

        Performs detection, crops each detection, computes embeddings,
        and searches the catalog. Embedding and search operations run
        concurrently for all detections.

        :param frame_bytes: Raw image bytes (JPEG, PNG, etc.)
        :param top_k: Maximum number of matches per detection
        :param min_similarity: Minimum similarity threshold (uses service default if None)
        :return: List of retrieval results, one per detection
        :raises ValueError: If frame cannot be decoded
        """
        # Decode frame
        image = self._decode_frame(frame_bytes)
        pil_image = Image.fromarray(image)
        width, height = pil_image.size

        # Run detection (with lock)
        detections = await self.detect_frame(frame_bytes)

        if not detections:
            logger.debug("No cameras detected in frame")
            return []

        # Use provided similarity threshold or service default
        similarity_threshold = (
            min_similarity
            if min_similarity is not None
            else self._sync_service.min_similarity
        )

        # Process all detections concurrently
        async def _process_detection(detection: CameraDetection) -> RetrievalResult:
            # Same expand-then-clamp crop behaviour as the sync service
            expanded_bbox = IdentificationService._expand_bbox(
                detection.bbox, self._sync_service.crop_margin
            )
            clamped_bbox = IdentificationService._clamp_bbox(
                expanded_bbox, width, height
            )

            if not IdentificationService._is_valid_bbox(clamped_bbox):
                logger.debug(f"Skipping degenerate bbox: {detection.bbox}")
                return RetrievalResult(detection=detection, matches=[])

            # Crop detection
            crop = pil_image.crop(
                (
                    clamped_bbox.x_min,
                    clamped_bbox.y_min,
                    clamped_bbox.x_max,
                    clamped_bbox.y_max,
                )
            )

            # Compute embedding in thread pool
            embedding = await asyncio.to_thread(
                AsyncIdentificationService._embed_pil_image, crop
            )

            # Search catalog and enrich matches in thread pool
            def _search():
                results = self._sync_service.index.search(embedding, top_k=top_k)
                return self._sync_service._filter_and_enrich_matches(
                    results, similarity_threshold
                )

            matches = await asyncio.to_thread(_search)

            return RetrievalResult(detection=detection, matches=matches)

        # Process all detections concurrently
        results = await asyncio.gather(*[_process_detection(d) for d in detections])

        logger.debug(f"Identified {len(results)} detections in frame")
        return results

    @property
    def detector(self):
        """Access to underlying detector (for compatibility)."""
        return self._sync_service.detector

    @property
    def index(self):
        """Access to underlying catalog index (for compatibility)."""
        return self._sync_service.index

    @property
    def catalog(self):
        """Access to underlying catalog (for compatibility)."""
        return self._sync_service.catalog
