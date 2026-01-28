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

from src.identification.embeddings import _get_clip_components
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

        This is similar to embed_image() but works with in-memory PIL Images
        instead of file paths, avoiding disk I/O for streaming use cases.

        :param pil_image: PIL Image object in RGB format
        :return: Normalized embedding vector representation
        :raises RuntimeError: If embedding dependencies are missing
        """
        model, preprocess, device = _get_clip_components()

        # Imports only needed when embeddings are used
        import torch  # type: ignore[import]

        with torch.no_grad():
            tensor = preprocess(pil_image).unsqueeze(0)
            if device != "cpu":
                tensor = tensor.to(device)
            features = model.encode_image(tensor)
            # L2-normalize
            features = features / features.norm(dim=-1, keepdim=True)

        embedding = features.detach().cpu().numpy().astype("float32")[0]
        return embedding

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
            # Clamp bounding box to image bounds
            x_min = max(0, detection.bbox.x_min)
            y_min = max(0, detection.bbox.y_min)
            x_max = min(width, detection.bbox.x_max)
            y_max = min(height, detection.bbox.y_max)

            if x_max <= x_min or y_max <= y_min:
                logger.debug(f"Skipping degenerate bbox: {detection.bbox}")
                return RetrievalResult(detection=detection, matches=[])

            # Crop detection
            crop = pil_image.crop((x_min, y_min, x_max, y_max))

            # Compute embedding in thread pool
            embedding = await asyncio.to_thread(
                AsyncIdentificationService._embed_pil_image, crop
            )

            # Search catalog in thread pool
            def _search():
                results = self._sync_service.index.search(embedding, top_k=top_k)
                # Filter by similarity threshold
                filtered = [
                    (camera_id, similarity)
                    for camera_id, similarity in results
                    if similarity >= similarity_threshold
                ]
                return filtered

            search_results = await asyncio.to_thread(_search)

            # Enrich with catalog metadata
            matches = []
            for camera_id, similarity in search_results:
                if camera_id in self._sync_service.catalog:
                    record = self._sync_service.catalog[camera_id]
                    from src.models.identification import CameraMatch

                    # Get catalog image path if available
                    catalog_image_path = None
                    if record.image_files and len(record.image_files) > 0:
                        catalog_image_path = Path(record.image_files[0])

                    matches.append(
                        CameraMatch(
                            camera_id=camera_id,
                            score=similarity,
                            catalog_image_path=catalog_image_path,
                            source=record.source,
                            record=record,
                        )
                    )

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
