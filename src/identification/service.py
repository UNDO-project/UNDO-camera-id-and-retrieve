"""High-level camera identification service.

This module orchestrates detection, optional cropping, embedding, and
catalog search to produce camera identification results for input
images.
"""

from pathlib import Path
from typing import List, Optional

from loguru import logger
from PIL import Image

from src.config import paths
from src.identification.catalog import load_catalog
from src.identification.detector import Detector
from src.identification.embeddings import embed_image
from src.identification.index import CatalogIndex
from src.models.identification import CameraMatch, RetrievalResult


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
        model_path: Optional[Path | str] = None,
        parquet_path: Optional[Path | str] = None,
        embeddings_path: Optional[Path | str] = None,
        conf_threshold: float = 0.25,
        min_similarity: float = 0.3,
        crop_dir: Optional[Path | str] = None,
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

    def identify_from_image(
        self, image_path: Path | str, top_k: int = 5
    ) -> List[RetrievalResult]:
        r"""Identify cameras in the given image.

        The pipeline is:

        1. Run YOLOv8 detector to obtain camera detections.
        2. For each detection, crop the corresponding patch.
        3. Compute an embedding for the patch.
        4. Query the catalog index for nearest neighbors.
        5. Filter matches below :attr:`min_similarity`.
        6. Return :class:`RetrievalResult` objects, one per detection.

        :param image_path: Path to the input image
        :param top_k: Maximum number of matches to return per detection
        :return: List of retrieval results, one per detection
        :raises FileNotFoundError: If the input image does not exist
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Running detection on image: {image_path}")
        detections = self.detector.detect_from_path(image_path)

        if not detections:
            logger.info("No cameras detected in image")
            return []

        # Open image once and reuse for all crops.
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        results: List[RetrievalResult] = []

        for idx, detection in enumerate(detections):
            # Clamp bounding box to image bounds to avoid PIL errors.
            x_min = max(0, detection.bbox.x_min)
            y_min = max(0, detection.bbox.y_min)
            x_max = min(width, detection.bbox.x_max)
            y_max = min(height, detection.bbox.y_max)

            if x_max <= x_min or y_max <= y_min:
                logger.debug(
                    "Skipping degenerate bbox for detection %d: (%d, %d, %d, %d)",
                    idx,
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                )
                continue

            crop = image.crop((x_min, y_min, x_max, y_max))

            crop_path: Optional[Path] = None
            if self.save_crops:
                crop_filename = f"{image_path.stem}_det{idx}.png"
                crop_path = self.crop_dir / crop_filename
                crop.save(crop_path)

            # Embed the cropped patch. We reuse the path-based interface
            # for simplicity by saving the crop if needed.
            if crop_path is not None:
                query_vec = embed_image(crop_path)
            else:
                # Fallback: save to a temporary file under crop_dir
                tmp_path = self.crop_dir / f"{image_path.stem}_det{idx}_tmp.png"
                crop.save(tmp_path)
                query_vec = embed_image(tmp_path)
                tmp_path.unlink(missing_ok=True)

            # Run nearest-neighbor search in the catalog index.
            matches = self.index.search(query_vec, top_k=top_k)

            # Filter by similarity threshold and enrich with full records.
            filtered_matches: List[CameraMatch] = []
            for match in matches:
                if match.score < self.min_similarity:
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

            # Update detection with crop path (if saved) and build result.
            if crop_path is not None:
                detection.crop_path = crop_path

            results.append(
                RetrievalResult(
                    detection=detection,
                    matches=filtered_matches,
                )
            )

        return results
