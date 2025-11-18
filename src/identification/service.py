"""High-level camera identification service (skeleton).

This module will orchestrate detection, embedding, and catalog search to
produce camera identification results for input images.
"""

from pathlib import Path
from typing import List

from src.models.identification import RetrievalResult


class IdentificationService:
    r"""Camera identification service.

    Combines detector, embedding model, and catalog index into a single
    high-level interface.

    The actual implementation will be added in later steps.
    """

    def __init__(
        self,
        model_path: Path | str,
        parquet_path: Path | str,
        embeddings_path: Path | str,
    ) -> None:
        r"""Initialize identification service.

        :param model_path: Path to YOLOv8 model weights
        :param parquet_path: Path to products parquet dataset
        :param embeddings_path: Path to catalog embeddings artifact
        """
        self.model_path = Path(model_path)
        self.parquet_path = Path(parquet_path)
        self.embeddings_path = Path(embeddings_path)

    def identify_from_image(self, image_path: Path | str) -> List[RetrievalResult]:
        r"""Identify cameras in the given image.

        Placeholder implementation that will later run detection and
        retrieval over the catalog.

        :param image_path: Path to the input image
        :return: List of identification results
        :raises NotImplementedError: Always, until implemented
        """
        raise NotImplementedError(
            "IdentificationService.identify_from_image is not implemented yet"
        )
