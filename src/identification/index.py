"""Catalog indexing and search for camera identification (skeleton).

This module will be responsible for building and querying an index of
catalog images, using precomputed embeddings for nearest-neighbor search.
"""

from pathlib import Path
from typing import Any, List

from src.models.identification import CameraMatch


class CatalogIndex:
    r"""In-memory index over catalog image embeddings.

    The concrete implementation will be added in a later step. For now,
    this class serves as a placeholder and API sketch.

    :ivar embeddings_path: Path to the serialized embeddings artifact
    """

    def __init__(self, embeddings_path: Path | str) -> None:
        r"""Initialize index from a serialized embeddings file.

        :param embeddings_path: Path to the stored embeddings
        """
        self.embeddings_path = Path(embeddings_path)

    def search(self, query_vector: Any, top_k: int = 5) -> List[CameraMatch]:
        r"""Search the index for nearest catalog items.

        Placeholder implementation to be replaced with real similarity
        search over embedding vectors.

        :param query_vector: Embedding vector for the query image
        :param top_k: Number of nearest neighbors to return
        :return: List of search result objects
        :raises NotImplementedError: Always, until implemented
        """
        raise NotImplementedError("CatalogIndex.search is not implemented yet")
