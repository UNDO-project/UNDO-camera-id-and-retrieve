"""Catalog indexing and search for camera identification.

This module builds and queries an index of catalog images using
precomputed embeddings stored in a NumPy ``.npz`` artifact. The
index is designed to be simple, fast to load, and easy to rebuild
whenever the underlying catalog changes.
"""

from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

from src.config import paths
from src.identification.catalog import iter_reference_image_files, load_catalog
from src.identification.embeddings import embed_image
from src.models.identification import CameraMatch


def build_catalog_embeddings(
    parquet_path: Path | str | None = None,
    embeddings_path: Path | str | None = None,
    max_images_per_camera: int = 1,
) -> Path:
    r"""Build catalog image embeddings and serialize them to disk.

    This function:

    1. Loads the catalog from the products parquet file.
    2. Iterates over reference image files for each camera.
    3. Computes CLIP embeddings for each image.
    4. Stores the resulting arrays in a compressed ``.npz`` artifact
       containing:

       - ``embeddings``: ``(N, D)`` float32 array of image embeddings.
       - ``camera_ids``: ``(N,)`` array of camera identifiers.
       - ``image_paths``: ``(N,)`` array of absolute image paths.
       - ``sources``: ``(N,)`` array of camera source strings.

    :param parquet_path: Optional path to products parquet
    :param embeddings_path: Optional path to output ``.npz`` file
    :param max_images_per_camera: Maximum number of images per camera to embed
    :return: Path to the saved embeddings artifact
    :raises RuntimeError: If no embeddings could be produced
    """
    if embeddings_path is None:
        embeddings_path = paths.output_dir / "catalog_embeddings.npz"
    else:
        embeddings_path = Path(embeddings_path)

    catalog = load_catalog(parquet_path)

    vectors: list[np.ndarray] = []
    camera_ids: list[str] = []
    image_paths: list[str] = []
    sources: list[str] = []

    logger.info("Building catalog embeddings for identification index")

    for camera_id, image_path in iter_reference_image_files(
        catalog, max_images_per_camera=max_images_per_camera
    ):
        try:
            vec = embed_image(image_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to embed image for {camera_id}: {image_path} ({exc})")
            continue

        vectors.append(vec)
        camera_ids.append(camera_id)
        image_paths.append(str(image_path))
        sources.append(catalog[camera_id].source)

    if not vectors:
        raise RuntimeError("No embeddings were produced from catalog images")

    embeddings = np.stack(vectors).astype("float32")
    camera_ids_arr = np.array(camera_ids, dtype="U256")
    image_paths_arr = np.array(image_paths, dtype="U1024")
    sources_arr = np.array(sources, dtype="U128")

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        camera_ids=camera_ids_arr,
        image_paths=image_paths_arr,
        sources=sources_arr,
    )

    logger.info(
        "Saved catalog embeddings: %s (N=%d, D=%d)",
        embeddings_path,
        embeddings.shape[0],
        embeddings.shape[1],
    )

    return embeddings_path


class CatalogIndex:
    r"""In-memory index over catalog image embeddings.

    Loads precomputed embeddings from a ``.npz`` file and provides
    cosine-similarity search over all catalog images.

    :ivar embeddings_path: Path to the serialized embeddings artifact
    :ivar embeddings: ``(N, D)`` array of catalog image embeddings
    :ivar camera_ids: List of camera identifiers (length ``N``)
    :ivar image_paths: List of catalog image paths (length ``N``)
    :ivar sources: List of camera sources (length ``N``)
    """

    def __init__(self, embeddings_path: Path | str | None = None) -> None:
        r"""Initialize index from a serialized embeddings file.

        If ``embeddings_path`` is not provided, the default
        ``output/catalog_embeddings.npz`` under :data:`OUTPUT_DIR` is
        used.

        :param embeddings_path: Path to the stored embeddings
        :raises FileNotFoundError: If the embeddings file does not exist
        """
        if embeddings_path is None:
            embeddings_path = paths.output_dir / "catalog_embeddings.npz"
        self.embeddings_path = Path(embeddings_path)

        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {self.embeddings_path}"
            )

        logger.info(f"Loading catalog index from {self.embeddings_path}")
        data = np.load(self.embeddings_path)

        self.embeddings: np.ndarray = data["embeddings"].astype("float32")
        self.camera_ids: list[str] = data["camera_ids"].astype(str).tolist()
        self.image_paths: list[Path] = [
            Path(p) for p in data["image_paths"].astype(str).tolist()
        ]
        self.sources: list[str] = data["sources"].astype(str).tolist()

        if self.embeddings.ndim != 2:
            raise ValueError("Embeddings array must be 2D (N, D)")

        if not (
            len(self.camera_ids)
            == len(self.image_paths)
            == len(self.sources)
            == self.embeddings.shape[0]
        ):
            raise ValueError("Embeddings and metadata arrays have inconsistent lengths")

        logger.info(
            "Catalog index loaded: N=%d, D=%d",
            self.embeddings.shape[0],
            self.embeddings.shape[1],
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[CameraMatch]:
        r"""Search the index for nearest catalog items.

        Uses cosine similarity between the query vector and all catalog
        embeddings. The query is L2-normalized before comparison.

        :param query_vector: Embedding vector for the query image
        :param top_k: Number of nearest neighbors to return
        :return: List of :class:`CameraMatch` objects ordered by score
        :raises ValueError: If the query dimensionality does not match
        """
        if self.embeddings.size == 0:
            return []

        v = np.asarray(query_vector, dtype="float32").ravel()
        if v.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {v.shape[0]} does not match index dimension "
                f"{self.embeddings.shape[1]}"
            )

        # L2-normalize query
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            raise ValueError("Query embedding has zero norm")
        v /= norm

        # Cosine similarity reduces to dot product for normalized vectors
        scores = self.embeddings @ v  # shape: (N,)

        n = scores.shape[0]
        if n == 0:
            return []

        k = min(max(top_k, 0), n)
        if k == 0:
            return []

        # Partial sort for top-k, then sort those indices by score
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        matches: list[CameraMatch] = []
        for i in idx:
            matches.append(
                CameraMatch(
                    camera_id=self.camera_ids[int(i)],
                    score=float(scores[int(i)]),
                    catalog_image_path=self.image_paths[int(i)],
                    source=self.sources[int(i)],
                    record=None,
                )
            )

        return matches
