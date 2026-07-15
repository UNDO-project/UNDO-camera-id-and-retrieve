"""Catalog indexing and search for camera identification.

This module builds and queries an index of catalog images using
precomputed embeddings stored in a NumPy ``.npz`` artifact. The
index is designed to be simple, fast to load, and easy to rebuild
whenever the underlying catalog changes.

The artifact may optionally hold several embedding rows per product
(catalogue augmentation) plus a catalogue mean vector; both are
optional keys, so artifacts built before these features still load.
"""

import zlib
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from src.config import matching, paths
from src.identification.catalog import iter_reference_image_files, load_catalog
from src.identification.embeddings import embed_image
from src.models.identification import CameraMatch


def _augmentation_variants(
    image: Image.Image, camera_id: str, k: int
) -> list[tuple[str, Image.Image]]:
    r"""Generate K mildly degraded variants of a reference image.

    Variants reuse the eval degradations at mild severities so
    measurement (WP2) and augmentation share one code path. Randomness
    is derived from the camera ID, so rebuilding the index reproduces
    the same variants regardless of catalog ordering.

    :param image: Reference image to derive variants from
    :param camera_id: Stable identifier used to seed the variants
    :param k: Number of variants to generate (capped by the recipe list)
    :return: List of ``(variant_tag, image)`` pairs
    """
    # Imported lazily so non-augmented builds don't require cv2
    from src.identification.eval.degradations import (
        AUGMENTATION_RECIPES,
        apply_degradation,
    )

    variants: list[tuple[str, Image.Image]] = []
    id_seed = zlib.crc32(camera_id.encode("utf-8"))

    for variant_index, (name, severity) in enumerate(AUGMENTATION_RECIPES[:k]):
        rng = np.random.default_rng([id_seed, variant_index])
        degraded = apply_degradation(name, image, severity, rng)
        variants.append((f"{name}{severity}", degraded))

    return variants


def build_catalog_embeddings(
    parquet_path: Path | str | None = None,
    embeddings_path: Path | str | None = None,
    max_images_per_camera: int = 1,
    augment: bool | None = None,
    augment_k: int | None = None,
) -> Path:
    r"""Build catalog image embeddings and serialize them to disk.

    This function:

    1. Loads the catalog from the products parquet file.
    2. Iterates over reference image files for each camera.
    3. Computes CLIP embeddings for each image — plus, when
       augmentation is enabled, for K mildly degraded variants of it.
    4. Stores the resulting arrays in a compressed ``.npz`` artifact
       containing:

       - ``embeddings``: ``(N, D)`` float32 array of image embeddings.
       - ``camera_ids``: ``(N,)`` array of camera identifiers.
       - ``image_paths``: ``(N,)`` array of absolute image paths.
       - ``sources``: ``(N,)`` array of camera source strings.
       - ``mean_vector``: ``(D,)`` catalogue mean embedding (for
         optional query-time mean-centring).
       - ``variant_tags``: ``(N,)`` per-row variant labels (``"orig"``
         or e.g. ``"lighting1"``) — only present when augmentation is
         enabled.

    :param parquet_path: Optional path to products parquet
    :param embeddings_path: Optional path to output ``.npz`` file
    :param max_images_per_camera: Maximum number of images per camera to embed
    :param augment: Whether to embed K degraded variants per reference
        image. Defaults to ``matching.augment_enabled``
        (``CIDAR_MATCH_AUGMENT_ENABLED``)
    :param augment_k: Number of variants per reference image (1-8).
        Defaults to ``matching.augment_k`` (``CIDAR_MATCH_AUGMENT_K``)
    :return: Path to the saved embeddings artifact
    :raises RuntimeError: If no embeddings could be produced
    """
    if embeddings_path is None:
        embeddings_path = paths.output_dir / "catalog_embeddings.npz"
    else:
        embeddings_path = Path(embeddings_path)

    if augment is None:
        augment = matching.augment_enabled
    if augment_k is None:
        augment_k = matching.augment_k

    catalog = load_catalog(parquet_path)

    vectors: list[np.ndarray] = []
    camera_ids: list[str] = []
    image_paths: list[str] = []
    sources: list[str] = []
    variant_tags: list[str] = []

    logger.info(
        "Building catalog embeddings for identification index (augment={}, k={})",
        augment,
        augment_k if augment else "-",
    )

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
        variant_tags.append("orig")

        if not augment:
            continue

        try:
            reference_image = Image.open(image_path).convert("RGB")
        except OSError as exc:  # pragma: no cover - defensive logging
            logger.error(f"Failed to open image for augmentation: {image_path} ({exc})")
            continue

        for tag, variant_image in _augmentation_variants(
            reference_image, camera_id, augment_k
        ):
            try:
                variant_vec = embed_image(variant_image)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error(f"Failed to embed variant {tag} for {camera_id}: {exc}")
                continue

            vectors.append(variant_vec)
            camera_ids.append(camera_id)
            image_paths.append(str(image_path))
            sources.append(catalog[camera_id].source)
            variant_tags.append(tag)

    if not vectors:
        raise RuntimeError("No embeddings were produced from catalog images")

    embeddings = np.stack(vectors).astype("float32")
    camera_ids_arr = np.array(camera_ids, dtype="U256")
    image_paths_arr = np.array(image_paths, dtype="U1024")
    sources_arr = np.array(sources, dtype="U128")
    mean_vector = embeddings.mean(axis=0).astype("float32")

    arrays: dict[str, np.ndarray] = {
        "embeddings": embeddings,
        "camera_ids": camera_ids_arr,
        "image_paths": image_paths_arr,
        "sources": sources_arr,
        "mean_vector": mean_vector,
    }
    if augment:
        arrays["variant_tags"] = np.array(variant_tags, dtype="U64")

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embeddings_path, **arrays)

    logger.info(
        "Saved catalog embeddings: {} (N={}, D={}, products={})",
        embeddings_path,
        embeddings.shape[0],
        embeddings.shape[1],
        len(set(camera_ids)),
    )

    return embeddings_path


class CatalogIndex:
    r"""In-memory index over catalog image embeddings.

    Loads precomputed embeddings from a ``.npz`` file and provides
    cosine-similarity search over all catalog images. When the artifact
    holds several rows per product (catalogue augmentation), search
    scores collapse to the best variant per product, so a product never
    appears twice in the results.

    :ivar embeddings_path: Path to the serialized embeddings artifact
    :ivar embeddings: ``(N, D)`` array of catalog image embeddings
    :ivar camera_ids: List of camera identifiers (length ``N``)
    :ivar image_paths: List of catalog image paths (length ``N``)
    :ivar sources: List of camera sources (length ``N``)
    :ivar variant_tags: Optional per-row variant labels (length ``N``)
    :ivar mean_vector: Optional catalogue mean embedding (length ``D``)
    :ivar mean_center: Whether query-time mean-centring is requested
    """

    def __init__(
        self,
        embeddings_path: Path | str | None = None,
        mean_center: bool | None = None,
    ) -> None:
        r"""Initialize index from a serialized embeddings file.

        If ``embeddings_path`` is not provided, the default
        ``output/catalog_embeddings.npz`` under :data:`OUTPUT_DIR` is
        used.

        :param embeddings_path: Path to the stored embeddings
        :param mean_center: Whether to mean-centre embeddings before
            cosine similarity. Defaults to ``matching.mean_center``
            (``CIDAR_MATCH_MEAN_CENTER``). Only takes effect when the
            artifact contains a ``mean_vector``
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

        # Optional keys — artifacts built before augmentation/centring
        # support don't have them and must keep loading.
        self.variant_tags: list[str] | None = (
            data["variant_tags"].astype(str).tolist()
            if "variant_tags" in data.files
            else None
        )
        self.mean_vector: np.ndarray | None = (
            data["mean_vector"].astype("float32")
            if "mean_vector" in data.files
            else None
        )

        if self.embeddings.ndim != 2:
            raise ValueError("Embeddings array must be 2D (N, D)")

        if not (
            len(self.camera_ids)
            == len(self.image_paths)
            == len(self.sources)
            == self.embeddings.shape[0]
        ):
            raise ValueError("Embeddings and metadata arrays have inconsistent lengths")

        if self.variant_tags is not None and len(self.variant_tags) != len(
            self.camera_ids
        ):
            raise ValueError("variant_tags length does not match embeddings")

        self.mean_center = (
            mean_center if mean_center is not None else matching.mean_center
        )
        if self.mean_center and self.mean_vector is None:
            logger.warning(
                "Mean-centring requested but artifact has no mean_vector; "
                "centring is disabled. Rebuild the index to enable it."
            )
        self._center_active = self.mean_center and self.mean_vector is not None

        # Matrix actually used for search: mean-centred and re-normalized
        # when centring is active, the raw embeddings otherwise.
        if self._center_active:
            centered = self.embeddings - self.mean_vector
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            self._search_matrix: np.ndarray = (centered / norms).astype("float32")
        else:
            self._search_matrix = self.embeddings

        # Group rows by product for best-variant-per-product collapse.
        self._unique_camera_ids, self._camera_inverse = np.unique(
            np.array(self.camera_ids), return_inverse=True
        )

        logger.info(
            "Catalog index loaded: N={}, D={}, products={}, "
            "variants={}, mean_center={}",
            self.embeddings.shape[0],
            self.embeddings.shape[1],
            len(self._unique_camera_ids),
            "yes" if self.variant_tags is not None else "no",
            "on" if self._center_active else "off",
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[CameraMatch]:
        r"""Search the index for nearest catalog items.

        Uses cosine similarity between the query vector and all catalog
        embeddings. The query is L2-normalized before comparison (after
        mean-centring, when active). Scores are collapsed to the best
        variant per product before top-k selection, so each
        ``camera_id`` appears at most once in the results.

        :param query_vector: Embedding vector for the query image
        :param top_k: Number of nearest products to return
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

        if self._center_active:
            v = v - self.mean_vector

        # L2-normalize query
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            raise ValueError("Query embedding has zero norm")
        v = v / norm

        # Cosine similarity reduces to dot product for normalized vectors
        scores = self._search_matrix @ v  # shape: (N,)

        n_products = self._unique_camera_ids.shape[0]
        if n_products == 0:
            return []

        # Collapse to the best variant per product before top-k
        best_scores = np.full(n_products, -np.inf, dtype="float64")
        np.maximum.at(best_scores, self._camera_inverse, scores.astype("float64"))

        k = min(max(top_k, 0), n_products)
        if k == 0:
            return []

        # Partial sort for top-k products, then sort those by score
        group_idx = np.argpartition(-best_scores, k - 1)[:k]
        group_idx = group_idx[np.argsort(-best_scores[group_idx])]

        matches: list[CameraMatch] = []
        for group in group_idx:
            rows = np.flatnonzero(self._camera_inverse == group)
            row = int(rows[np.argmax(scores[rows])])
            matches.append(
                CameraMatch(
                    camera_id=self.camera_ids[row],
                    score=float(scores[row]),
                    catalog_image_path=self.image_paths[row],
                    source=self.sources[row],
                    record=None,
                )
            )

        return matches
