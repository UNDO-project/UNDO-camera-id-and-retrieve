r"""Synthetic robustness harness for catalogue retrieval.

Degrades each catalogue reference image in controlled ways, embeds the
degraded copy, queries the index, and records whether the correct
product is recovered at top-1/top-5 — a measurement of retrieval
robustness that needs zero street labels.

Determinism: every (image, degradation, severity) triple derives its
own :class:`numpy.random.Generator` from the run seed, so two runs
with the same seed produce identical reports regardless of iteration
order or subsetting.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from src.identification.eval.degradations import (
    DEGRADATIONS,
    SEVERITIES,
    apply_degradation,
)
from src.identification.eval.report import EvalReport, EvalRow
from src.identification.index import CatalogIndex

TOP_K = 5


def iter_reference_images(index: CatalogIndex) -> list[tuple[str, Path]]:
    r"""List one reference image per product in the index.

    Augmented indexes (WP3) hold several rows per ``camera_id``; the
    original reference row is preferred via its variant tag, falling
    back to the first row seen for that product.

    :param index: Loaded catalog index
    :return: List of ``(camera_id, image_path)`` pairs, index order
    """
    variant_tags = getattr(index, "variant_tags", None)
    chosen: dict[str, Path] = {}

    for position, (camera_id, image_path) in enumerate(
        zip(index.camera_ids, index.image_paths)
    ):
        is_original = variant_tags is None or variant_tags[position] == "orig"
        if camera_id not in chosen and is_original:
            chosen[camera_id] = image_path

    # Products whose original row was somehow untagged: fall back to first row
    for camera_id, image_path in zip(index.camera_ids, index.image_paths):
        chosen.setdefault(camera_id, image_path)

    return list(chosen.items())


def run_synthetic_eval(
    index: CatalogIndex,
    seed: int = 42,
    degradation_names: list[str] | None = None,
    severities: tuple[int, ...] = SEVERITIES,
    limit: int | None = None,
    embed_fn: Callable[[Image.Image], np.ndarray] | None = None,
) -> EvalReport:
    r"""Run the synthetic robustness evaluation against an index.

    :param index: Loaded catalog index to query
    :param seed: Run seed; identical seeds yield identical reports
    :param degradation_names: Subset of degradations to run (default: all)
    :param severities: Severity levels to run (default: 1-3)
    :param limit: Evaluate only the first N products (quick runs)
    :param embed_fn: Embedding function override, mainly for tests
        (default: :func:`src.identification.embeddings.embed_image`)
    :return: Aggregated evaluation report
    :raises KeyError: If an unknown degradation name is requested
    """
    if embed_fn is None:
        from src.identification.embeddings import embed_image

        embed_fn = embed_image

    if degradation_names is None:
        degradation_names = list(DEGRADATIONS)
    for name in degradation_names:
        if name not in DEGRADATIONS:
            raise KeyError(
                f"Unknown degradation '{name}'. Available: {sorted(DEGRADATIONS)}"
            )

    references = iter_reference_images(index)
    total_products = len(references)
    if limit is not None:
        references = references[:limit]

    logger.info(
        "Synthetic eval: {} products ({} in index), {} degradations, severities {}",
        len(references),
        total_products,
        len(degradation_names),
        list(severities),
    )

    hits: dict[tuple[str, int], list[int]] = {
        (name, severity): [0, 0, 0]  # [n, top1_hits, top5_hits]
        for name in degradation_names
        for severity in severities
    }
    skipped_images = 0
    n_queries = 0

    for image_index, (camera_id, image_path) in enumerate(references):
        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            logger.warning(f"Skipping unreadable reference image {image_path}: {exc}")
            skipped_images += 1
            continue

        for degradation_index, name in enumerate(degradation_names):
            for severity in severities:
                rng = np.random.default_rng(
                    [seed, image_index, degradation_index, severity]
                )
                degraded = apply_degradation(name, image, severity, rng)
                query_vector = embed_fn(degraded)
                matches = index.search(query_vector, top_k=TOP_K)

                counters = hits[(name, severity)]
                counters[0] += 1
                if matches and matches[0].camera_id == camera_id:
                    counters[1] += 1
                if any(match.camera_id == camera_id for match in matches):
                    counters[2] += 1
                n_queries += 1

        if (image_index + 1) % 100 == 0:
            logger.info(f"Synthetic eval progress: {image_index + 1}/{len(references)}")

    rows = [
        EvalRow(
            group=name,
            severity=severity,
            n=counters[0],
            top1_hits=counters[1],
            top5_hits=counters[2],
        )
        for (name, severity), counters in sorted(hits.items())
    ]

    return EvalReport(
        mode="synthetic",
        embeddings_path=str(index.embeddings_path),
        n_queries=n_queries,
        seed=seed,
        rows=rows,
        notes={
            "products_evaluated": len(references) - skipped_images,
            "products_in_index": total_products,
            "skipped_images": skipped_images,
            "top_k": TOP_K,
        },
    )
