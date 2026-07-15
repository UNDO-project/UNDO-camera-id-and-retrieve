r"""Real-probe evaluation for catalogue retrieval.

Runs a small, hand-labelled set of real street crops through the same
embed-and-search path as production and reports top-k recovery in the
shared report format. This is the external-validity check the
synthetic harness cannot provide: synthetic curves can look great
while real street crops still fail.

Probe set format (``probe_set.jsonl``, one JSON object per line)::

    {"image": "data/eval_probe/img_001.jpg", "camera_id": "...",
     "vendor": "Axis", "model": "..."}

``image`` paths are resolved relative to the project root. Probe
images stay out of git; only the jsonl format and the
``data/eval_probe/`` directory convention are part of the repo.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from src.config import paths
from src.identification.eval.report import EvalReport, EvalRow
from src.identification.index import CatalogIndex

TOP_K = 5


def default_probe_set_path() -> Path:
    """Return the conventional probe-set location.

    :return: ``data/eval_probe/probe_set.jsonl`` under the project root
    """
    return paths.data_dir / "eval_probe" / "probe_set.jsonl"


def load_probe_set(probe_set_path: Path | str) -> list[dict]:
    r"""Load and validate probe entries from a jsonl file.

    Lines that are blank are ignored; lines that are not valid JSON or
    lack ``image``/``camera_id`` raise, since a malformed probe set
    should fail loudly rather than skew results.

    :param probe_set_path: Path to the ``probe_set.jsonl`` file
    :return: List of probe entry dictionaries
    :raises FileNotFoundError: If the probe set file does not exist
    :raises ValueError: If a line is malformed or misses required keys
    """
    probe_set_path = Path(probe_set_path)
    if not probe_set_path.exists():
        raise FileNotFoundError(f"Probe set not found: {probe_set_path}")

    entries: list[dict] = []
    for line_number, line in enumerate(
        probe_set_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{probe_set_path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if "image" not in entry or "camera_id" not in entry:
            raise ValueError(
                f"{probe_set_path}:{line_number}: entry must contain "
                "'image' and 'camera_id'"
            )
        entries.append(entry)

    return entries


def run_probe_eval(
    index: CatalogIndex,
    probe_set_path: Path | str | None = None,
    embed_fn: Callable[[Image.Image], np.ndarray] | None = None,
) -> EvalReport:
    r"""Evaluate the index against a labelled real-image probe set.

    Missing or unreadable probe images are skipped and counted in the
    report notes rather than failing the run, since probe images are
    intentionally not tracked in git.

    :param index: Loaded catalog index to query
    :param probe_set_path: Path to ``probe_set.jsonl``
        (default: ``data/eval_probe/probe_set.jsonl``)
    :param embed_fn: Embedding function override, mainly for tests
        (default: :func:`src.identification.embeddings.embed_image`)
    :return: Aggregated evaluation report; ``notes`` records the probe
        set size and the number of skipped images
    :raises FileNotFoundError: If the probe set file does not exist
    :raises ValueError: If the probe set file is malformed
    """
    if embed_fn is None:
        from src.identification.embeddings import embed_image

        embed_fn = embed_image

    if probe_set_path is None:
        probe_set_path = default_probe_set_path()
    entries = load_probe_set(probe_set_path)

    logger.info(f"Probe eval: {len(entries)} labelled entries from {probe_set_path}")

    project_root = paths.project_root
    n = 0
    top1_hits = 0
    top5_hits = 0
    skipped_images = 0

    for entry in entries:
        image_path = Path(entry["image"])
        if not image_path.is_absolute():
            image_path = project_root / image_path

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError) as exc:
            logger.warning(f"Skipping missing probe image {image_path}: {exc}")
            skipped_images += 1
            continue

        query_vector = embed_fn(image)
        matches = index.search(query_vector, top_k=TOP_K)

        n += 1
        if matches and matches[0].camera_id == entry["camera_id"]:
            top1_hits += 1
        if any(match.camera_id == entry["camera_id"] for match in matches):
            top5_hits += 1

    return EvalReport(
        mode="probe",
        embeddings_path=str(index.embeddings_path),
        n_queries=n,
        seed=None,
        rows=[
            EvalRow(
                group="probe",
                severity=0,
                n=n,
                top1_hits=top1_hits,
                top5_hits=top5_hits,
            )
        ],
        notes={
            "probe_set_path": str(probe_set_path),
            "probe_set_size": len(entries),
            "skipped_images": skipped_images,
            "top_k": TOP_K,
        },
    )
