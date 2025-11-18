"""Command-line interface for camera identification.

This module exposes a CLI entry point for running the camera
identification pipeline on one or more input images.
"""

import argparse
from pathlib import Path
from typing import Optional

from loguru import logger

from src.identification.service import IdentificationService


def main() -> None:
    r"""Entry point for the camera identification CLI.

    Example usage::

        python -m src.identification.cli --image path/to/photo.jpg
    """
    parser = argparse.ArgumentParser(
        description="Camera Identification & Retrieval CLI",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of matches to return per detection (default: 5)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.3,
        help="Minimum cosine similarity to keep a match (default: 0.3)",
    )
    parser.add_argument(
        "--no-save-crops",
        action="store_true",
        help="Do not persist cropped detection patches to disk",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Optional path to products parquet file (default: output/products.parquet)",
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help=(
            "Optional path to catalog embeddings .npz file "
            "(default: output/catalog_embeddings.npz)"
        ),
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Input image not found: {image_path}")

    parquet_path: Optional[Path] = Path(args.parquet) if args.parquet else None
    embeddings_path: Optional[Path] = Path(args.embeddings) if args.embeddings else None

    logger.info("Camera Identification & Retrieval CLI")
    logger.info(f"Image: {image_path}")

    service = IdentificationService(
        parquet_path=parquet_path,
        embeddings_path=embeddings_path,
        min_similarity=args.min_similarity,
        save_crops=not args.no_save_crops,
    )

    results = service.identify_from_image(image_path, top_k=args.top_k)

    if not results:
        print("No cameras detected.")
        return

    for det_idx, result in enumerate(results, start=1):
        det = result.detection
        print(f"Detection {det_idx}:")
        print(
            f"  BBox: ({det.bbox.x_min}, {det.bbox.y_min}, {det.bbox.x_max}, {det.bbox.y_max})"
        )
        print(f"  Confidence: {det.confidence:.3f}")
        if det.crop_path is not None:
            print(f"  Crop path: {det.crop_path}")

        if not result.matches:
            print("  No matches above similarity threshold.")
            continue

        print("  Matches:")
        for rank, match in enumerate(result.matches, start=1):
            record = match.record
            model_name = record.model_name if record is not None else "<unknown>"
            source = match.source
            print(
                f"    {rank}. {match.camera_id} | score={match.score:.3f} | "
                f"model={model_name} | source={source}"
            )
            if match.catalog_image_path is not None:
                print(f"       image: {match.catalog_image_path}")


if __name__ == "__main__":
    main()
