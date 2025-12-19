"""Stage 2: Build parquet dataset from filesystem.

This is an independent stage that reads scraped data from the filesystem
and creates the parquet dataset.

Usage:
    python build_dataset.py
"""

import argparse
from pathlib import Path

from loguru import logger

from src.config import OUTPUT_DIR
from src.pipeline.dataset_builder import DatasetBuilder


def main() -> None:
    r"""
    Build dataset from filesystem.
    """
    parser = argparse.ArgumentParser(
        description="Build parquet dataset from scraped data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset with default paths (prompts for overwrite if exists)
  python build_dataset.py

  # Force overwrite without confirmation
  python build_dataset.py --force

  # Append new data to existing dataset (update duplicates)
  python build_dataset.py --append

  # Append with skip strategy (keep original data for duplicates)
  python build_dataset.py --append --merge-strategy skip

  # Append with error on duplicates
  python build_dataset.py --append --merge-strategy error

  # Custom paths with append
  python build_dataset.py --manifest data/manifest.json --output data/dataset.parquet --append
        """,
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default="output/verification_manifest.json",
        help="Path to verification manifest (default: output/verification_manifest.json)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output/products.parquet",
        help="Path to output parquet file (default: output/products.parquet)",
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing dataset instead of overwriting",
    )

    parser.add_argument(
        "--merge-strategy",
        choices=["update", "skip", "error"],
        default="update",
        help="Strategy for handling duplicate camera_ids (default: update). "
        "'update' overwrites duplicates with new data, 'skip' keeps original data, "
        "'error' fails if duplicates are found.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompts (useful for automation)",
    )

    args = parser.parse_args()

    # Resolve paths
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = OUTPUT_DIR.parent / manifest_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR.parent / output_path

    logger.info("CCTV Dataset Building Stage")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Output: {output_path}")
    if args.append:
        logger.info(f"Mode: Append with merge strategy '{args.merge_strategy}'")
    else:
        logger.info("Mode: Overwrite")

    # Build dataset
    builder = DatasetBuilder(
        manifest_path=manifest_path,
        output_path=output_path,
        append=args.append,
        merge_strategy=args.merge_strategy,
        force=args.force,
    )

    if builder.build_and_save():
        logger.success("Dataset built successfully")
        exit(0)
    else:
        logger.error("Failed to build dataset")
        exit(1)


if __name__ == "__main__":
    main()
