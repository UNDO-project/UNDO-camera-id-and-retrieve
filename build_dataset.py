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
  # Build dataset with default paths
  python build_dataset.py

  # Custom manifest path
  python build_dataset.py --manifest data/custom_manifest.json

  # Custom output parquet path
  python build_dataset.py --output data/custom_dataset.parquet

  # Both custom paths
  python build_dataset.py --manifest data/manifest.json --output data/dataset.parquet
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

    # Build dataset
    builder = DatasetBuilder(manifest_path=manifest_path, output_path=output_path)

    if builder.build_and_save():
        logger.success("Dataset built successfully")
        exit(0)
    else:
        logger.error("Failed to build dataset")
        exit(1)


if __name__ == "__main__":
    main()
