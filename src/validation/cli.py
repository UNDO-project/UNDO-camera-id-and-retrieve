"""Command-line interface for dataset validation."""

import argparse
from pathlib import Path

from loguru import logger

from src.config import OUTPUT_DIR
from src.validation.validator import DatasetValidator


def main() -> None:
    r"""
    CLI entry point for dataset validation.
    """
    parser = argparse.ArgumentParser(
        description="Validate CCTV camera dataset integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # Full validation with all 5 layers
  python -m src.validation.cli

  # Validation with verbose output
  python -m src.validation.cli --verbose

  # Generate baseline manifest from existing dataset
  python -m src.validation.cli --generate-manifest

  # Custom paths
  python -m src.validation.cli --parquet data/custom.parquet --manifest data/manifest.json

  # Validate without generating manifest if missing
  python -m src.validation.cli --no-auto-manifest

  # Generate manifest only (don't validate)
  python -m src.validation.cli --manifest-only
        """,
    )

    parser.add_argument(
        "--parquet",
        type=str,
        default="output/products.parquet",
        help="Path to parquet file (default: output/products.parquet)",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default="output/verification_manifest.json",
        help="Path to verification manifest (default: output/verification_manifest.json)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed errors and warnings",
    )

    parser.add_argument(
        "--generate-manifest",
        action="store_true",
        help="Generate manifest from parquet if missing",
    )

    parser.add_argument(
        "--no-auto-manifest",
        action="store_true",
        help="Skip automatic manifest generation if missing",
    )

    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only generate manifest, skip validation",
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root directory for file path resolution (default: current directory)",
    )

    args = parser.parse_args()

    # Resolve paths
    parquet_path = Path(args.parquet)
    if not parquet_path.is_absolute():
        parquet_path = OUTPUT_DIR.parent / parquet_path

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = OUTPUT_DIR.parent / manifest_path

    project_root = None
    if args.project_root:
        project_root = Path(args.project_root)

    logger.info("CCTV Dataset Validator")
    logger.info(f"Parquet: {parquet_path}")
    logger.info(f"Manifest: {manifest_path}")

    # Initialize validator
    validator = DatasetValidator(parquet_path=parquet_path, manifest_path=manifest_path)

    # Manifest-only mode
    if args.manifest_only:
        logger.info("Running in manifest-only mode...")
        if not validator.load_dataset():
            logger.error("Failed to load dataset")
            return

        logger.info("Generating manifest from parquet...")
        if validator.generate_manifest_from_parquet():
            logger.success("Manifest generated successfully")
        else:
            logger.error("Failed to generate manifest")
        return

    # Full validation mode
    logger.info("Running full validation...")

    # Handle manifest
    if not manifest_path.exists():
        if args.generate_manifest or not args.no_auto_manifest:
            logger.info("Manifest not found. Generating from parquet...")
            if not validator.load_dataset():
                logger.error("Failed to load dataset")
                return

            if not validator.generate_manifest_from_parquet():
                logger.error("Failed to generate manifest")
                return

            if not validator.load_manifest():
                logger.warning("Could not reload generated manifest")
        else:
            logger.warning("Manifest not found and auto-generation disabled")

    # Run validation
    success = validator.validate_all(verbose=args.verbose, project_root=project_root)

    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
