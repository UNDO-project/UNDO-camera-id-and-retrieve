"""Stage 2: Build parquet dataset from filesystem.

This is an independent stage that reads scraped data from the filesystem
and creates the parquet dataset.

"""

import argparse
from pathlib import Path

from loguru import logger

from src.config import OUTPUT_DIR
from src.building.builder import DatasetBuilder
from src.storage.versioning import DatasetVersionManager


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

  # Build with auto-versioning
  python build_dataset.py --version-mode auto

  # Append with versioning
  python build_dataset.py --append --version-mode auto

  # Build with manual version number
  python build_dataset.py --version-mode manual --version 5

  # List all versions
  python build_dataset.py --list-versions

  # Clean up old versions (keep last 3)
  python build_dataset.py --cleanup-versions 3
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

    parser.add_argument(
        "--version-mode",
        choices=["auto", "manual", "none"],
        default="none",
        help="Versioning mode: 'auto' increments version automatically, "
        "'manual' requires --version, 'none' disables versioning (default: none)",
    )

    parser.add_argument(
        "--version",
        type=int,
        help="Manual version number (requires --version-mode manual)",
    )

    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="List all dataset versions and exit",
    )

    parser.add_argument(
        "--cleanup-versions",
        type=int,
        metavar="KEEP_N",
        help="Clean up old versions, keeping last N versions",
    )

    args = parser.parse_args()

    # Resolve paths
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = OUTPUT_DIR.parent / manifest_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = OUTPUT_DIR.parent / output_path

    # Handle special commands
    version_manager = DatasetVersionManager(output_path.parent)

    if args.list_versions:
        # List all versions and exit
        logger.info("📊 Dataset Version History")
        versions = version_manager.list_versions()

        if not versions:
            logger.info("No versions found")
            exit(0)

        for v in versions:
            logger.info(f"\nVersion {v['version']}:")
            logger.info(f"  Timestamp: {v['timestamp']}")
            logger.info(f"  Records: {v['record_count']}")
            logger.info(
                f"  Manifest: {v['manifest_file']} ({v['manifest_hash'][:16]}...)"
            )
            logger.info(f"  Append mode: {v.get('append_mode', False)}")
            if v.get("append_mode"):
                logger.info(
                    f"  Added: {v.get('records_added', 0)}, Updated: {v.get('records_updated', 0)}"
                )

        logger.info(f"\n📌 Current version: {version_manager.get_current_version()}")
        exit(0)

    if args.cleanup_versions is not None:
        # Cleanup old versions
        logger.info(
            f"Cleaning up old versions, keeping last {args.cleanup_versions}..."
        )
        version_manager.cleanup_old_versions(args.cleanup_versions)
        logger.success("Cleanup completed")
        exit(0)

    # Normal build operation
    logger.info("CCTV Dataset Building Stage")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Output: {output_path}")

    if args.append:
        logger.info(f"Mode: Append with merge strategy '{args.merge_strategy}'")
    else:
        logger.info("Mode: Overwrite")

    if args.version_mode != "none":
        logger.info(f"Versioning: {args.version_mode}")
        if args.version_mode == "manual" and args.version:
            logger.info(f"Target version: {args.version}")

    # Build dataset
    builder = DatasetBuilder(
        manifest_path=manifest_path,
        output_path=output_path,
        append=args.append,
        merge_strategy=args.merge_strategy,
        force=args.force,
        version_mode=args.version_mode,
        version_number=args.version,
    )

    if builder.build_and_save():
        logger.success("Dataset built successfully")
        exit(0)
    else:
        logger.error("Failed to build dataset")
        exit(1)


if __name__ == "__main__":
    main()
