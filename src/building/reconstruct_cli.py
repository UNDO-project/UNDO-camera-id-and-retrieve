"""CLI for manifest reconstruction."""

import sys
from pathlib import Path

import click
from loguru import logger

from src.building.manifest_reconstruction import ManifestReconstructor
from src.config import paths


@click.command()
@click.option(
    "--cache-db",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=f"Path to download_cache.db (default: {paths.output_dir / 'download_cache.db'})",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=f"Path to data directory (default: {paths.data_dir})",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Output path for manifest (default: {paths.output_dir / 'verification_manifest.json'})",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite existing manifest without confirmation",
)
def main(
    cache_db: Path | None,
    data_dir: Path | None,
    output: Path | None,
    force: bool,
) -> None:
    """
    Reconstruct verification manifest from cache and filesystem.

    This tool rebuilds the manifest without rescraping by using:
    - Filesystem structure (data/images/ and data/pdfs/)
    - Product metadata cache (output/download_cache.db)

    Use this when:
    - Manifest file is missing or corrupted
    - Scraping completed but manifest creation failed
    - Need to regenerate manifest from existing data
    """
    logger.info("Manifest Reconstruction Tool")
    logger.info("=" * 50)

    # Check if output file exists
    output_path = output or paths.output_dir / "verification_manifest.json"
    if output_path.exists() and not force:
        click.echo(f"\nManifest already exists at: {output_path}")
        if not click.confirm("Overwrite existing manifest?", default=False):
            logger.info("Operation cancelled by user")
            sys.exit(0)

    # Initialize reconstructor
    reconstructor = ManifestReconstructor(
        cache_db_path=cache_db,
        data_dir=data_dir,
        output_path=output_path,
    )

    # Perform reconstruction
    success = reconstructor.reconstruct()

    if success:
        logger.success("Manifest reconstruction completed successfully!")
        logger.info("You can now run: cidar-build")
        sys.exit(0)
    else:
        logger.error("Manifest reconstruction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
