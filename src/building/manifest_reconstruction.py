"""Reconstruct verification manifest from cache and filesystem."""

import json
import sqlite3
from datetime import datetime, UTC
from pathlib import Path

from loguru import logger

from src.config import paths


class ManifestReconstructor:
    r"""
    Reconstructs verification manifest from download cache and filesystem.

    This tool allows rebuilding the manifest without rescraping when:
    - Manifest file is deleted or corrupted
    - Scraping completed but manifest creation failed
    - Need to regenerate manifest from existing data

    Strategy:
    1. Scan filesystem to discover category/series/product structure
    2. Load product metadata from product_cache table in download_cache.db
    3. Rebuild manifest structure matching original format
    """

    def __init__(
        self,
        cache_db_path: Path | str | None = None,
        data_dir: Path | str | None = None,
        output_path: Path | str | None = None,
    ) -> None:
        r"""
        Initialize manifest reconstructor.

        :param cache_db_path: Path to download_cache.db
        :param data_dir: Path to data/ directory
        :param output_path: Path to save reconstructed manifest
        """
        if cache_db_path is None:
            cache_db_path = paths.output_dir / "download_cache.db"
        else:
            cache_db_path = Path(cache_db_path)

        if data_dir is None:
            data_dir = paths.data_dir
        else:
            data_dir = Path(data_dir)

        if output_path is None:
            output_path = paths.output_dir / "verification_manifest.json"
        else:
            output_path = Path(output_path)

        self.cache_db_path = cache_db_path
        self.data_dir = data_dir
        self.output_path = output_path
        self.manifest_data: dict = {}

    def _scan_filesystem_structure(self) -> dict[str, dict[str, list[str]]]:
        r"""
        Scan filesystem to discover category/series/product structure.

        Filesystem structure is:
        data/images/{VENDOR}/{CATEGORY}/{SERIES}/{CAMERA_ID}/

        :return: Dict mapping {category_name: {series_name: [camera_id, ...]}}
        """
        logger.info("Scanning filesystem structure...")

        structure: dict[str, dict[str, list[str]]] = {}
        images_dir = self.data_dir / "images"

        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return structure

        # Iterate through vendor directories (e.g., AXIS_COMMUNICATIONS, HIKVISION)
        for vendor_dir in sorted(images_dir.iterdir()):
            if not vendor_dir.is_dir():
                continue

            # Iterate through category directories within each vendor
            for category_dir in sorted(vendor_dir.iterdir()):
                if not category_dir.is_dir():
                    continue

                category_name = category_dir.name.replace("_", " ").title()

                # Initialize category if not exists (may have products from multiple vendors)
                if category_name not in structure:
                    structure[category_name] = {}

                # Iterate through series directories
                for series_dir in sorted(category_dir.iterdir()):
                    if not series_dir.is_dir():
                        continue

                    series_name = series_dir.name.replace("_", " ")

                    # Initialize series if not exists
                    if series_name not in structure[category_name]:
                        structure[category_name][series_name] = []

                    # Iterate through product directories (camera_id folders)
                    for product_dir in sorted(series_dir.iterdir()):
                        if not product_dir.is_dir():
                            continue

                        camera_id = product_dir.name
                        structure[category_name][series_name].append(camera_id)

        total_series = sum(len(s) for s in structure.values())
        total_products = sum(
            len(cameras) for series in structure.values() for cameras in series.values()
        )
        logger.info(
            f"Found {len(structure)} categories with {total_series} series "
            f"and {total_products} products"
        )
        return structure

    def _load_product_cache(self) -> dict[str, dict]:
        r"""
        Load product metadata from cache database.

        :return: Dict mapping camera_id to product metadata
        """
        logger.info(f"Loading product cache from {self.cache_db_path}")

        if not self.cache_db_path.exists():
            logger.error(f"Cache database not found: {self.cache_db_path}")
            return {}

        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT product_url, model_name, image_urls,
                       datasheet_url, specifications_html
                FROM product_cache
                """
            )

            cache = {}
            for row in cursor.fetchall():
                product_url, model_name, image_urls, datasheet_url, specs_html = row

                # Extract camera_id from model name (convert to slug)
                camera_id = model_name.lower().replace(" ", "-")

                # Parse JSON strings
                try:
                    image_urls_list = json.loads(image_urls) if image_urls else []
                    specs_dict = json.loads(specs_html) if specs_html else {}
                except json.JSONDecodeError:
                    image_urls_list = []
                    specs_dict = {}

                cache[camera_id] = {
                    "product_url": product_url,
                    "model_name": model_name,
                    "image_urls": image_urls_list,
                    "datasheet_url": datasheet_url,
                    "specifications_html": specs_dict,
                }

            conn.close()
            logger.info(f"Loaded {len(cache)} products from cache")
            return cache

        except Exception as e:
            logger.error(f"Failed to load product cache: {e}")
            return {}

    def _build_manifest_structure(
        self,
        fs_structure: dict[str, dict[str, list[str]]],
        product_cache: dict[str, dict],
    ) -> dict:
        r"""
        Build manifest structure from filesystem and cache data.

        :param fs_structure: Filesystem structure from _scan_filesystem_structure
        :param product_cache: Product cache from _load_product_cache
        :return: Complete manifest dictionary
        """
        logger.info("Building manifest structure...")

        manifest = {
            "scrape_timestamp": datetime.now(UTC).isoformat(),
            "reconstruction_timestamp": datetime.now(UTC).isoformat(),
            "reconstructed": True,
            "total_categories": 0,
            "total_series": 0,
            "total_products": 0,
            "total_images": 0,
            "total_pdfs_with_urls": 0,
            "total_specs_populated": 0,
            "categories": {},
        }

        for category_name, series_dict in fs_structure.items():
            manifest["categories"][category_name] = {
                "series_count": 0,
                "product_count": 0,
                "image_count": 0,
                "pdf_count": 0,
                "series": {},
            }
            manifest["total_categories"] += 1

            for series_name, camera_ids in series_dict.items():
                manifest["categories"][category_name]["series"][series_name] = {
                    "product_count": 0,
                    "image_count": 0,
                    "pdf_count": 0,
                    "specs_count": 0,
                    "products": [],
                }
                manifest["categories"][category_name]["series_count"] += 1
                manifest["total_series"] += 1

                for camera_id in camera_ids:
                    # Try to get product metadata from cache
                    product_data = product_cache.get(camera_id, {})

                    # Count images in filesystem
                    image_count = self._count_images(
                        category_name, series_name, camera_id
                    )

                    # Check for PDF file
                    has_pdf = self._check_pdf_exists(
                        category_name, series_name, camera_id
                    )

                    product_entry = {
                        "camera_id": camera_id,
                        "model_name": product_data.get("model_name", camera_id),
                        "image_urls": product_data.get("image_urls", []),
                        "image_count": image_count,
                        "has_datasheet": has_pdf
                        or bool(product_data.get("datasheet_url")),
                        "datasheet_url": product_data.get("datasheet_url"),
                        "has_specs": bool(product_data.get("specifications_html")),
                        "specifications_html": product_data.get(
                            "specifications_html", {}
                        ),
                    }

                    manifest["categories"][category_name]["series"][series_name][
                        "products"
                    ].append(product_entry)

                    # Update counts
                    manifest["categories"][category_name]["series"][series_name][
                        "product_count"
                    ] += 1
                    manifest["categories"][category_name]["product_count"] += 1
                    manifest["total_products"] += 1

                    if image_count > 0:
                        manifest["categories"][category_name]["series"][series_name][
                            "image_count"
                        ] += image_count
                        manifest["categories"][category_name]["image_count"] += (
                            image_count
                        )
                        manifest["total_images"] += image_count

                    if has_pdf or product_data.get("datasheet_url"):
                        manifest["categories"][category_name]["series"][series_name][
                            "pdf_count"
                        ] += 1
                        manifest["total_pdfs_with_urls"] += 1

                    if product_data.get("specifications_html"):
                        manifest["categories"][category_name]["series"][series_name][
                            "specs_count"
                        ] += 1
                        manifest["total_specs_populated"] += 1

        logger.info(
            f"Built manifest with {manifest['total_products']} products, "
            f"{manifest['total_images']} images, {manifest['total_pdfs_with_urls']} PDFs"
        )
        return manifest

    def _count_images(
        self, category_name: str, series_name: str, camera_id: str
    ) -> int:
        r"""
        Count image files for a product.

        Searches all vendor directories for the product.

        :param category_name: Category name
        :param series_name: Series name
        :param camera_id: Camera ID
        :return: Number of image files
        """
        category_dir = category_name.upper().replace(" ", "_")
        series_dir = series_name.replace(" ", "_")
        images_dir = self.data_dir / "images"

        # Search in all vendor directories
        for vendor_dir in images_dir.iterdir():
            if not vendor_dir.is_dir():
                continue

            product_dir = vendor_dir / category_dir / series_dir / camera_id
            if product_dir.exists():
                return len(list(product_dir.glob("*.webp")))

        return 0

    def _check_pdf_exists(
        self, category_name: str, series_name: str, camera_id: str
    ) -> bool:
        r"""
        Check if PDF file exists for a product.

        Searches all vendor directories for the product PDF.

        :param category_name: Category name
        :param series_name: Series name
        :param camera_id: Camera ID
        :return: True if PDF exists
        """
        category_dir = category_name.upper().replace(" ", "_")
        series_dir = series_name.replace(" ", "_")
        pdfs_dir = self.data_dir / "pdfs"

        # Search in all vendor directories
        for vendor_dir in pdfs_dir.iterdir():
            if not vendor_dir.is_dir():
                continue

            pdf_file = vendor_dir / category_dir / series_dir / f"{camera_id}.pdf"
            if pdf_file.exists():
                return True

        return False

    def reconstruct(self) -> bool:
        r"""
        Reconstruct manifest from cache and filesystem.

        :return: True if successful
        """
        logger.info("Starting manifest reconstruction...")

        # Step 1: Scan filesystem
        fs_structure = self._scan_filesystem_structure()
        if not fs_structure:
            logger.error("No data found in filesystem")
            return False

        # Step 2: Load product cache
        product_cache = self._load_product_cache()

        # Step 3: Build manifest
        self.manifest_data = self._build_manifest_structure(fs_structure, product_cache)

        # Step 4: Save manifest
        return self._save_manifest()

    def _save_manifest(self) -> bool:
        r"""
        Save reconstructed manifest to file.

        :return: True if successful
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest_data, f, indent=2, ensure_ascii=False)

            logger.success(f"Reconstructed manifest saved to {self.output_path}")
            logger.info(
                f"  Categories: {self.manifest_data['total_categories']}, "
                f"Series: {self.manifest_data['total_series']}, "
                f"Products: {self.manifest_data['total_products']}"
            )
            logger.info(
                f"  Images: {self.manifest_data['total_images']}, "
                f"PDFs: {self.manifest_data['total_pdfs_with_urls']}, "
                f"Specs: {self.manifest_data['total_specs_populated']}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            return False
