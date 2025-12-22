"""Manifest recording for dataset verification."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import paths
from src.models.camera import CameraRecord


class ManifestRecorder:
    r"""
    Records scraping progress and generates verification manifest.

    Tracks categories, series, and products as they are processed
    to create a verification manifest for later validation.

    :ivar manifest_path: Path where manifest JSON will be saved
    :ivar manifest_data: Dictionary containing manifest structure
    """

    def __init__(self, manifest_path: Path | str | None = None) -> None:
        r"""
        Initialize ManifestRecorder.

        :param manifest_path: Path where manifest JSON will be saved
        """
        if manifest_path is None:
            manifest_path = paths.output_dir / "verification_manifest.json"
        else:
            manifest_path = Path(manifest_path)

        self.manifest_path = manifest_path
        self.manifest_data: dict[str, Any] = {
            "scrape_timestamp": datetime.utcnow().isoformat(),
            "total_categories": 0,
            "total_series": 0,
            "total_products": 0,
            "total_images": 0,
            "total_pdfs_with_urls": 0,
            "total_specs_populated": 0,
            "categories": {},
        }

    def record_category(self, category_name: str) -> None:
        r"""
        Record the start of processing a category.

        :param category_name: Name of the category being processed
        """
        if category_name not in self.manifest_data["categories"]:
            self.manifest_data["categories"][category_name] = {
                "series_count": 0,
                "product_count": 0,
                "image_count": 0,
                "pdf_count": 0,
                "series": {},
            }
            self.manifest_data["total_categories"] += 1
            logger.debug(f"Manifest: Recorded category {category_name}")

    def record_series(self, category_name: str, series_name: str) -> None:
        r"""
        Record the start of processing a series within a category.

        :param category_name: Parent category name
        :param series_name: Series name being processed
        """
        if category_name not in self.manifest_data["categories"]:
            self.record_category(category_name)

        if series_name not in self.manifest_data["categories"][category_name]["series"]:
            self.manifest_data["categories"][category_name]["series"][series_name] = {
                "product_count": 0,
                "image_count": 0,
                "pdf_count": 0,
                "specs_count": 0,
                "products": [],
            }
            self.manifest_data["categories"][category_name]["series_count"] += 1
            self.manifest_data["total_series"] += 1
            logger.debug(f"Manifest: Recorded series {series_name} in {category_name}")

    def record_product(self, record: CameraRecord) -> None:
        r"""
        Record a product that was successfully scraped.

        :param record: CameraRecord containing product information
        """
        category = record.product_category
        series = record.product_series

        # Ensure category and series are initialized
        self.record_series(category, series)

        product_entry = {
            "camera_id": record.camera_id,
            "model_name": record.model_name,
            "image_urls": record.images if record.images else [],
            "image_count": len(record.images) if record.images else 0,
            "has_datasheet": record.datasheet_url is not None,
            "datasheet_url": record.datasheet_url,
            "has_specs": len(record.specifications_html) > 0
            if record.specifications_html
            else False,
            "specifications_html": record.specifications_html
            if record.specifications_html
            else {},
        }

        self.manifest_data["categories"][category]["series"][series]["products"].append(
            product_entry
        )

        # Update counts
        self.manifest_data["categories"][category]["series"][series][
            "product_count"
        ] += 1
        self.manifest_data["categories"][category]["product_count"] += 1

        self.manifest_data["total_products"] += 1

        # Image count
        if record.images:
            image_count = len(record.images)
            self.manifest_data["categories"][category]["series"][series][
                "image_count"
            ] += image_count
            self.manifest_data["categories"][category]["image_count"] += image_count
            self.manifest_data["total_images"] += image_count

        # PDF count
        if record.datasheet_url:
            self.manifest_data["categories"][category]["series"][series][
                "pdf_count"
            ] += 1
            self.manifest_data["total_pdfs_with_urls"] += 1

        # Specs count
        if record.specifications_html:
            self.manifest_data["categories"][category]["series"][series][
                "specs_count"
            ] += 1
            self.manifest_data["total_specs_populated"] += 1

        logger.debug(f"Manifest: Recorded product {record.camera_id}")

    def save_manifest(self) -> None:
        r"""
        Save the manifest to JSON file.

        Creates output directory if needed and writes manifest with formatting.
        """
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Verification manifest saved to {self.manifest_path}")
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

    def load_manifest(self) -> bool:
        r"""
        Load existing manifest from file.

        :return: True if manifest loaded successfully
        """
        if not self.manifest_path.exists():
            logger.warning(f"Manifest file not found: {self.manifest_path}")
            return False

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest_data = json.load(f)
            logger.info(f"Loaded manifest from {self.manifest_path}")
            return True
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load manifest: {e}")
            return False
