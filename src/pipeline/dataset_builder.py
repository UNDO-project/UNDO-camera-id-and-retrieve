"""Dataset building stage: Create parquet from filesystem data."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import DATA_DIR, OUTPUT_DIR
from src.models.camera import CameraRecord


class DatasetBuilder:
    r"""
    Builds parquet dataset from scraped filesystem data.

    Reads:
    - Images from data/images/{category}/{series}/{product_id}/
    - PDFs from data/pdfs/{category}/{series}/
    - Manifest from output/verification_manifest.json

    Creates parquet with all product metadata.

    :ivar manifest_path: Path to verification_manifest.json
    :ivar output_path: Path to output parquet file
    :ivar records: List of CameraRecord objects
    """

    def __init__(
        self,
        manifest_path: Path | str | None = None,
        output_path: Path | str | None = None,
    ) -> None:
        r"""
        Initialize dataset builder.

        :param manifest_path: Path to verification manifest
        :param output_path: Path to output parquet (default: output/products.parquet)
        """
        if manifest_path is None:
            manifest_path = OUTPUT_DIR / "verification_manifest.json"
        else:
            manifest_path = Path(manifest_path)

        if output_path is None:
            output_path = OUTPUT_DIR / "products.parquet"
        else:
            output_path = Path(output_path)

        self.manifest_path = manifest_path
        self.output_path = output_path
        self.records: list[CameraRecord] = []
        self.manifest_data: dict = {}

    def load_manifest(self) -> bool:
        r"""
        Load verification manifest.

        :return: True if loaded successfully
        """
        if not self.manifest_path.exists():
            logger.error(f"Manifest not found: {self.manifest_path}")
            return False

        try:
            logger.info(f"Loading manifest from {self.manifest_path}")
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.manifest_data = json.load(f)
            logger.info("Manifest loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return False

    def build_dataset(self) -> bool:
        r"""
        Build dataset from filesystem and manifest.

        Iterates through manifest structure and creates CameraRecord objects
        for each product, populating file paths from filesystem.

        :return: True if dataset built successfully
        """
        logger.info("Building dataset from filesystem...")

        manifest = self.manifest_data
        total_products = 0

        try:
            for category_name, category_data in manifest.get("categories", {}).items():
                for series_name, series_data in category_data.get("series", {}).items():
                    for product_info in series_data.get("products", []):
                        camera_id = product_info.get("camera_id")
                        model_name = product_info.get("model_name")

                        if not camera_id or not model_name:
                            logger.warning(
                                "Skipping product with missing camera_id or model_name"
                            )
                            continue

                        # Find image files
                        image_files = self._find_image_files(
                            category_name, series_name, camera_id
                        )

                        # Find PDF file
                        pdf_file = self._find_pdf_file(
                            category_name, series_name, camera_id
                        )

                        # Load specifications from manifest
                        specifications_html = self._load_specifications_from_manifest(
                            product_info
                        )

                        # Create record
                        record = CameraRecord(
                            camera_id=camera_id,
                            model_name=model_name,
                            display_name=model_name,
                            description=None,
                            specifications={},
                            source="Axis Communications",
                            category="Network Camera",
                            product_category=category_name,
                            product_series=series_name,
                            images=[],  # Image URLs from manifest
                            image_url=None,
                            image_files=image_files,
                            datasheet_url=None,  # URL from manifest
                            datasheet_file=pdf_file,
                            specifications_html=specifications_html,
                        )

                        self.records.append(record)
                        total_products += 1
                        logger.debug(f"Created record for {camera_id}")

            logger.info(f"Built dataset with {total_products} products")
            return True

        except Exception as e:
            logger.error(f"Failed to build dataset: {e}")
            return False

    def _find_image_files(
        self, category_name: str, series_name: str, camera_id: str
    ) -> list[str]:
        r"""
        Find all image files for a product in filesystem.

        :param category_name: Product category
        :param series_name: Product series
        :param camera_id: Camera ID
        :return: List of relative paths to image files
        """
        category_dir = category_name.upper().replace(" ", "_")
        series_dir = series_name.replace(" ", "_")

        product_dir = DATA_DIR / "images" / category_dir / series_dir / camera_id

        if not product_dir.exists():
            return []

        image_files = []
        for img_file in sorted(product_dir.glob("*.webp")):
            relative_path = img_file.relative_to(Path.cwd())
            image_files.append(str(relative_path))

        return image_files

    def _find_pdf_file(
        self, category_name: str, series_name: str, camera_id: str
    ) -> str | None:
        r"""
        Find PDF file for a product in filesystem.

        :param category_name: Product category
        :param series_name: Product series
        :param camera_id: Camera ID
        :return: Relative path to PDF or None
        """
        category_dir = category_name.upper().replace(" ", "_")
        series_dir = series_name.replace(" ", "_")

        pdf_file = DATA_DIR / "pdfs" / category_dir / series_dir / f"{camera_id}.pdf"

        if pdf_file.exists():
            relative_path = pdf_file.relative_to(Path.cwd())
            return str(relative_path)

        return None

    def _load_specifications_from_manifest(
        self, product_info: dict
    ) -> dict[str, dict[str, str]]:
        r"""
        Load specifications from product info in manifest.

        :param product_info: Product entry from manifest
        :return: Specifications dictionary (section_name -> {spec_name -> spec_value})
        """
        return product_info.get("specifications_html", {})

    def save_dataset(self) -> bool:
        r"""
        Save dataset to parquet file.

        :return: True if saved successfully
        """
        if not self.records:
            logger.warning("No records to save")
            return False

        logger.info(f"Serializing {len(self.records)} records to parquet...")

        try:
            # Convert records to dictionaries
            data = []
            for record in self.records:
                data.append(
                    {
                        "camera_id": record.camera_id,
                        "model_name": record.model_name,
                        "display_name": record.display_name,
                        "description": record.description,
                        "source": record.source,
                        "category": record.category,
                        "product_category": record.product_category,
                        "product_series": record.product_series,
                        "image_urls": json.dumps(record.images),
                        "image_files": json.dumps(
                            record.image_files
                            if hasattr(record, "image_files") and record.image_files
                            else []
                        ),
                        "datasheet_url": record.datasheet_url,
                        "datasheet_file": record.datasheet_file
                        if hasattr(record, "datasheet_file")
                        else None,
                        "specifications": json.dumps(record.specifications_html),
                    }
                )

            df = pd.DataFrame(data)

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.output_path, index=False, engine="pyarrow")

            logger.success(f"Dataset saved to {self.output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            return False

    def build_and_save(self) -> bool:
        r"""
        Build and save dataset in one step.

        :return: True if successful
        """
        if not self.load_manifest():
            return False

        if not self.build_dataset():
            return False

        if not self.save_dataset():
            return False

        return True
