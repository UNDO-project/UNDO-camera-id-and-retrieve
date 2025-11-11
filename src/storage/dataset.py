"""Dataset management and parquet serialization."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.config import IMAGES_DIR, OUTPUT_DIR, PDFS_DIR
from src.models.camera import CameraRecord
from src.storage.download_cache import DownloadCache


class DatasetManager:
    r"""
    Manages dataset creation, storage, and serialization to parquet format.

    Handles:
    - Organizing downloaded images by product ID
    - Tracking PDF file locations
    - Building and serializing parquet dataset with file paths
    \n
    :ivar dataset_path: Path to the output parquet file
    :ivar records: List of CameraRecord objects to be serialized
    """

    def __init__(self, dataset_path: Path | str | None = None) -> None:
        r"""
        Initialize DatasetManager.

        :param dataset_path: Path where parquet file will be saved
        """
        if dataset_path is None:
            dataset_path = OUTPUT_DIR / "products.parquet"
        else:
            dataset_path = Path(dataset_path)

        self.dataset_path = dataset_path
        self.records: list[CameraRecord] = []

    def add_record(self, record: CameraRecord) -> None:
        r"""
        Add a camera record to the dataset.

        :param record: CameraRecord to add
        """
        self.records.append(record)

    @staticmethod
    def organize_images(
        record: CameraRecord,
        image_data_list: list[tuple[str, bytes] | bytes],
        download_cache: DownloadCache | None = None,
    ) -> list[str]:
        r"""
        Save downloaded images and return local file paths.

        Creates directory structure: images/{category}/{series}/{product_id}/{index}.webp

        :param record: CameraRecord with camera_id, product_category, product_series
        :param image_data_list: List of downloaded image bytes or tuples of (url, bytes)
        :param download_cache: Optional cache to mark downloads as completed
        :return: List of local file paths relative to project root
        """
        if not image_data_list:
            return []

        # Normalize category and series names for directory structure
        category_dir = record.product_category.upper().replace(" ", "_")
        series_dir = record.product_series.replace(" ", "_")

        product_dir = IMAGES_DIR / category_dir / series_dir / record.camera_id
        product_dir.mkdir(parents=True, exist_ok=True)

        local_paths = []
        for idx, item in enumerate(image_data_list):
            # Handle both old format (bytes) and new format (url, bytes)
            if isinstance(item, tuple):
                image_url, image_data = item
            else:
                image_url = None
                image_data = item

            file_path = product_dir / f"{idx}.webp"
            with open(file_path, "wb") as f:
                f.write(image_data)

            # Record in cache if provided
            if download_cache and image_url:
                download_cache.mark_downloaded(
                    image_url, file_path, "image", file_content=image_data
                )

            # Store relative path for portability
            relative_path = file_path.relative_to(Path.cwd())
            local_paths.append(str(relative_path))
            logger.debug(f"Saved image {idx} for {record.camera_id}")

        return local_paths

    @staticmethod
    def save_pdf(
        record: CameraRecord,
        pdf_data: bytes,
        download_cache: DownloadCache | None = None,
        pdf_url: str | None = None,
    ) -> str:
        r"""
        Save downloaded PDF and return local file path.

        Creates file: pdfs/{category}/{series}/{product_id}.pdf

        :param record: CameraRecord with camera_id, product_category, product_series
        :param pdf_data: Downloaded PDF bytes
        :param download_cache: Optional cache to mark download as completed
        :param pdf_url: URL of the PDF for cache tracking
        :return: Local file path relative to project root
        """
        # Normalize category and series names for directory structure
        category_dir = record.product_category.upper().replace(" ", "_")
        series_dir = record.product_series.replace(" ", "_")

        pdf_parent_dir = PDFS_DIR / category_dir / series_dir
        pdf_parent_dir.mkdir(parents=True, exist_ok=True)

        file_path = pdf_parent_dir / f"{record.camera_id}.pdf"
        with open(file_path, "wb") as f:
            f.write(pdf_data)

        # Record in cache if provided
        if download_cache and pdf_url:
            download_cache.mark_downloaded(
                pdf_url, file_path, "pdf", file_content=pdf_data
            )

        relative_path = file_path.relative_to(Path.cwd())
        logger.debug(f"Saved PDF for {record.camera_id}")
        return str(relative_path)

    @staticmethod
    def _serialize_record(record: CameraRecord) -> dict[str, Any]:
        r"""
        Convert CameraRecord to a dictionary suitable for parquet serialization.

        :param record: CameraRecord to serialize
        :return: Dictionary representation
        """
        return {
            "camera_id": record.camera_id,
            "model_name": record.model_name,
            "display_name": record.display_name,
            "description": record.description,
            "source": record.source,
            "category": record.category,
            "product_category": record.product_category,
            "product_series": record.product_series,
            "image_urls": json.dumps(record.images),
            "image_files": json.dumps(record.image_files),
            "datasheet_url": record.datasheet_url,
            "datasheet_file": record.datasheet_file,
            "specifications": json.dumps(record.specifications_html),
        }

    def save_dataset(self) -> None:
        r"""
        Serialize all records to parquet format.

        Creates a parquet file with columns for all product metadata.
        """
        if not self.records:
            logger.warning("No records to save")
            return

        logger.info(f"Serializing {len(self.records)} records to parquet")

        # Convert records to dictionaries
        data = [self._serialize_record(record) for record in self.records]

        # Create DataFrame
        df = pd.DataFrame(data)

        # Ensure output directory exists
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to parquet
        df.to_parquet(self.dataset_path, index=False, engine="pyarrow")
        logger.info(f"Dataset saved to {self.dataset_path}")

    @staticmethod
    def update_record_with_files(
        record: CameraRecord,
        image_files: list[str] | None = None,
        pdf_file: str | None = None,
    ) -> None:
        r"""
        Update a record with downloaded file paths.

        Should be called after downloading images/PDFs.

        :param record: CameraRecord to update
        :param image_files: List of local image file paths
        :param pdf_file: Local PDF file path
        """
        if image_files:
            record.images = image_files
        if pdf_file:
            record.datasheet_pdf = pdf_file  # type: ignore
