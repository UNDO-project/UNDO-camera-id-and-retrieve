"""Dataset building stage: Create parquet from filesystem data."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.building.merge_strategies import MergeStrategyFactory
from src.config import paths
from src.models.camera import CameraRecord
from src.storage.versioning import DatasetVersionManager


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
        append: bool = False,
        merge_strategy: str = "update",
        force: bool = False,
        version_mode: str = "none",
        version_number: int | None = None,
    ) -> None:
        r"""
        Initialize dataset builder.

        :param manifest_path: Path to verification manifest
        :param output_path: Path to output parquet (default: output/products.parquet)
        :param append: If True, append to existing dataset instead of overwriting
        :param merge_strategy: Strategy for handling duplicate camera_ids (update|skip|error)
        :param force: If True, skip user confirmation prompts
        :param version_mode: Versioning mode (auto|manual|none)
        :param version_number: Manual version number (requires version_mode=manual)
        """
        if manifest_path is None:
            manifest_path = paths.output_dir / "verification_manifest.json"
        else:
            manifest_path = Path(manifest_path)

        if output_path is None:
            output_path = paths.output_dir / "products.parquet"
        else:
            output_path = Path(output_path)

        self.manifest_path = manifest_path
        self.output_path = output_path
        self.records: list[CameraRecord] = []
        self.manifest_data: dict = {}
        self.append = append
        self.merge_strategy = merge_strategy
        self.force = force
        self.version_mode = version_mode
        self.version_number = version_number
        self.merge_stats: dict[str, int] = {
            "records_added": 0,
            "records_updated": 0,
            "records_skipped": 0,
        }

        # Initialize version manager if versioning is enabled
        if self.version_mode != "none":
            self.version_manager = DatasetVersionManager(self.output_path.parent)
        else:
            self.version_manager = None

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

                        # Find image files and detect vendor
                        image_files, vendor_source = self._find_image_files_and_vendor(
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
                        image_urls = product_info.get("image_urls", [])
                        datasheet_url = product_info.get("datasheet_url")
                        record = CameraRecord(
                            camera_id=camera_id,
                            model_name=model_name,
                            display_name=model_name,
                            description=None,
                            specifications={},
                            source=vendor_source or "Unknown",
                            category="Network Camera",
                            product_category=category_name,
                            product_series=series_name,
                            images=image_urls,
                            image_url=None,
                            image_files=image_files,
                            datasheet_url=datasheet_url,
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

    @staticmethod
    def _find_image_files_and_vendor(
        category_name: str, series_name: str, camera_id: str
    ) -> tuple[list[str], str | None]:
        r"""
        Find all image files for a product and detect vendor from filesystem.

        Filesystem structure: data/images/{VENDOR}/{CATEGORY}/{SERIES}/{CAMERA_ID}/

        :param category_name: Product category
        :param series_name: Product series
        :param camera_id: Camera ID
        :return: Tuple of (image file paths, vendor source name)
        """
        category_dir = category_name.upper().replace(" ", "_")
        series_dir = series_name.replace(" ", "_")
        images_dir = paths.data_dir / "images"

        # Vendor name mapping
        vendor_map = {
            "AXIS_COMMUNICATIONS": "Axis Communications",
            "HIKVISION": "HikVision",
        }

        # Search in all vendor directories
        for vendor_dir in images_dir.iterdir():
            if not vendor_dir.is_dir():
                continue

            product_dir = vendor_dir / category_dir / series_dir / camera_id

            if product_dir.exists():
                image_files = []
                for img_file in sorted(product_dir.glob("*.webp")):
                    relative_path = img_file.relative_to(Path.cwd())
                    image_files.append(str(relative_path))

                # Detect vendor from directory name
                vendor_name = vendor_map.get(vendor_dir.name, vendor_dir.name)
                return image_files, vendor_name

        return [], None

    def _find_image_files(
        self, category_name: str, series_name: str, camera_id: str
    ) -> list[str]:
        r"""
        Find all image files for a product in filesystem.

        Filesystem structure: data/images/{VENDOR}/{CATEGORY}/{SERIES}/{CAMERA_ID}/

        :param category_name: Product category
        :param series_name: Product series
        :param camera_id: Camera ID
        :return: List of relative paths to image files
        """
        image_files, _ = self._find_image_files_and_vendor(
            category_name, series_name, camera_id
        )
        return image_files

    @staticmethod
    def _find_pdf_file(
        category_name: str, series_name: str, camera_id: str
    ) -> str | None:
        r"""
        Find PDF file for a product in filesystem.

        Filesystem structure: data/pdfs/{VENDOR}/{CATEGORY}/{SERIES}/{CAMERA_ID}.pdf

        :param category_name: Product category
        :param series_name: Product series
        :param camera_id: Camera ID
        :return: Relative path to PDF or None
        """
        category_dir = category_name.upper().replace(" ", "_")
        series_dir = series_name.replace(" ", "_")
        pdfs_dir = paths.data_dir / "pdfs"

        # Search in all vendor directories
        for vendor_dir in pdfs_dir.iterdir():
            if not vendor_dir.is_dir():
                continue

            pdf_file = vendor_dir / category_dir / series_dir / f"{camera_id}.pdf"

            if pdf_file.exists():
                relative_path = pdf_file.relative_to(Path.cwd())
                return str(relative_path)

        return None

    @staticmethod
    def _load_specifications_from_manifest(
        product_info: dict,
    ) -> dict[str, dict[str, str]]:
        r"""
        Load specifications from product info in manifest.

        :param product_info: Product entry from manifest
        :return: Specifications dictionary (section_name -> {spec_name -> spec_value})
        """
        return product_info.get("specifications_html", {})

    def _confirm_overwrite(self) -> bool:
        r"""
        Prompt user to confirm overwrite of existing dataset.

        :return: True if user confirms or file doesn't exist
        """
        if not self.output_path.exists():
            return True

        if self.force:
            logger.info("Force mode enabled, skipping confirmation")
            return True

        try:
            response = (
                input(f"\nDataset exists at {self.output_path}. Overwrite? [y/N]: ")
                .strip()
                .lower()
            )
            return response in ["y", "yes"]
        except (EOFError, KeyboardInterrupt):
            logger.info("\nOperation cancelled by user")
            return False

    def _load_existing_dataset(self) -> pd.DataFrame | None:
        r"""
        Load existing dataset from parquet file.

        :return: Existing DataFrame, or None if file doesn't exist
        :raises Exception: If file exists but cannot be read
        """
        if not self.output_path.exists():
            return None

        try:
            existing_df = pd.read_parquet(self.output_path)
            logger.info(f"Loaded existing dataset with {len(existing_df)} records")
            return existing_df
        except Exception as e:
            logger.error(f"Failed to read existing dataset: {e}")
            raise

    def _merge_with_existing(self, new_df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Merge new records with existing dataset using configured strategy.

        :param new_df: DataFrame with new records
        :return: Merged DataFrame
        :raises ValueError: If merge_strategy is invalid or duplicates found in error mode
        """
        # Check if existing dataset exists
        existing_df = self._load_existing_dataset()
        if existing_df is None:
            logger.info("No existing dataset found, creating new one")
            self.merge_stats["records_added"] = len(new_df)
            return new_df

        # Get and apply merge strategy
        strategy = MergeStrategyFactory.get_strategy(self.merge_strategy)
        return strategy.merge(new_df, existing_df, self)

    @staticmethod
    def _record_to_dict(record: CameraRecord) -> dict:
        r"""
        Convert single CameraRecord to dictionary.

        :param record: CameraRecord to convert
        :return: Dictionary representation of record
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

    def _serialize_records_to_dataframe(self) -> pd.DataFrame | None:
        r"""
        Convert CameraRecord objects to DataFrame.

        :return: DataFrame with serialized records, or None if no records
        """
        if not self.records:
            logger.warning("No records to save")
            return None

        logger.info(f"Serializing {len(self.records)} records to parquet...")

        try:
            data = [self._record_to_dict(record) for record in self.records]
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to serialize records: {e}")
            return None

    def _apply_merge_strategy(self, df: pd.DataFrame) -> pd.DataFrame | None:
        r"""
        Handle append mode and merge strategies.

        :param df: DataFrame with new records
        :return: DataFrame after applying merge strategy, or None if cancelled
        """
        if self.append:
            logger.info("Append mode enabled, merging with existing dataset")
            return self._merge_with_existing(df)

        # Check if file exists and confirm overwrite
        if not self._confirm_overwrite():
            logger.warning("Operation cancelled by user")
            return None

        self.merge_stats["records_added"] = len(df)
        return df

    def _persist_with_versioning(self, df: pd.DataFrame) -> bool:
        r"""
        Save DataFrame with version management.

        :param df: DataFrame to save
        :return: True if saved successfully
        """
        if self.version_mode == "auto":
            return self._persist_auto_version(df)
        elif self.version_mode == "manual":
            return self._persist_manual_version(df)
        else:
            logger.error(f"Unknown version mode: {self.version_mode}")
            return False

    def _persist_auto_version(self, df: pd.DataFrame) -> bool:
        r"""
        Save DataFrame with automatic version management.

        :param df: DataFrame to save
        :return: True if saved successfully
        """
        version = self.version_manager.create_version(
            record_count=len(df),
            manifest_path=self.manifest_path,
            append_mode=self.append,
            merge_strategy=self.merge_strategy if self.append else None,
            records_added=self.merge_stats["records_added"],
            records_updated=self.merge_stats["records_updated"],
        )
        versioned_path = self.version_manager.get_version_path(version)

        # Save to versioned file
        df.to_parquet(versioned_path, index=False, engine="pyarrow")
        logger.success(f"Dataset saved as version {version}: {versioned_path}")

        # Update symlinks
        self.version_manager.update_symlinks(version)
        logger.info(f"Symlinks updated to version {version}")

        return True

    def _persist_manual_version(self, df: pd.DataFrame) -> bool:
        r"""
        Save DataFrame with manual version number.

        :param df: DataFrame to save
        :return: True if saved successfully
        """
        if self.version_number is None:
            logger.error("Manual version mode requires --version number")
            return False

        # Create version entry (side effect: adds to metadata, but we override version number below)
        self.version_manager.create_version(
            record_count=len(df),
            manifest_path=self.manifest_path,
            append_mode=self.append,
            merge_strategy=self.merge_strategy if self.append else None,
            records_added=self.merge_stats["records_added"],
            records_updated=self.merge_stats["records_updated"],
        )

        # Override version number in metadata with user-specified version
        metadata = self.version_manager.load_metadata()
        metadata["current_version"] = self.version_number
        metadata["versions"][-1]["version"] = self.version_number
        self.version_manager.save_metadata(metadata)

        # Use user-specified version number for file path and symlinks
        versioned_path = self.version_manager.get_version_path(self.version_number)
        df.to_parquet(versioned_path, index=False, engine="pyarrow")
        logger.success(
            f"Dataset saved as version {self.version_number}: {versioned_path}"
        )

        # Update symlinks to user-specified version
        self.version_manager.update_symlinks(self.version_number)
        logger.info(f"Symlinks updated to version {self.version_number}")

        return True

    def _persist_directly(self, df: pd.DataFrame) -> bool:
        r"""
        Save DataFrame directly without versioning.

        :param df: DataFrame to save
        :return: True if saved successfully
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False, engine="pyarrow")
        logger.success(f"Dataset saved to {self.output_path}")
        return True

    def _persist_dataframe(self, df: pd.DataFrame) -> bool:
        r"""
        Save DataFrame to disk with optional versioning.

        :param df: DataFrame to save
        :return: True if saved successfully
        """
        if self.version_manager:
            return self._persist_with_versioning(df)
        return self._persist_directly(df)

    def _log_save_results(self, df: pd.DataFrame) -> None:
        r"""
        Log save operation results.

        :param df: Saved DataFrame
        """
        logger.info(f"Total records in dataset: {len(df)}")
        if self.append:
            logger.info(
                f"Merge statistics - Added: {self.merge_stats['records_added']}, "
                f"Updated: {self.merge_stats['records_updated']}, "
                f"Skipped: {self.merge_stats['records_skipped']}"
            )

    def save_dataset(self) -> bool:
        r"""
        Save dataset to parquet file.

        Orchestrates the save process by delegating to specialized methods:
        1. Serialize records to DataFrame
        2. Apply merge strategy (if append mode)
        3. Persist DataFrame (with or without versioning)
        4. Log results

        :return: True if saved successfully
        """
        try:
            # Step 1: Serialize records to DataFrame
            df = self._serialize_records_to_dataframe()
            if df is None:
                return False

            # Step 2: Apply merge strategy
            df = self._apply_merge_strategy(df)
            if df is None:
                return False

            # Step 3: Persist DataFrame
            if not self._persist_dataframe(df):
                return False

            # Step 4: Log results
            self._log_save_results(df)

            return True

        except ValueError as e:
            # Re-raise merge strategy errors
            logger.error(f"Merge failed: {e}")
            return False
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
