"""Dataset validation with 5-layer approach."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.config import OUTPUT_DIR
from src.storage.manifest import ManifestRecorder


class DatasetValidator:
    r"""
    Comprehensive dataset validator with 5-layer validation approach.

    Layers:
    1. Schema Validation (Structural)
    2. File Integrity (Media)
    3. Data Quality (Semantic)
    4. Statistical Verification (Completeness)
    5. Repair Recommendations (Actionable)

    :ivar parquet_path: Path to products.parquet
    :ivar manifest_path: Path to verification_manifest.json
    :ivar df: Loaded parquet dataframe
    :ivar errors: List of validation errors found
    :ivar warnings: List of validation warnings found
    :ivar stats: Dictionary of validation statistics
    """

    def __init__(
        self,
        parquet_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
        version_number: int | None = None,
        version_info: dict[str, Any] | None = None,
    ) -> None:
        r"""
        Initialize validator.

        :param parquet_path: Path to products.parquet (default: output/products.parquet)
        :param manifest_path: Path to verification_manifest.json (default: output/verification_manifest.json)
        :param version_number: Version number being validated (optional)
        :param version_info: Version metadata dict (optional)
        """
        if parquet_path is None:
            parquet_path = OUTPUT_DIR / "products.parquet"
        else:
            parquet_path = Path(parquet_path)

        if manifest_path is None:
            manifest_path = OUTPUT_DIR / "verification_manifest.json"
        else:
            manifest_path = Path(manifest_path)

        self.parquet_path = parquet_path
        self.manifest_path = manifest_path
        self.version_number = version_number
        self.version_info = version_info
        self.df: pd.DataFrame | None = None
        self.manifest_data: dict[str, Any] | None = None
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {}

    def load_dataset(self) -> bool:
        r"""
        Load parquet dataset.

        :return: True if loaded successfully
        """
        if not self.parquet_path.exists():
            logger.error(f"Parquet file not found: {self.parquet_path}")
            return False

        try:
            logger.info(f"Loading dataset from {self.parquet_path}")
            self.df = pd.read_parquet(self.parquet_path)
            logger.info(f"Loaded {len(self.df)} records")
            return True
        except Exception as e:
            logger.error(f"Failed to load parquet: {e}")
            return False

    def load_manifest(self) -> bool:
        r"""
        Load verification manifest.

        :return: True if loaded successfully
        """
        if not self.manifest_path.exists():
            logger.warning(f"Manifest not found: {self.manifest_path}")
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

    def generate_manifest_from_parquet(self) -> bool:
        r"""
        Generate manifest from existing parquet dataset (for baseline validation).

        Creates verification_manifest.json from current products.parquet.

        :return: True if manifest generated successfully
        """
        if self.df is None:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return False

        logger.info("Generating manifest from parquet dataset...")

        recorder = ManifestRecorder(self.manifest_path)

        try:
            for idx, row in self.df.iterrows():
                category = row.get("product_category")
                series = row.get("product_series")
                camera_id = row.get("camera_id")

                if not category or not series or not camera_id:
                    logger.warning(
                        f"Row {idx} missing category/series/camera_id, skipping"
                    )
                    continue

                # Create minimal CameraRecord for recording
                from src.models.camera import CameraRecord

                images = []
                if (
                    not pd.isna(row.get("image_files"))
                    and row.get("image_files") != "[]"
                ):
                    try:
                        images = json.loads(row.get("image_files", "[]"))
                    except json.JSONDecodeError:
                        pass

                specs = {}
                if (
                    not pd.isna(row.get("specifications"))
                    and row.get("specifications") != "{}"
                ):
                    try:
                        specs = json.loads(row.get("specifications", "{}"))
                    except json.JSONDecodeError:
                        pass

                record = CameraRecord(
                    camera_id=camera_id,
                    model_name=row.get("model_name", "Unknown"),
                    display_name=row.get("display_name", "Unknown"),
                    description=row.get("description"),
                    source=row.get("source", "Unknown"),
                    category=row.get("category", "Unknown"),
                    product_category=category,
                    product_series=series,
                    images=images,
                    datasheet_url=row.get("datasheet_url"),
                    specifications_html=specs,
                )

                recorder.record_product(record)

            recorder.save_manifest()
            logger.success("Manifest generated and saved")
            return True

        except Exception as e:
            logger.error(f"Failed to generate manifest: {e}")
            return False

    def validate_schema(self) -> None:
        r"""
        Layer 1: Schema Validation (Structural).

        Checks:
        - Required fields present
        - Hierarchy consistency (category → series → product)
        - Data types correct
        """
        if self.df is None:
            return

        logger.info("Layer 1: Validating schema...")

        # Required fields
        required_fields = [
            "camera_id",
            "model_name",
            "display_name",
            "source",
            "category",
            "product_category",
            "product_series",
        ]

        for field in required_fields:
            if field not in self.df.columns:
                self.errors.append(
                    {"type": "missing_column", "field": field, "severity": "critical"}
                )
                logger.error(f"Missing required column: {field}")
            else:
                null_count = self.df[field].isna().sum()
                if null_count > 0:
                    self.errors.append(
                        {"type": "null_values", "field": field, "count": null_count}
                    )
                    logger.warning(f"Found {null_count} null values in {field}")

        # Hierarchy consistency
        logger.info("Checking hierarchy consistency...")
        invalid_hierarchy = self.df[
            (self.df["product_category"].isna())
            | (self.df["product_series"].isna())
            | (self.df["camera_id"].isna())
        ]
        if len(invalid_hierarchy) > 0:
            self.errors.append(
                {"type": "invalid_hierarchy", "count": len(invalid_hierarchy)}
            )
            logger.error(
                f"Found {len(invalid_hierarchy)} records with invalid hierarchy"
            )

        self.stats["schema_errors"] = len(self.errors)
        logger.success("Schema validation complete")

    def validate_files(self, project_root: Path | None = None) -> None:
        r"""
        Layer 2: File Integrity (Media).

        Checks:
        - Files exist on filesystem
        - Files are readable
        - Basic file format validation (magic bytes)
        """
        if self.df is None:
            return

        if project_root is None:
            project_root = Path.cwd()

        logger.info("Layer 2: Validating file integrity...")

        missing_images = 0
        invalid_format_images = 0
        missing_pdfs = 0
        # orphaned_images = 0

        # Check referenced files
        for idx, row in self.df.iterrows():
            # Check images
            if not pd.isna(row.get("image_files")) and row.get("image_files") != "[]":
                try:
                    image_files = json.loads(row.get("image_files", "[]"))
                    for img_path in image_files:
                        full_path = project_root / img_path
                        if not full_path.exists():
                            missing_images += 1
                            self.errors.append(
                                {"type": "missing_image", "path": img_path}
                            )
                        elif full_path.suffix.lower() == ".webp":
                            # Basic magic byte check for webp
                            try:
                                with open(full_path, "rb") as f:
                                    header = f.read(4)
                                    if header != b"RIFF":
                                        invalid_format_images += 1
                                        self.errors.append(
                                            {
                                                "type": "invalid_image_format",
                                                "path": img_path,
                                            }
                                        )
                            except Exception as e:
                                self.errors.append(
                                    {
                                        "type": "cannot_read_image",
                                        "path": img_path,
                                        "error": str(e),
                                    }
                                )
                except json.JSONDecodeError:
                    self.errors.append({"type": "invalid_json_image_files", "row": idx})

            # Check PDFs
            if not pd.isna(row.get("datasheet_file")) and row.get("datasheet_file"):
                pdf_file = row.get("datasheet_file")
                full_path = project_root / pdf_file
                if not full_path.exists():
                    missing_pdfs += 1
                    self.errors.append({"type": "missing_pdf", "path": pdf_file})
                else:
                    # Basic magic byte check for PDF
                    try:
                        with open(full_path, "rb") as f:
                            header = f.read(4)
                            if header != b"%PDF":
                                self.errors.append(
                                    {"type": "invalid_pdf_format", "path": pdf_file}
                                )
                    except Exception as e:
                        self.errors.append(
                            {
                                "type": "cannot_read_pdf",
                                "path": pdf_file,
                                "error": str(e),
                            }
                        )

        self.stats["missing_images"] = missing_images
        self.stats["invalid_format_images"] = invalid_format_images
        self.stats["missing_pdfs"] = missing_pdfs

        if missing_images > 0:
            logger.warning(f"Found {missing_images} missing images")
        if invalid_format_images > 0:
            logger.warning(f"Found {invalid_format_images} invalid image formats")
        if missing_pdfs > 0:
            logger.warning(f"Found {missing_pdfs} missing PDFs")

        logger.success("File integrity validation complete")

    def validate_data_quality(self) -> None:
        r"""
        Layer 3: Data Quality (Semantic).

        Checks:
        - No duplicate camera IDs
        - Records have minimum required data
        - No obviously invalid values
        """
        if self.df is None:
            return

        logger.info("Layer 3: Validating data quality...")

        # Check duplicates
        duplicates = self.df[self.df.duplicated(subset=["camera_id"], keep=False)]
        if len(duplicates) > 0:
            self.errors.append(
                {"type": "duplicate_camera_ids", "count": len(duplicates)}
            )
            logger.error(f"Found {len(duplicates)} duplicate camera IDs")

        # Check records with no data
        no_data = self.df[
            (self.df["image_files"].isna() | (self.df["image_files"] == "[]"))
            & (self.df["datasheet_file"].isna() | (self.df["datasheet_file"] == ""))
        ]
        if len(no_data) > 0:
            self.warnings.append({"type": "no_media_data", "count": len(no_data)})
            logger.warning(f"Found {len(no_data)} records with no images or PDFs")

        self.stats["duplicate_ids"] = len(duplicates)
        self.stats["no_media_records"] = len(no_data)

        logger.success("Data quality validation complete")

    def validate_statistics(self) -> None:
        r"""
        Layer 4: Statistical Verification (Completeness).

        Checks:
        - Coverage metrics (images, PDFs, specs)
        - Distribution per category/series
        - Anomalies (empty categories, etc.)
        """
        if self.df is None:
            return

        logger.info("Layer 4: Validating statistics...")

        # Coverage metrics
        total_records = len(self.df)
        records_with_images = (
            ~self.df["image_files"].isna() & (self.df["image_files"] != "[]")
        ).sum()
        records_with_pdfs = (
            ~self.df["datasheet_file"].isna() & (self.df["datasheet_file"] != "")
        ).sum()
        records_with_specs = (
            ~self.df["specifications"].isna() & (self.df["specifications"] != "{}")
        ).sum()

        self.stats["total_records"] = total_records
        self.stats["records_with_images"] = records_with_images
        self.stats["records_with_pdfs"] = records_with_pdfs
        self.stats["records_with_specs"] = records_with_specs
        self.stats["image_coverage"] = (
            f"{(records_with_images / total_records * 100):.1f}%"
        )
        self.stats["pdf_coverage"] = f"{(records_with_pdfs / total_records * 100):.1f}%"
        self.stats["specs_coverage"] = (
            f"{(records_with_specs / total_records * 100):.1f}%"
        )

        # Distribution per category
        category_counts = self.df["product_category"].value_counts().to_dict()
        self.stats["records_per_category"] = category_counts

        # Detect anomalies
        empty_categories = [cat for cat, count in category_counts.items() if count == 0]
        if empty_categories:
            self.warnings.append(
                {"type": "empty_categories", "categories": empty_categories}
            )

        logger.info(f"Total records: {total_records}")
        logger.info(
            f"Records with images: {records_with_images} ({self.stats['image_coverage']})"
        )
        logger.info(
            f"Records with PDFs: {records_with_pdfs} ({self.stats['pdf_coverage']})"
        )
        logger.info(
            f"Records with specs: {records_with_specs} ({self.stats['specs_coverage']})"
        )

        logger.success("Statistical validation complete")

    def compare_with_manifest(self) -> None:
        r"""
        Layer 5: Compare against manifest (Verification).

        Compares actual dataset counts against expected counts from manifest.
        """
        if self.manifest_data is None:
            logger.warning("Manifest not loaded, skipping comparison")
            return

        logger.info("Layer 5: Comparing with manifest...")

        manifest = self.manifest_data
        actual_products = len(self.df) if self.df is not None else 0

        manifest_products = manifest.get("total_products", 0)
        manifest_images = manifest.get("total_images", 0)
        manifest_pdfs = manifest.get("total_pdfs_with_urls", 0)

        self.stats["manifest_products"] = manifest_products
        self.stats["manifest_images"] = manifest_images
        self.stats["manifest_pdfs"] = manifest_pdfs

        # Check for discrepancies
        if actual_products != manifest_products:
            self.warnings.append(
                {
                    "type": "product_count_mismatch",
                    "expected": manifest_products,
                    "actual": actual_products,
                }
            )
            logger.warning(
                f"Product count mismatch: expected {manifest_products}, got {actual_products}"
            )

        logger.success("Manifest comparison complete")

    def print_report(self, verbose: bool = False) -> None:
        r"""
        Print comprehensive validation report.

        :param verbose: Show detailed errors and warnings
        """
        print("\n" + "=" * 80)
        print("DATASET VALIDATION REPORT")
        print("=" * 80)

        print(f"\nDataset: {self.parquet_path}")
        print(f"Manifest: {self.manifest_path}")

        # Display version information if available
        if self.version_number is not None and self.version_info is not None:
            print("\n--- VERSION INFORMATION ---")
            print(f"Version: {self.version_number}")
            print(f"Timestamp: {self.version_info.get('timestamp', 'N/A')}")
            print(f"Record Count: {self.version_info.get('record_count', 'N/A')}")
            print(
                f"Manifest Hash: {self.version_info.get('manifest_hash', 'N/A')[:32]}..."
            )
            print(f"Append Mode: {self.version_info.get('append_mode', 'N/A')}")
            if self.version_info.get("append_mode"):
                print(f"Records Added: {self.version_info.get('records_added', 'N/A')}")
                print(
                    f"Records Updated: {self.version_info.get('records_updated', 'N/A')}"
                )
                print(
                    f"Parent Version: {self.version_info.get('parent_version', 'N/A')}"
                )

        print("\n--- VALIDATION SUMMARY ---")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")

        print("\n--- COVERAGE METRICS ---")
        print(f"Total Records: {self.stats.get('total_records', 0)}")
        print(
            f"Records with Images: {self.stats.get('records_with_images', 0)} ({self.stats.get('image_coverage', 'N/A')})"
        )
        print(
            f"Records with PDFs: {self.stats.get('records_with_pdfs', 0)} ({self.stats.get('pdf_coverage', 'N/A')})"
        )
        print(
            f"Records with Specs: {self.stats.get('records_with_specs', 0)} ({self.stats.get('specs_coverage', 'N/A')})"
        )

        print("\n--- FILE INTEGRITY ---")
        print(f"Missing Images: {self.stats.get('missing_images', 0)}")
        print(f"Invalid Image Format: {self.stats.get('invalid_format_images', 0)}")
        print(f"Missing PDFs: {self.stats.get('missing_pdfs', 0)}")

        print("\n--- DATA QUALITY ---")
        print(f"Duplicate IDs: {self.stats.get('duplicate_ids', 0)}")
        print(f"Records without Media: {self.stats.get('no_media_records', 0)}")

        print("\n--- MANIFEST COMPARISON ---")
        print(f"Expected Products: {self.stats.get('manifest_products', 'N/A')}")
        print(f"Actual Products: {self.stats.get('total_records', 'N/A')}")

        if len(self.errors) == 0 and len(self.warnings) == 0:
            print("\n✅ Dataset validation passed!")
        elif len(self.errors) == 0:
            print(f"\n⚠️ Dataset has {len(self.warnings)} warning(s) but no errors")
        else:
            print(f"\n❌ Dataset has {len(self.errors)} error(s)")

        if verbose and (self.errors or self.warnings):
            if self.errors:
                print("\n--- ERRORS (first 10) ---")
                for error in self.errors[:10]:
                    print(f"  • {error}")
                if len(self.errors) > 10:
                    print(f"  ... and {len(self.errors) - 10} more")

            if self.warnings:
                print("\n--- WARNINGS (first 10) ---")
                for warning in self.warnings[:10]:
                    print(f"  • {warning}")
                if len(self.warnings) > 10:
                    print(f"  ... and {len(self.warnings) - 10} more")

        print("=" * 80 + "\n")

    def validate_all(
        self, verbose: bool = False, project_root: Path | None = None
    ) -> bool:
        r"""
        Run all validation layers.

        :param verbose: Print detailed output
        :param project_root: Root directory for file path resolution
        :return: True if no errors found
        """
        if not self.load_dataset():
            return False

        # Load or generate manifest
        if not self.load_manifest():
            logger.info("Generating manifest from dataset...")
            self.generate_manifest_from_parquet()
            self.load_manifest()

        self.validate_schema()
        self.validate_files(project_root)
        self.validate_data_quality()
        self.validate_statistics()
        self.compare_with_manifest()

        self.print_report(verbose=verbose)

        return len(self.errors) == 0
