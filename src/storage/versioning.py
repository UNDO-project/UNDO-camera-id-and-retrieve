"""Dataset versioning and metadata tracking."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


class DatasetVersionManager:
    r"""
    Manages dataset versioning and metadata tracking.

    Responsibilities:
    - Create new dataset versions
    - Track version metadata (timestamps, record counts, manifest info)
    - Manage symlinks to current version
    - List and query version history
    - Clean up old versions (manual)

    :ivar output_dir: Directory where datasets and metadata are stored
    :ivar metadata_path: Path to dataset_metadata.json
    """

    def __init__(self, output_dir: Path | str) -> None:
        r"""
        Initialize DatasetVersionManager.

        :param output_dir: Directory where versioned datasets will be stored
        """
        self.output_dir = Path(output_dir)
        self.metadata_path = self.output_dir / "dataset_metadata.json"

    def load_metadata(self) -> dict:
        r"""
        Load version metadata from JSON file.

        :return: Metadata dictionary with version history
        """
        if not self.metadata_path.exists():
            return {"current_version": 0, "versions": []}

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {"current_version": 0, "versions": []}

    def save_metadata(self, metadata: dict) -> None:
        r"""
        Save version metadata to JSON file.

        :param metadata: Metadata dictionary to save
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.debug(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise

    def get_current_version(self) -> int:
        r"""
        Get current version number.

        :return: Current version number (0 if no versions exist)
        """
        metadata = self.load_metadata()
        return metadata.get("current_version", 0)

    @staticmethod
    def _build_version_info(
        version: int,
        record_count: int,
        versioned_manifest: Path,
        manifest_hash: str,
        append_mode: bool,
        merge_strategy: str | None,
        records_added: int,
        records_updated: int,
        parent_version: int | None,
    ) -> dict:
        r"""
        Build version metadata dictionary.

        :param version: Version number
        :param record_count: Total number of records
        :param versioned_manifest: Path to versioned manifest file
        :param manifest_hash: SHA-256 hash of manifest file
        :param append_mode: Whether this version was created in append mode
        :param merge_strategy: Merge strategy used
        :param records_added: Number of records added (if append mode)
        :param records_updated: Number of records updated (if append mode)
        :param parent_version: Parent version number (if append mode)
        :return: Version info dictionary
        """
        version_info = {
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "file": f"products_v{version}.parquet",
            "record_count": record_count,
            "manifest_file": versioned_manifest.name,
            "manifest_hash": manifest_hash,
            "created_by": "build_dataset.py",
            "append_mode": append_mode,
            "merge_strategy": merge_strategy,
        }

        # Add append-specific stats
        if append_mode:
            version_info["records_added"] = records_added
            version_info["records_updated"] = records_updated
            if parent_version is not None:
                version_info["parent_version"] = parent_version

        return version_info

    def _update_and_save_metadata(self, metadata: dict, version_info: dict) -> None:
        r"""
        Update metadata with new version and save to disk.

        :param metadata: Current metadata dictionary
        :param version_info: New version info to add
        """
        metadata["current_version"] = version_info["version"]
        metadata["versions"].append(version_info)
        self.save_metadata(metadata)

    def create_version(
        self,
        record_count: int,
        manifest_path: Path,
        append_mode: bool,
        merge_strategy: str | None,
        records_added: int = 0,
        records_updated: int = 0,
    ) -> int:
        r"""
        Create new dataset version and update metadata.

        Orchestrates the version creation process:
        1. Load existing metadata and compute new version number
        2. Copy and version the manifest file
        3. Compute manifest file hash
        4. Build version info dictionary
        5. Update and save metadata

        :param record_count: Total number of records in the new version
        :param manifest_path: Path to the manifest used for this version
        :param append_mode: Whether this version was created in append mode
        :param merge_strategy: Merge strategy used (update/skip/error or None)
        :param records_added: Number of records added (if append mode)
        :param records_updated: Number of records updated (if append mode)
        :return: New version number
        """
        # Load metadata and compute new version
        metadata = self.load_metadata()
        new_version = metadata["current_version"] + 1

        # Copy and version the manifest
        versioned_manifest = self.copy_and_version_manifest(manifest_path, new_version)
        manifest_hash = self._compute_file_hash(versioned_manifest)

        # Determine parent version for append mode
        parent_version = (
            metadata["current_version"]
            if append_mode and metadata["versions"]
            else None
        )

        # Build version info
        version_info = self._build_version_info(
            version=new_version,
            record_count=record_count,
            versioned_manifest=versioned_manifest,
            manifest_hash=manifest_hash,
            append_mode=append_mode,
            merge_strategy=merge_strategy,
            records_added=records_added,
            records_updated=records_updated,
            parent_version=parent_version,
        )

        # Update and save metadata
        self._update_and_save_metadata(metadata, version_info)

        logger.info(f"Created version {new_version} with {record_count} records")
        return new_version

    def get_version_path(self, version: int) -> Path:
        r"""
        Get path to specific version's parquet file.

        :param version: Version number
        :return: Path to versioned parquet file
        """
        return self.output_dir / f"products_v{version}.parquet"

    def get_version_info(self, version: int) -> dict | None:
        r"""
        Get metadata for specific version.

        :param version: Version number
        :return: Version metadata dictionary or None if not found
        """
        metadata = self.load_metadata()
        for version_info in metadata["versions"]:
            if version_info["version"] == version:
                return version_info
        return None

    def list_versions(self) -> list[dict]:
        r"""
        List all versions with metadata.

        :return: List of version metadata dictionaries
        """
        metadata = self.load_metadata()
        return metadata.get("versions", [])

    def update_symlinks(self, version: int) -> None:
        r"""
        Update products.parquet and verification_manifest.json symlinks.

        Creates symlinks pointing to the specified version's files.
        On Windows, falls back to file copy if symlink creation fails.

        :param version: Version number to link to
        """
        version_info = self.get_version_info(version)
        if not version_info:
            logger.error(f"Version {version} not found")
            return

        # Dataset symlink
        dataset_symlink = self.output_dir / "products.parquet"
        dataset_target = self.output_dir / version_info["file"]

        # Manifest symlink
        manifest_symlink = self.output_dir / "verification_manifest.json"
        manifest_target = self.output_dir / version_info["manifest_file"]

        # Update dataset symlink
        self._create_symlink(dataset_target, dataset_symlink, "dataset")

        # Update manifest symlink
        self._create_symlink(manifest_target, manifest_symlink, "manifest")

        logger.info(f"Symlinks updated to version {version}")

    def _create_symlink(self, target: Path, link: Path, description: str) -> None:
        r"""
        Create a symlink, with fallback to file copy on Windows.

        :param target: Target file (what the symlink points to)
        :param link: Symlink path (the symlink itself)
        :param description: Description for logging (e.g., "dataset", "manifest")
        """
        # Remove existing symlink or file
        if link.exists() or link.is_symlink():
            link.unlink()

        try:
            # Try to create symlink (Unix/Linux/MacOS)
            link.symlink_to(target.name)  # Use relative path
            logger.debug(f"Created {description} symlink: {link} -> {target.name}")
        except (OSError, NotImplementedError):
            # Fallback to file copy (Windows or filesystems without symlink support)
            shutil.copy2(target, link)
            logger.warning(
                f"Symlink not supported, copied {description} file instead: {link}"
            )

    def copy_and_version_manifest(self, source_manifest: Path, version: int) -> Path:
        r"""
        Copy manifest to versioned filename.

        :param source_manifest: Current manifest path
        :param version: Version number
        :return: Path to versioned manifest
        """
        versioned_manifest = self.output_dir / f"verification_manifest_v{version}.json"

        if not source_manifest.exists():
            logger.warning(f"Source manifest not found: {source_manifest}")
            return versioned_manifest

        try:
            shutil.copy2(source_manifest, versioned_manifest)
            logger.debug(f"Copied manifest to {versioned_manifest}")
        except Exception as e:
            logger.error(f"Failed to copy manifest: {e}")
            raise

        return versioned_manifest

    def cleanup_old_versions(self, keep_last_n: int) -> None:
        r"""
        Delete old dataset and manifest versions (manual cleanup).

        Keeps the last N versions and deletes older ones.

        :param keep_last_n: Number of recent versions to preserve
        """
        metadata = self.load_metadata()
        versions = metadata.get("versions", [])

        if len(versions) <= keep_last_n:
            logger.info(f"Only {len(versions)} versions exist, nothing to clean up")
            return

        # Sort versions by version number
        versions.sort(key=lambda v: v["version"])

        # Determine which versions to delete
        versions_to_delete = versions[:-keep_last_n]

        deleted_count = 0
        for version_info in versions_to_delete:
            version_num = version_info["version"]

            # Delete dataset file
            dataset_file = self.output_dir / version_info["file"]
            if dataset_file.exists():
                dataset_file.unlink()
                logger.debug(f"Deleted dataset: {dataset_file}")
                deleted_count += 1

            # Delete manifest file
            manifest_file = self.output_dir / version_info["manifest_file"]
            if manifest_file.exists():
                manifest_file.unlink()
                logger.debug(f"Deleted manifest: {manifest_file}")

            # Remove from metadata
            metadata["versions"] = [
                v for v in metadata["versions"] if v["version"] != version_num
            ]

        # Save updated metadata
        self.save_metadata(metadata)

        logger.info(f"Cleaned up {deleted_count} old versions, kept last {keep_last_n}")

    def _compute_file_hash(self, file_path: Path) -> str:
        r"""
        Compute SHA-256 hash of a file.

        :param file_path: Path to file
        :return: Hexadecimal SHA-256 hash string
        """
        if not file_path.exists():
            return ""

        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return f"sha256:{sha256_hash.hexdigest()}"
        except Exception as e:
            logger.error(f"Failed to compute hash for {file_path}: {e}")
            return ""
