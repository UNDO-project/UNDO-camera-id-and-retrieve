"""Test validation integration with dataset versioning."""

import json

import pytest
from loguru import logger

from src.pipeline.dataset_builder import DatasetBuilder
from src.storage.versioning import DatasetVersionManager
from src.validation.validator import DatasetValidator

logger.add(lambda msg: print(msg, end=""))


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for testing."""
    return tmp_path


@pytest.fixture
def sample_manifest(tmp_path):
    """Create sample manifest file."""
    manifest = {
        "categories": {
            "Dome cameras": {
                "series": {
                    "AXIS M30 Dome Camera Series": {
                        "products": [
                            {
                                "camera_id": "axis-m3057-plr",
                                "model_name": "AXIS M3057-PLR",
                                "image_urls": [],
                                "datasheet_url": "http://example.com/m3057.pdf",
                                "specifications_html": {
                                    "Camera": {"Resolution": "6 MP"}
                                },
                            }
                        ]
                    }
                }
            }
        }
    }

    manifest_path = tmp_path / "verification_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path


def test_validate_with_version_info(temp_dir, sample_manifest):
    """Test validation with version information."""
    output_path = temp_dir / "products.parquet"

    # Create version 1
    builder = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        version_mode="auto",
        force=True,
    )
    builder.build_and_save()

    # Get version info
    version_manager = DatasetVersionManager(temp_dir)
    version_info = version_manager.get_version_info(1)

    # Validate with version info
    versioned_path = version_manager.get_version_path(1)
    versioned_manifest = temp_dir / version_info["manifest_file"]

    validator = DatasetValidator(
        parquet_path=versioned_path,
        manifest_path=versioned_manifest,
        version_number=1,
        version_info=version_info,
    )

    # Check version info was stored
    assert validator.version_number == 1
    assert validator.version_info is not None
    assert validator.version_info["version"] == 1
    assert validator.version_info["record_count"] == 1


def test_validate_without_version_info(temp_dir, sample_manifest):
    """Test validation without version information (backward compatibility)."""
    output_path = temp_dir / "products.parquet"

    # Create dataset without versioning
    builder = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        version_mode="none",
        force=True,
    )
    builder.build_and_save()

    # Validate without version info
    validator = DatasetValidator(
        parquet_path=output_path,
        manifest_path=sample_manifest,
    )

    # Check version info is None
    assert validator.version_number is None
    assert validator.version_info is None


def test_validation_report_with_version(temp_dir, sample_manifest, capsys):
    """Test that validation report displays version information."""
    output_path = temp_dir / "products.parquet"

    # Create version 1
    builder = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        version_mode="auto",
        force=True,
    )
    builder.build_and_save()

    # Get version info
    version_manager = DatasetVersionManager(temp_dir)
    version_info = version_manager.get_version_info(1)

    # Validate with version info
    versioned_path = version_manager.get_version_path(1)
    versioned_manifest = temp_dir / version_info["manifest_file"]

    validator = DatasetValidator(
        parquet_path=versioned_path,
        manifest_path=versioned_manifest,
        version_number=1,
        version_info=version_info,
    )

    success = validator.validate_all(verbose=False, project_root=temp_dir.parent)
    assert success

    # Capture printed output
    captured = capsys.readouterr()

    # Check version info appears in report
    assert "VERSION INFORMATION" in captured.out
    assert "Version: 1" in captured.out
    assert "Timestamp:" in captured.out
    assert "Record Count: 1" in captured.out
    assert "Manifest Hash:" in captured.out
    assert "Append Mode: False" in captured.out


def test_validation_report_without_version(temp_dir, sample_manifest, capsys):
    """Test that validation report works without version information."""
    output_path = temp_dir / "products.parquet"

    # Create dataset without versioning
    builder = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        version_mode="none",
        force=True,
    )
    builder.build_and_save()

    # Validate without version info
    validator = DatasetValidator(
        parquet_path=output_path,
        manifest_path=sample_manifest,
    )

    success = validator.validate_all(verbose=False, project_root=temp_dir.parent)
    assert success

    # Capture printed output
    captured = capsys.readouterr()

    # Check version info does NOT appear in report
    assert "VERSION INFORMATION" not in captured.out


def test_validation_with_append_version(temp_dir, sample_manifest):
    """Test validation report displays append mode information."""
    output_path = temp_dir / "products.parquet"

    # Create version 1
    builder1 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        version_mode="auto",
        force=True,
    )
    builder1.build_and_save()

    # Create version 2 with append
    builder2 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        append=True,
        merge_strategy="update",
        version_mode="auto",
        force=True,
    )
    builder2.build_and_save()

    # Get version 2 info
    version_manager = DatasetVersionManager(temp_dir)
    version_info = version_manager.get_version_info(2)

    # Validate version 2
    versioned_path = version_manager.get_version_path(2)
    versioned_manifest = temp_dir / version_info["manifest_file"]

    validator = DatasetValidator(
        parquet_path=versioned_path,
        manifest_path=versioned_manifest,
        version_number=2,
        version_info=version_info,
    )

    # Check version info includes append details
    assert validator.version_info["append_mode"] is True
    assert validator.version_info["parent_version"] == 1
    assert "records_added" in validator.version_info
    assert "records_updated" in validator.version_info


def test_validate_specific_version_multiple_versions(temp_dir, sample_manifest):
    """Test validating a specific version when multiple versions exist."""
    output_path = temp_dir / "products.parquet"

    # Create 3 versions
    for i in range(3):
        builder = DatasetBuilder(
            manifest_path=sample_manifest,
            output_path=output_path,
            version_mode="auto",
            force=True,
        )
        builder.build_and_save()

    # Validate version 2 specifically
    version_manager = DatasetVersionManager(temp_dir)
    version_info = version_manager.get_version_info(2)

    versioned_path = version_manager.get_version_path(2)
    versioned_manifest = temp_dir / version_info["manifest_file"]

    validator = DatasetValidator(
        parquet_path=versioned_path,
        manifest_path=versioned_manifest,
        version_number=2,
        version_info=version_info,
    )

    assert validator.version_number == 2
    assert validator.version_info["version"] == 2

    # Validate successfully
    success = validator.validate_all(verbose=False, project_root=temp_dir.parent)
    assert success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
