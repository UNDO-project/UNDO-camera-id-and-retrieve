"""Test dataset versioning functionality."""

import json

import pytest
from loguru import logger

from src.building.builder import DatasetBuilder
from src.storage.versioning import DatasetVersionManager

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


def test_version_manager_initialization(temp_dir):
    """Test DatasetVersionManager initialization."""
    vm = DatasetVersionManager(temp_dir)

    assert vm.output_dir == temp_dir
    assert vm.metadata_path == temp_dir / "dataset_metadata.json"


def test_load_metadata_no_file(temp_dir):
    """Test loading metadata when file doesn't exist."""
    vm = DatasetVersionManager(temp_dir)
    metadata = vm.load_metadata()

    assert metadata == {"current_version": 0, "versions": []}


def test_save_and_load_metadata(temp_dir):
    """Test saving and loading metadata."""
    vm = DatasetVersionManager(temp_dir)

    test_metadata = {
        "current_version": 1,
        "versions": [
            {
                "version": 1,
                "timestamp": "2025-01-01T00:00:00Z",
                "file": "products_v1.parquet",
                "record_count": 100,
            }
        ],
    }

    vm.save_metadata(test_metadata)
    loaded = vm.load_metadata()

    assert loaded == test_metadata


def test_get_current_version(temp_dir):
    """Test getting current version number."""
    vm = DatasetVersionManager(temp_dir)

    # No versions
    assert vm.get_current_version() == 0

    # After creating metadata
    vm.save_metadata({"current_version": 5, "versions": []})
    assert vm.get_current_version() == 5


def test_create_version(temp_dir, sample_manifest):
    """Test creating a new version."""
    vm = DatasetVersionManager(temp_dir)

    version = vm.create_version(
        record_count=100,
        manifest_path=sample_manifest,
        append_mode=False,
        merge_strategy=None,
    )

    assert version == 1

    # Verify metadata
    metadata = vm.load_metadata()
    assert metadata["current_version"] == 1
    assert len(metadata["versions"]) == 1

    v = metadata["versions"][0]
    assert v["version"] == 1
    assert v["record_count"] == 100
    assert v["append_mode"] is False
    assert v["merge_strategy"] is None
    assert "timestamp" in v
    assert "manifest_file" in v
    assert "manifest_hash" in v


def test_create_multiple_versions(temp_dir, sample_manifest):
    """Test creating multiple versions."""
    vm = DatasetVersionManager(temp_dir)

    v1 = vm.create_version(
        record_count=100,
        manifest_path=sample_manifest,
        append_mode=False,
        merge_strategy=None,
    )

    v2 = vm.create_version(
        record_count=150,
        manifest_path=sample_manifest,
        append_mode=True,
        merge_strategy="update",
        records_added=30,
        records_updated=20,
    )

    assert v1 == 1
    assert v2 == 2

    metadata = vm.load_metadata()
    assert metadata["current_version"] == 2
    assert len(metadata["versions"]) == 2

    # Check version 2 has append stats
    v2_info = metadata["versions"][1]
    assert v2_info["append_mode"] is True
    assert v2_info["records_added"] == 30
    assert v2_info["records_updated"] == 20
    assert v2_info["parent_version"] == 1


def test_get_version_path(temp_dir):
    """Test getting version file path."""
    vm = DatasetVersionManager(temp_dir)

    path = vm.get_version_path(1)
    assert path == temp_dir / "products_v1.parquet"

    path = vm.get_version_path(5)
    assert path == temp_dir / "products_v5.parquet"


def test_get_version_info(temp_dir, sample_manifest):
    """Test getting version information."""
    vm = DatasetVersionManager(temp_dir)

    vm.create_version(
        record_count=100,
        manifest_path=sample_manifest,
        append_mode=False,
        merge_strategy=None,
    )

    info = vm.get_version_info(1)
    assert info is not None
    assert info["version"] == 1
    assert info["record_count"] == 100

    # Non-existent version
    info = vm.get_version_info(999)
    assert info is None


def test_list_versions(temp_dir, sample_manifest):
    """Test listing all versions."""
    vm = DatasetVersionManager(temp_dir)

    # No versions
    versions = vm.list_versions()
    assert versions == []

    # Create versions
    vm.create_version(100, sample_manifest, False, None)
    vm.create_version(150, sample_manifest, True, "update", 30, 20)

    versions = vm.list_versions()
    assert len(versions) == 2
    assert versions[0]["version"] == 1
    assert versions[1]["version"] == 2


def test_copy_and_version_manifest(temp_dir, sample_manifest):
    """Test copying manifest to versioned file."""
    vm = DatasetVersionManager(temp_dir)

    versioned = vm.copy_and_version_manifest(sample_manifest, 1)

    assert versioned == temp_dir / "verification_manifest_v1.json"
    assert versioned.exists()

    # Verify content matches
    with open(sample_manifest) as f1, open(versioned) as f2:
        assert json.load(f1) == json.load(f2)


def test_cleanup_old_versions(temp_dir, sample_manifest):
    """Test cleanup of old versions."""
    vm = DatasetVersionManager(temp_dir)

    # Create 5 versions
    for i in range(1, 6):
        vm.create_version(
            record_count=100 + i,
            manifest_path=sample_manifest,
            append_mode=False,
            merge_strategy=None,
        )

        # Create dummy dataset file
        version_path = vm.get_version_path(i)
        version_path.write_text("dummy data")

    # Verify 5 versions exist
    assert len(vm.list_versions()) == 5

    # Cleanup, keep last 3
    vm.cleanup_old_versions(3)

    # Verify only 3 versions remain
    versions = vm.list_versions()
    assert len(versions) == 3
    assert [v["version"] for v in versions] == [3, 4, 5]

    # Verify files deleted
    assert not vm.get_version_path(1).exists()
    assert not vm.get_version_path(2).exists()
    assert vm.get_version_path(3).exists()
    assert vm.get_version_path(4).exists()
    assert vm.get_version_path(5).exists()


def test_dataset_builder_with_auto_versioning(sample_manifest, temp_dir):
    """Test DatasetBuilder with auto versioning."""
    output_path = temp_dir / "products.parquet"

    builder = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        version_mode="auto",
        force=True,
    )

    success = builder.build_and_save()
    assert success

    # Check versioned file was created
    versioned_path = temp_dir / "products_v1.parquet"
    assert versioned_path.exists()

    # Check metadata
    vm = DatasetVersionManager(temp_dir)
    assert vm.get_current_version() == 1
    assert len(vm.list_versions()) == 1

    # Check manifest was versioned
    versioned_manifest = temp_dir / "verification_manifest_v1.json"
    assert versioned_manifest.exists()


def test_dataset_builder_versioning_with_append(sample_manifest, temp_dir):
    """Test versioning with append mode."""
    output_path = temp_dir / "products.parquet"

    # Create initial version
    builder1 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        version_mode="auto",
        force=True,
    )
    builder1.build_and_save()

    # Append to create version 2
    builder2 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        append=True,
        merge_strategy="update",
        version_mode="auto",
        force=True,
    )
    builder2.build_and_save()

    # Verify 2 versions exist
    vm = DatasetVersionManager(temp_dir)
    assert vm.get_current_version() == 2
    assert len(vm.list_versions()) == 2

    # Verify version 2 has parent_version
    v2_info = vm.get_version_info(2)
    assert v2_info["parent_version"] == 1
    assert v2_info["append_mode"] is True


def test_dataset_builder_no_versioning(sample_manifest, temp_dir):
    """Test DatasetBuilder without versioning (default)."""
    output_path = temp_dir / "products.parquet"

    builder = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        version_mode="none",
        force=True,
    )

    success = builder.build_and_save()
    assert success

    # Check regular file was created (not versioned)
    assert output_path.exists()

    # Check no versioned files
    versioned_path = temp_dir / "products_v1.parquet"
    assert not versioned_path.exists()

    # Check no metadata
    metadata_path = temp_dir / "dataset_metadata.json"
    assert not metadata_path.exists()


def test_compute_file_hash(temp_dir):
    """Test file hash computation."""
    vm = DatasetVersionManager(temp_dir)

    # Create test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("Hello, World!")

    hash_result = vm._compute_file_hash(test_file)

    assert hash_result.startswith("sha256:")
    assert len(hash_result) > 7  # "sha256:" + hash

    # Same content should produce same hash
    hash_result2 = vm._compute_file_hash(test_file)
    assert hash_result == hash_result2


def test_update_symlinks(temp_dir, sample_manifest):
    """Test symlink creation and updates."""
    vm = DatasetVersionManager(temp_dir)

    # Create version 1
    vm.create_version(100, sample_manifest, False, None)

    # Create dummy dataset file
    version1_path = vm.get_version_path(1)
    version1_path.write_text("version 1 data")

    # Update symlinks
    vm.update_symlinks(1)

    # Check symlinks exist
    dataset_symlink = temp_dir / "products.parquet"
    manifest_symlink = temp_dir / "verification_manifest.json"

    assert dataset_symlink.exists()
    assert manifest_symlink.exists()

    # Verify symlinks point to correct files (or are copies on Windows)
    if dataset_symlink.is_symlink():
        assert dataset_symlink.resolve() == version1_path.resolve()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
