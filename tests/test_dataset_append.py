"""Test dataset append functionality with merge strategies."""

import json

import pandas as pd
import pytest
from loguru import logger

from src.building.builder import DatasetBuilder

logger.add(lambda msg: print(msg, end=""))


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


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
                                "datasheet_url": "http://example.com/datasheet.pdf",
                                "specifications_html": {
                                    "Camera": {"Resolution": "6 MP"}
                                },
                            },
                            {
                                "camera_id": "axis-m3085-v",
                                "model_name": "AXIS M3085-V",
                                "image_urls": [],
                                "datasheet_url": None,
                                "specifications_html": {
                                    "Camera": {"Resolution": "2 MP"}
                                },
                            },
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


@pytest.fixture
def updated_manifest(tmp_path):
    """Create updated manifest with one new camera and one existing camera."""
    manifest = {
        "categories": {
            "Dome cameras": {
                "series": {
                    "AXIS M30 Dome Camera Series": {
                        "products": [
                            {
                                "camera_id": "axis-m3085-v",  # Existing camera with updated specs
                                "model_name": "AXIS M3085-V Updated",
                                "image_urls": [],
                                "datasheet_url": "http://example.com/new_datasheet.pdf",
                                "specifications_html": {
                                    "Camera": {
                                        "Resolution": "4 MP"
                                    }  # Updated from 2 MP
                                },
                            },
                            {
                                "camera_id": "axis-m4000-new",  # New camera
                                "model_name": "AXIS M4000-NEW",
                                "image_urls": [],
                                "datasheet_url": None,
                                "specifications_html": {
                                    "Camera": {"Resolution": "8 MP"}
                                },
                            },
                        ]
                    }
                }
            }
        }
    }

    manifest_path = tmp_path / "updated_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    return manifest_path


def test_merge_strategy_update(sample_manifest, updated_manifest, temp_output_dir):
    """Test update strategy: overwrites duplicates with new data."""
    output_path = temp_output_dir / "products.parquet"

    # Step 1: Create initial dataset
    builder1 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        force=True,
    )
    assert builder1.build_and_save()

    # Verify initial dataset
    df1 = pd.read_parquet(output_path)
    assert len(df1) == 2
    assert set(df1["camera_id"]) == {"axis-m3057-plr", "axis-m3085-v"}

    # Step 2: Append with update strategy
    builder2 = DatasetBuilder(
        manifest_path=updated_manifest,
        output_path=output_path,
        append=True,
        merge_strategy="update",
        force=True,
    )
    assert builder2.build_and_save()

    # Verify merged dataset
    df2 = pd.read_parquet(output_path)
    assert len(df2) == 3  # 2 original + 1 new (axis-m3085-v updated, not added)
    assert set(df2["camera_id"]) == {"axis-m3057-plr", "axis-m3085-v", "axis-m4000-new"}

    # Verify axis-m3085-v was updated
    m3085_row = df2[df2["camera_id"] == "axis-m3085-v"].iloc[0]
    assert m3085_row["model_name"] == "AXIS M3085-V Updated"
    specs = json.loads(m3085_row["specifications"])
    assert specs["Camera"]["Resolution"] == "4 MP"

    # Verify merge stats
    assert builder2.merge_stats["records_added"] == 1  # axis-m4000-new
    assert builder2.merge_stats["records_updated"] == 1  # axis-m3085-v

    logger.success("test_merge_strategy_update passed")


def test_merge_strategy_skip(sample_manifest, updated_manifest, temp_output_dir):
    """Test skip strategy: keeps original data for duplicates."""
    output_path = temp_output_dir / "products.parquet"

    # Step 1: Create initial dataset
    builder1 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        force=True,
    )
    assert builder1.build_and_save()

    # Step 2: Append with skip strategy
    builder2 = DatasetBuilder(
        manifest_path=updated_manifest,
        output_path=output_path,
        append=True,
        merge_strategy="skip",
        force=True,
    )
    assert builder2.build_and_save()

    # Verify merged dataset
    df2 = pd.read_parquet(output_path)
    assert len(df2) == 3

    # Verify axis-m3085-v was NOT updated (original kept)
    m3085_row = df2[df2["camera_id"] == "axis-m3085-v"].iloc[0]
    assert m3085_row["model_name"] == "AXIS M3085-V"  # Original name
    specs = json.loads(m3085_row["specifications"])
    assert specs["Camera"]["Resolution"] == "2 MP"  # Original resolution

    # Verify merge stats
    assert builder2.merge_stats["records_added"] == 1  # axis-m4000-new
    assert builder2.merge_stats["records_skipped"] == 1  # axis-m3085-v

    logger.success("test_merge_strategy_skip passed")


def test_merge_strategy_error(sample_manifest, updated_manifest, temp_output_dir):
    """Test error strategy: fails if duplicates are found."""
    output_path = temp_output_dir / "products.parquet"

    # Step 1: Create initial dataset
    builder1 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        force=True,
    )
    assert builder1.build_and_save()

    # Step 2: Attempt to append with error strategy (should fail)
    builder2 = DatasetBuilder(
        manifest_path=updated_manifest,
        output_path=output_path,
        append=True,
        merge_strategy="error",
        force=True,
    )

    # Should fail because axis-m3085-v is a duplicate
    result = builder2.build_and_save()
    assert result is False

    # Verify original dataset is unchanged
    df = pd.read_parquet(output_path)
    assert len(df) == 2

    logger.success("test_merge_strategy_error passed")


def test_append_to_nonexistent_dataset(sample_manifest, temp_output_dir):
    """Test appending when no existing dataset exists (should create new)."""
    output_path = temp_output_dir / "new_products.parquet"

    builder = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        append=True,
        merge_strategy="update",
        force=True,
    )

    assert builder.build_and_save()

    # Verify dataset was created
    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert len(df) == 2
    assert builder.merge_stats["records_added"] == 2

    logger.success("test_append_to_nonexistent_dataset passed")


def test_overwrite_without_append(sample_manifest, temp_output_dir):
    """Test that overwrite mode replaces the dataset entirely."""
    output_path = temp_output_dir / "products.parquet"

    # Create initial dataset with 2 records
    builder1 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        force=True,
    )
    assert builder1.build_and_save()

    df1 = pd.read_parquet(output_path)
    assert len(df1) == 2

    # Overwrite with same data (should still have 2 records, not 4)
    builder2 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        append=False,  # Overwrite mode
        force=True,
    )
    assert builder2.build_and_save()

    df2 = pd.read_parquet(output_path)
    assert len(df2) == 2  # Not 4!

    logger.success("test_overwrite_without_append passed")


def test_force_flag_skips_confirmation(sample_manifest, temp_output_dir):
    """Test that force flag bypasses user confirmation."""
    output_path = temp_output_dir / "products.parquet"

    # Create initial dataset
    builder1 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        force=True,
    )
    assert builder1.build_and_save()

    # Overwrite with force flag (should not prompt)
    builder2 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        force=True,
    )
    result = builder2.build_and_save()
    assert result is True

    logger.success("test_force_flag_skips_confirmation passed")


def test_merge_stats_tracking(sample_manifest, updated_manifest, temp_output_dir):
    """Test that merge statistics are correctly tracked."""
    output_path = temp_output_dir / "products.parquet"

    # Initial dataset
    builder1 = DatasetBuilder(
        manifest_path=sample_manifest,
        output_path=output_path,
        force=True,
    )
    builder1.build_and_save()

    # Append with update strategy
    builder2 = DatasetBuilder(
        manifest_path=updated_manifest,
        output_path=output_path,
        append=True,
        merge_strategy="update",
        force=True,
    )
    builder2.build_and_save()

    # Verify stats
    stats = builder2.merge_stats
    assert stats["records_added"] == 1  # axis-m4000-new
    assert stats["records_updated"] == 1  # axis-m3085-v
    assert stats["records_skipped"] == 0

    logger.success("test_merge_stats_tracking passed")


def test_empty_manifest_append(temp_output_dir, tmp_path):
    """Test appending an empty manifest (no products)."""
    output_path = temp_output_dir / "products.parquet"

    # Create initial dataset
    initial_manifest = {
        "categories": {
            "Dome cameras": {
                "series": {
                    "AXIS M30 Dome Camera Series": {
                        "products": [
                            {
                                "camera_id": "axis-m3057-plr",
                                "model_name": "AXIS M3057-PLR",
                                "image_urls": [],
                                "datasheet_url": None,
                                "specifications_html": {},
                            }
                        ]
                    }
                }
            }
        }
    }

    manifest_path1 = tmp_path / "manifest1.json"
    with open(manifest_path1, "w") as f:
        json.dump(initial_manifest, f)

    builder1 = DatasetBuilder(
        manifest_path=manifest_path1,
        output_path=output_path,
        force=True,
    )
    builder1.build_and_save()

    # Create empty manifest
    empty_manifest = {"categories": {}}
    manifest_path2 = tmp_path / "manifest2.json"
    with open(manifest_path2, "w") as f:
        json.dump(empty_manifest, f)

    # Attempt to append empty manifest
    builder2 = DatasetBuilder(
        manifest_path=manifest_path2,
        output_path=output_path,
        append=True,
        force=True,
    )
    result = builder2.build_and_save()

    # Should fail because no records to save
    assert result is False

    # Original dataset should remain unchanged
    df = pd.read_parquet(output_path)
    assert len(df) == 1

    logger.success("test_empty_manifest_append passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
