"""Tests for catalog loading and indexing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.identification.catalog import (
    load_catalog,
    iter_reference_image_files,
    _parse_json_field,
    _row_to_camera_record,
)
from src.identification.index import CatalogIndex
from src.models.camera import CameraRecord


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Create a sample parquet file for testing."""
    data = {
        "camera_id": ["axis-m3027", "axis-m3028", "hik-ds2de"],
        "model_name": ["AXIS M3027-PVE", "AXIS M3028-PVE", "DS-2DE2A404IW-DE3"],
        "display_name": ["AXIS M3027-PVE", "AXIS M3028-PVE", "DS-2DE2A404IW-DE3"],
        "description": ["5 MP camera", "8 MP camera", "4 MP PTZ"],
        "image_urls": [
            json.dumps(["https://example.com/axis-m3027.jpg"]),
            json.dumps(
                [
                    "https://example.com/axis-m3028-1.jpg",
                    "https://example.com/axis-m3028-2.jpg",
                ]
            ),
            json.dumps([]),
        ],
        "image_files": [
            json.dumps(["data/images/axis-m3027/img1.jpg"]),
            json.dumps(
                ["data/images/axis-m3028/img1.jpg", "data/images/axis-m3028/img2.jpg"]
            ),
            json.dumps([]),
        ],
        "datasheet_url": [
            "https://example.com/axis-m3027.pdf",
            None,
            "https://example.com/ds2de.pdf",
        ],
        "datasheet_file": ["data/pdfs/axis-m3027.pdf", None, None],
        "specifications": [
            json.dumps({"Resolution": {"value": "5 MP"}}),
            json.dumps({"Resolution": {"value": "8 MP"}}),
            "{}",
        ],
        "source": ["Axis Communications", "Axis Communications", "HikVision"],
        "category": ["Network Camera", "Network Camera", "PTZ Camera"],
        "product_category": ["Network Cameras", "Network Cameras", "PTZ Cameras"],
        "product_series": ["AXIS M3000", "AXIS M3000", "DS-2DE"],
    }

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_products.parquet"
    df.to_parquet(parquet_path)

    return parquet_path


def test_load_catalog(sample_parquet: Path) -> None:
    """Test loading catalog from parquet file."""
    catalog = load_catalog(sample_parquet)

    assert len(catalog) == 3
    assert "axis-m3027" in catalog
    assert "axis-m3028" in catalog
    assert "hik-ds2de" in catalog

    # Check first camera record
    axis_cam = catalog["axis-m3027"]
    assert axis_cam.model_name == "AXIS M3027-PVE"
    assert axis_cam.source == "Axis Communications"
    assert axis_cam.category == "Network Camera"
    assert len(axis_cam.image_files) == 1
    assert axis_cam.datasheet_url == "https://example.com/axis-m3027.pdf"


def test_load_catalog_missing_file(tmp_path: Path) -> None:
    """Test loading catalog with non-existent parquet file."""
    nonexistent_path = tmp_path / "nonexistent.parquet"

    with pytest.raises(FileNotFoundError) as exc_info:
        load_catalog(nonexistent_path)

    assert "not found" in str(exc_info.value)


def test_load_catalog_with_empty_camera_id(tmp_path: Path) -> None:
    """Test loading catalog skips rows with empty camera_id."""
    data = {
        "camera_id": ["valid-cam", "", None],
        "model_name": ["Valid Camera", "Invalid 1", "Invalid 2"],
        "display_name": ["Valid", "Invalid 1", "Invalid 2"],
        "description": ["desc", "desc", "desc"],
        "image_urls": ["[]", "[]", "[]"],
        "image_files": ["[]", "[]", "[]"],
        "datasheet_url": [None, None, None],
        "datasheet_file": [None, None, None],
        "specifications": ["{}", "{}", "{}"],
        "source": ["Test", "Test", "Test"],
        "category": ["Test", "Test", "Test"],
        "product_category": ["Test", "Test", "Test"],
        "product_series": ["Test", "Test", "Test"],
    }

    df = pd.DataFrame(data)
    parquet_path = tmp_path / "test_invalid.parquet"
    df.to_parquet(parquet_path)

    catalog = load_catalog(parquet_path)

    # Should only load the valid camera
    assert len(catalog) == 1
    assert "valid-cam" in catalog


def test_parse_json_field_valid_json() -> None:
    """Test parsing valid JSON field."""
    result = _parse_json_field('["a", "b", "c"]', default=[])
    assert result == ["a", "b", "c"]

    result = _parse_json_field('{"key": "value"}', default={})
    assert result == {"key": "value"}


def test_parse_json_field_invalid_json() -> None:
    """Test parsing invalid JSON returns default."""
    result = _parse_json_field("not json", default=[])
    assert result == []

    result = _parse_json_field("{invalid}", default={})
    assert result == {}


def test_parse_json_field_none_value() -> None:
    """Test parsing None value returns default."""
    result = _parse_json_field(None, default=[])
    assert result == []

    result = _parse_json_field(pd.NA, default={})
    assert result == {}


def test_parse_json_field_empty_string() -> None:
    """Test parsing empty string returns default."""
    result = _parse_json_field("", default=[])
    assert result == []

    result = _parse_json_field("   ", default={})
    assert result == {}


def test_row_to_camera_record() -> None:
    """Test converting dataframe row to CameraRecord."""
    row = pd.Series(
        {
            "camera_id": "test-cam-123",
            "model_name": "Test Camera Model",
            "display_name": "Test Cam",
            "description": "A test camera",
            "image_urls": json.dumps(["https://example.com/img1.jpg"]),
            "image_files": json.dumps(["data/images/test.jpg"]),
            "datasheet_url": "https://example.com/datasheet.pdf",
            "datasheet_file": "data/pdfs/datasheet.pdf",
            "specifications": json.dumps({"key": {"subkey": "value"}}),
            "source": "Test Source",
            "category": "Test Category",
            "product_category": "Test Product Category",
            "product_series": "Test Series",
        }
    )

    record = _row_to_camera_record(row)

    assert isinstance(record, CameraRecord)
    assert record.camera_id == "test-cam-123"
    assert record.model_name == "Test Camera Model"
    assert record.source == "Test Source"
    assert len(record.images) == 1
    assert len(record.image_files) == 1


def test_iter_reference_image_files(sample_parquet: Path, tmp_path: Path) -> None:
    """Test iterating over reference image files."""
    # Create actual image files referenced in catalog
    image_dir = tmp_path / "data" / "images"

    axis_m3027_dir = image_dir / "axis-m3027"
    axis_m3027_dir.mkdir(parents=True)
    (axis_m3027_dir / "img1.jpg").write_text("fake image")

    axis_m3028_dir = image_dir / "axis-m3028"
    axis_m3028_dir.mkdir(parents=True)
    (axis_m3028_dir / "img1.jpg").write_text("fake image")
    (axis_m3028_dir / "img2.jpg").write_text("fake image")

    # Load catalog and update image paths to use tmp_path
    catalog = load_catalog(sample_parquet)

    # Update paths to point to tmp_path
    for camera_id in catalog:
        catalog[camera_id].image_files = [
            str(tmp_path / f) for f in catalog[camera_id].image_files
        ]

    # Iterate with max_images_per_camera=1
    results = list(iter_reference_image_files(catalog, max_images_per_camera=1))

    # Should get 1 image per camera (2 cameras have images)
    assert len(results) == 2

    camera_ids = [camera_id for camera_id, _ in results]
    assert "axis-m3027" in camera_ids
    assert "axis-m3028" in camera_ids

    # Iterate with max_images_per_camera=2
    results = list(iter_reference_image_files(catalog, max_images_per_camera=2))

    # axis-m3027 has 1 image, axis-m3028 has 2 images
    assert len(results) == 3


def test_iter_reference_image_files_missing_files(sample_parquet: Path) -> None:
    """Test iterating skips missing image files."""
    catalog = load_catalog(sample_parquet)

    # Image files don't exist, should skip them
    results = list(iter_reference_image_files(catalog, max_images_per_camera=5))

    # Should return empty since no files actually exist
    assert len(results) == 0


def test_catalog_index_initialization(tmp_path: Path) -> None:
    """Test CatalogIndex initialization."""
    # Create fake embeddings file
    embeddings = np.random.rand(10, 512).astype("float32")
    camera_ids = np.array([f"cam-{i}" for i in range(10)], dtype="U256")
    image_paths = np.array([f"/fake/path/img{i}.jpg" for i in range(10)], dtype="U1024")
    sources = np.array(["Test Source"] * 10, dtype="U128")

    embeddings_path = tmp_path / "test_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        camera_ids=camera_ids,
        image_paths=image_paths,
        sources=sources,
    )

    # Initialize index
    index = CatalogIndex(embeddings_path)

    assert index.embeddings is not None
    assert index.embeddings.shape == (10, 512)
    assert len(index.camera_ids) == 10
    assert len(index.image_paths) == 10
    assert len(index.sources) == 10


def test_catalog_index_search(tmp_path: Path) -> None:
    """Test CatalogIndex search functionality."""
    # Create fake embeddings
    embeddings = np.random.rand(20, 128).astype("float32")

    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    camera_ids = np.array([f"cam-{i}" for i in range(20)], dtype="U256")
    image_paths = np.array([f"/fake/img{i}.jpg" for i in range(20)], dtype="U1024")
    sources = np.array(["Source A"] * 10 + ["Source B"] * 10, dtype="U128")

    embeddings_path = tmp_path / "test_embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        camera_ids=camera_ids,
        image_paths=image_paths,
        sources=sources,
    )

    index = CatalogIndex(embeddings_path)

    # Create a query vector (normalized)
    query = np.random.rand(128).astype("float32")
    query = query / np.linalg.norm(query)

    # Search with top_k=5
    matches = index.search(query, top_k=5)

    assert len(matches) == 5
    assert all(hasattr(m, "camera_id") for m in matches)
    assert all(hasattr(m, "score") for m in matches)
    assert all(hasattr(m, "catalog_image_path") for m in matches)
    assert all(hasattr(m, "source") for m in matches)

    # Scores should be in descending order
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_catalog_index_search_top_k_larger_than_catalog(tmp_path: Path) -> None:
    """Test searching with top_k larger than catalog size."""
    embeddings = np.random.rand(5, 64).astype("float32")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    camera_ids = np.array([f"cam-{i}" for i in range(5)], dtype="U256")
    image_paths = np.array([f"/fake/img{i}.jpg" for i in range(5)], dtype="U1024")
    sources = np.array(["Test"] * 5, dtype="U128")

    embeddings_path = tmp_path / "small_catalog.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        camera_ids=camera_ids,
        image_paths=image_paths,
        sources=sources,
    )

    index = CatalogIndex(embeddings_path)

    query = np.random.rand(64).astype("float32")
    query = query / np.linalg.norm(query)

    # Search with top_k=10 but only 5 items in catalog
    matches = index.search(query, top_k=10)

    # Should return all 5 items
    assert len(matches) == 5


def test_build_catalog_embeddings_basic(
    sample_parquet: Path, tmp_path: Path, monkeypatch
) -> None:
    """Test building catalog embeddings from parquet file."""
    # Create actual image files
    image_dir = tmp_path / "data" / "images"

    axis_m3027_dir = image_dir / "axis-m3027"
    axis_m3027_dir.mkdir(parents=True)

    # Use PIL to create a real image
    from PIL import Image

    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    img.save(axis_m3027_dir / "img1.jpg")

    # Update parquet to use tmp_path
    catalog = load_catalog(sample_parquet)
    for camera_id in catalog:
        catalog[camera_id].image_files = [
            str(tmp_path / f) for f in catalog[camera_id].image_files
        ]

    # Mock load_catalog to return our updated catalog
    monkeypatch.setattr("src.identification.index.load_catalog", lambda _: catalog)

    # Mock embed_image to return fake embeddings
    fake_embedding = np.random.rand(512).astype("float32")
    monkeypatch.setattr(
        "src.identification.index.embed_image", lambda _: fake_embedding
    )

    embeddings_path = tmp_path / "output" / "embeddings.npz"

    from src.identification.index import build_catalog_embeddings

    result_path = build_catalog_embeddings(
        parquet_path=sample_parquet,
        embeddings_path=embeddings_path,
        max_images_per_camera=1,
    )

    assert result_path == embeddings_path
    assert embeddings_path.exists()

    # Load and verify embeddings file
    data = np.load(embeddings_path)
    assert "embeddings" in data
    assert "camera_ids" in data
    assert "image_paths" in data
    assert "sources" in data


def test_build_catalog_embeddings_no_valid_images(
    sample_parquet: Path, tmp_path: Path, monkeypatch
) -> None:
    """Test building embeddings when no valid images exist."""
    # Mock load_catalog to return catalog with no image files
    catalog = load_catalog(sample_parquet)
    for camera_id in catalog:
        catalog[camera_id].image_files = []

    monkeypatch.setattr("src.identification.index.load_catalog", lambda _: catalog)

    embeddings_path = tmp_path / "embeddings.npz"

    from src.identification.index import build_catalog_embeddings

    with pytest.raises(RuntimeError) as exc_info:
        build_catalog_embeddings(
            parquet_path=sample_parquet,
            embeddings_path=embeddings_path,
        )

    assert "No embeddings were produced" in str(exc_info.value)
