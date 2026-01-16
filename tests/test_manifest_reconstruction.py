"""Tests for manifest reconstruction functionality."""

import json
import sqlite3
from pathlib import Path

import pytest

from src.building.manifest_reconstruction import ManifestReconstructor


@pytest.fixture
def mock_filesystem(tmp_path: Path) -> Path:
    """Create a mock filesystem structure for testing.

    Creates structure:
    data/images/{VENDOR}/{CATEGORY}/{SERIES}/{CAMERA_ID}/
    data/pdfs/{VENDOR}/{CATEGORY}/{SERIES}/{CAMERA_ID}/
    """
    data_dir = tmp_path / "data"

    # Create image directories for Axis cameras
    axis_images = (
        data_dir / "images" / "AXIS_COMMUNICATIONS" / "network_cameras" / "AXIS_M3000"
    )
    (axis_images / "axis-m3027-pve").mkdir(parents=True)
    (axis_images / "axis-m3028-pve").mkdir(parents=True)

    # Create some dummy image files
    (axis_images / "axis-m3027-pve" / "image1.webp").write_text("fake image")
    (axis_images / "axis-m3027-pve" / "image2.webp").write_text("fake image")
    (axis_images / "axis-m3028-pve" / "image1.webp").write_text("fake image")

    # Create PDF directories
    axis_pdfs = (
        data_dir / "pdfs" / "AXIS_COMMUNICATIONS" / "network_cameras" / "AXIS_M3000"
    )
    (axis_pdfs / "axis-m3027-pve").mkdir(parents=True)
    (axis_pdfs / "axis-m3028-pve").mkdir(parents=True)

    # Create some dummy PDF files
    (axis_pdfs / "axis-m3027-pve" / "datasheet.pdf").write_text("fake pdf")
    (axis_pdfs / "axis-m3028-pve" / "datasheet.pdf").write_text("fake pdf")

    # Create HikVision structure
    hik_images = data_dir / "images" / "HIKVISION" / "ptz_cameras" / "DS-2DE"
    (hik_images / "ds-2de2a404iw-de3").mkdir(parents=True)
    (hik_images / "ds-2de2a404iw-de3" / "image1.webp").write_text("fake image")

    return data_dir


@pytest.fixture
def mock_cache_db(tmp_path: Path) -> Path:
    """Create a mock SQLite cache database for testing."""
    cache_db = tmp_path / "output" / "download_cache.db"
    cache_db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(cache_db)
    cursor = conn.cursor()

    # Create product_cache table (matching actual schema used by reconstructor)
    cursor.execute("""
        CREATE TABLE product_cache (
            product_url TEXT PRIMARY KEY,
            model_name TEXT,
            image_urls TEXT,
            datasheet_url TEXT,
            specifications_html TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Insert test data (matching actual schema)
    test_products = [
        (
            "https://example.com/axis-m3027-pve",
            "AXIS M3027-PVE",
            '["https://example.com/axis-m3027-1.jpg", "https://example.com/axis-m3027-2.jpg"]',
            "https://example.com/axis-m3027.pdf",
            '{"Resolution": "5 MP", "Type": "Fixed Dome"}',
        ),
        (
            "https://example.com/axis-m3028-pve",
            "AXIS M3028-PVE",
            '["https://example.com/axis-m3028.jpg"]',
            "https://example.com/axis-m3028.pdf",
            '{"Resolution": "8 MP", "Type": "Fixed Dome"}',
        ),
        (
            "https://example.com/ds-2de2a404iw-de3",
            "DS-2DE2A404IW-DE3",
            '["https://example.com/ds-2de2a404iw.jpg"]',
            "https://example.com/ds-2de2a404iw.pdf",
            '{"Resolution": "4 MP", "Type": "PTZ"}',
        ),
    ]

    cursor.executemany(
        """
        INSERT INTO product_cache (
            product_url, model_name, image_urls, datasheet_url, specifications_html
        ) VALUES (?, ?, ?, ?, ?)
        """,
        test_products,
    )

    conn.commit()
    conn.close()

    return cache_db


def test_reconstructor_initialization(tmp_path: Path) -> None:
    """Test ManifestReconstructor initialization with custom paths."""
    cache_db = tmp_path / "custom_cache.db"
    data_dir = tmp_path / "custom_data"
    output_path = tmp_path / "custom_manifest.json"

    reconstructor = ManifestReconstructor(
        cache_db_path=cache_db,
        data_dir=data_dir,
        output_path=output_path,
    )

    assert reconstructor.cache_db_path == cache_db
    assert reconstructor.data_dir == data_dir
    assert reconstructor.output_path == output_path
    assert reconstructor.manifest_data == {}


def test_reconstructor_initialization_with_defaults() -> None:
    """Test ManifestReconstructor initialization with default paths."""
    reconstructor = ManifestReconstructor()

    # Should use config paths by default
    assert reconstructor.cache_db_path.name == "download_cache.db"
    assert reconstructor.data_dir.name == "data"
    assert reconstructor.output_path.name == "verification_manifest.json"


def test_scan_filesystem_structure(mock_filesystem: Path) -> None:
    """Test filesystem scanning discovers correct structure."""
    reconstructor = ManifestReconstructor(
        data_dir=mock_filesystem,
        cache_db_path=Path("/dummy/cache.db"),
        output_path=Path("/dummy/manifest.json"),
    )

    structure = reconstructor._scan_filesystem_structure()

    # Should find 2 categories (Network Cameras, PTZ Cameras)
    assert len(structure) == 2
    assert "Network Cameras" in structure
    assert "Ptz Cameras" in structure

    # Network Cameras should have AXIS M3000 series
    assert "AXIS M3000" in structure["Network Cameras"]
    assert len(structure["Network Cameras"]["AXIS M3000"]) == 2
    assert "axis-m3027-pve" in structure["Network Cameras"]["AXIS M3000"]
    assert "axis-m3028-pve" in structure["Network Cameras"]["AXIS M3000"]

    # PTZ Cameras should have DS-2DE series
    assert "DS-2DE" in structure["Ptz Cameras"]
    assert len(structure["Ptz Cameras"]["DS-2DE"]) == 1
    assert "ds-2de2a404iw-de3" in structure["Ptz Cameras"]["DS-2DE"]


def test_scan_filesystem_empty_directory(tmp_path: Path) -> None:
    """Test filesystem scanning with empty data directory."""
    empty_data_dir = tmp_path / "empty_data"
    empty_data_dir.mkdir()

    reconstructor = ManifestReconstructor(
        data_dir=empty_data_dir,
        cache_db_path=Path("/dummy/cache.db"),
        output_path=Path("/dummy/manifest.json"),
    )

    structure = reconstructor._scan_filesystem_structure()

    # Should return empty structure
    assert structure == {}


def test_load_cache(mock_cache_db: Path) -> None:
    """Test loading product data from SQLite cache."""
    reconstructor = ManifestReconstructor(
        cache_db_path=mock_cache_db,
        data_dir=Path("/dummy/data"),
        output_path=Path("/dummy/manifest.json"),
    )

    cache_data = reconstructor._load_product_cache()

    # Should load 3 products from cache
    assert len(cache_data) == 3
    assert "axis-m3027-pve" in cache_data
    assert "axis-m3028-pve" in cache_data
    assert "ds-2de2a404iw-de3" in cache_data

    # Check product data structure
    axis_product = cache_data["axis-m3027-pve"]
    assert axis_product["model_name"] == "AXIS M3027-PVE"
    assert "product_url" in axis_product
    assert "image_urls" in axis_product
    assert "datasheet_url" in axis_product


def test_load_cache_missing_database(tmp_path: Path) -> None:
    """Test loading cache when database doesn't exist."""
    missing_db = tmp_path / "nonexistent.db"

    reconstructor = ManifestReconstructor(
        cache_db_path=missing_db,
        data_dir=Path("/dummy/data"),
        output_path=Path("/dummy/manifest.json"),
    )

    cache_data = reconstructor._load_product_cache()

    # Should return empty dict and log error
    assert cache_data == {}


def test_count_images(mock_filesystem: Path) -> None:
    """Test image counting for a product."""
    reconstructor = ManifestReconstructor(
        data_dir=mock_filesystem,
        cache_db_path=Path("/dummy/cache.db"),
        output_path=Path("/dummy/manifest.json"),
    )

    # Test Axis product
    image_count = reconstructor._count_images(
        "Network Cameras", "AXIS M3000", "axis-m3027-pve"
    )

    assert image_count == 2  # 2 image files

    # Test HikVision product
    image_count = reconstructor._count_images(
        "Ptz Cameras", "DS-2DE", "ds-2de2a404iw-de3"
    )

    assert image_count == 1  # 1 image file


def test_full_reconstruction(
    mock_filesystem: Path, mock_cache_db: Path, tmp_path: Path
) -> None:
    """Test complete manifest reconstruction process."""
    output_path = tmp_path / "reconstructed_manifest.json"

    reconstructor = ManifestReconstructor(
        cache_db_path=mock_cache_db,
        data_dir=mock_filesystem,
        output_path=output_path,
    )

    success = reconstructor.reconstruct()

    assert success is True
    assert output_path.exists()

    # Load and validate manifest structure
    with open(output_path) as f:
        manifest = json.load(f)

    # Check manifest metadata
    assert "scrape_timestamp" in manifest
    assert "reconstruction_timestamp" in manifest
    assert "reconstructed" in manifest
    assert "categories" in manifest

    # The manifest structure is nested dict, not flat with total counts
    # Check that categories exist
    categories = manifest["categories"]
    assert isinstance(categories, dict)

    # Check categories structure (simplified - manifest structure is nested dict)
    categories = manifest["categories"]
    assert len(categories) == 2

    # Check that Network Cameras category exists
    assert "Network Cameras" in categories
    network_cam_category = categories["Network Cameras"]
    assert "product_count" in network_cam_category
    assert "series" in network_cam_category

    # Check series structure (nested dict)
    assert isinstance(network_cam_category["series"], dict)
    assert "AXIS M3000" in network_cam_category["series"]

    axis_series = network_cam_category["series"]["AXIS M3000"]
    assert "products" in axis_series
    assert isinstance(axis_series["products"], list)
    assert len(axis_series["products"]) == 2


def test_reconstruct_without_cache(mock_filesystem: Path, tmp_path: Path) -> None:
    """Test reconstruction when cache database is missing."""
    missing_cache = tmp_path / "missing_cache.db"
    output_path = tmp_path / "manifest.json"

    reconstructor = ManifestReconstructor(
        cache_db_path=missing_cache,
        data_dir=mock_filesystem,
        output_path=output_path,
    )

    success = reconstructor.reconstruct()

    # Should still succeed but with limited data (no cached metadata)
    assert success is True
    assert output_path.exists()

    with open(output_path) as f:
        manifest = json.load(f)

    # Should have filesystem structure but products may lack detailed metadata
    assert "categories" in manifest
    assert len(manifest["categories"]) == 2


def test_reconstruct_with_existing_output_file(
    mock_filesystem: Path, mock_cache_db: Path, tmp_path: Path
) -> None:
    """Test reconstruction overwrites existing manifest file."""
    output_path = tmp_path / "existing_manifest.json"

    # Create pre-existing file
    output_path.write_text('{"old": "data"}')
    assert output_path.exists()

    reconstructor = ManifestReconstructor(
        cache_db_path=mock_cache_db,
        data_dir=mock_filesystem,
        output_path=output_path,
    )

    success = reconstructor.reconstruct()

    assert success is True

    # Should have new content, not old data
    with open(output_path) as f:
        manifest = json.load(f)

    assert "old" not in manifest
    assert "scrape_timestamp" in manifest
    assert "reconstructed" in manifest
    assert manifest["reconstructed"] is True
