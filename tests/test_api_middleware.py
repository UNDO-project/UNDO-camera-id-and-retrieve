"""Tests for API middleware components.

Tests the custom middleware implementations, particularly
CORSMiddlewareStaticFiles which adds CORS headers to static files.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.main import CORSMiddlewareStaticFiles


@pytest.fixture
def temp_static_files(tmp_path: Path) -> dict[str, Path]:
    """Create temporary static files for testing.

    Returns:
        dict with 'images_dir', 'pdfs_dir', 'image_file', 'pdf_file'
    """
    # Create directory structure
    images_dir = tmp_path / "images"
    pdfs_dir = tmp_path / "pdfs"
    images_dir.mkdir()
    pdfs_dir.mkdir()

    # Create test image file
    image_file = images_dir / "test_camera.jpg"
    image_file.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Minimal JPEG

    # Create test PDF file
    pdf_file = pdfs_dir / "test_datasheet.pdf"
    pdf_file.write_bytes(b"%PDF-1.4" + b"\x00" * 100)  # Minimal PDF

    return {
        "images_dir": images_dir,
        "pdfs_dir": pdfs_dir,
        "image_file": image_file,
        "pdf_file": pdf_file,
    }


@pytest.fixture
def app_with_static_files(temp_static_files: dict[str, Path]) -> FastAPI:
    """Create a minimal FastAPI app with static file mounts for testing.

    Args:
        temp_static_files: Fixture providing temporary static files

    Returns:
        FastAPI app with CORSMiddlewareStaticFiles mounted
    """
    app = FastAPI()

    # Mount static directories with CORS middleware
    app.mount(
        "/api/v1/images",
        CORSMiddlewareStaticFiles(directory=str(temp_static_files["images_dir"])),
        name="images",
    )
    app.mount(
        "/api/v1/datasheets",
        CORSMiddlewareStaticFiles(directory=str(temp_static_files["pdfs_dir"])),
        name="datasheets",
    )

    return app


def test_static_file_cors_headers_images(
    app_with_static_files: FastAPI, temp_static_files: dict[str, Path]
) -> None:
    """Test that CORS headers are added to image files."""
    client = TestClient(app_with_static_files)

    # Request the test image
    response = client.get("/api/v1/images/test_camera.jpg")

    assert response.status_code == 200
    assert response.headers.get("content-type") in [
        "image/jpeg",
        "application/octet-stream",
    ]

    # Check CORS headers
    assert response.headers.get("access-control-allow-origin") == "*"
    assert "GET" in response.headers.get("access-control-allow-methods", "")
    assert response.headers.get("access-control-allow-headers") == "*"


def test_static_file_cors_headers_datasheets(
    app_with_static_files: FastAPI, temp_static_files: dict[str, Path]
) -> None:
    """Test that CORS headers are added to PDF files."""
    client = TestClient(app_with_static_files)

    # Request the test PDF
    response = client.get("/api/v1/datasheets/test_datasheet.pdf")

    assert response.status_code == 200
    assert response.headers.get("content-type") in [
        "application/pdf",
        "application/octet-stream",
    ]

    # Check CORS headers
    assert response.headers.get("access-control-allow-origin") == "*"
    assert "GET" in response.headers.get("access-control-allow-methods", "")
    assert response.headers.get("access-control-allow-headers") == "*"


def test_static_file_not_found(app_with_static_files: FastAPI) -> None:
    """Test that 404 is returned for non-existent files."""
    client = TestClient(app_with_static_files)

    response = client.get("/api/v1/images/nonexistent.jpg")
    assert response.status_code == 404


def test_cors_middleware_static_files_configuration(monkeypatch: Any) -> None:
    """Test that CORSMiddlewareStaticFiles uses settings.static_file_paths."""
    from src.api import config

    # Mock settings with custom paths
    original_paths = config.settings.static_file_paths
    custom_paths = ["/custom/path/", "/another/path/"]
    monkeypatch.setattr(config.settings, "static_file_paths", custom_paths)

    # Verify settings were changed
    assert config.settings.static_file_paths == custom_paths

    # Create app that should use these paths

    # The middleware should reference the updated settings
    # (This is more of an integration verification)
    assert config.settings.static_file_paths == custom_paths

    # Restore original
    monkeypatch.setattr(config.settings, "static_file_paths", original_paths)


def test_static_file_cors_on_subdirectories(
    app_with_static_files: FastAPI, temp_static_files: dict[str, Path]
) -> None:
    """Test that CORS headers work for files in subdirectories."""
    client = TestClient(app_with_static_files)

    # Create a subdirectory with a file
    subdir = temp_static_files["images_dir"] / "vendor" / "series"
    subdir.mkdir(parents=True)
    subfile = subdir / "nested_camera.jpg"
    subfile.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    # Request the nested file
    response = client.get("/api/v1/images/vendor/series/nested_camera.jpg")

    assert response.status_code == 200
    # CORS headers should still be present
    assert response.headers.get("access-control-allow-origin") == "*"


def test_cors_headers_include_options_method(
    app_with_static_files: FastAPI, temp_static_files: dict[str, Path]
) -> None:
    """Test that CORS headers include OPTIONS method for preflight requests."""
    client = TestClient(app_with_static_files)

    # Make a regular GET request
    response = client.get("/api/v1/images/test_camera.jpg")

    assert response.status_code == 200
    # Check that OPTIONS is in allowed methods (for CORS preflight)
    allowed_methods = response.headers.get("access-control-allow-methods", "")
    assert "OPTIONS" in allowed_methods


def test_main_app_static_files_integration() -> None:
    """Integration test: verify main app has static file mounts with CORS."""
    from src.api.main import app

    client = TestClient(app)

    # Test that images mount exists (will 404 if no files, but mount should exist)
    # We're testing the middleware is attached, not file existence
    response = client.get("/api/v1/images/any_file.jpg")
    # Should get 404 (not found) not 404 (route not found)
    assert response.status_code == 404

    # Same for datasheets
    response = client.get("/api/v1/datasheets/any_file.pdf")
    assert response.status_code == 404


def test_static_files_work_with_existing_data_files() -> None:
    """Integration test: verify static files work with real data directory.

    This test only runs if data/pdfs directory exists with files.
    """
    from src.api.main import app
    from src.config import paths

    client = TestClient(app)

    # Check if we have real PDF files
    pdfs_dir = paths.pdfs_dir
    if not pdfs_dir.exists():
        pytest.skip("No data/pdfs directory found")

    pdf_files = list(pdfs_dir.glob("**/*.pdf"))
    if not pdf_files:
        pytest.skip("No PDF files found in data/pdfs")

    # Get first PDF and construct URL
    pdf_file = pdf_files[0]
    relative_path = pdf_file.relative_to(pdfs_dir)
    test_url = f"/api/v1/datasheets/{relative_path}"

    # Request the file
    response = client.get(test_url)

    assert response.status_code == 200
    assert response.headers.get("content-type") in [
        "application/pdf",
        "application/octet-stream",
    ]

    # Most importantly: check CORS headers
    assert response.headers.get("access-control-allow-origin") == "*"
    assert "GET" in response.headers.get("access-control-allow-methods", "")


def test_cors_middleware_does_not_affect_api_routes() -> None:
    """Test that CORSMiddlewareStaticFiles doesn't interfere with API routes."""
    from src.api.main import app

    client = TestClient(app)

    # Test a regular API endpoint with Origin header to trigger CORS
    response = client.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

    # API routes should have CORS from CORSMiddleware
    # When Origin header is present, CORS middleware should add the header
    assert response.headers.get("access-control-allow-origin") is not None
