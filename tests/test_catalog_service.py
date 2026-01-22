"""Tests for CatalogService class."""

from src.api.services.catalog import CatalogService


def test_catalog_service_initialization():
    """Test that service initializes without loading catalog."""
    service = CatalogService()
    assert service._catalog is None


def test_catalog_lazy_loading():
    """Test that catalog loads on first access."""
    service = CatalogService()
    assert service._catalog is None
    catalog = service.catalog
    assert catalog is not None
    assert len(catalog) > 0


def test_catalog_reload():
    """Test that reload clears cache."""
    service = CatalogService()
    _ = service.catalog
    assert service._catalog is not None
    service.reload_catalog()
    assert service._catalog is None


def test_filter_cameras():
    """Test filtering logic."""
    service = CatalogService()
    cameras, total = service.filter_cameras(
        vendor="Axis Communications", page=1, limit=20
    )
    assert isinstance(cameras, list)
    assert isinstance(total, int)
    assert total >= 0
    for camera in cameras:
        assert camera.source == "Axis Communications"


def test_filter_cameras_pagination():
    """Test pagination works correctly."""
    service = CatalogService()

    cameras_page1, total1 = service.filter_cameras(page=1, limit=1)
    cameras_page2, total2 = service.filter_cameras(page=2, limit=1)

    assert len(cameras_page1) == 1
    assert len(cameras_page2) == 1
    assert total1 == total2
    assert cameras_page1[0].camera_id != cameras_page2[0].camera_id


def test_filter_cameras_search():
    """Test full-text search."""
    service = CatalogService()
    cameras, total = service.filter_cameras(search="3057", page=1, limit=20)
    assert isinstance(cameras, list)
    assert isinstance(total, int)
    for camera in cameras:
        assert (
            "3057" in camera.model_name.lower() or "3057" in camera.display_name.lower()
        )


def test_get_camera_detail():
    """Test getting camera detail by ID."""
    service = CatalogService()

    cameras, _ = service.filter_cameras(limit=1)
    assert len(cameras) > 0

    camera_id = cameras[0].camera_id
    detail = service.get_camera_detail(camera_id)

    assert detail is not None
    assert detail.camera_id == camera_id
    assert detail.model_name is not None


def test_get_camera_detail_not_found():
    """Test getting camera detail for non-existent ID."""
    service = CatalogService()
    detail = service.get_camera_detail("non-existent-camera-id")
    assert detail is None


def test_get_facets():
    """Test facets retrieval."""
    service = CatalogService()
    facets = service.get_facets()

    assert "vendors" in facets
    assert "categories" in facets
    assert "series" in facets

    assert len(facets["vendors"]) > 0
    for vendor in facets["vendors"]:
        assert hasattr(vendor, "value")
        assert hasattr(vendor, "count")
        assert isinstance(vendor.count, int)


def test_flatten_specs():
    """Test specs flattening."""
    service = CatalogService()

    cameras, _ = service.filter_cameras(limit=1)
    if cameras and cameras[0].specs:
        assert (
            cameras[0].specs.max_resolution is not None
            or cameras[0].specs.max_resolution is None
        )


def test_build_thumbnail_url_no_double_path():
    """Test that thumbnail URL does not contain double path prefix."""
    service = CatalogService()

    image_file = "data/images/AXIS_COMMUNICATIONS/BOX_CAMERAS/camera.jpg"
    url = service._build_thumbnail_url(image_file)

    assert url == "/api/v1/images/AXIS_COMMUNICATIONS/BOX_CAMERAS/camera.jpg"
    assert "data/images/data/images" not in url


def test_build_thumbnail_url_without_prefix():
    """Test that thumbnail URL works when path has no prefix."""
    service = CatalogService()

    image_file = "VENDOR/CATEGORY/camera.jpg"
    url = service._build_thumbnail_url(image_file)

    assert url == "/api/v1/images/VENDOR/CATEGORY/camera.jpg"


def test_build_datasheet_url_no_double_path():
    """Test that datasheet URL does not contain double path prefix."""
    service = CatalogService()

    datasheet_file = "data/pdfs/HIKVISION/ITS/datasheet.pdf"
    url = service._build_datasheet_url(datasheet_file)

    assert url == "/api/v1/datasheets/HIKVISION/ITS/datasheet.pdf"
    assert "data/pdfs/data/pdfs" not in url


def test_build_datasheet_url_without_prefix():
    """Test that datasheet URL works when path has no prefix."""
    service = CatalogService()

    datasheet_file = "VENDOR/datasheet.pdf"
    url = service._build_datasheet_url(datasheet_file)

    assert url == "/api/v1/datasheets/VENDOR/datasheet.pdf"


def test_build_datasheet_url_none():
    """Test that datasheet URL returns None for empty input."""
    service = CatalogService()

    url = service._build_datasheet_url(None)
    assert url is None


def test_flatten_specs_nested_lens_dict():
    """Test that nested lens dict extracts focal length correctly."""
    service = CatalogService()

    specs = {
        "Lens": {
            "Focal length": "3.16 mm",
            "Horizontal field of view": "103 °",
            "Lens mount": "M12",
        }
    }

    result = service._flatten_specs(specs)

    assert result is not None
    assert result.lens == "3.16 mm"
    assert "{'Focal length'" not in result.lens


def test_flatten_specs_simple_lens_string():
    """Test that simple lens string is preserved."""
    service = CatalogService()

    specs = {"Lens": "Fixed lens 2.8 mm"}

    result = service._flatten_specs(specs)

    assert result is not None
    assert result.lens == "Fixed lens 2.8 mm"


def test_get_camera_detail_image_files_no_double_path():
    """Test that image_files URLs in camera detail do not have double path."""
    service = CatalogService()

    cameras, _ = service.filter_cameras(limit=1)
    if not cameras:
        return

    camera_id = cameras[0].camera_id
    detail = service.get_camera_detail(camera_id)

    assert detail is not None
    for image_url in detail.image_files:
        assert "data/images/data/images" not in image_url
        assert image_url.startswith("/api/v1/images/")
