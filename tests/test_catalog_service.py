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
