"""Tests for configuration management."""

from src.config import scraper, paths, axis, hikvision
from src.config.scraper import ScraperSettings
from src.config.paths import PathSettings


class TestScraperSettings:
    """Test scraper configuration."""

    def test_default_values(self):
        """Test default scraper settings."""
        assert scraper.min_request_delay == 2.0
        assert scraper.max_request_delay == 5.0
        assert "Mozilla" in scraper.user_agent

    def test_default_headers_structure(self):
        """Test default headers are properly constructed."""
        headers = scraper.default_headers
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert headers["DNT"] == "1"

    def test_env_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("CIDAR_SCRAPER_MIN_REQUEST_DELAY", "10.0")
        monkeypatch.setenv("CIDAR_SCRAPER_MAX_REQUEST_DELAY", "20.0")

        settings = ScraperSettings()
        assert settings.min_request_delay == 10.0
        assert settings.max_request_delay == 20.0


class TestPathSettings:
    """Test path configuration."""

    def test_computed_paths(self):
        """Test path settings computed fields."""
        assert paths.data_dir == paths.project_root / "data"
        assert paths.images_dir == paths.data_dir / "images"
        assert paths.pdfs_dir == paths.data_dir / "pdfs"
        assert paths.output_dir == paths.project_root / "output"
        assert paths.models_dir == paths.project_root / "model_weights"

    def test_yolo_weights_path(self):
        """Test YOLO weights path construction."""
        expected = paths.models_dir / "yolov8_camera.pt"
        assert paths.yolo_camera_weights_default == expected

    def test_project_root_is_absolute(self):
        """Test project root is resolved to absolute path."""
        assert paths.project_root.is_absolute()

    def test_ensure_directories_exist(self, tmp_path, monkeypatch):
        """Test directory creation."""
        monkeypatch.setenv("CIDAR_PATH_PROJECT_ROOT", str(tmp_path))

        settings = PathSettings()
        settings.ensure_directories_exist()

        assert settings.data_dir.exists()
        assert settings.images_dir.exists()
        assert settings.pdfs_dir.exists()
        assert settings.output_dir.exists()
        assert settings.models_dir.exists()


class TestVendorSettings:
    """Test vendor configuration."""

    def test_axis_urls(self):
        """Test Axis URL construction."""
        assert axis.base_url == "https://www.axis.com"
        assert axis.products_url == "https://www.axis.com/products/network-cameras"

    def test_hikvision_urls(self):
        """Test HikVision URL construction."""
        assert hikvision.base_url == "https://www.hikvision.com"
        assert hikvision.region == "europe"
        assert "europe" in hikvision.ip_products_url
        assert "IP-Products" in hikvision.ip_products_url
        assert "traffic-cameras" in hikvision.its_products_url

    def test_hikvision_pagination_settings(self):
        """Test HikVision pagination settings."""
        assert hikvision.page_load_timeout == 30000
        assert hikvision.products_per_page == 12


class TestConstants:
    """Test static constants."""

    def test_hikvision_selectors_exist(self):
        """Test HikVision CSS selectors are defined."""
        from src.config import HIKVISION_SELECTORS

        assert "search_list" in HIKVISION_SELECTORS
        assert "product_link" in HIKVISION_SELECTORS
        assert "pagination" in HIKVISION_SELECTORS

    def test_hikvision_subcategories_exist(self):
        """Test HikVision subcategories are defined."""
        from src.config import HIKVISION_IP_SUBCATEGORIES

        assert "Network Cameras" in HIKVISION_IP_SUBCATEGORIES
        assert "PTZ Cameras" in HIKVISION_IP_SUBCATEGORIES
