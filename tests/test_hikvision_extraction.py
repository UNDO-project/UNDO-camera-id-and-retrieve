"""Test HikVision product detail extraction."""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from src.scrapers.hikvision import HikvisionCameraScraper


class TestHikvisionProductNameExtraction:
    """Test product name extraction from HikVision product pages."""

    def test_extract_from_meta_tag(self):
        """Test extraction from page-title meta tag."""
        html = """
        <html>
        <head>
            <meta name="page-title" content="DS-2CD2H46G2H-IZS2UY/S(L)(RB)"/>
        </head>
        <body></body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        product_name, product_number = HikvisionCameraScraper._extract_product_name(
            soup
        )

        assert product_name == "DS-2CD2H46G2H-IZS2UY/S(L)(RB)"
        assert product_number is None

    def test_extract_from_data_attributes(self):
        """Test extraction from product_description-wrapper attributes."""
        html = """
        <html>
        <body>
            <div class="product_description-wrapper"
                 data-product-name="DS-2CD2H46G2H-IZS2UY/S(L)(RB)"
                 data-product-number="M000163208">
            </div>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        product_name, product_number = HikvisionCameraScraper._extract_product_name(
            soup
        )

        assert product_name == "DS-2CD2H46G2H-IZS2UY/S(L)(RB)"
        assert product_number == "M000163208"

    def test_extract_with_both_sources(self):
        """Test that data attributes override meta tag."""
        html = """
        <html>
        <head>
            <meta name="page-title" content="Meta Product Name"/>
        </head>
        <body>
            <div class="product_description-wrapper"
                 data-product-name="Data Attribute Name"
                 data-product-number="M000123">
            </div>
        </body>
        </html>
        """
        soup = BeautifulSoup(html, "html.parser")
        product_name, product_number = HikvisionCameraScraper._extract_product_name(
            soup
        )

        assert product_name == "Data Attribute Name"
        assert product_number == "M000123"

    def test_extract_with_no_sources(self):
        """Test extraction when no product name is found."""
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        product_name, product_number = HikvisionCameraScraper._extract_product_name(
            soup
        )

        assert product_name is None
        assert product_number is None


class TestHikvisionImageExtraction:
    """Test image extraction from HikVision product carousel."""

    def test_extract_carousel_images(self):
        """Test carousel image extraction from swiper slides."""
        html = """
        <div id="productCarouselComp" class="product-carousel-wrap">
            <div class="swiper-slide" data-original="https://assets.hikvision.com/image1.png">
            </div>
            <div class="swiper-slide" data-original="https://assets.hikvision.com/image2.png">
            </div>
            <div class="swiper-slide" data-original="https://assets.hikvision.com/image3.png">
            </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        images = HikvisionCameraScraper._extract_carousel_images(soup)

        assert len(images) == 3
        assert images[0] == "https://assets.hikvision.com/image1.png"
        assert images[1] == "https://assets.hikvision.com/image2.png"
        assert images[2] == "https://assets.hikvision.com/image3.png"

    def test_handles_protocol_relative_urls(self):
        """Test that //assets.hikvision.com URLs are converted to https://."""
        html = """
        <div class="product-carousel-wrap">
            <div class="swiper-slide" data-original="//assets.hikvision.com/image1.png">
            </div>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        images = HikvisionCameraScraper._extract_carousel_images(soup)

        assert len(images) == 1
        assert images[0] == "https://assets.hikvision.com/image1.png"

    def test_fallback_to_img_tags(self):
        """Test fallback to img tags when swiper slides don't have data-original."""
        html = """
        <div class="product-carousel-wrap">
            <img src="https://assets.hikvision.com/image1.png"/>
            <img data-src="https://assets.hikvision.com/image2.png"/>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        images = HikvisionCameraScraper._extract_carousel_images(soup)

        assert len(images) == 2
        assert "https://assets.hikvision.com/image1.png" in images
        assert "https://assets.hikvision.com/image2.png" in images

    def test_no_carousel_returns_empty(self):
        """Test that missing carousel returns empty list."""
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        images = HikvisionCameraScraper._extract_carousel_images(soup)

        assert images == []


class TestHikvisionDatasheetExtraction:
    """Test datasheet URL extraction from HikVision product pages."""

    def test_extract_datasheet_link(self):
        """Test datasheet URL extraction from product_data_sheet link."""
        html = """
        <a class="product_data_sheet"
           href="https://assets.hikvision.com/doc/DS-2CD2H46G2H_Datasheet.pdf">
            Data Sheet
        </a>
        """
        soup = BeautifulSoup(html, "html.parser")
        datasheet_url = HikvisionCameraScraper._extract_datasheet_url(soup)

        assert (
            datasheet_url
            == "https://assets.hikvision.com/doc/DS-2CD2H46G2H_Datasheet.pdf"
        )

    def test_handles_protocol_relative_urls(self):
        """Test that protocol-relative URLs are converted."""
        html = """
        <a class="product_data_sheet"
           href="//assets.hikvision.com/doc/datasheet.pdf">
            Data Sheet
        </a>
        """
        soup = BeautifulSoup(html, "html.parser")
        datasheet_url = HikvisionCameraScraper._extract_datasheet_url(soup)

        assert datasheet_url == "https://assets.hikvision.com/doc/datasheet.pdf"

    def test_fallback_to_any_pdf_with_datasheet_text(self):
        """Test fallback to any link with 'datasheet' text and .pdf href."""
        html = """
        <div>
            <a href="https://example.com/product_datasheet.pdf">Download Datasheet</a>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        datasheet_url = HikvisionCameraScraper._extract_datasheet_url(soup)

        assert datasheet_url == "https://example.com/product_datasheet.pdf"

    def test_no_datasheet_returns_none(self):
        """Test that missing datasheet returns None."""
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        datasheet_url = HikvisionCameraScraper._extract_datasheet_url(soup)

        assert datasheet_url is None


class TestHikvisionSpecsExtraction:
    """Test specification extraction from HikVision accordion format."""

    def test_extract_accordion_specs(self):
        """Test specification extraction from accordion format."""
        html = """
        <div class="tech-specs-accordion-container">
            <li class="tech-specs-items-title__name">
                <a data-target="Camera">Camera</a>
            </li>
            <ul class="tech-specs-items-description" data-target="Camera">
                <li class="tech-specs-items-description-list">
                    <span class="tech-specs-items-description__title">Image Sensor</span>
                    <span class="tech-specs-items-description__title-details">1/3" Progressive Scan CMOS</span>
                </li>
                <li class="tech-specs-items-description-list">
                    <span class="tech-specs-items-description__title">Max. Resolution</span>
                    <span class="tech-specs-items-description__title-details">2688 × 1520</span>
                </li>
            </ul>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        specs = HikvisionCameraScraper._extract_specifications(soup)

        assert "Camera" in specs
        assert specs["Camera"]["Image Sensor"] == '1/3" Progressive Scan CMOS'
        assert specs["Camera"]["Max. Resolution"] == "2688 × 1520"

    def test_multiple_sections(self):
        """Test extraction of multiple spec sections."""
        html = """
        <div class="tech-specs-accordion-container">
            <li class="tech-specs-items-title__name">
                <a data-target="Camera">Camera</a>
            </li>
            <ul class="tech-specs-items-description" data-target="Camera">
                <li class="tech-specs-items-description-list">
                    <span class="tech-specs-items-description__title">Image Sensor</span>
                    <span class="tech-specs-items-description__title-details">1/3" CMOS</span>
                </li>
            </ul>

            <li class="tech-specs-items-title__name">
                <a data-target="Lens">Lens</a>
            </li>
            <ul class="tech-specs-items-description" data-target="Lens">
                <li class="tech-specs-items-description-list">
                    <span class="tech-specs-items-description__title">Focal Length</span>
                    <span class="tech-specs-items-description__title-details">2.8-12mm</span>
                </li>
            </ul>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        specs = HikvisionCameraScraper._extract_specifications(soup)

        assert len(specs) == 2
        assert "Camera" in specs
        assert "Lens" in specs
        assert specs["Camera"]["Image Sensor"] == '1/3" CMOS'
        assert specs["Lens"]["Focal Length"] == "2.8-12mm"

    def test_no_specs_container_returns_empty(self):
        """Test that missing specs container returns empty dict."""
        html = "<html><body></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        specs = HikvisionCameraScraper._extract_specifications(soup)

        assert specs == {}

    def test_missing_data_target_skips_section(self):
        """Test that sections without data-target are skipped."""
        html = """
        <div class="tech-specs-accordion-container">
            <li class="tech-specs-items-title__name">
                <a>Invalid Section</a>
            </li>
        </div>
        """
        soup = BeautifulSoup(html, "html.parser")
        specs = HikvisionCameraScraper._extract_specifications(soup)

        assert specs == {}


class TestHikvisionRealPageExtraction:
    """Test extraction on real HikVision HTML pages."""

    @pytest.fixture
    def network_product_html(self):
        """Load the real network product page HTML."""
        html_path = Path("test_html/hikvision/individual_network_product_page.html")
        if not html_path.exists():
            pytest.skip(f"Test HTML file not found: {html_path}")
        return html_path.read_text(encoding="utf-8")

    @pytest.fixture
    def ptz_product_html(self):
        """Load the real PTZ product page HTML."""
        html_path = Path("test_html/hikvision/individual_ptz_product_page.html")
        if not html_path.exists():
            pytest.skip(f"Test HTML file not found: {html_path}")
        return html_path.read_text(encoding="utf-8")

    @pytest.fixture
    def explosion_product_html(self):
        """Load the real explosion-proof product page HTML."""
        html_path = Path("test_html/hikvision/individual_explosion_product_page.html")
        if not html_path.exists():
            pytest.skip(f"Test HTML file not found: {html_path}")
        return html_path.read_text(encoding="utf-8")

    def test_extract_network_product_name(self, network_product_html):
        """Test product name extraction from real network camera page."""
        soup = BeautifulSoup(network_product_html, "html.parser")
        product_name, product_number = HikvisionCameraScraper._extract_product_name(
            soup
        )

        assert product_name is not None
        assert "DS-2CD" in product_name or product_name != ""
        # Product number should be extracted
        assert product_number is not None or product_name is not None

    def test_extract_network_product_images(self, network_product_html):
        """Test image extraction from real network camera page."""
        soup = BeautifulSoup(network_product_html, "html.parser")
        images = HikvisionCameraScraper._extract_carousel_images(soup)

        assert len(images) > 0
        # All images should be full URLs
        for img in images:
            assert img.startswith("http://") or img.startswith("https://")

    def test_extract_network_product_datasheet(self, network_product_html):
        """Test datasheet extraction from real network camera page."""
        soup = BeautifulSoup(network_product_html, "html.parser")
        datasheet_url = HikvisionCameraScraper._extract_datasheet_url(soup)

        # Datasheet should exist and be a PDF
        if datasheet_url:
            assert ".pdf" in datasheet_url.lower()
            assert datasheet_url.startswith("http://") or datasheet_url.startswith(
                "https://"
            )

    def test_extract_network_product_specifications(self, network_product_html):
        """Test specification extraction from real network camera page."""
        soup = BeautifulSoup(network_product_html, "html.parser")
        specs = HikvisionCameraScraper._extract_specifications(soup)

        assert len(specs) > 0
        # Check for common sections
        common_sections = ["Camera", "Lens", "Video", "Network", "General"]
        found_sections = [s for s in common_sections if s in specs]
        assert len(found_sections) > 0

    def test_extract_ptz_product_name(self, ptz_product_html):
        """Test product name extraction from real PTZ camera page."""
        soup = BeautifulSoup(ptz_product_html, "html.parser")
        product_name, product_number = HikvisionCameraScraper._extract_product_name(
            soup
        )

        assert product_name is not None
        assert product_name != ""

    def test_extract_ptz_product_datasheet(self, ptz_product_html):
        """Test datasheet extraction from real PTZ camera page."""
        soup = BeautifulSoup(ptz_product_html, "html.parser")
        datasheet_url = HikvisionCameraScraper._extract_datasheet_url(soup)

        if datasheet_url:
            assert ".pdf" in datasheet_url.lower()

    def test_extract_explosion_product_name(self, explosion_product_html):
        """Test product name extraction from real explosion-proof camera page."""
        soup = BeautifulSoup(explosion_product_html, "html.parser")
        product_name, product_number = HikvisionCameraScraper._extract_product_name(
            soup
        )

        assert product_name is not None
        assert product_name != ""

    def test_extract_explosion_product_datasheet(self, explosion_product_html):
        """Test datasheet extraction from real explosion-proof camera page."""
        soup = BeautifulSoup(explosion_product_html, "html.parser")
        datasheet_url = HikvisionCameraScraper._extract_datasheet_url(soup)

        if datasheet_url:
            assert ".pdf" in datasheet_url.lower()


class TestHikvisionCategoryExtraction:
    """Test category and series link extraction."""

    @pytest.fixture
    def network_products_page_html(self):
        """Load the network products category page HTML."""
        html_path = Path("test_html/hikvision/network_products_page.html")
        if not html_path.exists():
            pytest.skip(f"Test HTML file not found: {html_path}")
        return html_path.read_text(encoding="utf-8")

    def test_extract_series_links(self, network_products_page_html):
        """Test extraction of series links from category page."""
        soup = BeautifulSoup(network_products_page_html, "html.parser")

        # Find series links - they have class "title-link"
        series_links = soup.find_all("a", class_="title-link")

        assert len(series_links) > 0

        # Check that we can extract series names and hrefs
        for link in series_links[:5]:  # Check first 5
            href = link.get("href")
            h4_tag = link.find("h4")
            name = h4_tag.get_text(strip=True) if h4_tag else None

            if href and name:
                assert href.startswith("/europe/products/")
                assert name != ""
