"""Scraper for Axis Communications network cameras."""

from bs4 import BeautifulSoup
from loguru import logger

from src.config import axis
from src.models.camera import CameraRecord, CategoryLink
from src.scrapers.base import CameraScraperBase
from src.scrapers.managers import (
    DownloadManager,
    ProductDetailExtractor,
    RelativeURLNormalizer,
)
from src.storage.download_cache import DownloadCache


class _AxisProductDetailExtractor(ProductDetailExtractor):
    """Axis-specific selectors for product detail extraction."""

    def _extract_name(self, soup: BeautifulSoup) -> str | None:
        h1_tag = soup.find("h1", class_="title-attention")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        return None

    def _extract_images(self, soup: BeautifulSoup) -> list[str]:
        images: list[str] = []
        carousel = soup.find("div", class_="product-img-carousel--main")
        if not carousel:
            return images

        for img in carousel.find_all("img"):
            src = img.get("src")
            if src:
                images.append(src)
        return images

    def _extract_datasheet_url(self, soup: BeautifulSoup) -> str | None:
        for link in soup.find_all("a"):
            link_text = link.get_text(strip=True).lower()
            if "datasheet" in link_text and "pdf" in link_text:
                href = link.get("href")
                if href:
                    return href
        return None

    def _extract_specifications(self, soup: BeautifulSoup) -> dict[str, dict[str, str]]:
        specifications: dict[str, dict[str, str]] = {}

        for table in soup.find_all("table", class_="ac-table"):
            caption = table.find("caption", class_="ac-table__caption")
            section_name = caption.get_text(strip=True) if caption else "Unknown"

            tbody = table.find("tbody", class_="ac-table__body")
            if not tbody:
                continue

            section_specs: dict[str, str] = {}
            for row in tbody.find_all("tr", class_="ac-table__row"):
                cells = row.find_all("td", class_="ac-table__cell")
                if len(cells) >= 2:
                    spec_name = cells[0].get_text(strip=True)
                    spec_value = cells[1].get_text(strip=True)
                    section_specs[spec_name] = spec_value

            if section_specs:
                specifications[section_name] = section_specs

        return specifications


class AxisCameraScraper(CameraScraperBase):
    r"""
    Scraper for Axis Communications network cameras.

    Handles fetching camera categories and individual camera records.
    Uses composition pattern with DownloadManager for handling downloads.
    """

    def __init__(self, download_cache: DownloadCache | None = None) -> None:
        r"""
        Initialize Axis scraper.

        :param download_cache: Optional DownloadCache for skipping downloaded content
        """
        super().__init__(axis.base_url)
        self.download_cache = download_cache

        # Composition: Inject download manager with strategy
        self.url_normalizer = RelativeURLNormalizer()
        self.download_manager = DownloadManager(
            url_normalizer=self.url_normalizer, download_cache=download_cache
        )
        self.product_extractor = _AxisProductDetailExtractor(self, download_cache)

    async def fetch_categories(self) -> list[CategoryLink]:
        r"""
        Fetch available camera categories from Axis product page.

        :return: List of category links
        """
        logger.info("Fetching Axis product categories")
        html = await self.fetch_html(axis.products_url)
        soup = BeautifulSoup(html, "html.parser")

        categories = []

        # Find all nav-card elements (camera categories)
        nav_cards = soup.find_all("a", class_="nav-card")
        logger.debug(f"Found {len(nav_cards)} nav-card elements")

        for card in nav_cards:
            href = card.get("href")
            node_id = card.get("data-history-node-id")

            if not href:
                continue

            # Extract category name from the h4 tag
            h4_tag = card.find("h4")
            name = h4_tag.get_text(strip=True) if h4_tag else None

            if name:
                category = CategoryLink(
                    name=name,
                    href=href,
                    node_id=node_id,
                )
                categories.append(category)

        logger.info(f"Found {len(categories)} camera categories")
        return categories

    async def fetch_cameras(self, category: CategoryLink) -> list[CategoryLink]:
        r"""
        Fetch all product series within a specific category.
        Returns series links to be processed further.

        :param category: Category to scrape
        :return: List of product series links
        """
        category_url = f"{axis.base_url}{category.href}"
        html = await self.fetch_html(category_url)
        soup = BeautifulSoup(html, "html.parser")

        series = []

        # Find the "Products within" section specifically (exclude "Collections within")
        for heading in soup.find_all("h3"):
            if "Products within" in heading.get_text():
                # Find the next nav-card__coll container
                nav_coll = heading.find_next("div", class_="nav-card__coll")
                if nav_coll:
                    # Extract all nav-card elements from this container
                    nav_cards = nav_coll.find_all("a", class_="nav-card")
                    break
        else:
            # Fallback: if no "Products within" heading found, get all nav-cards
            nav_cards = soup.find_all("a", class_="nav-card")

        for card in nav_cards:
            href = card.get("href")
            if not href:
                continue

            # Extract product series info
            h4_tag = card.find("h4")
            name = h4_tag.get_text(strip=True) if h4_tag else None

            if name:
                series_link = CategoryLink(
                    name=name,
                    href=href,
                    node_id=card.get("data-history-node-id"),
                )
                series.append(series_link)

        logger.info(f"Found {len(series)} product series in {category.name}")
        return series

    async def fetch_products_in_series(self, series: CategoryLink) -> list[str]:
        r"""
        Fetch individual product links within a product series page.
        If the series page has no sub-products, returns the series itself as a product.

        :param series: Product series link to scrape
        :return: List of product URLs
        """
        series_url = f"{axis.base_url}{series.href}"
        html = await self.fetch_html(series_url)
        soup = BeautifulSoup(html, "html.parser")

        products = []

        # Find the "Products within" section specifically
        for heading in soup.find_all("h3"):
            if "Products within" in heading.get_text():
                # Find the next nav-card__coll container
                nav_coll = heading.find_next("div", class_="nav-card__coll")
                if nav_coll:
                    # Extract all nav-card elements from this container
                    nav_cards = nav_coll.find_all("a", class_="nav-card")
                    break
        else:
            # Fallback: if no "Products within" heading found, get all nav-cards
            nav_cards = soup.find_all("a", class_="nav-card")

        for card in nav_cards:
            href = card.get("href")
            if href:
                products.append(href)

        # If no sub-products found, the series page itself is a product
        if not products:
            products.append(series.href)

        logger.info(f"Found {len(products)} product(s) in {series.name}")
        return products

    async def fetch_product_details(
        self, product_url: str, category_name: str, series_name: str | None = None
    ) -> CameraRecord:
        r"""
        Fetch detailed information about a specific product from its page.

        Delegates HTML extraction and caching to the product extractor;
        assembles the vendor-specific :class:`CameraRecord` here.

        :param product_url: Relative URL to the product page
        :param category_name: Top-level category (e.g., "DOME CAMERAS")
        :param series_name: Product series name (e.g., "AXIS M30 Dome Camera Series")
        :return: CameraRecord with detailed product information
        """
        product_page_url = f"{axis.base_url}{product_url}"
        details = await self.product_extractor.extract(product_page_url)

        return CameraRecord(
            camera_id=details.name.lower().replace(" ", "-"),
            model_name=details.name,
            display_name=details.name,
            description=None,
            specifications={},
            image_url=details.images[0] if details.images else None,
            images=details.images,
            datasheet_url=details.datasheet_url,
            specifications_html=details.specifications_html,
            source="Axis Communications",
            category="Network Camera",
            product_category=category_name,
            product_series=series_name,
        )

    async def download(self, url: str) -> bytes:
        """
        Implement ContentDownloader protocol for DownloadManager.

        Simple HTTP download using base class method.

        :param url: Absolute URL to download from
        :return: Downloaded content as bytes
        """
        return await self.download_image(url)

    async def download_and_organize_images(self, record: CameraRecord) -> list[str]:
        """
        Download and organize images using the download manager.

        Delegates to DownloadManager which handles caching, deduplication,
        and organization.

        :param record: CameraRecord containing image URLs
        :return: List of local file paths
        """
        return await self.download_manager.download_and_organize_images(
            record=record, image_downloader=self, base_url=self.base_url
        )

    async def download_and_save_pdf(self, record: CameraRecord) -> str | None:
        """
        Download and save PDF using the download manager.

        Delegates to DownloadManager which handles caching and deduplication.

        :param record: CameraRecord containing datasheet URL
        :return: Local file path or None if failed
        """
        return await self.download_manager.download_and_save_pdf(
            record=record, pdf_downloader=self, base_url=self.base_url
        )
