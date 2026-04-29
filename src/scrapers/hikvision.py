"""Scraper for HikVision network cameras."""

import re

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import async_playwright, Browser

from src.config import (
    hikvision,
    scraper,
    HIKVISION_SELECTORS,
    HIKVISION_IP_SUBCATEGORIES,
)
from src.models.camera import CategoryLink, CameraRecord
from src.scrapers.base import CameraScraperBase
from src.scrapers.managers import DownloadManager, ProtocolRelativeURLNormalizer
from src.storage.download_cache import DownloadCache


class HikvisionCameraScraper(CameraScraperBase):
    r"""
    Scraper for HikVision network cameras.

    Handles fetching camera categories, product listings,
    and individual camera records from HikVision Europe site.
    Uses composition pattern with DownloadManager for handling downloads.
    """

    def __init__(self, download_cache: DownloadCache | None = None) -> None:
        r"""
        Initialize HikVision scraper.

        :param download_cache: Optional DownloadCache for skipping downloaded content
        """
        super().__init__(hikvision.base_url)
        self.download_cache = download_cache
        self._browser: Browser | None = None
        self._playwright = None

        # Composition: Inject download manager with protocol-relative URL strategy
        self.url_normalizer = ProtocolRelativeURLNormalizer()
        self.download_manager = DownloadManager(
            url_normalizer=self.url_normalizer, download_cache=download_cache
        )

    async def _get_browser(self) -> Browser:
        """Lazily initialize Playwright browser."""
        if self._browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=False)
        return self._browser

    @staticmethod
    def _normalize_url(url: str | None) -> str | None:
        """
        Normalize protocol-relative URLs to HTTPS.

        :param url: URL that may start with '//'
        :return: Normalized URL with https: prefix, or None if input is None
        """
        if not url:
            return None
        if url.startswith("//"):
            return f"https:{url}"
        return url

    async def close(self) -> None:
        """Close HTTP client and Playwright browser."""
        await self.client.aclose()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @staticmethod
    async def _read_product_count(page) -> int:
        """
        Read the displayed product count from the page.

        :param page: Playwright page
        :return: Product count, or 0 if not parseable
        """
        count_locator = page.locator(HIKVISION_SELECTORS["product_count"])
        count_texts = await count_locator.all_text_contents()
        counts: list[int] = []
        for t in count_texts:
            digits = re.sub(r"[^\d]", "", (t or "").strip())
            if digits:
                counts.append(int(digits))
        return max(counts) if counts else 0

    @staticmethod
    async def _collect_links_on_page(page, product_urls: list[str]) -> None:
        """
        Append unseen product hrefs from the current page to ``product_urls``.

        :param page: Playwright page
        :param product_urls: Accumulator list (mutated in place)
        """
        links = page.locator(HIKVISION_SELECTORS["product_link"])
        count = await links.count()
        for i in range(count):
            href = await links.nth(i).get_attribute("href")
            if href and href not in product_urls:
                product_urls.append(href)

    @staticmethod
    async def _advance_to_next_page(page) -> bool:
        """
        Click the pagination "Next" button if visible.

        :param page: Playwright page
        :return: True if navigation occurred, False if no next page
        """
        next_btn = page.locator(HIKVISION_SELECTORS["next_page_btn"])
        if not await next_btn.is_visible():
            return False
        await next_btn.click()
        await page.wait_for_load_state("networkidle")
        await page.wait_for_selector(HIKVISION_SELECTORS["product_grid"])
        return True

    async def _extract_product_urls_with_playwright(
        self,
        url: str,
        label: str,
        apply_filter=None,
    ) -> list[str]:
        """
        Generic product URL extraction with optional pre-filter step.

        Navigates to ``url``, optionally applies a filter via
        ``apply_filter``, then collects product links across paginated
        results.

        :param url: Listing page URL
        :param label: Human-readable label used in log messages
        :param apply_filter: Optional async callable ``(page) -> None``
            invoked before pagination starts. Used for IP-products
            subcategory filtering.
        :return: List of product detail page URLs
        """
        browser = await self._get_browser()
        page = await browser.new_page()
        product_urls: list[str] = []

        try:
            logger.info(f"Navigating to {label} page")
            await page.goto(url, wait_until="networkidle")

            if apply_filter is not None:
                await apply_filter(page)

            await page.wait_for_selector(HIKVISION_SELECTORS["product_grid"])
            await page.wait_for_timeout(2000)

            total_count = await self._read_product_count(page)
            logger.info(f"Found {total_count} {label} products")

            while True:
                await self._collect_links_on_page(page, product_urls)
                logger.info(
                    f"Extracted {len(product_urls)}/"
                    f"{total_count if total_count > 0 else '?'} {label} product URLs"
                )

                if 0 < total_count <= len(product_urls):
                    break

                if not await self._advance_to_next_page(page):
                    logger.info(f"No more {label} pages available")
                    break

        finally:
            await page.close()

        logger.info(f"Total {label} products found: {len(product_urls)}")
        return product_urls

    async def fetch_product_urls_with_playwright(self, subcategory: str) -> list[str]:
        """
        Use Playwright to navigate, filter by subcategory, and extract IP product URLs.

        :param subcategory: Subcategory filter value (e.g., "Network Cameras")
        :return: List of product detail page URLs
        """

        async def apply_subcategory_filter(page) -> None:
            await page.wait_for_selector(HIKVISION_SELECTORS["search_list"])
            await page.locator(HIKVISION_SELECTORS["subcategory_dropdown"]).click()
            radio_selector = HIKVISION_SELECTORS["subcategory_radio"].format(
                subcategory=subcategory
            )
            await page.locator(radio_selector).check()

        return await self._extract_product_urls_with_playwright(
            url=hikvision.ip_products_url,
            label=f"IP/{subcategory}",
            apply_filter=apply_subcategory_filter,
        )

    async def fetch_its_product_urls(self) -> list[str]:
        """
        Fetch all ITS product URLs without filtering.

        :return: List of product detail page URLs
        """
        return await self._extract_product_urls_with_playwright(
            url=hikvision.its_products_url,
            label="ITS",
        )

    async def fetch_thermal_product_urls(self) -> list[str]:
        """
        Fetch all Thermal product URLs without filtering.

        :return: List of product detail page URLs
        """
        return await self._extract_product_urls_with_playwright(
            url=hikvision.thermal_products_url,
            label="Thermal",
        )

    async def fetch_categories(self) -> list[CategoryLink]:
        r"""
        Return predefined HikVision camera categories (IP + ITS + Thermal products).

        :return: List of category links
        """
        logger.info("Fetching HikVision product categories")

        categories = []

        # Add IP product subcategories
        for name, filter_value in HIKVISION_IP_SUBCATEGORIES.items():
            categories.append(
                CategoryLink(
                    name=f"IP - {name}",  # Prefix for clarity
                    href=filter_value,  # Store filter value
                    node_id="IP",  # Tag to identify product type
                )
            )

        # Add single ITS category (no filtering needed)
        categories.append(
            CategoryLink(
                name="ITS - Traffic Cameras",
                href=hikvision.its_products_url,  # Direct URL to ITS products
                node_id="ITS",  # Tag to identify product type
            )
        )

        # Add single Thermal category (no filtering needed)
        categories.append(
            CategoryLink(
                name="Thermal - All Products",
                href=hikvision.thermal_products_url,  # Direct URL to Thermal products
                node_id="THERMAL",  # Tag to identify product type
            )
        )

        logger.info(f"Found {len(categories)} HikVision categories")
        return categories

    async def fetch_cameras(self, category: CategoryLink) -> list[CategoryLink]:
        r"""
        Fetch all product URLs for a category using Playwright.

        Routes to appropriate method based on product type (IP vs ITS vs Thermal).

        :param category: Category with subcategory filter value in href
        :return: List of CategoryLink objects (each pointing to a product)

        """
        # Route to appropriate fetcher based on product type
        if category.node_id == "ITS":
            # Fetch ITS products directly (no filtering)
            logger.info(f"Fetching ITS products for category: {category.name}")
            product_urls = await self.fetch_its_product_urls()
        elif category.node_id == "THERMAL":
            # Fetch Thermal products directly (no filtering)
            logger.info(f"Fetching Thermal products for category: {category.name}")
            product_urls = await self.fetch_thermal_product_urls()
        else:
            # Fetch IP products with subcategory filtering
            subcategory_filter = category.href
            logger.info(f"Fetching IP products for category: {category.name}")
            product_urls = await self.fetch_product_urls_with_playwright(
                subcategory_filter
            )

        # Wrap each URL as a CategoryLink for compatibility with main.py
        product_links = []
        for url in product_urls:
            product_links.append(
                CategoryLink(
                    name=url,
                    href=url,
                    node_id=None,
                )
            )

        logger.info(f"Found {len(product_links)} products in {category.name}")
        return product_links

    @staticmethod
    async def fetch_products_in_series(series: CategoryLink) -> list[str]:
        r"""
        Return the product URL directly (series IS the product).

        :param series: Product series link to scrape
        :return: List of product URLs
        """
        return [series.href]

    async def fetch_product_details(
        self, product_url: str, category_name: str, series_name: str | None = None
    ) -> CameraRecord:
        r"""
        Fetch detailed information about a specific product from its page.

        :param series_name:
        :param product_url: Relative URL to the product page
        :param category_name: Top-level category (e.g., "Network Cameras")
        :return: CameraRecord with detailed product information
        """
        product_page_url = f"{hikvision.base_url}{product_url}"

        # Check cache first
        if self.download_cache and self.download_cache.has_cached_product(
            product_page_url
        ):
            logger.info(f"Using cached product details for {product_url}")
            cached_data = self.download_cache.get_cached_product(product_page_url)
            product_name = cached_data["model_name"]
            images = cached_data["image_urls"]
            datasheet_url = cached_data["datasheet_url"]
            specifications_html = cached_data["specifications_html"]
        else:
            # Fetch and parse product page
            html = await self.fetch_html(product_page_url)
            soup = BeautifulSoup(html, "html.parser")

            # Extract product details
            product_name, product_number = self._extract_product_name(soup)
            if not product_name:
                logger.warning(f"Could not extract product name from {product_url}")
                product_name = "Unknown Product"

            # Extract carousel images
            images = self._extract_carousel_images(soup)

            # Extract datasheet URL
            datasheet_url = self._extract_datasheet_url(soup)

            # Extract technical specifications
            specifications_html = self._extract_specifications(soup)

            # Cache the extracted product details
            if self.download_cache:
                self.download_cache.cache_product(
                    product_page_url,
                    product_name,
                    images,
                    datasheet_url,
                    specifications_html,
                )

            logger.info(f"Extracted {len(images)} images for {product_name}")
            if datasheet_url:
                logger.info(f"Found datasheet: {datasheet_url}")
            logger.info(f"Found {len(specifications_html)} specification sections")

        # Generate camera_id from product name
        camera_id = product_name.lower().replace(" ", "-").replace("/", "-")

        # Create camera record with extracted data
        camera_record = CameraRecord(
            camera_id=camera_id,
            model_name=product_name,
            display_name=product_name,
            description=None,
            specifications={},
            image_url=images[0] if images else None,
            images=images,
            datasheet_url=datasheet_url,
            specifications_html=specifications_html,
            source="HikVision",
            category="Network Camera",
            product_category=category_name,
            product_series=category_name,
        )

        return camera_record

    @staticmethod
    def _extract_product_name(soup: BeautifulSoup) -> tuple[str | None, str | None]:
        r"""
        Extract product name and number from meta tags or data attributes.

        :param soup: BeautifulSoup parsed HTML
        :return: Tuple of (product_name, product_number)
        """
        product_name = None
        product_number = None

        # Try meta tag first
        meta_title = soup.find("meta", attrs={"name": "page-title"})
        if meta_title and meta_title.get("content"):
            product_name = meta_title["content"]

        # Try data attributes on product wrapper
        product_wrapper = soup.find("div", class_="product_description-wrapper")
        if product_wrapper:
            product_name = product_wrapper.get("data-product-name", product_name)
            product_number = product_wrapper.get("data-product-number")

        return product_name, product_number

    @staticmethod
    def _extract_carousel_images(soup: BeautifulSoup) -> list[str]:
        r"""
        Extract all image URLs from the product carousel.

        :param soup: BeautifulSoup parsed HTML
        :return: List of image URLs
        """
        images = []

        # Find carousel container
        carousel = soup.find("div", id="productCarouselComp")
        if not carousel:
            carousel = soup.find("div", class_="product-carousel-wrap")

        if carousel:
            # Find all swiper slides with data-original attribute
            slides = carousel.find_all("div", class_="swiper-slide")
            for slide in slides:
                img_url = HikvisionCameraScraper._normalize_url(
                    slide.get("data-original")
                )
                if img_url:
                    images.append(img_url)

        # Fallback: look for img tags with src/data-src
        if not images and carousel:
            for img in carousel.find_all("img"):
                src = HikvisionCameraScraper._normalize_url(
                    img.get("src") or img.get("data-src")
                )
                if src:
                    images.append(src)

        return images

    @staticmethod
    def _extract_datasheet_url(soup: BeautifulSoup) -> str | None:
        r"""
        Extract the datasheet PDF URL from the product page.

        :param soup: BeautifulSoup parsed HTML
        :return: Datasheet URL or None if not found
        """
        # Look for the product_data_sheet link
        datasheet_link = soup.find("a", class_="product_data_sheet")
        if datasheet_link:
            href = HikvisionCameraScraper._normalize_url(datasheet_link.get("href"))
            if href:
                return href

        # Fallback: look for any PDF link with "datasheet" in text
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True).lower()
            if ".pdf" in href.lower() and "datasheet" in text:
                return HikvisionCameraScraper._normalize_url(href)

        return None

    @staticmethod
    def _extract_specifications(
        soup: BeautifulSoup,
    ) -> dict[str, dict[str, str]]:
        r"""
        Extract technical specifications from accordion-style container.

        :param soup: BeautifulSoup parsed HTML
        :return: Dictionary with section names as keys and spec tables as values
        """
        specifications = {}

        # Find the tech specs accordion container
        container = soup.find("div", class_="tech-specs-accordion-container")
        if not container:
            return specifications

        # Find all section headers
        section_headers = container.find_all(
            "li", class_="tech-specs-items-title__name"
        )

        for header in section_headers:
            # Get section name from link with data-target
            link = header.find("a", attrs={"data-target": True})
            if not link:
                continue

            section_name = link.get_text(strip=True)
            data_target = link.get("data-target")

            # Find corresponding spec list
            spec_list = container.find(
                "ul",
                class_="tech-specs-items-description",
                attrs={"data-target": data_target},
            )

            if not spec_list:
                continue

            section_specs = {}
            spec_items = spec_list.find_all(
                "li", class_="tech-specs-items-description-list"
            )

            for item in spec_items:
                name_span = item.find(
                    "span", class_="tech-specs-items-description__title"
                )
                value_span = item.find(
                    "span", class_="tech-specs-items-description__title-details"
                )

                if name_span and value_span:
                    spec_name = name_span.get_text(strip=True)
                    spec_value = value_span.get_text(strip=True)
                    if spec_name and spec_value:
                        section_specs[spec_name] = spec_value

            if section_specs:
                specifications[section_name] = section_specs

        return specifications

    async def _download_image_httpx(self, image_url: str) -> bytes:
        """
        Download image using httpx but with anti-hotlink headers Hikvision commonly expects.
        """
        headers = {
            **scraper.default_headers,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": f"{hikvision.base_url}/",
            "Origin": hikvision.base_url,
        }

        response = await self.client.get(
            image_url, headers=headers, follow_redirects=True
        )
        response.raise_for_status()
        return response.content

    async def _download_image_playwright(self, image_url: str) -> bytes:
        """
        Download image via Chromium network stack (often bypasses CDN 403s that block httpx).
        """
        browser = await self._get_browser()
        context = await browser.new_context(
            extra_http_headers={
                **scraper.default_headers,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": f"{hikvision.base_url}/",
                "Origin": hikvision.base_url,
            }
        )
        try:
            resp = await context.request.get(image_url)
            if resp.status >= 400:
                raise httpx.HTTPStatusError(
                    f"Playwright download failed: HTTP {resp.status}",
                    request=None,
                    response=None,
                )
            return await resp.body()
        finally:
            await context.close()

    async def download(self, url: str) -> bytes:
        """
        Implement ContentDownloader protocol for DownloadManager.

        Hikvision-specific implementation with fallback to Playwright on 403.

        :param url: Absolute URL to download from
        :return: Downloaded content as bytes
        """
        try:
            return await self._download_image_httpx(url)
        except httpx.HTTPStatusError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 403:
                logger.warning(f"Got 403 from httpx; retrying via Playwright: {url}")
                return await self._download_image_playwright(url)
            raise

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
