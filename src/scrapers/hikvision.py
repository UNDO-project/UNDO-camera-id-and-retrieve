"""Scraper for HikVision network cameras."""

from typing import List, Tuple, Dict
import re

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import async_playwright, Browser

from src.config import (
    HIKVISION_BASE_URL,
    HIKVISION_SELECTORS,
    HIKVISION_IP_PRODUCTS_URL,
    HIKVISION_SUBCATEGORIES,
    DEFAULT_HEADERS,
)
from src.models.camera import CategoryLink, CameraRecord
from src.scrapers.base import CameraScraperBase
from src.storage.download_cache import DownloadCache


class HikvisionCameraScraper(CameraScraperBase):
    r"""
    Scraper for HikVision network cameras.

    Handles fetching camera categories, product listings,
    and individual camera records from HikVision Europe site.
    """

    def __init__(self, download_cache: DownloadCache | None = None) -> None:
        r"""
        Initialize HikVision scraper.

        :param download_cache: Optional DownloadCache for skipping downloaded content
        """
        super().__init__(HIKVISION_BASE_URL)
        self.download_cache = download_cache
        self._browser: Browser | None = None
        self._playwright = None

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

    async def fetch_product_urls_with_playwright(self, subcategory: str) -> List[str]:
        """
        Use Playwright to navigate, filter, and extract product URLs.

        :param subcategory: Subcategory filter value (e.g., "Network Cameras")
        :return: List of product detail page URLs
        """
        browser = await self._get_browser()
        page = await browser.new_page()
        product_urls = []

        try:
            # 1. Navigate to IP Products page
            await page.goto(HIKVISION_IP_PRODUCTS_URL, wait_until="networkidle")

            # 2. Wait for search list to load
            await page.wait_for_selector(HIKVISION_SELECTORS["search_list"])

            # 3. Click subcategory dropdown to expand
            subcategory_dropdown = page.locator(
                HIKVISION_SELECTORS["subcategory_dropdown"]
            )
            await subcategory_dropdown.click()

            # 4. Select the subcategory radio
            radio_selector = HIKVISION_SELECTORS["subcategory_radio"].format(
                subcategory=subcategory
            )
            await page.locator(radio_selector).check()

            # 5. Wait for products to load
            await page.wait_for_selector(HIKVISION_SELECTORS["product_grid"])
            await page.wait_for_timeout(2000)  # Extra wait for dynamic content

            # 6. Get initial product count
            count_locator = page.locator(HIKVISION_SELECTORS["product_count"])
            count_texts = await count_locator.all_text_contents()
            counts = []
            for t in count_texts:
                digits = re.sub(r"[^\d]", "", (t or "").strip())
                if digits:
                    counts.append(int(digits))

            total_count = max(counts) if counts else 0
            logger.info(f"Found {total_count} products for {subcategory}")

            # 7. Extract products with pagination
            while True:
                # Extract product links from current page
                links = page.locator(HIKVISION_SELECTORS["product_link"])
                count = await links.count()

                for i in range(count):
                    href = await links.nth(i).get_attribute("href")
                    if href and href not in product_urls:
                        product_urls.append(href)

                logger.info(f"Extracted {len(product_urls)}/{total_count} product URLs")

                # Check if we have all products
                if 0 < total_count <= len(product_urls):
                    break

                # Click "Next" button to go to next page
                next_btn = page.locator(HIKVISION_SELECTORS["next_page_btn"])
                if await next_btn.is_visible():
                    await next_btn.click()
                    # Wait for navigation and network to settle
                    await page.wait_for_load_state("networkidle")
                    await page.wait_for_selector(HIKVISION_SELECTORS["product_grid"])
                else:
                    logger.info("No more pages available")
                    break  # No more pages to load

        finally:
            await page.close()

        return product_urls

    async def fetch_categories(self) -> List[CategoryLink]:
        r"""
        Return predefined HikVision camera categories.

        :return: List of category links
        """
        logger.info("Fetching HikVision product categories")

        categories = []
        for name, filter_value in HIKVISION_SUBCATEGORIES.items():
            categories.append(
                CategoryLink(
                    name=name,
                    href=filter_value,  # Store filter value
                    node_id=None,
                )
            )

        logger.info(f"Found {len(categories)} HikVision categories")
        return categories

    async def fetch_cameras(self, category: CategoryLink) -> List[CategoryLink]:
        r"""
        Fetch all product URLs for a category using Playwright.

        :param category: Category with subcategory filter value in href
        :return: List of CategoryLink objects (each pointing to a product)

        """
        subcategory_filter = category.href

        # Use Playwright to get all product URLs
        product_urls = await self.fetch_product_urls_with_playwright(subcategory_filter)

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
    async def fetch_products_in_series(series: CategoryLink) -> List[str]:
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
        product_page_url = f"{HIKVISION_BASE_URL}{product_url}"

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
    def _extract_product_name(soup: BeautifulSoup) -> Tuple[str | None, str | None]:
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
    def _extract_carousel_images(soup: BeautifulSoup) -> List[str]:
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
    ) -> Dict[str, Dict[str, str]]:
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
            **DEFAULT_HEADERS,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Referer": f"{HIKVISION_BASE_URL}/",
            "Origin": HIKVISION_BASE_URL,
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
                **DEFAULT_HEADERS,
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Referer": f"{HIKVISION_BASE_URL}/",
                "Origin": HIKVISION_BASE_URL,
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

    async def download_and_organize_images(self, record: CameraRecord) -> List[str]:
        r"""
        Download all carousel images and organize them by product ID.

        Skips images already in cache to reduce server burden.

        :param record: CameraRecord containing image URLs
        :return: List of local file paths
        """
        if not record.images:
            return []

        logger.info(f"Processing {len(record.images)} images for {record.model_name}")
        image_data_list = []
        skipped_count = 0

        for idx, image_url in enumerate(record.images):
            try:
                # Normalize URL (handle protocol-relative URLs)
                full_url = self._normalize_url(image_url)
                if not full_url:
                    continue

                # Check cache before downloading
                if self.download_cache and self.download_cache.has_downloaded(full_url):
                    logger.debug(f"Skipping cached image: {full_url}")
                    skipped_count += 1
                    continue

                try:
                    image_data = await self._download_image_httpx(full_url)
                except httpx.HTTPStatusError as e:
                    status = getattr(getattr(e, "response", None), "status_code", None)
                    if status == 403:
                        logger.warning(
                            f"Got 403 from httpx for image; retrying via Playwright: {full_url}"
                        )
                        image_data = await self._download_image_playwright(full_url)
                    else:
                        raise

                # Check for duplicate content
                if self.download_cache:
                    duplicate_path = self.download_cache.check_content_duplicate(
                        image_data
                    )
                    if duplicate_path:
                        logger.info(f"Image content already stored at {duplicate_path}")
                        image_data_list.append((full_url, image_data))
                        skipped_count += 1
                        continue

                image_data_list.append((full_url, image_data))
                logger.debug(f"Downloaded image {idx + 1}/{len(record.images)}")

            except Exception as e:
                logger.error(f"Failed to download image {image_url}: {e}")
                continue

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} cached image(s)")

        # Store downloaded images using DatasetManager
        from src.storage.dataset import DatasetManager

        dataset_manager = DatasetManager()
        local_paths = dataset_manager.organize_images(
            record, image_data_list, self.download_cache
        )

        logger.info(
            f"Saved {len(local_paths)} images for {record.model_name} "
            f"(skipped {skipped_count})"
        )
        return local_paths

    # download_and_save_pdf() inherited from CameraScraperBase

    def _normalize_pdf_url(self, url: str) -> str | None:
        """
        Override base implementation to handle Hikvision's protocol-relative URLs.

        Hikvision uses protocol-relative URLs (starting with '//').

        :param url: URL that may be protocol-relative or regular
        :return: Normalized absolute URL, or None if URL is invalid
        """
        return self._normalize_url(url)
