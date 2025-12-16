"""Scraper for HikVision network cameras."""

from bs4 import BeautifulSoup
from loguru import logger

from src.config import HIKVISION_BASE_URL, HIKVISION_CATEGORIES
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

    async def fetch_categories(self) -> list[CategoryLink]:
        r"""
        Return predefined HikVision camera categories.

        :return: List of category links
        """
        logger.info("Fetching HikVision product categories")

        categories = []
        for name, href in HIKVISION_CATEGORIES.items():
            categories.append(
                CategoryLink(
                    name=name,
                    href=href,
                    node_id=None,  # HikVision doesn't use Drupal node IDs
                )
            )

        logger.info(f"Found {len(categories)} HikVision categories")
        return categories

    async def fetch_cameras(self, category: CategoryLink) -> list[CategoryLink]:
        r"""
        Fetch all product series within a specific category.
        Returns series links to be processed further.

        :param category: Category to scrape
        :return: List of product series links
        """
        category_url = f"{HIKVISION_BASE_URL}{category.href}"
        html = await self.fetch_html(category_url)
        soup = BeautifulSoup(html, "html.parser")

        series = []

        # Find series links (subcategories) - they have class "title-link"
        series_links = soup.find_all("a", class_="title-link")

        for link in series_links:
            href = link.get("href")
            if not href:
                continue

            # Extract series name from h4 tag
            h4_tag = link.find("h4")
            name = h4_tag.get_text(strip=True) if h4_tag else None

            if name and href:
                series_link = CategoryLink(
                    name=name,
                    href=href,
                    node_id=None,
                )
                series.append(series_link)

        logger.info(f"Found {len(series)} product series in {category.name}")
        return series

    async def fetch_products_in_series(self, series: CategoryLink) -> list[str]:
        r"""
        Fetch individual product links within a product series page.

        :param series: Product series link to scrape
        :return: List of product URLs
        """
        series_url = f"{HIKVISION_BASE_URL}{series.href}"
        html = await self.fetch_html(series_url)
        soup = BeautifulSoup(html, "html.parser")

        products = []

        # Look for product links - need to identify the pattern
        # Based on the individual product page URLs, they follow pattern like:
        # /europe/products/IP-Products/Network-Cameras/Pro-Series-EasyIP-/ds-2cd2h46g2h-izs2uy-s-l--rb-/

        # Try finding links that contain product model patterns
        for link in soup.find_all("a", href=True):
            href = link.get("href")
            # Product pages are deeper in the hierarchy (more path segments)
            # and typically end with a model name slug
            if href and href.startswith(series.href) and href != series.href:
                # Check if this is likely a product page (not another category)
                # Product URLs typically have more path segments
                if href.count("/") > series.href.count("/"):
                    if href not in products:
                        products.append(href)

        logger.info(f"Found {len(products)} product(s) in {series.name}")
        return products

    async def fetch_product_details(
        self, product_url: str, category_name: str
    ) -> CameraRecord:
        r"""
        Fetch detailed information about a specific product from its page.

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
                img_url = slide.get("data-original")
                if img_url:
                    # Ensure full URL
                    if img_url.startswith("//"):
                        img_url = f"https:{img_url}"
                    images.append(img_url)

        # Fallback: look for img tags with src/data-src
        if not images and carousel:
            for img in carousel.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if src:
                    if src.startswith("//"):
                        src = f"https:{src}"
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
            href = datasheet_link.get("href")
            if href:
                # Ensure full URL
                if href.startswith("//"):
                    href = f"https:{href}"
                return href

        # Fallback: look for any PDF link with "datasheet" in text
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            text = link.get_text(strip=True).lower()
            if ".pdf" in href.lower() and "datasheet" in text:
                if href.startswith("//"):
                    href = f"https:{href}"
                return href

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

    async def download_and_organize_images(self, record: CameraRecord) -> list[str]:
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
                # HikVision images are full URLs from assets.hikvision.com
                full_url = image_url
                if full_url.startswith("//"):
                    full_url = f"https:{full_url}"

                # Check cache before downloading
                if self.download_cache and self.download_cache.has_downloaded(full_url):
                    logger.debug(f"Skipping cached image: {full_url}")
                    skipped_count += 1
                    continue

                image_data = await self.download_image(full_url)

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

    async def download_and_save_pdf(self, record: CameraRecord) -> str | None:
        r"""
        Download datasheet PDF and save to organized location.

        Skips PDFs already in cache to reduce server burden.

        :param record: CameraRecord containing datasheet URL
        :return: Local file path or None if failed
        """
        if not record.datasheet_url:
            return None

        logger.info(f"Processing PDF for {record.model_name}")
        try:
            # Handle protocol-relative URLs
            full_url = record.datasheet_url
            if full_url.startswith("//"):
                full_url = f"https:{full_url}"

            # Check cache before downloading
            if self.download_cache and self.download_cache.has_downloaded(full_url):
                cached_path = self.download_cache.get_downloaded_path(full_url)
                logger.info(f"Using cached PDF for {record.model_name}: {cached_path}")
                return cached_path

            pdf_data = await self.download_pdf(full_url)

            # Check for duplicate content
            if self.download_cache:
                duplicate_path = self.download_cache.check_content_duplicate(pdf_data)
                if duplicate_path:
                    logger.info(
                        f"PDF content already stored at {duplicate_path}, "
                        f"reusing for {record.model_name}"
                    )
                    return duplicate_path

            # Store PDF using DatasetManager
            from src.storage.dataset import DatasetManager

            dataset_manager = DatasetManager()
            local_path = dataset_manager.save_pdf(
                record, pdf_data, self.download_cache, full_url
            )

            logger.info(f"Saved PDF for {record.model_name}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download PDF for {record.model_name}: {e}")
            return None
