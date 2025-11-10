"""Scraper for Axis Communications network cameras."""

from bs4 import BeautifulSoup
from loguru import logger

from src.config import AXIS_BASE_URL, AXIS_PRODUCTS_URL
from src.models.camera import CategoryLink, CameraRecord
from src.scrapers.base import CameraScraperBase
from src.storage.download_cache import DownloadCache


class AxisCameraScraper(CameraScraperBase):
    r"""
    Scraper for Axis Communications network cameras.

    Handles fetching camera categories and individual camera records.
    """

    def __init__(self, download_cache: DownloadCache | None = None) -> None:
        r"""
        Initialize Axis scraper.

        :param download_cache: Optional DownloadCache for skipping downloaded content
        """
        super().__init__(AXIS_BASE_URL)
        self.download_cache = download_cache

    async def fetch_categories(self) -> list[CategoryLink]:
        r"""
        Fetch available camera categories from Axis product page.

        :return: List of category links
        """
        logger.info("Fetching Axis product categories")
        html = await self.fetch_html(AXIS_PRODUCTS_URL)
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
        category_url = f"{AXIS_BASE_URL}{category.href}"
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
        series_url = f"{AXIS_BASE_URL}{series.href}"
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
        self, product_url: str, category_name: str, series_name: str
    ) -> CameraRecord:
        r"""
        Fetch detailed information about a specific product from its page.
        Extracts product name, carousel images, datasheet link, and technical specifications.
        Uses cache to skip re-fetching product pages.

        :param product_url: Relative URL to the product page
        :param category_name: Top-level category (e.g., "DOME CAMERAS")
        :param series_name: Product series name (e.g., "AXIS M30 Dome Camera Series")
        :return: CameraRecord with detailed product information
        """
        product_page_url = f"{AXIS_BASE_URL}{product_url}"

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

            # Extract product name from the page itself
            product_name = self._extract_product_name(soup)
            if not product_name:
                logger.warning(f"Could not extract product name from {product_url}")
                product_name = "Unknown Product"

            # Extract carousel images
            images = self._extract_carousel_images(soup)

            # Extract datasheet URL
            datasheet_url = self._extract_datasheet_url(soup)

            # Extract technical specifications from HTML tables
            specifications_html = self._extract_specifications_tables(soup)

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

        # Create camera record with extracted data
        camera_record = CameraRecord(
            camera_id=product_name.lower().replace(" ", "-"),
            model_name=product_name,
            display_name=product_name,
            description=None,
            specifications={},
            image_url=images[0] if images else None,
            images=images,
            datasheet_url=datasheet_url,
            specifications_html=specifications_html,
            source="Axis Communications",
            category="Network Camera",
            product_category=category_name,
            product_series=series_name,
        )

        return camera_record

    @staticmethod
    def _extract_product_name(soup: BeautifulSoup) -> str | None:
        r"""
        Extract the main product name/title from the product page.

        :param soup: BeautifulSoup parsed HTML
        :return: Product name or None if not found
        """
        # Look for h1 with title class
        h1_tag = soup.find("h1", class_="title-attention")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        # Fallback to first h1
        h1_tag = soup.find("h1")
        if h1_tag:
            return h1_tag.get_text(strip=True)

        return None

    @staticmethod
    def _extract_carousel_images(soup: BeautifulSoup) -> list[str]:
        r"""
        Extract all image URLs from the product carousel.

        :param soup: BeautifulSoup parsed HTML
        :return: List of image URLs
        """
        images = []
        carousel = soup.find("div", class_="product-img-carousel--main")

        if not carousel:
            return images

        # Find all img tags within the carousel
        img_tags = carousel.find_all("img")
        for img in img_tags:
            src = img.get("src")
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
        # Look for a link with text containing "Datasheet" and "pdf"
        for link in soup.find_all("a"):
            link_text = link.get_text(strip=True).lower()
            if "datasheet" in link_text and "pdf" in link_text:
                href = link.get("href")
                if href:
                    return href

        return None

    @staticmethod
    def _extract_specifications_tables(
        soup: BeautifulSoup,
    ) -> dict[str, dict[str, str]]:
        r"""
        Extract technical specifications from HTML tables.

        :param soup: BeautifulSoup parsed HTML
        :return: Dictionary with section names as keys and spec tables as values
        """
        specifications = {}

        # Find all tables with class "ac-table"
        tables = soup.find_all("table", class_="ac-table")

        for table in tables:
            # Get the table caption (section name)
            caption = table.find("caption", class_="ac-table__caption")
            section_name = caption.get_text(strip=True) if caption else "Unknown"

            # Extract rows from tbody
            tbody = table.find("tbody", class_="ac-table__body")
            if not tbody:
                continue

            section_specs = {}
            rows = tbody.find_all("tr", class_="ac-table__row")

            for row in rows:
                cells = row.find_all("td", class_="ac-table__cell")
                if len(cells) >= 2:
                    # First cell is the spec name, second is the value
                    spec_name = cells[0].get_text(strip=True)
                    spec_value = cells[1].get_text(strip=True)
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
                # Handle relative URLs
                if image_url.startswith("/"):
                    full_url = f"{AXIS_BASE_URL}{image_url}"
                else:
                    full_url = image_url

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
            # Handle relative URLs
            if record.datasheet_url.startswith("/"):
                full_url = f"{AXIS_BASE_URL}{record.datasheet_url}"
            else:
                full_url = record.datasheet_url

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
