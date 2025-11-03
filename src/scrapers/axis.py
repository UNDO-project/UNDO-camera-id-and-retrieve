"""Scraper for Axis Communications network cameras."""

from bs4 import BeautifulSoup
from loguru import logger

from src.config import AXIS_BASE_URL, AXIS_PRODUCTS_URL
from src.models.camera import CameraRecord, CategoryLink
from src.scraper_base import CameraScraperBase


class AxisCameraScraper(CameraScraperBase):
    r"""
    Scraper for Axis Communications network cameras.

    Handles fetching camera categories and individual camera records.
    """

    def __init__(self) -> None:
        r"""
        Initialize Axis scraper.
        """
        super().__init__(AXIS_BASE_URL)

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
                logger.debug(f"Added category: {name}")

        logger.info(f"Found {len(categories)} camera categories")
        return categories

    async def fetch_cameras(self, category: CategoryLink) -> list[CameraRecord]:
        r"""
        Fetch cameras within a specific category.

        :param category: Category to scrape
        :return: List of camera records
        """
        category_url = f"{AXIS_BASE_URL}{category.href}"
        logger.info(f"Fetching cameras from category: {category.name}")
        html = await self.fetch_html(category_url)
        soup = BeautifulSoup(html, "html.parser")

        cameras = []

        # Find all product cards in the category
        product_cards = soup.find_all("a", class_="product-card")
        logger.debug(f"Found {len(product_cards)} product cards")

        for idx, card in enumerate(product_cards):
            product_url = card.get("href")

            if not product_url:
                continue

            # Extract camera info from card
            product_title = card.find("h3", class_="product-card__title")
            product_name = (
                product_title.get_text(strip=True) if product_title else f"Camera-{idx}"
            )

            product_desc = card.find("span", class_="product-card__tagline")
            description = product_desc.get_text(strip=True) if product_desc else None

            # Extract image URL from picture/img
            img_tag = card.find("img")
            image_url = img_tag.get("src") if img_tag else None

            # Create camera record
            camera = CameraRecord(
                camera_id=f"axis-{category.name.lower().replace(' ', '-')}-{idx}",
                model_name=product_name,
                display_name=product_name,
                description=description,
                specifications={},
                image_url=image_url,
                source="Axis Communications",
                category=category.name,
            )
            cameras.append(camera)
            logger.debug(f"Added camera: {product_name}")

        logger.info(f"Found {len(cameras)} cameras in {category.name}")
        return cameras
