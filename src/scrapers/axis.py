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

        logger.info(f"Found {len(categories)} camera categories")
        return categories

    async def fetch_cameras(self, category: CategoryLink) -> list[CameraRecord]:
        r"""
        Fetch all products (series and individual) within a specific category.
        Excludes collections/accessories sections.

        :param category: Category to scrape
        :return: List of camera records (one per nav-card element)
        """
        category_url = f"{AXIS_BASE_URL}{category.href}"
        html = await self.fetch_html(category_url)
        soup = BeautifulSoup(html, "html.parser")

        cameras = []

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

        for idx, card in enumerate(nav_cards):
            product_url = card.get("href")
            if not product_url:
                continue

            # Extract product info from card
            h4_tag = card.find("h4")
            product_name = h4_tag.get_text(strip=True) if h4_tag else f"Product-{idx}"

            # Extract description
            tagline_tag = card.find("span", class_="nav-card__tagline")
            description = tagline_tag.get_text(strip=True) if tagline_tag else None

            # Extract image URL from picture/img
            img_tag = card.find("img")
            image_url = img_tag.get("src") if img_tag else None

            # Create camera record for all nav-card elements (series and individual products)
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

        logger.info(f"Found {len(cameras)} products in {category.name}")
        return cameras
