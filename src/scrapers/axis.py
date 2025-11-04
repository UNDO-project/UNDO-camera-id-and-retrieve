"""Scraper for Axis Communications network cameras."""

from bs4 import BeautifulSoup
from loguru import logger

from src.config import AXIS_BASE_URL, AXIS_PRODUCTS_URL
from src.models.camera import CategoryLink
from src.scrapers.base import CameraScraperBase


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
