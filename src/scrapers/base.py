"""Base scraper interface."""

import asyncio
from abc import ABC, abstractmethod

import httpx

from src.config import scraper
from src.utils.timing import get_random_delay
from src.models.camera import CategoryLink, CameraRecord


class CameraScraperBase(ABC):
    """
    Abstract base class defining the scraper interface.

    Provides HTTP client and common utility methods for fetching content.
    All complex business logic (downloading, caching, organizing) is delegated
    to manager classes via composition.

    This follows the Interface Segregation Principle and Single Responsibility Principle.
    """

    def __init__(self, base_url: str) -> None:
        """
        Initialize scraper with base URL and HTTP client.

        :param base_url: Base URL for the target website
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(headers=scraper.default_headers, timeout=30.0)

    async def close(self) -> None:
        """Close the HTTP client connection."""
        await self.client.aclose()

    # ========== Common Utility Methods ==========

    async def fetch_html(self, url: str) -> str:
        """
        Fetch HTML content from a URL with human-like delays.

        :param url: URL to fetch
        :return: HTML content as string
        :raises httpx.HTTPError: If request fails
        """
        delay = get_random_delay()
        await asyncio.sleep(delay)

        response = await self.client.get(url)
        response.raise_for_status()
        return response.text

    async def download_image(self, image_url: str) -> bytes:
        """
        Download image bytes from URL.

        This is a basic implementation. Subclasses can override or use
        their own methods for vendor-specific download logic.

        :param image_url: URL to the image
        :return: Image bytes
        :raises httpx.HTTPError: If download fails
        """
        delay = get_random_delay()
        await asyncio.sleep(delay)

        response = await self.client.get(image_url)
        response.raise_for_status()
        return response.content

    async def download_pdf(self, pdf_url: str) -> bytes:
        """
        Download PDF bytes from URL.

        This is a basic implementation. Subclasses can override or use
        their own methods for vendor-specific download logic.

        :param pdf_url: URL to the PDF
        :return: PDF bytes
        :raises httpx.HTTPError: If download fails
        """
        delay = get_random_delay()
        await asyncio.sleep(delay)

        response = await self.client.get(pdf_url)
        response.raise_for_status()
        return response.content

    # ========== Abstract Methods (Interface) ==========

    @abstractmethod
    async def fetch_categories(self) -> list[CategoryLink]:
        """
        Fetch available product categories.

        :return: List of category links
        """
        pass

    @abstractmethod
    async def fetch_cameras(self, category: CategoryLink) -> list[CategoryLink]:
        """
        Fetch cameras within a specific category.

        :param category: Category to scrape
        :return: List of camera records or series links
        """
        pass

    @abstractmethod
    async def fetch_product_details(
        self, product_url: str, category_name: str, series_name: str | None = None
    ) -> CameraRecord:
        """
        Fetch detailed information about a specific product.

        :param product_url: URL to the product page
        :param category_name: Top-level category name
        :param series_name: Product series name (optional)
        :return: CameraRecord with product details
        """
        pass
