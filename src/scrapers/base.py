"""Base scraper class."""

import asyncio
from abc import ABC, abstractmethod

import httpx

from src.config import DEFAULT_HEADERS, get_random_delay
from src.models.camera import CategoryLink, CameraRecord


class CameraScraperBase(ABC):
    r"""
    Abstract base class for camera scrapers.

    Provides common interface for all vendor-specific scrapers.
    """

    def __init__(self, base_url: str) -> None:
        r"""
        Initialize scraper with base URL and HTTP client.

        :param base_url: Base URL for the target website
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(headers=DEFAULT_HEADERS, timeout=30.0)

    async def close(self) -> None:
        r"""
        Close the HTTP client connection.
        """
        await self.client.aclose()

    async def fetch_html(self, url: str) -> str:
        r"""
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
        r"""
        Download image bytes from URL.

        :param image_url: URL to the image
        :return: Image bytes
        :raises httpx.HTTPError: If download fails
        """
        delay = get_random_delay()
        await asyncio.sleep(delay)

        response = await self.client.get(image_url)
        response.raise_for_status()
        return response.content

    @abstractmethod
    async def fetch_categories(self) -> list[CategoryLink]:
        r"""
        Fetch available product categories.

        :return: List of category links
        """
        pass

    @abstractmethod
    async def fetch_cameras(self, category: CategoryLink) -> list[CameraRecord]:
        r"""
        Fetch cameras within a specific category.

        :param category: Category to scrape
        :return: List of camera records
        """
        pass
