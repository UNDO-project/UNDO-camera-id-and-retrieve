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

    async def download_pdf(self, pdf_url: str) -> bytes:
        r"""
        Download PDF bytes from URL.

        :param pdf_url: URL to the PDF
        :return: PDF bytes
        :raises httpx.HTTPError: If download fails
        """
        delay = get_random_delay()
        await asyncio.sleep(delay)

        response = await self.client.get(pdf_url)
        response.raise_for_status()
        return response.content

    async def download_and_save_pdf(self, record: CameraRecord) -> str | None:
        r"""
        Download datasheet PDF and save to organized location.

        Skips PDFs already in cache to reduce server burden.
        Subclasses can override _normalize_pdf_url() to handle vendor-specific URL formats.

        :param record: CameraRecord containing datasheet URL
        :return: Local file path or None if failed
        """
        if not record.datasheet_url:
            return None

        from loguru import logger

        logger.info(f"Processing PDF for {record.model_name}")
        try:
            # Normalize URL (subclasses can override _normalize_pdf_url)
            full_url = self._normalize_pdf_url(record.datasheet_url)
            if not full_url:
                return None

            # Check cache before downloading
            download_cache = getattr(self, "download_cache", None)
            if download_cache and download_cache.has_downloaded(full_url):
                cached_path = download_cache.get_downloaded_path(full_url)
                logger.info(f"Using cached PDF for {record.model_name}: {cached_path}")
                return cached_path

            pdf_data = await self.download_pdf(full_url)

            # Check for duplicate content
            if download_cache:
                duplicate_path = download_cache.check_content_duplicate(pdf_data)
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
                record, pdf_data, download_cache, full_url
            )

            logger.info(f"Saved PDF for {record.model_name}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download PDF for {record.model_name}: {e}")
            return None

    def _normalize_pdf_url(self, url: str) -> str | None:
        r"""
        Normalize PDF URL to absolute URL.

        Subclasses can override this to handle vendor-specific URL formats.
        Default implementation handles relative URLs starting with '/'.

        :param url: URL that may be relative
        :return: Normalized absolute URL, or None if URL is invalid
        """
        if not url:
            return None
        if url.startswith("/"):
            return f"{self.base_url}{url}"
        return url

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
