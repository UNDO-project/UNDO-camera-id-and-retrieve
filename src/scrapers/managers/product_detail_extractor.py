"""Cache-aware product detail extraction template.

This module defines :class:`ProductDetailExtractor`, a small Template
Method that handles the common flow shared by every vendor scraper's
``fetch_product_details``:

1. Check the download cache for the product page URL and return cached
   details if available.
2. Otherwise, fetch and parse the HTML.
3. Extract name, images, datasheet URL, and specifications using
   vendor-specific selectors implemented by subclasses.
4. Cache the extracted details.

The extractor does **not** build a :class:`CameraRecord` — that is the
caller's responsibility because identity fields (camera_id, source,
product_series) are vendor-specific and naturally belong with the
scraper.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bs4 import BeautifulSoup
from loguru import logger

from src.storage.download_cache import DownloadCache

if TYPE_CHECKING:
    from src.scrapers.base import CameraScraperBase


@dataclass
class ProductDetails:
    """Raw extracted product details from a vendor page.

    :ivar name: Product display name
    :ivar images: List of image URLs (in the vendor's own scheme)
    :ivar datasheet_url: Datasheet PDF URL, or None if not advertised
    :ivar specifications_html: Specs grouped by section name
    """

    name: str
    images: list[str]
    datasheet_url: str | None
    specifications_html: dict[str, dict[str, str]]


class ProductDetailExtractor(ABC):
    """Template for cache-aware product detail extraction.

    Subclasses implement four vendor-specific extraction hooks. The
    template :meth:`extract` orchestrates cache lookup, HTML fetching,
    extraction, and write-back caching.

    :ivar scraper: Scraper used for the underlying HTTP fetch
    :ivar download_cache: Optional cache for product details
    """

    def __init__(
        self,
        scraper: "CameraScraperBase",
        download_cache: DownloadCache | None,
    ) -> None:
        """
        :param scraper: Scraper providing :meth:`fetch_html`
        :param download_cache: Optional product-details cache
        """
        self.scraper = scraper
        self.download_cache = download_cache

    async def extract(self, product_page_url: str) -> ProductDetails:
        """Fetch (or retrieve from cache) the product details for a page.

        :param product_page_url: Absolute URL of the product page
        :return: Extracted product details
        """
        cached = self._load_from_cache(product_page_url)
        if cached is not None:
            logger.info(f"Using cached product details for {product_page_url}")
            return cached

        html = await self.scraper.fetch_html(product_page_url)
        soup = BeautifulSoup(html, "html.parser")

        name = self._extract_name(soup) or "Unknown Product"
        if name == "Unknown Product":
            logger.warning(f"Could not extract product name from {product_page_url}")

        images = self._extract_images(soup)
        datasheet_url = self._extract_datasheet_url(soup)
        specifications_html = self._extract_specifications(soup)

        details = ProductDetails(
            name=name,
            images=images,
            datasheet_url=datasheet_url,
            specifications_html=specifications_html,
        )

        self._store_in_cache(product_page_url, details)

        logger.info(f"Extracted {len(images)} images for {name}")
        if datasheet_url:
            logger.info(f"Found datasheet: {datasheet_url}")
        logger.info(f"Found {len(specifications_html)} specification sections")

        return details

    def _load_from_cache(self, product_page_url: str) -> ProductDetails | None:
        """Return cached details for the URL, or None if absent.

        :param product_page_url: Absolute URL of the product page
        :return: Cached ProductDetails, or None
        """
        if not self.download_cache or not self.download_cache.has_cached_product(
            product_page_url
        ):
            return None

        cached = self.download_cache.get_cached_product(product_page_url)
        if cached is None:
            return None

        return ProductDetails(
            name=cached["model_name"],
            images=cached["image_urls"],
            datasheet_url=cached["datasheet_url"],
            specifications_html=cached["specifications_html"],
        )

    def _store_in_cache(self, product_page_url: str, details: ProductDetails) -> None:
        """Persist freshly-extracted details to the cache, if configured.

        :param product_page_url: Absolute URL of the product page
        :param details: Extracted details to cache
        """
        if not self.download_cache:
            return

        self.download_cache.cache_product(
            product_page_url,
            details.name,
            details.images,
            details.datasheet_url,
            details.specifications_html,
        )

    @abstractmethod
    def _extract_name(self, soup: BeautifulSoup) -> str | None:
        """Extract the product name from the page."""

    @abstractmethod
    def _extract_images(self, soup: BeautifulSoup) -> list[str]:
        """Extract product image URLs from the page."""

    @abstractmethod
    def _extract_datasheet_url(self, soup: BeautifulSoup) -> str | None:
        """Extract the datasheet PDF URL from the page."""

    @abstractmethod
    def _extract_specifications(self, soup: BeautifulSoup) -> dict[str, dict[str, str]]:
        """Extract grouped specifications from the page."""
