"""Tests for ProductDetailExtractor template."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from bs4 import BeautifulSoup

from src.scrapers.managers.product_detail_extractor import (
    ProductDetailExtractor,
    ProductDetails,
)


class StubExtractor(ProductDetailExtractor):
    """Concrete extractor returning fixed data; used to test the template."""

    def _extract_name(self, soup: BeautifulSoup) -> str | None:
        return "Stub Camera"

    def _extract_images(self, soup: BeautifulSoup) -> list[str]:
        return ["https://example.com/img1.jpg"]

    def _extract_datasheet_url(self, soup: BeautifulSoup) -> str | None:
        return "https://example.com/datasheet.pdf"

    def _extract_specifications(self, soup: BeautifulSoup) -> dict[str, dict[str, str]]:
        return {"General": {"Model": "Stub-1"}}


class NoNameExtractor(StubExtractor):
    def _extract_name(self, soup: BeautifulSoup) -> str | None:
        return None


@pytest.fixture
def scraper():
    s = MagicMock()
    s.fetch_html = AsyncMock(return_value="<html></html>")
    return s


class TestExtractWithoutCache:
    @pytest.mark.asyncio
    async def test_fetches_and_returns_details(self, scraper):
        extractor = StubExtractor(scraper, download_cache=None)
        details = await extractor.extract("https://example.com/product/1")

        assert isinstance(details, ProductDetails)
        assert details.name == "Stub Camera"
        assert details.images == ["https://example.com/img1.jpg"]
        assert details.datasheet_url == "https://example.com/datasheet.pdf"
        assert details.specifications_html == {"General": {"Model": "Stub-1"}}
        scraper.fetch_html.assert_awaited_once_with("https://example.com/product/1")

    @pytest.mark.asyncio
    async def test_unknown_product_when_name_missing(self, scraper):
        extractor = NoNameExtractor(scraper, download_cache=None)
        details = await extractor.extract("https://example.com/product/2")
        assert details.name == "Unknown Product"


class TestExtractWithCache:
    @pytest.mark.asyncio
    async def test_uses_cached_details_when_present(self, scraper):
        cache = MagicMock()
        cache.has_cached_product.return_value = True
        cache.get_cached_product.return_value = {
            "model_name": "Cached Camera",
            "image_urls": ["cached.jpg"],
            "datasheet_url": "cached.pdf",
            "specifications_html": {"S": {"k": "v"}},
        }

        extractor = StubExtractor(scraper, download_cache=cache)
        details = await extractor.extract("https://example.com/product/3")

        assert details.name == "Cached Camera"
        assert details.images == ["cached.jpg"]
        scraper.fetch_html.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_writes_freshly_extracted_details_to_cache(self, scraper):
        cache = MagicMock()
        cache.has_cached_product.return_value = False

        extractor = StubExtractor(scraper, download_cache=cache)
        await extractor.extract("https://example.com/product/4")

        cache.cache_product.assert_called_once_with(
            "https://example.com/product/4",
            "Stub Camera",
            ["https://example.com/img1.jpg"],
            "https://example.com/datasheet.pdf",
            {"General": {"Model": "Stub-1"}},
        )

    @pytest.mark.asyncio
    async def test_no_cache_write_when_cache_absent(self, scraper):
        extractor = StubExtractor(scraper, download_cache=None)
        # Should not raise even though there's no cache to consult/write to.
        await extractor.extract("https://example.com/product/5")
