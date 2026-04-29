"""Unit tests for Hikvision Playwright pagination helpers.

These helpers were extracted from three near-duplicate fetchers
(``fetch_product_urls_with_playwright``, ``fetch_its_product_urls``,
``fetch_thermal_product_urls``). The tests use AsyncMock objects to
simulate the Playwright Page surface so we don't need a real browser.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.scrapers.hikvision import HikvisionCameraScraper


def _fake_page_for_count(text_contents: list[str]) -> MagicMock:
    """Build a fake Playwright page whose product_count locator yields the given text."""
    locator = MagicMock()
    locator.all_text_contents = AsyncMock(return_value=text_contents)
    page = MagicMock()
    page.locator = MagicMock(return_value=locator)
    return page


class TestReadProductCount:
    """Tests for _read_product_count."""

    @pytest.mark.asyncio
    async def test_parses_digits_from_text(self):
        page = _fake_page_for_count(["123 products"])
        count = await HikvisionCameraScraper._read_product_count(page)
        assert count == 123

    @pytest.mark.asyncio
    async def test_returns_zero_when_no_digits(self):
        page = _fake_page_for_count(["", "no count"])
        count = await HikvisionCameraScraper._read_product_count(page)
        assert count == 0

    @pytest.mark.asyncio
    async def test_uses_max_when_multiple_values_present(self):
        page = _fake_page_for_count(["12", "34"])
        count = await HikvisionCameraScraper._read_product_count(page)
        assert count == 34

    @pytest.mark.asyncio
    async def test_handles_none_entries(self):
        page = _fake_page_for_count([None, "  ", "7"])  # type: ignore[list-item]
        count = await HikvisionCameraScraper._read_product_count(page)
        assert count == 7


class TestCollectLinksOnPage:
    """Tests for _collect_links_on_page."""

    @pytest.mark.asyncio
    async def test_appends_unique_links(self):
        link_a = MagicMock()
        link_a.get_attribute = AsyncMock(return_value="/p/a")
        link_b = MagicMock()
        link_b.get_attribute = AsyncMock(return_value="/p/b")

        locator = MagicMock()
        locator.count = AsyncMock(return_value=2)
        locator.nth = MagicMock(side_effect=[link_a, link_b])

        page = MagicMock()
        page.locator = MagicMock(return_value=locator)

        product_urls: list[str] = []
        await HikvisionCameraScraper._collect_links_on_page(page, product_urls)

        assert product_urls == ["/p/a", "/p/b"]

    @pytest.mark.asyncio
    async def test_skips_duplicates(self):
        link = MagicMock()
        link.get_attribute = AsyncMock(return_value="/p/a")

        locator = MagicMock()
        locator.count = AsyncMock(return_value=1)
        locator.nth = MagicMock(return_value=link)

        page = MagicMock()
        page.locator = MagicMock(return_value=locator)

        product_urls: list[str] = ["/p/a"]
        await HikvisionCameraScraper._collect_links_on_page(page, product_urls)

        assert product_urls == ["/p/a"]

    @pytest.mark.asyncio
    async def test_skips_falsy_hrefs(self):
        link_none = MagicMock()
        link_none.get_attribute = AsyncMock(return_value=None)
        link_empty = MagicMock()
        link_empty.get_attribute = AsyncMock(return_value="")
        link_real = MagicMock()
        link_real.get_attribute = AsyncMock(return_value="/p/x")

        locator = MagicMock()
        locator.count = AsyncMock(return_value=3)
        locator.nth = MagicMock(side_effect=[link_none, link_empty, link_real])

        page = MagicMock()
        page.locator = MagicMock(return_value=locator)

        product_urls: list[str] = []
        await HikvisionCameraScraper._collect_links_on_page(page, product_urls)

        assert product_urls == ["/p/x"]


class TestAdvanceToNextPage:
    """Tests for _advance_to_next_page."""

    @pytest.mark.asyncio
    async def test_returns_false_when_button_hidden(self):
        next_btn = MagicMock()
        next_btn.is_visible = AsyncMock(return_value=False)

        page = MagicMock()
        page.locator = MagicMock(return_value=next_btn)

        result = await HikvisionCameraScraper._advance_to_next_page(page)
        assert result is False

    @pytest.mark.asyncio
    async def test_clicks_and_waits_when_visible(self):
        next_btn = MagicMock()
        next_btn.is_visible = AsyncMock(return_value=True)
        next_btn.click = AsyncMock()

        page = MagicMock()
        page.locator = MagicMock(return_value=next_btn)
        page.wait_for_load_state = AsyncMock()
        page.wait_for_selector = AsyncMock()

        result = await HikvisionCameraScraper._advance_to_next_page(page)

        assert result is True
        next_btn.click.assert_awaited_once()
        page.wait_for_load_state.assert_awaited_once_with("networkidle")
        page.wait_for_selector.assert_awaited_once()


class TestExtractProductUrlsWithPlaywright:
    """Tests for the _extract_product_urls_with_playwright orchestrator."""

    @pytest.mark.asyncio
    async def test_iterates_until_total_count_reached(self, monkeypatch):
        scraper = HikvisionCameraScraper.__new__(HikvisionCameraScraper)

        # Single page with 2 products and total_count == 2 -> stops after first iteration.
        page = MagicMock()
        page.goto = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.close = AsyncMock()

        browser = MagicMock()
        browser.new_page = AsyncMock(return_value=page)

        async def fake_get_browser(self):
            return browser

        async def fake_read_count(p):
            return 2

        async def fake_collect(p, urls):
            urls.append("/p/a")
            urls.append("/p/b")

        async def fake_advance(p):
            return False  # not reached because count==len(urls)

        monkeypatch.setattr(HikvisionCameraScraper, "_get_browser", fake_get_browser)
        monkeypatch.setattr(
            HikvisionCameraScraper,
            "_read_product_count",
            staticmethod(fake_read_count),
        )
        monkeypatch.setattr(
            HikvisionCameraScraper,
            "_collect_links_on_page",
            staticmethod(fake_collect),
        )
        monkeypatch.setattr(
            HikvisionCameraScraper,
            "_advance_to_next_page",
            staticmethod(fake_advance),
        )

        urls = await scraper._extract_product_urls_with_playwright(
            url="https://example.com/cameras",
            label="ITS",
        )

        assert urls == ["/p/a", "/p/b"]
        page.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stops_when_advance_returns_false(self, monkeypatch):
        scraper = HikvisionCameraScraper.__new__(HikvisionCameraScraper)

        page = MagicMock()
        page.goto = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.close = AsyncMock()

        browser = MagicMock()
        browser.new_page = AsyncMock(return_value=page)

        async def fake_get_browser(self):
            return browser

        async def fake_read_count(p):
            # Unknown total: only stop when advance returns False.
            return 0

        collect_calls = {"count": 0}

        async def fake_collect(p, urls):
            collect_calls["count"] += 1
            urls.append(f"/p/{collect_calls['count']}")

        async def fake_advance(p):
            # Advance once, then stop.
            return collect_calls["count"] < 2

        monkeypatch.setattr(HikvisionCameraScraper, "_get_browser", fake_get_browser)
        monkeypatch.setattr(
            HikvisionCameraScraper,
            "_read_product_count",
            staticmethod(fake_read_count),
        )
        monkeypatch.setattr(
            HikvisionCameraScraper,
            "_collect_links_on_page",
            staticmethod(fake_collect),
        )
        monkeypatch.setattr(
            HikvisionCameraScraper,
            "_advance_to_next_page",
            staticmethod(fake_advance),
        )

        urls = await scraper._extract_product_urls_with_playwright(
            url="https://example.com/x",
            label="Thermal",
        )

        assert urls == ["/p/1", "/p/2"]

    @pytest.mark.asyncio
    async def test_apply_filter_runs_before_collection(self, monkeypatch):
        scraper = HikvisionCameraScraper.__new__(HikvisionCameraScraper)

        page = MagicMock()
        page.goto = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.wait_for_timeout = AsyncMock()
        page.close = AsyncMock()

        browser = MagicMock()
        browser.new_page = AsyncMock(return_value=page)

        order: list[str] = []

        async def fake_filter(p):
            order.append("filter")

        async def fake_get_browser(self):
            return browser

        async def fake_read_count(p):
            order.append("count")
            return 1

        async def fake_collect(p, urls):
            order.append("collect")
            urls.append("/p/x")

        async def fake_advance(p):
            return False

        monkeypatch.setattr(HikvisionCameraScraper, "_get_browser", fake_get_browser)
        monkeypatch.setattr(
            HikvisionCameraScraper,
            "_read_product_count",
            staticmethod(fake_read_count),
        )
        monkeypatch.setattr(
            HikvisionCameraScraper,
            "_collect_links_on_page",
            staticmethod(fake_collect),
        )
        monkeypatch.setattr(
            HikvisionCameraScraper,
            "_advance_to_next_page",
            staticmethod(fake_advance),
        )

        await scraper._extract_product_urls_with_playwright(
            url="https://example.com/x",
            label="IP/Network Cameras",
            apply_filter=fake_filter,
        )

        # Filter must precede the count read and collection.
        assert order.index("filter") < order.index("count") < order.index("collect")
