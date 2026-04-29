"""Tests for DownloadCacheStrategy."""

from pathlib import Path

import pytest

from src.scrapers.managers.download_manager import DownloadCacheStrategy
from src.storage.download_cache import DownloadCache


@pytest.fixture
def cache(tmp_path: Path) -> DownloadCache:
    """Create a DownloadCache backed by a tmp sqlite db."""
    return DownloadCache(cache_path=tmp_path / "cache.db")


class TestNoOpBehavior:
    """When constructed without a cache, all operations are no-ops."""

    def test_is_url_cached_returns_false(self):
        strategy = DownloadCacheStrategy(cache=None)
        assert strategy.is_url_cached("https://example.com/image.jpg") is False

    def test_cached_path_returns_none(self):
        strategy = DownloadCacheStrategy(cache=None)
        assert strategy.cached_path("https://example.com/image.jpg") is None

    def test_find_duplicate_content_returns_none(self):
        strategy = DownloadCacheStrategy(cache=None)
        assert strategy.find_duplicate_content(b"any data") is None


class TestWithCache:
    """When constructed with a cache, operations delegate to it."""

    def test_is_url_cached_when_unknown(self, cache: DownloadCache):
        strategy = DownloadCacheStrategy(cache=cache)
        assert strategy.is_url_cached("https://example.com/never-seen.jpg") is False

    def test_is_url_cached_when_recorded(self, cache: DownloadCache):
        url = "https://example.com/image.jpg"
        cache.mark_downloaded(url, "data/images/image.jpg", "image", b"some bytes")

        strategy = DownloadCacheStrategy(cache=cache)
        assert strategy.is_url_cached(url) is True

    def test_cached_path_returns_recorded_path(self, cache: DownloadCache):
        url = "https://example.com/image.jpg"
        local_path = "data/images/image.jpg"
        cache.mark_downloaded(url, local_path, "image", b"some bytes")

        strategy = DownloadCacheStrategy(cache=cache)
        assert strategy.cached_path(url) == local_path

    def test_cached_path_returns_none_for_unknown_url(self, cache: DownloadCache):
        strategy = DownloadCacheStrategy(cache=cache)
        assert strategy.cached_path("https://example.com/never-seen.jpg") is None

    def test_find_duplicate_content_returns_existing_path(self, cache: DownloadCache):
        data = b"some image bytes"
        cache.mark_downloaded(
            "https://example.com/a.jpg", "data/images/a.jpg", "image", data
        )

        strategy = DownloadCacheStrategy(cache=cache)
        assert strategy.find_duplicate_content(data) == "data/images/a.jpg"

    def test_find_duplicate_content_returns_none_for_new_content(
        self, cache: DownloadCache
    ):
        strategy = DownloadCacheStrategy(cache=cache)
        assert strategy.find_duplicate_content(b"unique bytes") is None
