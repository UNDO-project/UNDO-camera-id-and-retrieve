"""Tests for download cache functionality."""

import tempfile
from pathlib import Path

import pytest

from src.storage.download_cache import DownloadCache


def test_download_cache_initialization():
    r"""
    Test that download cache initializes correctly.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = DownloadCache(cache_path=cache_path)

        assert cache.cache_path == cache_path
        assert cache.cache_path.exists()
        cache.close()


def test_mark_and_check_downloaded():
    r"""
    Test marking a URL as downloaded and checking if it exists.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = DownloadCache(cache_path=cache_path)

        url = "https://example.com/image.png"
        local_path = Path(tmpdir) / "image.png"

        # Should not be cached initially
        assert not cache.has_downloaded(url)

        # Mark as downloaded
        cache.mark_downloaded(url, local_path, "image", file_content=b"fake image data")

        # Should be cached now
        assert cache.has_downloaded(url)
        cache.close()


def test_get_downloaded_path():
    r"""
    Test retrieving the local path of a downloaded file.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = DownloadCache(cache_path=cache_path)

        url = "https://example.com/document.pdf"
        local_path = Path(tmpdir) / "docs" / "document.pdf"

        cache.mark_downloaded(url, local_path, "pdf", file_content=b"fake pdf data")

        retrieved_path = cache.get_downloaded_path(url)
        assert retrieved_path == str(local_path)
        cache.close()


def test_cache_statistics():
    r"""
    Test cache statistics reporting.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = DownloadCache(cache_path=cache_path)

        # Add some test entries
        cache.mark_downloaded(
            "https://example.com/image1.png",
            Path(tmpdir) / "img1.png",
            "image",
            file_content=b"image data 1" * 85,  # ~1024 bytes
        )
        cache.mark_downloaded(
            "https://example.com/image2.png",
            Path(tmpdir) / "img2.png",
            "image",
            file_content=b"image data 2" * 171,  # ~2048 bytes
        )
        cache.mark_downloaded(
            "https://example.com/doc.pdf",
            Path(tmpdir) / "doc.pdf",
            "pdf",
            file_content=b"pdf data" * 640,  # ~5120 bytes
        )

        stats = cache.get_stats()

        assert stats["total_downloads"] == 3
        assert stats["images"] == 2
        assert stats["pdfs"] == 1
        assert stats["total_size_bytes"] == 8192

        cache.close()


def test_context_manager():
    r"""
    Test using DownloadCache as a context manager.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"

        with DownloadCache(cache_path=cache_path) as cache:
            url = "https://example.com/test.png"
            cache.mark_downloaded(
                url, Path(tmpdir) / "test.png", "image", file_content=b"test image"
            )
            assert cache.has_downloaded(url)

        # Connection should be closed after context
        # (Further operations would fail if attempted)


def test_duplicate_url_handling():
    r"""
    Test that marking the same URL twice logs a warning but doesn't fail.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = DownloadCache(cache_path=cache_path)

        url = "https://example.com/image.png"
        path1 = Path(tmpdir) / "image1.png"
        path2 = Path(tmpdir) / "image2.png"

        # Mark first time
        cache.mark_downloaded(url, path1, "image", file_content=b"image data")

        # Mark second time with different path (should not raise)
        cache.mark_downloaded(url, path2, "image", file_content=b"image data")

        # Should return the first path that was cached
        assert cache.get_downloaded_path(url) == str(path1)
        cache.close()


def test_product_caching():
    r"""
    Test caching and retrieving product specifications.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = DownloadCache(cache_path=cache_path)

        product_url = "https://example.com/products/camera-123"
        model_name = "AXIS M3004-V"
        image_urls = ["https://example.com/img1.png", "https://example.com/img2.png"]
        datasheet_url = "https://example.com/datasheet.pdf"
        specifications = {
            "Video": {"Resolution": "1280x720", "Frame rate": "30 fps"},
            "Audio": {"Built-in microphone": "Yes"},
        }

        # Should not be cached initially
        assert not cache.has_cached_product(product_url)
        assert cache.get_cached_product(product_url) is None

        # Cache the product
        cache.cache_product(
            product_url, model_name, image_urls, datasheet_url, specifications
        )

        # Should be cached now
        assert cache.has_cached_product(product_url)

        # Retrieve cached data
        cached_data = cache.get_cached_product(product_url)
        assert cached_data is not None
        assert cached_data["model_name"] == model_name
        assert cached_data["image_urls"] == image_urls
        assert cached_data["datasheet_url"] == datasheet_url
        assert cached_data["specifications_html"] == specifications

        cache.close()


def test_updated_cache_statistics():
    r"""
    Test cache statistics including cached products.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = DownloadCache(cache_path=cache_path)

        # Add file downloads
        cache.mark_downloaded(
            "https://example.com/image.png",
            Path(tmpdir) / "img.png",
            "image",
            file_content=b"image data" * 102,  # ~1024 bytes
        )

        # Add product cache
        cache.cache_product(
            "https://example.com/products/camera-123",
            "AXIS M3004-V",
            ["https://example.com/img.png"],
            "https://example.com/datasheet.pdf",
            {"Video": {"Resolution": "1280x720"}},
        )

        stats = cache.get_stats()

        assert stats["total_downloads"] == 1
        assert stats["images"] == 1
        assert stats["cached_products"] == 1

        cache.close()


def test_content_deduplication():
    r"""
    Test that identical content from different URLs is detected.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = DownloadCache(cache_path=cache_path)

        # Identical content from two different URLs
        content = b"fake image data 12345"
        url1 = "https://example.com/image1.png"
        url2 = "https://other.com/picture.png"
        path1 = Path(tmpdir) / "image1.png"
        path2 = Path(tmpdir) / "image2.png"

        # Download and cache from first URL
        cache.mark_downloaded(url1, path1, "image", file_content=content)
        assert cache.has_downloaded(url1)

        # Check if duplicate exists - should find the first one
        duplicate_path = cache.check_content_duplicate(content)
        assert duplicate_path == str(path1)

        # Second URL with same content should also find the duplicate
        cache.mark_downloaded(url2, path2, "image", file_content=content)
        duplicate_path = cache.check_content_duplicate(content)
        assert duplicate_path is not None

        cache.close()


def test_different_content_no_duplicate():
    r"""
    Test that different content is not flagged as duplicate.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = Path(tmpdir) / "test_cache.db"
        cache = DownloadCache(cache_path=cache_path)

        content1 = b"image data 1"
        content2 = b"image data 2"
        url1 = "https://example.com/image1.png"
        path1 = Path(tmpdir) / "image1.png"

        # Cache first content
        cache.mark_downloaded(url1, path1, "image", file_content=content1)

        # Check different content - should not find duplicate
        duplicate_path = cache.check_content_duplicate(content2)
        assert duplicate_path is None

        cache.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
