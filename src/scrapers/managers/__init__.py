"""Download and URL management utilities for scrapers."""

from src.scrapers.managers.download_manager import DownloadManager
from src.scrapers.managers.product_detail_extractor import (
    ProductDetailExtractor,
    ProductDetails,
)
from src.scrapers.managers.url_normalizer import (
    ProtocolRelativeURLNormalizer,
    RelativeURLNormalizer,
    URLNormalizer,
)

__all__ = [
    "URLNormalizer",
    "RelativeURLNormalizer",
    "ProtocolRelativeURLNormalizer",
    "DownloadManager",
    "ProductDetailExtractor",
    "ProductDetails",
]
