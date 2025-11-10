"""Download cache management using SQLite.

Tracks downloaded content to prevent re-downloading
the same files and reduce server burden.
Uses content-based deduplication to avoid storing identical files.
"""

import hashlib
import sqlite3
from pathlib import Path

from loguru import logger

from src.config import OUTPUT_DIR


class DownloadCache:
    r"""
    Tracks downloaded content to prevent re-downloading.

    Uses SQLite to persistently store downloaded URLs,
    allowing resumable scraping and intelligent skipping.

    :ivar cache_path: Path to SQLite database file
    :ivar connection: Active SQLite connection
    """

    def __init__(self, cache_path: Path | str | None = None) -> None:
        r"""
        Initialize DownloadCache.

        :param cache_path: Path to SQLite database file
        """
        if cache_path is None:
            cache_path = OUTPUT_DIR / "download_cache.db"
        else:
            cache_path = Path(cache_path)

        self.cache_path = cache_path
        self.connection = self._init_database()

    def _init_database(self) -> sqlite3.Connection:
        r"""
        Initialize SQLite database with schema.

        :return: SQLite connection
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(str(self.cache_path))
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS downloaded_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                content_hash TEXT NOT NULL,
                file_type TEXT NOT NULL,
                local_path TEXT NOT NULL,
                file_size_bytes INTEGER,
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Create index on content_hash for fast deduplication lookups
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_hash ON downloaded_content(content_hash)"
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS product_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_url TEXT NOT NULL UNIQUE,
                model_name TEXT NOT NULL,
                image_urls TEXT NOT NULL,
                datasheet_url TEXT,
                specifications_html TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.commit()
        logger.debug(f"Initialized download cache at {self.cache_path}")
        return connection

    @staticmethod
    def _compute_content_hash(content: bytes) -> str:
        r"""
        Compute SHA256 hash of file content for deduplication.

        :param content: File content bytes
        :return: Hex digest of SHA256 hash
        """
        return hashlib.sha256(content).hexdigest()

    def has_downloaded(self, url: str) -> bool:
        r"""
        Check if URL has been previously downloaded.

        :param url: URL to check
        :return: True if URL was downloaded before
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT 1 FROM downloaded_content WHERE url = ?",
            (url,),
        )
        result = cursor.fetchone()
        return result is not None

    def get_downloaded_path(self, url: str) -> str | None:
        r"""
        Get local path of previously downloaded content.

        :param url: URL to look up
        :return: Local file path or None if not downloaded
        """
        cursor = self.connection.cursor()

        cursor.execute(
            "SELECT local_path FROM downloaded_content WHERE url = ?",
            (url,),
        )
        result = cursor.fetchone()
        return result["local_path"] if result else None

    def mark_downloaded(
        self,
        url: str,
        local_path: str | Path,
        file_type: str,
        file_content: bytes,
    ) -> None:
        r"""
        Record that a URL has been downloaded.

        :param url: URL that was downloaded
        :param local_path: Local file path where content was saved
        :param file_type: Type of file (image, pdf)
        :param file_content: Actual file bytes for content-based deduplication
        """
        cursor = self.connection.cursor()
        local_path_str = str(local_path)

        # Compute content hash for deduplication
        content_hash = self._compute_content_hash(file_content)
        file_size_bytes = len(file_content)

        try:
            cursor.execute(
                """
                INSERT INTO downloaded_content
                (url, content_hash, file_type, local_path, file_size_bytes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (url, content_hash, file_type, local_path_str, file_size_bytes),
            )
            self.connection.commit()
            logger.debug(f"Cached download: {url} -> {local_path_str}")
        except sqlite3.IntegrityError:
            logger.warning(f"URL already cached: {url}")

    def check_content_duplicate(self, content: bytes) -> str | None:
        r"""
        Check if content with this hash already exists in cache.

        :param content: File content bytes to check
        :return: Local path of existing file or None if not found
        """
        content_hash = self._compute_content_hash(content)
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT local_path FROM downloaded_content WHERE content_hash = ? LIMIT 1",
            (content_hash,),
        )
        result = cursor.fetchone()
        if result:
            logger.info(f"Content duplicate found: {result['local_path']}")
            return result["local_path"]
        return None

    def has_cached_product(self, product_url: str) -> bool:
        r"""
        Check if product details are cached.

        :param product_url: Product page URL
        :return: True if product is cached
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT 1 FROM product_cache WHERE product_url = ?",
            (product_url,),
        )
        result = cursor.fetchone()
        return result is not None

    def get_cached_product(self, product_url: str) -> dict | None:
        r"""
        Retrieve cached product details.

        :param product_url: Product page URL
        :return: Dictionary with cached product data or None
        """
        import json

        cursor = self.connection.cursor()
        cursor.execute(
            """SELECT model_name, image_urls, datasheet_url, specifications_html
               FROM product_cache WHERE product_url = ?""",
            (product_url,),
        )
        result = cursor.fetchone()

        if result:
            return {
                "model_name": result["model_name"],
                "image_urls": json.loads(result["image_urls"]),
                "datasheet_url": result["datasheet_url"],
                "specifications_html": json.loads(result["specifications_html"]),
            }
        return None

    def cache_product(
        self,
        product_url: str,
        model_name: str,
        image_urls: list[str],
        datasheet_url: str | None,
        specifications_html: dict,
    ) -> None:
        r"""
        Cache product details after extraction.

        :param product_url: Product page URL
        :param model_name: Extracted model name
        :param image_urls: List of image URLs
        :param datasheet_url: Datasheet URL if available
        :param specifications_html: Extracted specifications dictionary
        """
        import json

        cursor = self.connection.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO product_cache
                (product_url, model_name, image_urls, datasheet_url, specifications_html)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    product_url,
                    model_name,
                    json.dumps(image_urls),
                    datasheet_url,
                    json.dumps(specifications_html),
                ),
            )
            self.connection.commit()
            logger.debug(f"Cached product details: {product_url}")
        except sqlite3.IntegrityError:
            logger.warning(f"Product already cached: {product_url}")

    def get_stats(self) -> dict[str, int]:
        r"""
        Get cache statistics.

        :return: Dictionary with cache stats
        """
        cursor = self.connection.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM downloaded_content")
        total = cursor.fetchone()["total"]

        cursor.execute(
            "SELECT COUNT(*) as images FROM downloaded_content WHERE file_type = 'image'"
        )
        images = cursor.fetchone()["images"]

        cursor.execute(
            "SELECT COUNT(*) as pdfs FROM downloaded_content WHERE file_type = 'pdf'"
        )
        pdfs = cursor.fetchone()["pdfs"]

        cursor.execute(
            "SELECT SUM(file_size_bytes) as total_size FROM downloaded_content"
        )
        total_size = cursor.fetchone()["total_size"] or 0

        cursor.execute("SELECT COUNT(*) as total FROM product_cache")
        cached_products = cursor.fetchone()["total"]

        return {
            "total_downloads": total,
            "images": images,
            "pdfs": pdfs,
            "total_size_bytes": total_size,
            "cached_products": cached_products,
        }

    def clear_cache(self) -> None:
        r"""
        Clear all entries from cache.

        Use with caution - this will allow all files to be re-downloaded.
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM downloaded_content")
        self.connection.commit()
        logger.warning("Cleared download cache")

    def close(self) -> None:
        r"""
        Close database connection.

        Should be called when done with cache.
        """
        if self.connection:
            self.connection.close()
            logger.debug("Closed download cache")

    def __enter__(self):
        r"""
        Context manager entry.

        :return: Self for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        r"""
        Context manager exit - closes connection.
        """
        self.close()
