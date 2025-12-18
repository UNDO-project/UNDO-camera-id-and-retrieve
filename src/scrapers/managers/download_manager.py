"""Download manager for images and PDFs with caching and deduplication."""

from typing import Protocol, List
from loguru import logger

from src.models.camera import CameraRecord
from src.storage.download_cache import DownloadCache
from src.storage.dataset import DatasetManager
from src.scrapers.managers.url_normalizer import URLNormalizer


class ContentDownloader(Protocol):
    """Protocol for content downloading strategies."""

    async def download(self, url: str) -> bytes:
        """
        Download content from URL.

        :param url: Absolute URL to download from
        :return: Downloaded content as bytes
        """
        ...


class DownloadManager:
    """
    Manages downloading, caching, and organizing images and PDFs.

    Uses composition to handle vendor-specific download strategies.
    Follows the Strategy pattern for URL normalization and downloading.
    """

    def __init__(
        self,
        url_normalizer: URLNormalizer,
        download_cache: DownloadCache | None = None,
    ):
        """
        Initialize download manager.

        :param url_normalizer: Strategy for normalizing URLs
        :param download_cache: Optional cache for tracking downloads
        """
        self.url_normalizer = url_normalizer
        self.download_cache = download_cache
        self.dataset_manager = DatasetManager()

    async def download_and_organize_images(
        self,
        record: CameraRecord,
        image_downloader: ContentDownloader,
        base_url: str,
    ) -> List[str]:
        """
        Download all images for a product and organize by product ID.

        Handles caching, deduplication, and error recovery automatically.

        :param record: CameraRecord containing image URLs
        :param image_downloader: Strategy for downloading images
        :param base_url: Base URL for normalizing relative URLs
        :return: List of local file paths
        """
        if not record.images:
            return []

        logger.info(f"Processing {len(record.images)} images for {record.model_name}")
        image_data_list = []
        skipped_count = 0

        for idx, image_url in enumerate(record.images):
            try:
                # Normalize URL using strategy
                full_url = self.url_normalizer.normalize(image_url, base_url)
                if not full_url:
                    continue

                # Check cache before downloading
                if self.download_cache and self.download_cache.has_downloaded(full_url):
                    logger.debug(f"Skipping cached image: {full_url}")
                    skipped_count += 1
                    continue

                # Download using provided strategy
                image_data = await image_downloader.download(full_url)

                # Check for duplicate content by hash
                if self.download_cache:
                    duplicate_path = self.download_cache.check_content_duplicate(
                        image_data
                    )
                    if duplicate_path:
                        logger.info(f"Image content already stored at {duplicate_path}")
                        image_data_list.append((full_url, image_data))
                        skipped_count += 1
                        continue

                image_data_list.append((full_url, image_data))
                logger.debug(f"Downloaded image {idx + 1}/{len(record.images)}")

            except Exception as e:
                logger.error(f"Failed to download image {image_url}: {e}")
                continue

        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} cached image(s)")

        # Store downloaded images using DatasetManager
        local_paths = self.dataset_manager.organize_images(
            record, image_data_list, self.download_cache
        )

        logger.info(
            f"Saved {len(local_paths)} images for {record.model_name} "
            f"(skipped {skipped_count})"
        )
        return local_paths

    async def download_and_save_pdf(
        self,
        record: CameraRecord,
        pdf_downloader: ContentDownloader,
        base_url: str,
    ) -> str | None:
        """
        Download datasheet PDF and save to organized location.

        Handles caching and deduplication automatically.

        :param record: CameraRecord containing datasheet URL
        :param pdf_downloader: Strategy for downloading PDFs
        :param base_url: Base URL for normalizing relative URLs
        :return: Local file path or None if failed
        """
        if not record.datasheet_url:
            return None

        logger.info(f"Processing PDF for {record.model_name}")
        try:
            # Normalize URL using strategy
            full_url = self.url_normalizer.normalize(record.datasheet_url, base_url)
            if not full_url:
                return None

            # Check cache before downloading
            if self.download_cache and self.download_cache.has_downloaded(full_url):
                cached_path = self.download_cache.get_downloaded_path(full_url)
                logger.info(f"Using cached PDF for {record.model_name}: {cached_path}")
                return cached_path

            # Download PDF using provided strategy
            pdf_data = await pdf_downloader.download(full_url)

            # Check for duplicate content by hash
            if self.download_cache:
                duplicate_path = self.download_cache.check_content_duplicate(pdf_data)
                if duplicate_path:
                    logger.info(
                        f"PDF content already stored at {duplicate_path}, "
                        f"reusing for {record.model_name}"
                    )
                    return duplicate_path

            # Store PDF using DatasetManager
            local_path = self.dataset_manager.save_pdf(
                record, pdf_data, self.download_cache, full_url
            )

            logger.info(f"Saved PDF for {record.model_name}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download PDF for {record.model_name}: {e}")
            return None
