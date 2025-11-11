"""Scraping stage: Download CCTV data and organize on filesystem."""

import asyncio

from loguru import logger

from src.scrapers.axis import AxisCameraScraper
from src.storage.download_cache import DownloadCache
from src.storage.manifest import ManifestRecorder


async def scrape_products() -> None:
    r"""
    Scrape CCTV products and save to filesystem.

    This stage:
    - Downloads images and PDFs
    - Organizes files by category/series/product
    - Creates verification manifest
    - Skips already-downloaded content to reduce server burden
    """
    # Initialize download cache
    download_cache = DownloadCache()
    cache_stats = download_cache.get_stats()
    logger.info(
        f"Loaded download cache: {cache_stats['total_downloads']} downloads cached"
    )

    scraper = AxisCameraScraper(download_cache=download_cache)
    manifest_recorder = ManifestRecorder()

    try:
        categories = await scraper.fetch_categories()
        logger.info(f"Processing {len(categories)} categories")

        for category in categories:
            logger.info(f"Category: {category.name}")
            series_list = await scraper.fetch_cameras(category)

            for series in series_list:
                logger.info(f"  Series: {series.name}")
                products = await scraper.fetch_products_in_series(series)

                for product_url in products:
                    logger.info(f"    Processing: {product_url}")
                    try:
                        # Extract product details with full category/series context
                        product_details = await scraper.fetch_product_details(
                            product_url,
                            category_name=category.name,
                            series_name=series.name,
                        )
                        logger.info(f"      Product: {product_details.model_name}")
                        logger.info(
                            f"      Found: {len(product_details.images)} images, {len(product_details.specifications_html)} spec sections"
                        )

                        # Download images
                        if product_details.images:
                            image_paths = await scraper.download_and_organize_images(
                                product_details
                            )
                            product_details.images = image_paths

                        # Download PDF
                        if product_details.datasheet_url:
                            pdf_path = await scraper.download_and_save_pdf(
                                product_details
                            )
                            if pdf_path:
                                product_details.datasheet_pdf = pdf_path

                        # Record for manifest
                        manifest_recorder.record_product(product_details)
                        logger.success(f"      Saved {product_details.model_name}")

                    except Exception as e:
                        logger.error(f"      Error processing product: {e}")
                        continue

        # Save manifest
        logger.info("Saving verification manifest...")
        manifest_recorder.save_manifest()
        logger.success("Scraping complete!")

        # Report cache statistics
        final_stats = download_cache.get_stats()
        logger.info(
            f"Final cache statistics: {final_stats['total_downloads']} total downloads, "
            f"{final_stats['images']} images, {final_stats['pdfs']} PDFs, "
            f"{final_stats['cached_products']} cached products, "
            f"{final_stats['total_size_bytes'] / (1024 * 1024):.2f}MB total"
        )

    except Exception as e:
        logger.error(f"Error during dataset building: {e}")
        raise
    finally:
        await scraper.close()
        download_cache.close()


def main() -> None:
    r"""
    Run the CCTV scraping stage.
    """
    logger.info("CCTV Scraping Stage started")
    asyncio.run(scrape_products())


if __name__ == "__main__":
    main()
