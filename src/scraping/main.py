"""Scraping stage: Download CCTV data and organize on filesystem."""

import argparse
import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

from src.scrapers import AxisCameraScraper, HikvisionCameraScraper
from src.storage.download_cache import DownloadCache
from src.storage.manifest import ManifestRecorder

if TYPE_CHECKING:
    from src.scrapers.base import CameraScraperBase


@dataclass
class ScrapingContext:
    """Context object holding scraping state."""

    scraper: "CameraScraperBase"
    download_cache: DownloadCache
    manifest_recorder: ManifestRecorder
    vendor: str


def _initialize_scraping_context(vendor: str) -> ScrapingContext:
    r"""
    Initialize scraping context with cache, scraper, and manifest recorder.

    :param vendor: Vendor name ('axis' or 'hikvision')
    :return: ScrapingContext with initialized components
    :raises ValueError: If vendor is not supported
    """
    download_cache = DownloadCache()
    cache_stats = download_cache.get_stats()
    logger.info(
        f"Loaded download cache: {cache_stats['total_downloads']} downloads cached"
    )

    if vendor.lower() == "axis":
        scraper = AxisCameraScraper(download_cache=download_cache)
        logger.info("Using Axis Communications scraper")
    elif vendor.lower() == "hikvision":
        scraper = HikvisionCameraScraper(download_cache=download_cache)
        logger.info("Using HikVision scraper")
    else:
        raise ValueError(
            f"Unknown vendor: {vendor}. Supported vendors: axis, hikvision, all"
        )

    manifest_recorder = ManifestRecorder()

    return ScrapingContext(
        scraper=scraper,
        download_cache=download_cache,
        manifest_recorder=manifest_recorder,
        vendor=vendor,
    )


async def _process_product(
    product_url: str,
    category_name: str,
    series_name: str,
    context: ScrapingContext,
) -> bool:
    r"""
    Process a single product: fetch details, download assets, record to manifest.

    :param product_url: URL of the product page
    :param category_name: Category name for organization
    :param series_name: Series name for organization
    :param context: Scraping context with scraper and recorders
    :return: True if product processed successfully
    """
    try:
        # Extract product details with full category/series context
        product_details = await context.scraper.fetch_product_details(
            product_url,
            category_name=category_name,
            series_name=series_name,
        )
        logger.info(f"      Product: {product_details.model_name}")
        logger.info(
            f"      Found: {len(product_details.images)} images, "
            f"{len(product_details.specifications_html)} spec sections"
        )

        # Download images
        if product_details.images:
            image_paths = await context.scraper.download_and_organize_images(
                product_details
            )
            product_details.image_files = image_paths

        # Download PDF
        if product_details.datasheet_url:
            pdf_path = await context.scraper.download_and_save_pdf(product_details)
            if pdf_path:
                product_details.datasheet_file = pdf_path

        # Record for manifest
        context.manifest_recorder.record_product(product_details)
        logger.success(f"      Saved {product_details.model_name}")
        return True

    except Exception as e:
        logger.error(f"      Error processing product: {e}")
        return False


async def _scrape_series(
    series,
    category_name: str,
    context: ScrapingContext,
) -> int:
    r"""
    Scrape all products in a series.

    :param series: Series object with name
    :param category_name: Category name for organization
    :param context: Scraping context
    :return: Number of products successfully processed
    """
    logger.info(f"  Series: {series.name}")
    products = await context.scraper.fetch_products_in_series(series)

    processed_count = 0
    for product_url in products:
        logger.info(f"    Processing: {product_url}")
        success = await _process_product(
            product_url, category_name, series.name, context
        )
        if success:
            processed_count += 1

    return processed_count


async def _scrape_category(category, context: ScrapingContext) -> int:
    r"""
    Scrape all series in a category.

    :param category: Category object with name
    :param context: Scraping context
    :return: Number of products successfully processed
    """
    logger.info(f"Category: {category.name}")
    series_list = await context.scraper.fetch_cameras(category)

    total_processed = 0
    for series in series_list:
        count = await _scrape_series(series, category.name, context)
        total_processed += count

    return total_processed


async def _scrape_vendor(context: ScrapingContext) -> int:
    r"""
    Scrape all categories for a single vendor.

    :param context: Scraping context with initialized scraper
    :return: Number of products successfully processed
    """
    categories = await context.scraper.fetch_categories()
    logger.info(f"Processing {len(categories)} categories")

    total_processed = 0
    for category in categories:
        count = await _scrape_category(category, context)
        total_processed += count

    return total_processed


def _report_final_stats(context: ScrapingContext) -> None:
    r"""
    Report final scraping statistics.

    :param context: Scraping context with download cache
    """
    final_stats = context.download_cache.get_stats()
    logger.info(
        f"Final cache statistics: {final_stats['total_downloads']} total downloads, "
        f"{final_stats['images']} images, {final_stats['pdfs']} PDFs, "
        f"{final_stats['cached_products']} cached products, "
        f"{final_stats['total_size_bytes'] / (1024 * 1024):.2f}MB total"
    )


async def scrape_products(vendor: str = "axis") -> None:
    r"""
    Scrape CCTV products and save to filesystem.

    Orchestrates the scraping process by delegating to specialized methods:
    1. Handle 'all' vendors option (recursive)
    2. Initialize scraping context (cache, scraper, manifest recorder)
    3. Scrape all categories for the vendor
    4. Save manifest and report statistics

    This stage:
    - Downloads images and PDFs
    - Organizes files by category/series/product
    - Creates verification manifest
    - Skips already-downloaded content to reduce server burden

    :param vendor: Vendor to scrape ('axis', 'hikvision', or 'all')
    """
    # Handle 'all' option - recursively scrape each vendor
    if vendor.lower() == "all":
        logger.info("Scraping all vendors: axis, hikvision")
        for vendor_name in ["axis", "hikvision"]:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Starting scrape for vendor: {vendor_name.upper()}")
            logger.info(f"{'=' * 60}\n")
            await scrape_products(vendor=vendor_name)
        return

    # Initialize scraping context
    context = _initialize_scraping_context(vendor)

    try:
        # Scrape all categories for this vendor
        total_products = await _scrape_vendor(context)
        logger.info(f"Successfully processed {total_products} products")

        # Save manifest
        logger.info("Saving verification manifest...")
        context.manifest_recorder.save_manifest()
        logger.success("Scraping complete!")

        # Report statistics
        _report_final_stats(context)

    except Exception as e:
        logger.error(f"Error during dataset building: {e}")
        raise
    finally:
        await context.scraper.close()
        context.download_cache.close()


def main() -> None:
    r"""
    Run the CCTV scraping stage.

    Supports command-line argument to specify vendor.
    """
    parser = argparse.ArgumentParser(
        description="Scrape CCTV camera products from vendor websites"
    )
    parser.add_argument(
        "--vendor",
        type=str,
        default="axis",
        choices=["axis", "hikvision", "all"],
        help="Vendor to scrape (default: axis). Use 'all' to scrape all vendors",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the download cache (downloaded_content table) and exit.",
    )
    parser.add_argument(
        "--clear-cache-all",
        action="store_true",
        help="Clear ALL cache tables (downloaded_content + product_cache) and exit.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (use with --clear-cache / --clear-cache-all).",
    )
    args = parser.parse_args()

    if args.clear_cache or args.clear_cache_all:
        if not args.yes:
            which = "ALL cache tables" if args.clear_cache_all else "download cache"
            reply = input(f"This will clear {which}. Continue? [y/N]: ").strip().lower()
            if reply not in ("y", "yes"):
                logger.info("Aborted cache clear.")
                return

        cache = DownloadCache()
        try:
            if args.clear_cache_all:
                cache.clear_all_cache()
            else:
                cache.clear_cache()
        finally:
            cache.close()
        return

    logger.info(f"CCTV Scraping Stage started - Vendor: {args.vendor}")
    asyncio.run(scrape_products(vendor=args.vendor))


if __name__ == "__main__":
    main()
