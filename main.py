"""Main entry point for cctv-scrapers."""

import asyncio

from loguru import logger

from src.scrapers.axis import AxisCameraScraper


async def test_axis_scraper() -> None:
    r"""
    Visit all categories and log all products in each category.
    """
    scraper = AxisCameraScraper()
    try:
        categories = await scraper.fetch_categories()

        for category in categories:
            logger.info(f"Visiting category: {category.name}")
            cameras = await scraper.fetch_cameras(category)
            logger.info(f"  Products in {category.name}: {len(cameras)}")

    except Exception as e:
        logger.error(f"Error during scraping: {e}")
        raise
    finally:
        await scraper.close()


def main() -> None:
    r"""
    Run the CCTV scraper pipeline.
    """
    logger.info("CCTV Scrapers Pipeline started")
    asyncio.run(test_axis_scraper())


if __name__ == "__main__":
    main()
