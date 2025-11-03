"""Main entry point for cctv-scrapers."""

import asyncio
import json
from pathlib import Path

from loguru import logger

from src.scrapers.axis import AxisCameraScraper


async def test_axis_scraper() -> None:
    r"""
    Test the Axis camera scraper.
    """
    scraper = AxisCameraScraper()
    try:
        logger.info("Fetching Axis camera categories...")
        categories = await scraper.fetch_categories()
        logger.info(f"Found {len(categories)} categories")
        for cat in categories:
            logger.debug(f"Category: {cat.name} ({cat.href})")

        if categories:
            logger.info(f"Fetching cameras from {categories[0].name}...")
            cameras = await scraper.fetch_cameras(categories[0])
            logger.info(f"Found {len(cameras)} cameras")

            if cameras:
                camera = cameras[0]
                logger.info(f"Sample camera - ID: {camera.camera_id}")
                logger.debug(f"  Model: {camera.model_name}")
                logger.debug(f"  Description: {camera.description}")
                logger.debug(f"  Image URL: {camera.image_url}")

                output_file = Path("test_output.json")
                with open(output_file, "w") as f:
                    json.dump(
                        [c.model_dump() for c in cameras[:3]],
                        f,
                        indent=2,
                        default=str,
                    )
                logger.success(f"Saved sample data to {output_file}")
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
