"""Main entry point for cctv-scrapers."""

import asyncio

from loguru import logger

from src.scrapers.axis import AxisCameraScraper


async def test_axis_scraper() -> None:
    r"""
    Visit all categories, then all product series within each category, then all individual products.
    Extract detailed information for each product.
    """
    scraper = AxisCameraScraper()
    try:
        categories = await scraper.fetch_categories()

        for category in categories:
            logger.info(f"Visiting category: {category.name}")
            series_list = await scraper.fetch_cameras(category)

            for series in series_list:
                logger.info(f"  Visiting series: {series.name}")
                products = await scraper.fetch_products_in_series(series)
                for product_url in products:
                    logger.info(f"    Processing product: {product_url}")
                    try:
                        product_details = await scraper.fetch_product_details(
                            product_url, series.name
                        )
                        logger.info(f"      Model: {product_details.model_name}")
                        logger.info(f"      Images: {len(product_details.images)}")
                        logger.info(
                            f"      Specs sections: {len(product_details.specifications_html)}"
                        )
                    except Exception as e:
                        logger.error(f"      Error fetching product details: {e}")

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
