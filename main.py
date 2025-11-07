"""Main entry point for cctv-scrapers."""

import asyncio

from loguru import logger

from src.scrapers.axis import AxisCameraScraper
from src.storage.dataset import DatasetManager


async def build_dataset() -> None:
    r"""
    Build complete CCTV camera dataset with images, PDFs, and specifications.
    Downloads all media files and creates parquet dataset.
    """
    scraper = AxisCameraScraper()
    dataset_manager = DatasetManager()

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
                            dataset_manager.update_record_with_files(
                                product_details, image_files=image_paths
                            )

                        # Download PDF
                        if product_details.datasheet_url:
                            pdf_path = await scraper.download_and_save_pdf(
                                product_details
                            )
                            if pdf_path:
                                dataset_manager.update_record_with_files(
                                    product_details, pdf_file=pdf_path
                                )

                        # Add to dataset
                        dataset_manager.add_record(product_details)
                        logger.success(f"      Saved {product_details.model_name}")

                    except Exception as e:
                        logger.error(f"      Error processing product: {e}")
                        continue

        # Save final dataset
        logger.info("Saving dataset to parquet...")
        dataset_manager.save_dataset()
        logger.success(
            f"Dataset complete! {len(dataset_manager.records)} products saved"
        )

    except Exception as e:
        logger.error(f"Error during dataset building: {e}")
        raise
    finally:
        await scraper.close()


def main() -> None:
    r"""
    Run the CCTV scraper dataset building pipeline.
    """
    logger.info("CCTV Dataset Builder started")
    asyncio.run(build_dataset())


if __name__ == "__main__":
    main()
