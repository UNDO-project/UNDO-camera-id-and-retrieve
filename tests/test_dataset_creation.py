"""Test dataset creation with sample product HTML."""

from pathlib import Path

from bs4 import BeautifulSoup
from loguru import logger

from src.models.camera import CameraRecord
from src.scrapers.axis import AxisCameraScraper
from src.storage.dataset import DatasetManager

logger.add(lambda msg: print(msg, end=""))


def test_dataset_manager(tmp_path: Path) -> None:
    r"""
    Test DatasetManager with sample CameraRecord.
    """
    logger.info("Testing DatasetManager functionality...")

    # Create sample records
    records = [
        CameraRecord(
            camera_id="axis-m3057-plr-mk-ii",
            model_name="AXIS M3057-PLR Mk II Dome Camera",
            display_name="AXIS M3057-PLR Mk II",
            description="6 MP onboard dome with 360° panoramic view",
            source="Axis Communications",
            category="Dome Camera",
            product_category="Dome cameras",
            product_series="AXIS M30 Dome Camera Series",
            images=[
                "/sites/axis/files/styles/square_500x500_/public/2022-09/m3057_MkII_ceiling_angle_right_2002.png.webp",
                "/sites/axis/files/styles/square_500x500_/public/2022-09/m3057_MkII_wall_angle_left_2002.png.webp",
            ],
            datasheet_url="https://www.axis.com/dam/public/28/40/34/datasheet-axis-m3057-plr-mk-ii-dome-camera-en-US-487290.pdf",
            specifications_html={
                "Camera": {
                    "Image sensor": "CMOS",
                    "Image sensor size": '1/1.8"',
                    "Lightfinder": "Lightfinder",
                },
                "Video": {
                    "Max video resolution": "2016x2016",
                    "Max frames per second": "50/60",
                    "Day and Night functionality": "Yes",
                },
            },
        ),
        CameraRecord(
            camera_id="axis-m3085-v",
            model_name="AXIS M3085-V Dome Camera",
            display_name="AXIS M3085-V",
            description="Fixed 2 MP mini dome with deep learning",
            source="Axis Communications",
            category="Dome Camera",
            product_category="Dome cameras",
            product_series="AXIS M30 Dome Camera Series",
            images=["/sites/axis/files/m3085v.png"],
            datasheet_url=None,
            specifications_html={"Camera": {"Resolution": "2 MP"}},
        ),
    ]

    # Create manager and add records (use tmp_path to avoid modifying production data)
    test_dataset_path = tmp_path / "products_test.parquet"
    dataset_manager = DatasetManager(dataset_path=test_dataset_path)
    for record in records:
        dataset_manager.add_record(record)

    logger.info(f"Added {len(records)} records to dataset manager")

    # Save dataset
    dataset_manager.save_dataset()
    logger.success("Dataset saved successfully")

    # Verify parquet file was created
    assert test_dataset_path.exists(), f"Parquet file not created: {test_dataset_path}"
    file_size = test_dataset_path.stat().st_size
    assert file_size > 0, "Parquet file is empty"
    logger.success(f"Parquet file created: {test_dataset_path} ({file_size} bytes)")


def test_product_extraction_with_dataset_integration() -> None:
    r"""
    Test extraction and dataset integration with sample HTML.
    """
    html_file = Path(
        "/Users/jnap/Code/UNDO/cctv-scrapers/test_html/"
        "AXIS M3057-PLR Mk II Dome Camera | Axis Communications.html"
    )

    if not html_file.exists():
        logger.error(f"Test HTML file not found: {html_file}")
        return

    logger.info("Testing product extraction with dataset integration...")

    with open(html_file, encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    scraper = AxisCameraScraper()
    dataset_manager = DatasetManager()

    # Extract product details
    images = scraper._extract_carousel_images(soup)
    datasheet = scraper._extract_datasheet_url(soup)
    specs = scraper._extract_specifications_tables(soup)

    logger.info(f"Extracted {len(images)} images")
    logger.info(f"Found datasheet: {datasheet is not None}")
    logger.info(f"Extracted {len(specs)} spec sections")

    # Create record
    record = CameraRecord(
        camera_id="axis-m3057-plr-mk-ii",
        model_name="AXIS M3057-PLR Mk II",
        display_name="AXIS M3057-PLR Mk II",
        description="6 MP onboard dome",
        source="Axis Communications",
        category="Dome Camera",
        product_category="Dome cameras",
        product_series="AXIS M30 Dome Camera Series",
        images=images,
        datasheet_url=datasheet,
        specifications_html=specs,
    )

    dataset_manager.add_record(record)
    dataset_manager.save_dataset()

    logger.success("Dataset integration test completed")


if __name__ == "__main__":
    test_dataset_manager()
    print("\n")
    test_product_extraction_with_dataset_integration()
