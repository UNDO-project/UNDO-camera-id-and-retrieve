"""Test script to verify product detail extraction."""

from pathlib import Path

from bs4 import BeautifulSoup
from loguru import logger

from src.scrapers.axis import AxisCameraScraper

logger.add(lambda msg: print(msg, end=""))


def test_product_extraction() -> None:
    r"""
    Test product detail extraction using local HTML file.
    """
    html_file = Path(
        "/test_html/AXIS M3057-PLR Mk II Dome Camera | Axis Communications.html"
    )

    if not html_file.exists():
        logger.error(f"Test HTML file not found: {html_file}")
        return

    logger.info(f"Loading test HTML: {html_file.name}")
    with open(html_file, encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")
    scraper = AxisCameraScraper()
    extractor = scraper.product_extractor

    # Test image extraction
    logger.info("Testing carousel image extraction...")
    images = extractor._extract_images(soup)
    logger.info(f"  Found {len(images)} carousel images")
    for idx, img_url in enumerate(images, 1):
        logger.info(f"    {idx}. {img_url[:80]}...")

    # Test datasheet extraction
    logger.info("Testing datasheet URL extraction...")
    datasheet = extractor._extract_datasheet_url(soup)
    if datasheet:
        logger.info(f"  Found datasheet: {datasheet}")
    else:
        logger.warning("  No datasheet found")

    # Test specifications extraction
    logger.info("Testing specifications extraction...")
    specs = extractor._extract_specifications(soup)
    logger.info(f"  Found {len(specs)} specification sections")
    for section, section_specs in specs.items():
        logger.info(f"    [{section}] - {len(section_specs)} specs")
        for spec_name, spec_value in list(section_specs.items())[:3]:
            logger.info(f"      {spec_name}: {spec_value}")
        if len(section_specs) > 3:
            logger.info(f"      ... and {len(section_specs) - 3} more")


if __name__ == "__main__":
    test_product_extraction()
