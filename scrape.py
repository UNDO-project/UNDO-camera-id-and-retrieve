"""Stage 1: Scrape CCTV data and save to filesystem.

This is the scraping stage that downloads images and PDFs,
organizes them by category/series/product, and creates a verification manifest.

Usage:
    python scrape.py
"""

import sys

from src.scraping import main

if __name__ == "__main__":
    sys.argv[0] = "scrape.py"
    main.main()
