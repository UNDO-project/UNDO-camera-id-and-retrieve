Scrapers
========

Base scraper interfaces and vendor-specific implementations.

Base interfaces
---------------

.. automodule:: src.scrapers.base
   :members:
   :undoc-members:
   :show-inheritance:

Axis Communications
-------------------

.. automodule:: src.scrapers.axis
   :members:
   :undoc-members:
   :show-inheritance:

HikVision
---------

.. automodule:: src.scrapers.hikvision
   :members:
   :undoc-members:
   :show-inheritance:

Download Managers
-----------------

Includes the orchestrator (``DownloadManager``) and the
``DownloadCacheStrategy`` wrapper that encapsulates URL cache lookups
and content deduplication with a no-op fallback when no cache is
configured.

.. automodule:: src.scrapers.managers.download_manager
   :members:
   :undoc-members:
   :show-inheritance:

URL Normalizers
---------------

Strategy pattern for vendor-specific URL handling (relative vs.
protocol-relative URLs).

.. automodule:: src.scrapers.managers.url_normalizer
   :members:
   :undoc-members:
   :show-inheritance:

Product Detail Extractor
------------------------

Cache-aware Template Method for extracting product details from vendor
pages. The base class drives the cache-check → fetch → parse →
cache-write flow; concrete extractors live alongside their vendor
scraper and implement only the selectors.

.. automodule:: src.scrapers.managers.product_detail_extractor
   :members:
   :undoc-members:
   :show-inheritance:

Scraping Pipeline
-----------------

Main entry point for the scraping pipeline.

.. automodule:: src.scraping.main
   :members:
   :undoc-members:
   :show-inheritance: