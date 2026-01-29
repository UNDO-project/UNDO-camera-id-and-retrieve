Data pipeline (Stages 1–3)
==========================

This section describes the scraping, dataset building, and validation pipeline.
For commands to run each stage, see the README.

Stage 1 – Scrape
----------------

.. automodule:: src.scrapers.axis
   :members: AxisCameraScraper
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.scrapers.hikvision
   :members: HikvisionCameraScraper
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.scrapers.managers.download_manager
   :members: ContentDownloader, DownloadManager
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.scrapers.managers.url_normalizer
   :members: URLNormalizer, RelativeURLNormalizer, ProtocolRelativeURLNormalizer
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.storage.manifest
   :members: ManifestRecorder
   :undoc-members:
   :show-inheritance:
   :noindex:

Stage 2 – Build dataset
-----------------------

.. automodule:: src.building.builder
   :members: DatasetBuilder
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.storage.dataset
   :members: DatasetManager
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.storage.versioning
   :members: DatasetVersionManager
   :undoc-members:
   :show-inheritance:
   :noindex:

Manifest reconstruction (recovery)
----------------------------------

.. automodule:: src.building.manifest_reconstruction
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.building.reconstruct_cli
   :members: main
   :undoc-members:
   :show-inheritance:
   :noindex:

Stage 3 – Validate dataset
--------------------------

.. automodule:: src.validation.validator
   :members: DatasetValidator
   :undoc-members:
   :show-inheritance:
   :noindex: