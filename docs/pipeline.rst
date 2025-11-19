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

.. automodule:: src.storage.manifest
   :members: ManifestRecorder
   :undoc-members:
   :show-inheritance:
   :noindex:

Stage 2 – Build dataset
-----------------------

.. automodule:: src.pipeline.dataset_builder
   :members: DatasetBuilder
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.storage.dataset
   :members: DatasetManager
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
