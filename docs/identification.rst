Camera Identification & Retrieval (Stage 4)
============================================

This section documents the camera identification pipeline built on top of
the scraped catalog and embeddings.

Models
------

.. automodule:: src.models.identification
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Detector
--------

.. automodule:: src.identification.detector
   :members: Detector
   :undoc-members:
   :show-inheritance:
   :noindex:

Catalog & embeddings
--------------------

.. automodule:: src.identification.catalog
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.identification.embeddings
   :members: embed_image
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.identification.index
   :members: build_catalog_embeddings, CatalogIndex
   :undoc-members:
   :show-inheritance:
   :noindex:

Service & CLI
-------------

.. automodule:: src.identification.service
   :members: IdentificationService, CropInfo
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.identification.cli
   :members: main
   :undoc-members:
   :show-inheritance:
   :noindex:
