Configuration
=============

Settings Modules
----------------

.. automodule:: src.config
   :members:
   :undoc-members:
   :show-inheritance:

Scraper Settings
~~~~~~~~~~~~~~~~

.. autoclass:: src.config.scraper.ScraperSettings
   :members:
   :undoc-members:
   :show-inheritance:

Path Settings
~~~~~~~~~~~~~

.. autoclass:: src.config.paths.PathSettings
   :members:
   :undoc-members:
   :show-inheritance:

Vendor Settings
~~~~~~~~~~~~~~~

.. autoclass:: src.config.vendors.AxisSettings
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.config.vendors.HikVisionSettings
   :members:
   :undoc-members:
   :show-inheritance:

Matching Settings
~~~~~~~~~~~~~~~~~

Retrieval behaviour flags (env prefix ``CIDAR_MATCH_``). All defaults
preserve pre-existing behaviour except ``crop_margin``; setting
``CIDAR_MATCH_CROP_MARGIN=0.0`` restores exact-bbox crops. Augmentation
and mean-centring stay off unless a ``cidar-eval synthetic`` comparison
justifies enabling them.

.. autoclass:: src.config.matching.MatchingSettings
   :members:
   :undoc-members:
   :show-inheritance: