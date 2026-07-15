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

Retrieval evaluation (``cidar-eval``)
-------------------------------------

Two evaluation modes share one report format. ``cidar-eval synthetic``
degrades catalogue images against the index itself (perspective warp,
small scale, blur, JPEG compression, lighting, background swap) and
reports top-1/top-5 recovery per (degradation, severity).
``cidar-eval probe`` runs a hand-labelled set of real street crops
(``data/eval_probe/probe_set.jsonl``; images are not tracked in git)
through the same retrieval path. Reports are written to
``output/eval/`` as JSON and CSV.

.. automodule:: src.identification.eval.degradations
   :members: apply_degradation
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.identification.eval.synthetic
   :members: run_synthetic_eval
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.identification.eval.probe
   :members: run_probe_eval, load_probe_set
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: src.identification.eval.report
   :members: EvalReport, EvalRow
   :undoc-members:
   :show-inheritance:
   :noindex:
