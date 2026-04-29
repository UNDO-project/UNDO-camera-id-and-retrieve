Validation Module APIs
======================

Components for validating dataset integrity and quality.

Dataset Validator
-----------------

.. automodule:: src.validation.validator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: src.validation.cli
   :members:
   :undoc-members:
   :show-inheritance:

File Validators
---------------

Strategy pattern implementation for file validation. Each validator
accumulates its own ``errors`` list and exposes ``get_stats()``; the
orchestrator (``DatasetValidator``) merges results after iteration.

.. automodule:: src.validation.file_validators
   :members:
   :undoc-members:
   :show-inheritance:

Validation Report
-----------------

Separates validation report data from formatting.
``ValidationReport`` is a plain dataclass holding all values the report
needs; ``ReportFormatter`` turns it into printable text. The validator
itself is responsible for printing.

.. automodule:: src.validation.report
   :members:
   :undoc-members:
   :show-inheritance:
