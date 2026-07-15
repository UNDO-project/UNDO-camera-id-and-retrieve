"""Retrieval evaluation harness for the identification pipeline.

Two evaluation modes share one report format:

- ``synthetic`` degrades catalogue images against the index itself,
  measuring top-k recovery per (degradation, severity) with zero
  street labels.
- ``probe`` runs a small hand-labelled set of real street crops
  through the same retrieval path.

Both are exposed through the ``cidar-eval`` console script.
"""

from src.identification.eval.degradations import DEGRADATIONS, SEVERITIES
from src.identification.eval.report import EvalReport, EvalRow

__all__ = ["DEGRADATIONS", "SEVERITIES", "EvalReport", "EvalRow"]
