r"""Result structures and emitters shared by the eval runners.

The synthetic harness and the real-probe runner both aggregate their
outcomes into an :class:`EvalReport` and write it through the same
JSON/CSV emitters, so reports from the two modes are directly
comparable.
"""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalRow:
    """Aggregated retrieval outcome for one evaluation group.

    For synthetic runs, ``group`` is the degradation name and
    ``severity`` its level. For probe runs there is a single row with
    ``group="probe"`` and ``severity=0``.
    """

    group: str
    severity: int
    n: int
    top1_hits: int
    top5_hits: int

    @property
    def top1_rate(self) -> float:
        """Fraction of queries whose correct product ranked first."""
        return self.top1_hits / self.n if self.n else 0.0

    @property
    def top5_rate(self) -> float:
        """Fraction of queries whose correct product ranked in the top 5."""
        return self.top5_hits / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the row, including derived rates.

        :return: JSON-ready dictionary
        """
        return {
            "group": self.group,
            "severity": self.severity,
            "n": self.n,
            "top1_hits": self.top1_hits,
            "top5_hits": self.top5_hits,
            "top1_rate": round(self.top1_rate, 4),
            "top5_rate": round(self.top5_rate, 4),
        }


@dataclass
class EvalReport:
    """Complete result of one evaluation run.

    :ivar mode: ``"synthetic"`` or ``"probe"``
    :ivar embeddings_path: Index artifact the run queried
    :ivar n_queries: Total number of queries executed
    :ivar seed: Seed used for synthetic degradations (None for probe)
    :ivar rows: Aggregated outcome rows
    :ivar notes: Free-form metadata (e.g. probe-set size, skipped images)
    :ivar generated_at: UTC timestamp set at construction
    """

    mode: str
    embeddings_path: str
    n_queries: int
    seed: int | None
    rows: list[EvalRow] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full report.

        :return: JSON-ready dictionary
        """
        return {
            "mode": self.mode,
            "generated_at": self.generated_at,
            "embeddings_path": self.embeddings_path,
            "n_queries": self.n_queries,
            "seed": self.seed,
            "notes": self.notes,
            "rows": [row.to_dict() for row in self.rows],
        }

    def save(self, out_dir: Path | str) -> tuple[Path, Path]:
        r"""Write the report as JSON and CSV.

        Files are named ``<mode>_report.json`` / ``<mode>_report.csv``
        inside ``out_dir``, which is created if missing.

        :param out_dir: Output directory
        :return: Tuple of ``(json_path, csv_path)``
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / f"{self.mode}_report.json"
        json_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

        csv_path = out_dir / f"{self.mode}_report.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "group",
                    "severity",
                    "n",
                    "top1_hits",
                    "top5_hits",
                    "top1_rate",
                    "top5_rate",
                ],
            )
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.to_dict())

        return json_path, csv_path

    def format_summary(self) -> str:
        """Render a small human-readable summary table.

        :return: Multi-line string for terminal output
        """
        lines = [
            f"Eval mode: {self.mode}",
            f"Index: {self.embeddings_path}",
            f"Queries: {self.n_queries}"
            + (f" (seed={self.seed})" if self.seed is not None else ""),
        ]
        for key, value in self.notes.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        lines.append(f"{'group':<20} {'sev':>3} {'n':>6} {'top-1':>7} {'top-5':>7}")
        for row in self.rows:
            lines.append(
                f"{row.group:<20} {row.severity:>3} {row.n:>6} "
                f"{row.top1_rate:>7.3f} {row.top5_rate:>7.3f}"
            )
        return "\n".join(lines)
