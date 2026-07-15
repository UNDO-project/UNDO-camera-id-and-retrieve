r"""Command-line interface for retrieval evaluation.

Exposes the synthetic robustness harness and the real-probe runner as
the ``cidar-eval`` console script.

Example usage::

    cidar-eval synthetic --seed 42
    cidar-eval synthetic --limit 50 --degradations blur,small_scale
    cidar-eval probe --probe-set data/eval_probe/probe_set.jsonl
"""

import argparse
from pathlib import Path

from loguru import logger

from src.config import paths
from src.identification.eval.degradations import DEGRADATIONS
from src.identification.eval.probe import run_probe_eval
from src.identification.eval.synthetic import run_synthetic_eval
from src.identification.index import CatalogIndex


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments common to both subcommands.

    :param parser: Subcommand parser to extend
    """
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help=(
            "Path to catalog embeddings .npz file "
            "(default: output/catalog_embeddings.npz)"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Report output directory (default: output/eval)",
    )


def main() -> None:
    r"""Entry point for the ``cidar-eval`` console script."""
    parser = argparse.ArgumentParser(
        prog="cidar-eval",
        description="Retrieval robustness evaluation for the cIDaR catalog index",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic_parser = subparsers.add_parser(
        "synthetic",
        help="Degrade catalogue images against the index and measure recovery",
    )
    synthetic_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Run seed; identical seeds produce identical reports (default: 42)",
    )
    synthetic_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N products (quick runs)",
    )
    synthetic_parser.add_argument(
        "--degradations",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of degradations to run "
            f"(default: all of {', '.join(sorted(DEGRADATIONS))})"
        ),
    )
    _add_shared_arguments(synthetic_parser)

    probe_parser = subparsers.add_parser(
        "probe",
        help="Run a hand-labelled set of real street crops through retrieval",
    )
    probe_parser.add_argument(
        "--probe-set",
        type=str,
        default=None,
        help=("Path to probe_set.jsonl (default: data/eval_probe/probe_set.jsonl)"),
    )
    _add_shared_arguments(probe_parser)

    args = parser.parse_args()

    embeddings_path = Path(args.embeddings) if args.embeddings else None
    out_dir = Path(args.out) if args.out else paths.output_dir / "eval"

    index = CatalogIndex(embeddings_path=embeddings_path)

    if args.command == "synthetic":
        degradation_names = (
            [name.strip() for name in args.degradations.split(",") if name.strip()]
            if args.degradations
            else None
        )
        report = run_synthetic_eval(
            index,
            seed=args.seed,
            degradation_names=degradation_names,
            limit=args.limit,
        )
    else:
        report = run_probe_eval(index, probe_set_path=args.probe_set)

    json_path, csv_path = report.save(out_dir)
    logger.info(f"Report written: {json_path}")
    logger.info(f"Report written: {csv_path}")

    print()
    print(report.format_summary())


if __name__ == "__main__":
    main()
