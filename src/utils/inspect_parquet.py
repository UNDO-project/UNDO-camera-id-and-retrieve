"""Inspect and visualize parquet dataset.

Usage:
    python inspect_parquet.py [--rows N] [--columns COL1,COL2] [--csv OUTPUT.csv]
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config import OUTPUT_DIR


def load_parquet(parquet_path: Path) -> pd.DataFrame | None:
    r"""
    Load parquet file into DataFrame.

    :param parquet_path: Path to parquet file
    :return: DataFrame or None if file doesn't exist
    """
    if not parquet_path.exists():
        logger.error(f"Parquet file not found: {parquet_path}")
        return None

    try:
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded parquet file: {parquet_path}")
        logger.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load parquet file: {e}")
        return None


def display_info(df: pd.DataFrame) -> None:
    r"""
    Display basic information about the dataset.

    :param df: DataFrame to inspect
    """
    print("\n" + "=" * 80)
    print("  DATASET INFORMATION")
    print("=" * 80 + "\n")

    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

    print("Columns:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        print(f"  {i:2d}. {col:<30} ({dtype}, {non_null}/{len(df)} non-null)")


def display_rows(df: pd.DataFrame, num_rows: int = 5) -> None:
    r"""
    Display first N rows of dataset.

    :param df: DataFrame to display
    :param num_rows: Number of rows to show
    """
    print("\n" + "=" * 80)
    print(f"  FIRST {num_rows} ROWS")
    print("=" * 80 + "\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", num_rows)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)

    print(df.head(num_rows).to_string())


def display_summary_stats(df: pd.DataFrame) -> None:
    r"""
    Display summary statistics.

    :param df: DataFrame to analyze
    """
    print("\n" + "=" * 80)
    print("  SUMMARY STATISTICS")
    print("=" * 80 + "\n")

    print(f"Total Products: {len(df)}")

    # Count by category
    if "product_category" in df.columns:
        print("\nProducts by Category:")
        category_counts = df["product_category"].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count}")

    # Count by series
    if "product_series" in df.columns:
        print("\nProducts by Series:")
        series_counts = df["product_series"].value_counts()
        for series, count in series_counts.head(10).items():
            print(f"  {series}: {count}")
        if len(series_counts) > 10:
            print(f"  ... and {len(series_counts) - 10} more series")

    # Image stats
    if "image_urls" in df.columns:
        try:
            import json

            total_images = 0
            for urls_json in df["image_urls"]:
                try:
                    urls = json.loads(urls_json)
                    total_images += len(urls) if isinstance(urls, list) else 0
                except Exception:
                    pass
            print(f"\nTotal Images Referenced: {total_images}")
            if len(df) > 0:
                print(f"Average Images per Product: {total_images / len(df):.1f}")
        except Exception as e:
            logger.debug(f"Could not parse image_urls: {e}")

    # Datasheet coverage
    if "datasheet_url" in df.columns:
        has_datasheet = df["datasheet_url"].notna().sum()
        coverage = (has_datasheet / len(df) * 100) if len(df) > 0 else 0
        print(f"\nDatasheets Found: {has_datasheet}/{len(df)} ({coverage:.1f}%)")

    # Specifications coverage
    if "specifications" in df.columns:
        has_specs = df["specifications"].notna().sum()
        coverage = (has_specs / len(df) * 100) if len(df) > 0 else 0
        print(f"Specifications: {has_specs}/{len(df)} ({coverage:.1f}%)")


def display_specific_columns(df: pd.DataFrame, columns: list[str]) -> None:
    r"""
    Display specific columns only.

    :param df: DataFrame to display
    :param columns: Column names to show
    """
    print("\n" + "=" * 80)
    print(f"  SELECTED COLUMNS: {', '.join(columns)}")
    print("=" * 80 + "\n")

    available_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        logger.warning(f"Columns not found: {missing_cols}")

    if available_cols:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 100)

        print(df[available_cols].head(10).to_string())


def export_csv(df: pd.DataFrame, output_path: str) -> None:
    r"""
    Export DataFrame to CSV.

    :param df: DataFrame to export
    :param output_path: Output CSV file path
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Exported to CSV: {output_path}")
        print(f"\nDataset exported to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")


def main() -> None:
    r"""
    Main inspection routine.
    """
    parser = argparse.ArgumentParser(description="Inspect CCTV scraper parquet dataset")
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows to display (default: 5)",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated columns to display",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export to CSV file",
    )

    args = parser.parse_args()

    parquet_path = OUTPUT_DIR / "products.parquet"

    df = load_parquet(parquet_path)
    if df is None:
        sys.exit(1)

    display_info(df)
    display_summary_stats(df)
    display_rows(df, num_rows=args.rows)

    if args.columns:
        columns = [col.strip() for col in args.columns.split(",")]
        display_specific_columns(df, columns)

    if args.csv:
        export_csv(df, args.csv)

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
