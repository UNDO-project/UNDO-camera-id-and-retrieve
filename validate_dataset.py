"""Stage 3: Validate dataset integrity.

This is an independent stage that validates the parquet dataset
against the filesystem and manifest.

Usage:
    python validate_dataset.py
"""

import sys

from src.validation.cli import main as validation_main


if __name__ == "__main__":
    sys.argv[0] = "validate_dataset.py"
    validation_main()
