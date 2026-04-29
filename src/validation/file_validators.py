"""File validators for dataset validation."""

import json
from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger


class FileValidator(ABC):
    """Abstract base class for file validators.

    Validators are independent of any orchestrator. They accumulate
    errors in :attr:`errors` and per-validator counters as they process
    rows. The caller collects results after iteration via
    :meth:`get_stats` and reading :attr:`errors`.
    """

    def __init__(self, project_root: Path) -> None:
        """
        Initialize file validator.

        :param project_root: Root directory for file path resolution
        """
        self.project_root = project_root
        self.errors: list[dict] = []

    @abstractmethod
    def validate(self, row: dict, row_idx: int) -> None:
        """
        Validate files for a dataset row, recording any errors found.

        :param row: DataFrame row as dictionary
        :param row_idx: Row index for error reporting
        """
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, int]:
        """
        Get validator statistics suitable for merging into the report.

        :return: Mapping of stat name to count
        """
        pass


class ImageFileValidator(FileValidator):
    """Validator for image file references."""

    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root)
        self.missing_count = 0
        self.invalid_format_count = 0

    def validate(self, row: dict, row_idx: int) -> None:
        """
        Validate image file references in a row.

        :param row: DataFrame row as dictionary
        :param row_idx: Row index for error reporting
        """
        image_files_str = row.get("image_files")
        if not image_files_str or image_files_str == "[]":
            return

        try:
            image_files = json.loads(image_files_str)
        except json.JSONDecodeError:
            self.errors.append({"type": "invalid_json_image_files", "row": row_idx})
            return

        for img_path in image_files:
            self._validate_image_file(img_path)

    def _validate_image_file(self, img_path: str) -> None:
        """
        Validate a single image file.

        :param img_path: Path to image file
        """
        full_path = self.project_root / img_path

        if not full_path.exists():
            self.missing_count += 1
            self.errors.append({"type": "missing_image", "path": img_path})
            return

        if full_path.suffix.lower() == ".webp":
            self._validate_webp_format(img_path, full_path)

    def _validate_webp_format(self, img_path: str, full_path: Path) -> None:
        """
        Validate WebP file format using magic bytes.

        :param img_path: Path to image file (for error reporting)
        :param full_path: Full path to image file
        """
        try:
            with open(full_path, "rb") as f:
                header = f.read(4)
        except OSError as e:
            self.errors.append(
                {
                    "type": "cannot_read_image",
                    "path": img_path,
                    "error": str(e),
                }
            )
            return

        if header != b"RIFF":
            self.invalid_format_count += 1
            self.errors.append(
                {
                    "type": "invalid_image_format",
                    "path": img_path,
                }
            )

    def get_stats(self) -> dict[str, int]:
        """
        Get validator statistics.

        :return: Stats dictionary with image counts
        """
        if self.missing_count > 0:
            logger.warning(f"Found {self.missing_count} missing images")
        if self.invalid_format_count > 0:
            logger.warning(f"Found {self.invalid_format_count} invalid image formats")

        return {
            "missing_images": self.missing_count,
            "invalid_format_images": self.invalid_format_count,
        }


class PdfFileValidator(FileValidator):
    """Validator for PDF file references."""

    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root)
        self.missing_count = 0
        self.invalid_format_count = 0

    def validate(self, row: dict, row_idx: int) -> None:
        """
        Validate PDF file reference in a row.

        :param row: DataFrame row as dictionary
        :param row_idx: Row index for error reporting
        """
        pdf_file: str | None = row.get("datasheet_file")
        if not pdf_file:
            return

        full_path = self.project_root / pdf_file

        if not full_path.exists():
            self.missing_count += 1
            self.errors.append({"type": "missing_pdf", "path": pdf_file})
            return

        self._validate_pdf_format(pdf_file, full_path)

    def _validate_pdf_format(self, pdf_path: str, full_path: Path) -> None:
        """
        Validate PDF file format using magic bytes.

        :param pdf_path: Path to PDF file (for error reporting)
        :param full_path: Full path to PDF file
        """
        try:
            with open(full_path, "rb") as f:
                header = f.read(4)
        except OSError as e:
            self.errors.append(
                {
                    "type": "cannot_read_pdf",
                    "path": pdf_path,
                    "error": str(e),
                }
            )
            return

        if header != b"%PDF":
            self.invalid_format_count += 1
            self.errors.append({"type": "invalid_pdf_format", "path": pdf_path})

    def get_stats(self) -> dict[str, int]:
        """
        Get validator statistics.

        :return: Stats dictionary with PDF counts
        """
        if self.missing_count > 0:
            logger.warning(f"Found {self.missing_count} missing PDFs")

        return {"missing_pdfs": self.missing_count}
