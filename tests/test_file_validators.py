"""Tests for file validators."""

import json
from pathlib import Path

import pytest

from src.validation.file_validators import ImageFileValidator, PdfFileValidator


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Create a temporary project root with test files."""
    images_dir = tmp_path / "data" / "images"
    images_dir.mkdir(parents=True)
    webp_file = images_dir / "test.webp"
    webp_file.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")

    pdfs_dir = tmp_path / "data" / "pdfs"
    pdfs_dir.mkdir(parents=True)
    pdf_file = pdfs_dir / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4")

    return tmp_path


class TestImageFileValidator:
    """Tests for ImageFileValidator."""

    def test_validate_no_images(self, project_root):
        validator = ImageFileValidator(project_root)
        row = {"image_files": "[]", "camera_id": "cam-001"}
        validator.validate(row, 0)

        assert validator.errors == []
        assert validator.missing_count == 0

    def test_validate_valid_images(self, project_root):
        validator = ImageFileValidator(project_root)
        image_files = json.dumps(["data/images/test.webp"])
        row = {"image_files": image_files, "camera_id": "cam-001"}
        validator.validate(row, 0)

        assert validator.errors == []
        assert validator.missing_count == 0

    def test_validate_missing_images(self, project_root):
        validator = ImageFileValidator(project_root)
        image_files = json.dumps(["data/images/nonexistent.webp"])
        row = {"image_files": image_files, "camera_id": "cam-001"}
        validator.validate(row, 0)

        assert len(validator.errors) == 1
        assert validator.errors[0]["type"] == "missing_image"
        assert validator.missing_count == 1

    def test_validate_invalid_image_format(self, project_root):
        invalid_webp = project_root / "data" / "images" / "invalid.webp"
        invalid_webp.write_bytes(b"NOT_RIFF_DATA")

        validator = ImageFileValidator(project_root)
        image_files = json.dumps(["data/images/invalid.webp"])
        row = {"image_files": image_files, "camera_id": "cam-001"}
        validator.validate(row, 0)

        assert len(validator.errors) == 1
        assert validator.errors[0]["type"] == "invalid_image_format"
        assert validator.invalid_format_count == 1

    def test_validate_invalid_json(self, project_root):
        validator = ImageFileValidator(project_root)
        row = {"image_files": "not valid json", "camera_id": "cam-001"}
        validator.validate(row, 5)

        assert len(validator.errors) == 1
        assert validator.errors[0]["type"] == "invalid_json_image_files"
        assert validator.errors[0]["row"] == 5

    def test_get_stats(self, project_root):
        validator = ImageFileValidator(project_root)
        validator.missing_count = 3
        validator.invalid_format_count = 2

        stats = validator.get_stats()

        assert stats["missing_images"] == 3
        assert stats["invalid_format_images"] == 2


class TestPdfFileValidator:
    """Tests for PdfFileValidator."""

    def test_validate_no_pdf(self, project_root):
        validator = PdfFileValidator(project_root)
        row = {"datasheet_file": None, "camera_id": "cam-001"}
        validator.validate(row, 0)

        assert validator.errors == []
        assert validator.missing_count == 0

    def test_validate_valid_pdf(self, project_root):
        validator = PdfFileValidator(project_root)
        row = {"datasheet_file": "data/pdfs/test.pdf", "camera_id": "cam-001"}
        validator.validate(row, 0)

        assert validator.errors == []
        assert validator.missing_count == 0

    def test_validate_missing_pdf(self, project_root):
        validator = PdfFileValidator(project_root)
        row = {"datasheet_file": "data/pdfs/nonexistent.pdf", "camera_id": "cam-001"}
        validator.validate(row, 0)

        assert len(validator.errors) == 1
        assert validator.errors[0]["type"] == "missing_pdf"
        assert validator.missing_count == 1

    def test_validate_invalid_pdf_format(self, project_root):
        invalid_pdf = project_root / "data" / "pdfs" / "invalid.pdf"
        invalid_pdf.write_bytes(b"NOT_PDF_DATA")

        validator = PdfFileValidator(project_root)
        row = {"datasheet_file": "data/pdfs/invalid.pdf", "camera_id": "cam-001"}
        validator.validate(row, 0)

        assert len(validator.errors) == 1
        assert validator.errors[0]["type"] == "invalid_pdf_format"

    def test_get_stats(self, project_root):
        validator = PdfFileValidator(project_root)
        validator.missing_count = 5

        stats = validator.get_stats()

        assert stats["missing_pdfs"] == 5
