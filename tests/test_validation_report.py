"""Tests for ValidationReport and ReportFormatter."""

from pathlib import Path

import pytest

from src.validation.report import ReportFormatter, ValidationReport


@pytest.fixture
def base_report() -> ValidationReport:
    return ValidationReport(
        parquet_path=Path("output/products.parquet"),
        manifest_path=Path("output/verification_manifest.json"),
        stats={
            "total_records": 1500,
            "records_with_images": 1450,
            "image_coverage": "96.7%",
            "records_with_pdfs": 1200,
            "pdf_coverage": "80.0%",
            "records_with_specs": 1500,
            "specs_coverage": "100.0%",
            "missing_images": 5,
            "invalid_format_images": 1,
            "missing_pdfs": 10,
            "duplicate_ids": 0,
            "no_media_records": 2,
            "manifest_products": 1500,
        },
    )


class TestReportFormatter:
    def test_passing_report_shows_success(self, base_report):
        output = ReportFormatter(base_report).format()
        assert "Dataset validation passed!" in output

    def test_warnings_only_shows_warning_summary(self, base_report):
        base_report.warnings = [{"type": "minor"}, {"type": "minor"}]
        output = ReportFormatter(base_report).format()
        assert "2 warning(s) but no errors" in output

    def test_errors_show_failure(self, base_report):
        base_report.errors = [{"type": "missing_image"}]
        output = ReportFormatter(base_report).format()
        assert "1 error(s)" in output

    def test_includes_dataset_paths(self, base_report):
        output = ReportFormatter(base_report).format()
        assert "output/products.parquet" in output
        assert "output/verification_manifest.json" in output

    def test_coverage_metrics_rendered(self, base_report):
        output = ReportFormatter(base_report).format()
        assert "Total Records: 1500" in output
        assert "Records with Images: 1450" in output
        assert "96.7%" in output

    def test_file_integrity_rendered(self, base_report):
        output = ReportFormatter(base_report).format()
        assert "Missing Images: 5" in output
        assert "Invalid Image Format: 1" in output
        assert "Missing PDFs: 10" in output

    def test_no_version_info_when_missing(self, base_report):
        output = ReportFormatter(base_report).format()
        assert "VERSION INFORMATION" not in output

    def test_version_info_rendered(self, base_report):
        base_report.version_number = 3
        base_report.version_info = {
            "timestamp": "2026-04-01T00:00:00Z",
            "record_count": 1500,
            "manifest_hash": "sha256:" + "a" * 64,
            "append_mode": False,
        }
        output = ReportFormatter(base_report).format()
        assert "VERSION INFORMATION" in output
        assert "Version: 3" in output
        assert "2026-04-01T00:00:00Z" in output

    def test_version_info_with_append_mode(self, base_report):
        base_report.version_number = 4
        base_report.version_info = {
            "timestamp": "2026-04-01T00:00:00Z",
            "record_count": 1500,
            "manifest_hash": "sha256:" + "b" * 64,
            "append_mode": True,
            "records_added": 50,
            "records_updated": 10,
            "parent_version": 3,
        }
        output = ReportFormatter(base_report).format()
        assert "Records Added: 50" in output
        assert "Records Updated: 10" in output
        assert "Parent Version: 3" in output

    def test_verbose_includes_error_details(self, base_report):
        base_report.errors = [{"type": "missing_image", "path": "data/images/foo.webp"}]
        non_verbose = ReportFormatter(base_report, verbose=False).format()
        verbose = ReportFormatter(base_report, verbose=True).format()
        assert "ERRORS (first 10)" not in non_verbose
        assert "ERRORS (first 10)" in verbose
        assert "data/images/foo.webp" in verbose

    def test_verbose_truncates_after_10(self, base_report):
        base_report.errors = [{"i": i} for i in range(15)]
        output = ReportFormatter(base_report, verbose=True).format()
        assert "and 5 more" in output

    def test_verbose_with_no_issues_omits_section(self, base_report):
        output = ReportFormatter(base_report, verbose=True).format()
        assert "ERRORS (first 10)" not in output
        assert "WARNINGS (first 10)" not in output


class TestValidationReportDataclass:
    def test_default_collections(self):
        report = ValidationReport(
            parquet_path=Path("a.parquet"),
            manifest_path=Path("a.json"),
        )
        assert report.errors == []
        assert report.warnings == []
        assert report.stats == {}
        assert report.version_number is None
        assert report.version_info is None
