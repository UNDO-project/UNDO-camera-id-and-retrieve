"""Validation report data and formatting.

Separates report generation into three concerns:

- :class:`ValidationReport` is a plain data carrier holding all values
  the report needs.
- :class:`ReportFormatter` turns a :class:`ValidationReport` into a
  printable string.
- The validator itself is responsible for printing, which keeps the
  formatting easy to test in isolation.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationReport:
    """Structured validation report data.

    :ivar parquet_path: Dataset parquet path
    :ivar manifest_path: Manifest path
    :ivar version_number: Version number, if validating a specific version
    :ivar version_info: Version metadata, if available
    :ivar errors: List of validation errors
    :ivar warnings: List of validation warnings
    :ivar stats: Aggregate statistics
    """

    parquet_path: Path
    manifest_path: Path
    version_number: int | None = None
    version_info: dict | None = None
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class ReportFormatter:
    """Format a :class:`ValidationReport` as printable text.

    All section helpers append their lines to an internal buffer. The
    public :meth:`format` method returns the assembled report as a
    single string. No I/O is performed here.
    """

    def __init__(self, report: ValidationReport, verbose: bool = False) -> None:
        """
        :param report: Report data to format
        :param verbose: Whether to include detailed errors and warnings
        """
        self.report = report
        self.verbose = verbose
        self._lines: list[str] = []

    def format(self) -> str:
        """
        Render the full report as a single string.

        :return: Formatted report text (newline-terminated)
        """
        self._lines = []
        self._append_header()
        self._append_version_info()
        self._append_summary()
        self._append_coverage_metrics()
        self._append_file_integrity()
        self._append_data_quality()
        self._append_manifest_comparison()
        self._append_result()
        if self.verbose:
            self._append_detailed_issues()
        self._lines.append("=" * 80 + "\n")
        return "\n".join(self._lines)

    def _append_header(self) -> None:
        self._lines.append("\n" + "=" * 80)
        self._lines.append("DATASET VALIDATION REPORT")
        self._lines.append("=" * 80)
        self._lines.append(f"\nDataset: {self.report.parquet_path}")
        self._lines.append(f"Manifest: {self.report.manifest_path}")

    def _append_version_info(self) -> None:
        if self.report.version_number is None or self.report.version_info is None:
            return

        info = self.report.version_info
        self._lines.append("\n--- VERSION INFORMATION ---")
        self._lines.append(f"Version: {self.report.version_number}")
        self._lines.append(f"Timestamp: {info.get('timestamp', 'N/A')}")
        self._lines.append(f"Record Count: {info.get('record_count', 'N/A')}")
        manifest_hash = info.get("manifest_hash", "N/A")
        self._lines.append(f"Manifest Hash: {manifest_hash[:32]}...")
        self._lines.append(f"Append Mode: {info.get('append_mode', 'N/A')}")
        if info.get("append_mode"):
            self._lines.append(f"Records Added: {info.get('records_added', 'N/A')}")
            self._lines.append(f"Records Updated: {info.get('records_updated', 'N/A')}")
            self._lines.append(f"Parent Version: {info.get('parent_version', 'N/A')}")

    def _append_summary(self) -> None:
        self._lines.append("\n--- VALIDATION SUMMARY ---")
        self._lines.append(f"Errors: {len(self.report.errors)}")
        self._lines.append(f"Warnings: {len(self.report.warnings)}")

    def _append_coverage_metrics(self) -> None:
        s = self.report.stats
        self._lines.append("\n--- COVERAGE METRICS ---")
        self._lines.append(f"Total Records: {s.get('total_records', 0)}")
        self._lines.append(
            f"Records with Images: {s.get('records_with_images', 0)} "
            f"({s.get('image_coverage', 'N/A')})"
        )
        self._lines.append(
            f"Records with PDFs: {s.get('records_with_pdfs', 0)} "
            f"({s.get('pdf_coverage', 'N/A')})"
        )
        self._lines.append(
            f"Records with Specs: {s.get('records_with_specs', 0)} "
            f"({s.get('specs_coverage', 'N/A')})"
        )

    def _append_file_integrity(self) -> None:
        s = self.report.stats
        self._lines.append("\n--- FILE INTEGRITY ---")
        self._lines.append(f"Missing Images: {s.get('missing_images', 0)}")
        self._lines.append(f"Invalid Image Format: {s.get('invalid_format_images', 0)}")
        self._lines.append(f"Missing PDFs: {s.get('missing_pdfs', 0)}")

    def _append_data_quality(self) -> None:
        s = self.report.stats
        self._lines.append("\n--- DATA QUALITY ---")
        self._lines.append(f"Duplicate IDs: {s.get('duplicate_ids', 0)}")
        self._lines.append(f"Records without Media: {s.get('no_media_records', 0)}")

    def _append_manifest_comparison(self) -> None:
        s = self.report.stats
        self._lines.append("\n--- MANIFEST COMPARISON ---")
        self._lines.append(f"Expected Products: {s.get('manifest_products', 'N/A')}")
        self._lines.append(f"Actual Products: {s.get('total_records', 'N/A')}")

    def _append_result(self) -> None:
        errors = self.report.errors
        warnings = self.report.warnings
        if not errors and not warnings:
            self._lines.append("\n✅ Dataset validation passed!")
        elif not errors:
            self._lines.append(
                f"\n⚠️ Dataset has {len(warnings)} warning(s) but no errors"
            )
        else:
            self._lines.append(f"\n❌ Dataset has {len(errors)} error(s)")

    def _append_detailed_issues(self) -> None:
        errors = self.report.errors
        warnings = self.report.warnings

        if not errors and not warnings:
            return

        if errors:
            self._lines.append("\n--- ERRORS (first 10) ---")
            for error in errors[:10]:
                self._lines.append(f"  • {error}")
            if len(errors) > 10:
                self._lines.append(f"  ... and {len(errors) - 10} more")

        if warnings:
            self._lines.append("\n--- WARNINGS (first 10) ---")
            for warning in warnings[:10]:
                self._lines.append(f"  • {warning}")
            if len(warnings) > 10:
                self._lines.append(f"  ... and {len(warnings) - 10} more")
