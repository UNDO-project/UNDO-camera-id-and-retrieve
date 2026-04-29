"""Unit tests for DatasetValidator orchestration helpers.

These tests cover ``_setup_validation`` and ``_run_all_validation_layers``
in isolation, using monkeypatch to stub out the heavy I/O paths
(load_dataset, load_manifest, validate_*) so we can verify the wiring.
"""

from pathlib import Path

import pytest

from src.validation.validator import DatasetValidator


@pytest.fixture
def validator(tmp_path: Path) -> DatasetValidator:
    return DatasetValidator(
        parquet_path=tmp_path / "products.parquet",
        manifest_path=tmp_path / "manifest.json",
    )


class TestSetupValidation:
    """Tests for _setup_validation."""

    def test_returns_false_when_load_dataset_fails(self, validator, monkeypatch):
        monkeypatch.setattr(DatasetValidator, "load_dataset", lambda self: False)
        assert validator._setup_validation() is False

    def test_returns_true_when_manifest_loads_directly(self, validator, monkeypatch):
        monkeypatch.setattr(DatasetValidator, "load_dataset", lambda self: True)
        monkeypatch.setattr(DatasetValidator, "load_manifest", lambda self: True)

        assert validator._setup_validation() is True

    def test_generates_manifest_when_load_manifest_fails(self, validator, monkeypatch):
        load_manifest_calls = {"count": 0}

        def fake_load_manifest(self):
            load_manifest_calls["count"] += 1
            # First call fails (no manifest on disk), second succeeds
            # (after generate_manifest_from_parquet writes one).
            return load_manifest_calls["count"] >= 2

        generated = {"called": False}

        def fake_generate(self):
            generated["called"] = True
            return True

        monkeypatch.setattr(DatasetValidator, "load_dataset", lambda self: True)
        monkeypatch.setattr(DatasetValidator, "load_manifest", fake_load_manifest)
        monkeypatch.setattr(
            DatasetValidator, "generate_manifest_from_parquet", fake_generate
        )

        assert validator._setup_validation() is True
        assert generated["called"] is True
        assert load_manifest_calls["count"] == 2

    def test_returns_false_when_manifest_generation_fails(self, validator, monkeypatch):
        monkeypatch.setattr(DatasetValidator, "load_dataset", lambda self: True)
        # Both load_manifest calls return False; generation didn't help.
        monkeypatch.setattr(DatasetValidator, "load_manifest", lambda self: False)
        monkeypatch.setattr(
            DatasetValidator,
            "generate_manifest_from_parquet",
            lambda self: True,
        )

        assert validator._setup_validation() is False


class TestRunAllValidationLayers:
    """Tests for _run_all_validation_layers."""

    def test_invokes_all_layers_in_order(self, validator, monkeypatch):
        order: list[str] = []

        monkeypatch.setattr(
            DatasetValidator,
            "validate_schema",
            lambda self: order.append("schema"),
        )
        monkeypatch.setattr(
            DatasetValidator,
            "validate_files",
            lambda self, project_root: order.append("files"),
        )
        monkeypatch.setattr(
            DatasetValidator,
            "validate_data_quality",
            lambda self: order.append("data_quality"),
        )
        monkeypatch.setattr(
            DatasetValidator,
            "validate_statistics",
            lambda self: order.append("statistics"),
        )
        monkeypatch.setattr(
            DatasetValidator,
            "compare_with_manifest",
            lambda self: order.append("manifest"),
        )

        validator._run_all_validation_layers(project_root=None)

        assert order == [
            "schema",
            "files",
            "data_quality",
            "statistics",
            "manifest",
        ]

    def test_passes_project_root_through_to_validate_files(
        self, validator, monkeypatch, tmp_path
    ):
        captured = {}

        def fake_validate_files(self, project_root):
            captured["project_root"] = project_root

        monkeypatch.setattr(DatasetValidator, "validate_schema", lambda self: None)
        monkeypatch.setattr(DatasetValidator, "validate_files", fake_validate_files)
        monkeypatch.setattr(
            DatasetValidator, "validate_data_quality", lambda self: None
        )
        monkeypatch.setattr(DatasetValidator, "validate_statistics", lambda self: None)
        monkeypatch.setattr(
            DatasetValidator, "compare_with_manifest", lambda self: None
        )

        validator._run_all_validation_layers(project_root=tmp_path)

        assert captured["project_root"] == tmp_path
