"""Unit tests for DatasetBuilder extracted helpers.

Each test focuses on one small private method in isolation, leaving the
larger integration scenarios to ``test_dataset_creation.py`` and
``test_dataset_append.py``.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.building.builder import DatasetBuilder
from src.models.camera import CameraRecord


def _make_record(camera_id: str = "cam-1", **overrides) -> CameraRecord:
    """Build a CameraRecord with sensible defaults for tests."""
    defaults = dict(
        camera_id=camera_id,
        model_name=f"Model {camera_id}",
        display_name=f"Model {camera_id}",
        description=None,
        specifications={},
        source="Test Vendor",
        category="Network Camera",
        product_category="DOME CAMERAS",
        product_series="M30",
        images=["https://example.com/a.jpg"],
        image_url=None,
        image_files=["data/images/a.webp"],
        datasheet_url="https://example.com/a.pdf",
        datasheet_file="data/pdfs/a.pdf",
        specifications_html={"General": {"Model": camera_id}},
    )
    defaults.update(overrides)
    return CameraRecord(**defaults)


@pytest.fixture
def builder(tmp_path: Path) -> DatasetBuilder:
    return DatasetBuilder(
        manifest_path=tmp_path / "manifest.json",
        output_path=tmp_path / "products.parquet",
        force=True,
    )


class TestRecordToDict:
    """Tests for the static _record_to_dict helper."""

    def test_serializes_lists_as_json(self):
        record = _make_record(images=["a.jpg", "b.jpg"], image_files=["x.webp"])

        row = DatasetBuilder._record_to_dict(record)

        assert row["camera_id"] == "cam-1"
        assert json.loads(row["image_urls"]) == ["a.jpg", "b.jpg"]
        assert json.loads(row["image_files"]) == ["x.webp"]

    def test_serializes_specifications_as_json(self):
        record = _make_record(specifications_html={"S1": {"k": "v"}})

        row = DatasetBuilder._record_to_dict(record)

        assert json.loads(row["specifications"]) == {"S1": {"k": "v"}}

    def test_handles_none_datasheet_file(self):
        record = _make_record(datasheet_file=None)

        row = DatasetBuilder._record_to_dict(record)

        assert row["datasheet_file"] is None
        assert row["datasheet_url"] == "https://example.com/a.pdf"


class TestSerializeRecordsToDataframe:
    """Tests for _serialize_records_to_dataframe."""

    def test_returns_none_when_no_records(self, builder):
        builder.records = []
        df = builder._serialize_records_to_dataframe()
        assert df is None

    def test_returns_dataframe_with_rows(self, builder):
        builder.records = [_make_record("cam-1"), _make_record("cam-2")]

        df = builder._serialize_records_to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert sorted(df["camera_id"].tolist()) == ["cam-1", "cam-2"]


class TestBuildCameraRecord:
    """Tests for the _build_camera_record helper."""

    def test_returns_none_when_camera_id_missing(self, builder, monkeypatch):
        product_info = {"model_name": "Some Camera"}
        record = builder._build_camera_record("cat", "ser", product_info)
        assert record is None

    def test_returns_none_when_model_name_missing(self, builder):
        product_info = {"camera_id": "cam-1"}
        record = builder._build_camera_record("cat", "ser", product_info)
        assert record is None

    def test_builds_record_with_filesystem_lookup(self, builder, monkeypatch):
        # Stub the filesystem lookups so the test doesn't depend on data/.
        monkeypatch.setattr(
            DatasetBuilder,
            "_find_image_files_and_vendor",
            staticmethod(
                lambda c, s, cid: (["data/images/x.webp"], "Axis Communications")
            ),
        )
        monkeypatch.setattr(
            DatasetBuilder,
            "_find_pdf_file",
            staticmethod(lambda c, s, cid: "data/pdfs/x.pdf"),
        )

        product_info = {
            "camera_id": "axis-m3057",
            "model_name": "AXIS M3057-PLR",
            "image_urls": ["https://example.com/m3057.jpg"],
            "datasheet_url": "https://example.com/m3057.pdf",
            "specifications_html": {"General": {"Model": "M3057"}},
        }

        record = builder._build_camera_record("Dome cameras", "M30", product_info)

        assert record is not None
        assert record.camera_id == "axis-m3057"
        assert record.model_name == "AXIS M3057-PLR"
        assert record.source == "Axis Communications"
        assert record.product_category == "Dome cameras"
        assert record.product_series == "M30"
        assert record.image_files == ["data/images/x.webp"]
        assert record.datasheet_file == "data/pdfs/x.pdf"
        assert record.specifications_html == {"General": {"Model": "M3057"}}

    def test_unknown_vendor_when_filesystem_empty(self, builder, monkeypatch):
        monkeypatch.setattr(
            DatasetBuilder,
            "_find_image_files_and_vendor",
            staticmethod(lambda c, s, cid: ([], None)),
        )
        monkeypatch.setattr(
            DatasetBuilder,
            "_find_pdf_file",
            staticmethod(lambda c, s, cid: None),
        )

        product_info = {"camera_id": "cam-x", "model_name": "Cam X"}
        record = builder._build_camera_record("c", "s", product_info)

        assert record is not None
        assert record.source == "Unknown"
        assert record.image_files == []
        assert record.datasheet_file is None


class TestApplyMergeStrategy:
    """Tests for _apply_merge_strategy."""

    def test_overwrite_mode_records_count_in_stats(self, builder):
        builder.append = False
        df = pd.DataFrame({"camera_id": ["a", "b", "c"]})

        result = builder._apply_merge_strategy(df)

        assert result is df
        assert builder.merge_stats["records_added"] == 3

    def test_append_mode_delegates_to_merge_with_existing(self, builder, monkeypatch):
        builder.append = True
        merged = pd.DataFrame({"camera_id": ["a"]})
        called = {}

        def fake_merge(self, df):
            called["df"] = df
            return merged

        monkeypatch.setattr(DatasetBuilder, "_merge_with_existing", fake_merge)

        df = pd.DataFrame({"camera_id": ["a"]})
        result = builder._apply_merge_strategy(df)

        assert result is merged
        assert called["df"] is df

    def test_overwrite_cancelled_returns_none(self, builder, monkeypatch):
        builder.append = False
        builder.force = False
        # Pretend the file exists and the user said no.
        monkeypatch.setattr(DatasetBuilder, "_confirm_overwrite", lambda self: False)

        result = builder._apply_merge_strategy(pd.DataFrame({"camera_id": ["a"]}))

        assert result is None


class TestPersistDirectly:
    """Tests for _persist_directly."""

    def test_writes_parquet(self, builder):
        df = pd.DataFrame({"camera_id": ["a", "b"], "model_name": ["A", "B"]})

        ok = builder._persist_directly(df)

        assert ok is True
        assert builder.output_path.exists()

        round_trip = pd.read_parquet(builder.output_path)
        assert len(round_trip) == 2
        assert sorted(round_trip["camera_id"].tolist()) == ["a", "b"]


class TestPersistDataframe:
    """Tests for _persist_dataframe routing."""

    def test_routes_to_direct_when_no_version_manager(self, builder, monkeypatch):
        builder.version_manager = None
        called = {}

        def fake_direct(self, df):
            called["called"] = True
            return True

        monkeypatch.setattr(DatasetBuilder, "_persist_directly", fake_direct)
        builder._persist_dataframe(pd.DataFrame({"camera_id": ["a"]}))

        assert called.get("called") is True

    def test_routes_to_versioning_when_version_manager_present(
        self, builder, monkeypatch
    ):
        builder.version_manager = object()
        called = {}

        def fake_versioned(self, df):
            called["called"] = True
            return True

        monkeypatch.setattr(DatasetBuilder, "_persist_with_versioning", fake_versioned)
        builder._persist_dataframe(pd.DataFrame({"camera_id": ["a"]}))

        assert called.get("called") is True
