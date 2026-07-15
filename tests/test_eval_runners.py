"""Tests for the synthetic and probe eval runners and the shared report."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.identification.eval.probe import load_probe_set, run_probe_eval
from src.identification.eval.report import EvalReport, EvalRow
from src.identification.eval.synthetic import iter_reference_images, run_synthetic_eval
from src.identification.index import CatalogIndex


def _color_embed(image: Image.Image) -> np.ndarray:
    """Fake embedding: normalized mean RGB + bias term.

    Color-dominant, so degraded copies of a solid-color image still land
    nearest their own catalogue entry — a real retrieval smoke test with
    no ML dependencies.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    vector = np.concatenate([array.mean(axis=(0, 1)), [255.0]])
    return (vector / np.linalg.norm(vector)).astype(np.float32)


COLORS = {
    "cam-red": (200, 30, 30),
    "cam-green": (30, 200, 30),
    "cam-blue": (30, 30, 200),
}


@pytest.fixture
def fixture_index(tmp_path: Path) -> CatalogIndex:
    """Tiny index of three solid-color products with real image files."""
    camera_ids = []
    image_paths = []
    vectors = []

    for camera_id, color in COLORS.items():
        image_path = tmp_path / f"{camera_id}.png"
        Image.new("RGB", (64, 64), color=color).save(image_path)
        camera_ids.append(camera_id)
        image_paths.append(str(image_path))
        vectors.append(_color_embed(Image.open(image_path)))

    embeddings_path = tmp_path / "embeddings.npz"
    np.savez_compressed(
        embeddings_path,
        embeddings=np.stack(vectors).astype("float32"),
        camera_ids=np.array(camera_ids, dtype="U256"),
        image_paths=np.array(image_paths, dtype="U1024"),
        sources=np.array(["Test"] * len(camera_ids), dtype="U128"),
    )
    return CatalogIndex(embeddings_path)


class TestSyntheticRunner:
    """Smoke and determinism tests for run_synthetic_eval."""

    def test_smoke_run_produces_full_grid(self, fixture_index):
        report = run_synthetic_eval(
            fixture_index,
            seed=42,
            degradation_names=["blur", "jpeg_compression"],
            severities=(1, 2),
            embed_fn=_color_embed,
        )

        assert report.mode == "synthetic"
        assert report.n_queries == 3 * 2 * 2
        assert len(report.rows) == 4
        groups = {(row.group, row.severity) for row in report.rows}
        assert groups == {
            ("blur", 1),
            ("blur", 2),
            ("jpeg_compression", 1),
            ("jpeg_compression", 2),
        }
        for row in report.rows:
            assert row.n == 3

    def test_mild_degradations_recover_solid_colors(self, fixture_index):
        """Color-preserving degradations must recover the correct product."""
        report = run_synthetic_eval(
            fixture_index,
            seed=42,
            degradation_names=["blur", "small_scale", "jpeg_compression"],
            severities=(1,),
            embed_fn=_color_embed,
        )

        for row in report.rows:
            assert row.top1_rate == 1.0
            assert row.top5_rate == 1.0

    def test_same_seed_same_report(self, fixture_index):
        kwargs = dict(
            seed=7,
            degradation_names=["perspective_warp", "lighting"],
            severities=(1, 3),
            embed_fn=_color_embed,
        )

        first = run_synthetic_eval(fixture_index, **kwargs)
        second = run_synthetic_eval(fixture_index, **kwargs)

        assert first.rows == second.rows
        assert first.n_queries == second.n_queries

    def test_limit_restricts_products(self, fixture_index):
        report = run_synthetic_eval(
            fixture_index,
            seed=1,
            degradation_names=["blur"],
            severities=(1,),
            limit=2,
            embed_fn=_color_embed,
        )

        assert report.n_queries == 2

    def test_unknown_degradation_rejected(self, fixture_index):
        with pytest.raises(KeyError):
            run_synthetic_eval(
                fixture_index, degradation_names=["nonsense"], embed_fn=_color_embed
            )

    def test_iter_reference_images_one_per_product(self, fixture_index):
        references = iter_reference_images(fixture_index)

        assert len(references) == 3
        assert {camera_id for camera_id, _ in references} == set(COLORS)


class TestProbeRunner:
    """Tests for the probe-set loader and runner."""

    @pytest.fixture
    def probe_set(self, tmp_path: Path, fixture_index) -> Path:
        """A 3-item probe set: two real images, one missing file."""
        entries = [
            {
                "image": str(tmp_path / "cam-red.png"),
                "camera_id": "cam-red",
                "vendor": "Test",
                "model": "Red",
            },
            {
                "image": str(tmp_path / "cam-green.png"),
                "camera_id": "cam-green",
                "vendor": "Test",
                "model": "Green",
            },
            {
                "image": str(tmp_path / "missing.png"),
                "camera_id": "cam-blue",
                "vendor": "Test",
                "model": "Blue",
            },
        ]
        probe_path = tmp_path / "probe_set.jsonl"
        probe_path.write_text(
            "\n".join(json.dumps(entry) for entry in entries) + "\n",
            encoding="utf-8",
        )
        return probe_path

    def test_probe_run_skips_and_counts_missing_images(self, fixture_index, probe_set):
        report = run_probe_eval(
            fixture_index, probe_set_path=probe_set, embed_fn=_color_embed
        )

        assert report.mode == "probe"
        assert report.n_queries == 2
        assert report.notes["probe_set_size"] == 3
        assert report.notes["skipped_images"] == 1
        assert len(report.rows) == 1
        assert report.rows[0].n == 2
        assert report.rows[0].top1_rate == 1.0

    def test_summary_states_probe_set_size(self, fixture_index, probe_set):
        report = run_probe_eval(
            fixture_index, probe_set_path=probe_set, embed_fn=_color_embed
        )

        assert "probe_set_size: 3" in report.format_summary()

    def test_missing_probe_set_raises(self, fixture_index, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_probe_eval(
                fixture_index,
                probe_set_path=tmp_path / "nope.jsonl",
                embed_fn=_color_embed,
            )

    def test_malformed_line_raises(self, tmp_path):
        probe_path = tmp_path / "bad.jsonl"
        probe_path.write_text("not json\n", encoding="utf-8")

        with pytest.raises(ValueError, match="invalid JSON"):
            load_probe_set(probe_path)

    def test_missing_required_keys_raises(self, tmp_path):
        probe_path = tmp_path / "incomplete.jsonl"
        probe_path.write_text(json.dumps({"image": "x.jpg"}) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="camera_id"):
            load_probe_set(probe_path)

    def test_blank_lines_ignored(self, tmp_path):
        probe_path = tmp_path / "gaps.jsonl"
        probe_path.write_text(
            "\n" + json.dumps({"image": "a.jpg", "camera_id": "c"}) + "\n\n",
            encoding="utf-8",
        )

        assert len(load_probe_set(probe_path)) == 1


class TestEvalReport:
    """Tests for report serialization."""

    @pytest.fixture
    def report(self) -> EvalReport:
        return EvalReport(
            mode="synthetic",
            embeddings_path="/fake/embeddings.npz",
            n_queries=20,
            seed=42,
            rows=[
                EvalRow(group="blur", severity=1, n=10, top1_hits=8, top5_hits=9),
                EvalRow(group="blur", severity=2, n=10, top1_hits=5, top5_hits=7),
            ],
            notes={"top_k": 5},
        )

    def test_rates_derived_from_hits(self, report):
        assert report.rows[0].top1_rate == 0.8
        assert report.rows[1].top5_rate == 0.7

    def test_save_writes_json_and_csv(self, report, tmp_path):
        json_path, csv_path = report.save(tmp_path / "eval")

        assert json_path.exists()
        assert csv_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["mode"] == "synthetic"
        assert payload["seed"] == 42
        assert len(payload["rows"]) == 2
        assert payload["rows"][0]["top1_rate"] == 0.8

        csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(csv_lines) == 3  # header + 2 rows
        assert csv_lines[0].startswith("group,severity,n")

    def test_zero_queries_yield_zero_rates(self):
        row = EvalRow(group="probe", severity=0, n=0, top1_hits=0, top5_hits=0)

        assert row.top1_rate == 0.0
        assert row.top5_rate == 0.0
