"""Unit tests for embedding hygiene and preprocessing parity.

These tests guard the invariants the retrieval pipeline relies on:
``embed_image`` always returns a unit-norm float32 1-D vector, and the
catalogue-build, sync-query, and async-query paths all resolve to the
same embedding function so preprocessing can never silently diverge.
"""

import numpy as np
import pytest
from PIL import Image

import src.identification.async_service as async_service_module
import src.identification.embeddings as embeddings_module
import src.identification.index as index_module
import src.identification.service as service_module
from src.identification.async_service import AsyncIdentificationService


class _FakeClipModel:
    """Minimal stand-in for a CLIP model returning a fixed raw vector."""

    def encode_image(self, tensor):
        import torch

        # Deliberately non-normalized and non-float32 so the test proves
        # embed_image itself normalizes and casts.
        return torch.tensor([[3.0, 4.0, 0.0, 0.0]], dtype=torch.float64)


def _fake_preprocess(pil_image):
    import torch

    return torch.zeros(3, 8, 8)


@pytest.fixture
def fake_clip(monkeypatch):
    """Patch CLIP components with a lightweight fake on CPU."""
    monkeypatch.setattr(
        embeddings_module,
        "_get_clip_components",
        lambda: (_FakeClipModel(), _fake_preprocess, "cpu"),
    )


class TestEmbedImageInvariants:
    """embed_image must return unit-norm, float32, 1-D vectors."""

    def test_unit_norm_float32_one_dimensional(self, fake_clip):
        vec = embeddings_module.embed_image(Image.new("RGB", (32, 32)))

        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.dtype == np.float32
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-6)

    def test_normalizes_raw_model_output(self, fake_clip):
        vec = embeddings_module.embed_image(Image.new("RGB", (32, 32)))

        # Raw model output is [3, 4, 0, 0]; normalized -> [0.6, 0.8, 0, 0]
        assert vec == pytest.approx([0.6, 0.8, 0.0, 0.0], abs=1e-6)

    def test_path_and_pil_inputs_share_one_code_path(self, fake_clip, tmp_path):
        image = Image.new("RGB", (32, 32), color=(10, 20, 30))
        image_path = tmp_path / "img.png"
        image.save(image_path)

        from_pil = embeddings_module.embed_image(image)
        from_path = embeddings_module.embed_image(image_path)

        assert np.array_equal(from_pil, from_path)

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            embeddings_module.embed_image(tmp_path / "does-not-exist.jpg")


class TestPreprocessingParity:
    """Catalogue build and all query paths must share one embedding function.

    If any of these identities break, catalogue and query embeddings can
    be produced by different preprocessing pipelines — the exact failure
    mode this regression test exists to prevent.
    """

    def test_catalog_build_uses_shared_embed_image(self):
        assert index_module.embed_image is embeddings_module.embed_image

    def test_sync_service_uses_shared_embed_image(self):
        assert service_module.embed_image is embeddings_module.embed_image

    def test_async_service_uses_shared_embed_image(self):
        assert async_service_module.embed_image is embeddings_module.embed_image

    def test_async_embed_pil_image_delegates_to_embed_image(self, monkeypatch):
        sentinel = np.zeros(4, dtype=np.float32)
        calls: list[object] = []

        def _spy(image):
            calls.append(image)
            return sentinel

        monkeypatch.setattr(async_service_module, "embed_image", _spy)

        image = Image.new("RGB", (16, 16))
        result = AsyncIdentificationService._embed_pil_image(image)

        assert result is sentinel
        assert calls == [image]
