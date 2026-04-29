"""Image embedding utilities for camera identification.

This module provides a function to compute vector representations of
images using a CLIP model from the ``open-clip-torch`` library. The
resulting embeddings can be used for nearest-neighbor search in the
camera catalog.

The implementation performs lazy imports so that importing this module
does not require the embedding dependencies to be installed until
:func:`embed_image` is actually called. It also supports running on
CUDA GPUs, Apple MPS, or CPU, selecting the best available device at
runtime.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image


_CLIP_MODEL_NAME = "ViT-B-32"
_CLIP_PRETRAINED = "laion2b_s34b_b79k"


def _select_device(torch_module: Any) -> str:
    r"""Select computation device with preference for CUDA, then MPS, then CPU.

    :param torch_module: Imported ``torch`` module
    :return: Device string understood by ``torch`` ("cuda", "mps" or "cpu")
    """
    if torch_module.cuda.is_available():
        return "cuda"

    # Apple Silicon (Metal Performance Shaders) support
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():  # type: ignore[union-attr]
        return "mps"

    return "cpu"


@lru_cache(maxsize=1)
def _get_clip_components() -> tuple[object, object, str]:
    r"""Load and cache the CLIP model and preprocessing pipeline.

    This function imports ``torch`` and ``open_clip`` lazily and creates
    a CLIP model suitable for computing image embeddings.

    :return: Tuple of ``(model, preprocess, device)``
    :raises RuntimeError: If the required dependencies are not installed
    """
    try:
        import torch  # type: ignore[import]
        import open_clip  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "Image embeddings require 'open-clip-torch' and its dependencies. "
            "Install them with `uv add open-clip-torch torch`."
        ) from exc

    device = _select_device(torch)

    model, _, preprocess = open_clip.create_model_and_transforms(  # type: ignore[attr-defined]
        _CLIP_MODEL_NAME,
        pretrained=_CLIP_PRETRAINED,
    )
    model.to(device)
    model.eval()

    logger.info(
        f"Loaded CLIP model {_CLIP_MODEL_NAME} ({_CLIP_PRETRAINED}) on {device}"
    )
    return model, preprocess, device


def embed_image(image: Path | str | Image.Image) -> np.ndarray:
    r"""Compute an embedding vector for the given image using CLIP.

    Accepts either a filesystem path or an in-memory PIL Image. The
    image is preprocessed and passed through a CLIP model. The output
    embedding is L2-normalized and returned as a one-dimensional
    :class:`numpy.ndarray` of type ``float32``.

    Device selection order is:

    1. CUDA GPU if available.
    2. Apple MPS backend if available.
    3. CPU otherwise.

    :param image: Path to an image file, or a PIL Image instance
    :return: Normalized embedding vector representation
    :raises FileNotFoundError: If a path is given and the file does not exist
    :raises RuntimeError: If embedding dependencies are missing
    """
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for embedding: {image_path}")
        pil_image = Image.open(image_path).convert("RGB")

    model, preprocess, device = _get_clip_components()

    # Imports only needed when embeddings are used
    import torch  # type: ignore[import]

    with torch.no_grad():
        tensor = preprocess(pil_image).unsqueeze(0)
        if device != "cpu":
            tensor = tensor.to(device)
        features = model.encode_image(tensor)
        # L2-normalize
        features = features / features.norm(dim=-1, keepdim=True)

    embedding = features.detach().cpu().numpy().astype("float32")[0]
    return embedding
