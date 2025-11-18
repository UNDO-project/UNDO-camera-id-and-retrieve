"""Image embedding utilities for camera identification (skeleton).

This module will provide functions to compute vector representations of
images, which are later used for nearest-neighbor search in the catalog.
"""

from pathlib import Path


def embed_image(image_path: Path | str) -> object:
    r"""Compute an embedding vector for the given image.

    This placeholder will be implemented using a suitable image embedding
    model (for example, CLIP or another vision backbone).

    :param image_path: Path to the image to embed
    :return: Embedding vector representation
    :raises NotImplementedError: Always, until implemented
    """
    raise NotImplementedError("embed_image is not implemented yet")
