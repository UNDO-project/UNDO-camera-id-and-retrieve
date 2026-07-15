r"""Deterministic image degradations for retrieval evaluation.

Each degradation is a pure function ``(image, severity, rng) -> image``
that simulates one way a street crop differs from a clean catalogue
photo. All randomness comes from the passed
:class:`numpy.random.Generator` — no global state — so a fixed seed
always reproduces the same degraded image.

Severities run from 1 (mild) to 3 (harsh). The mild end of the same
functions doubles as the catalogue augmentation recipe
(:data:`AUGMENTATION_RECIPES`), so measurement and augmentation share
one code path.
"""

import io
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

Degradation = Callable[[Image.Image, int, np.random.Generator], Image.Image]

SEVERITIES: tuple[int, ...] = (1, 2, 3)


def _check_severity(severity: int) -> None:
    """Validate that severity is one of the supported levels.

    :param severity: Severity level to validate
    :raises ValueError: If severity is not in :data:`SEVERITIES`
    """
    if severity not in SEVERITIES:
        raise ValueError(f"Severity must be one of {SEVERITIES}, got {severity}")


def perspective_warp(
    image: Image.Image, severity: int, rng: np.random.Generator
) -> Image.Image:
    r"""Warp the image as if photographed off-angle.

    Jitters the four corners by a fraction of the image size (larger at
    higher severity) and applies the resulting perspective transform.
    Border pixels are replicated so no black wedges are introduced.

    :param image: Input image
    :param severity: Severity level (1-3)
    :param rng: Random generator driving the corner jitter
    :return: Warped image, same size as the input
    """
    _check_severity(severity)
    jitter_fraction = {1: 0.04, 2: 0.08, 3: 0.14}[severity]

    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]

    src = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    max_dx = jitter_fraction * width
    max_dy = jitter_fraction * height
    offsets = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
    dst = src + offsets * np.array([max_dx, max_dy], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(warped)


def small_scale(
    image: Image.Image, severity: int, rng: np.random.Generator
) -> Image.Image:
    r"""Simulate a distant camera: shrink to a small size and blow back up.

    The long side is reduced to 96/64/48 pixels depending on severity,
    then the image is resized back to its original dimensions.

    :param image: Input image
    :param severity: Severity level (1-3)
    :param rng: Unused; accepted for a uniform degradation signature
    :return: Rescaled image, same size as the input
    """
    _check_severity(severity)
    target_long_side = {1: 96, 2: 64, 3: 48}[severity]

    image = image.convert("RGB")
    width, height = image.size
    scale = target_long_side / max(width, height)
    if scale >= 1.0:
        return image

    small = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    return small.resize((width, height), Image.Resampling.BILINEAR)


def blur(image: Image.Image, severity: int, rng: np.random.Generator) -> Image.Image:
    r"""Apply gaussian blur plus a simple linear motion kernel.

    Gaussian radius grows with severity; the motion kernel length grows
    with it, with the motion direction drawn from the rng.

    :param image: Input image
    :param severity: Severity level (1-3)
    :param rng: Random generator driving the motion direction
    :return: Blurred image
    """
    _check_severity(severity)
    gaussian_radius = {1: 1.0, 2: 2.0, 3: 3.5}[severity]
    kernel_length = {1: 3, 2: 5, 3: 9}[severity]

    blurred = image.convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=gaussian_radius)
    )

    # Horizontal, vertical, or diagonal motion streak
    kernel = np.zeros((kernel_length, kernel_length), dtype=np.float32)
    direction = int(rng.integers(0, 4))
    center = kernel_length // 2
    if direction == 0:  # horizontal
        kernel[center, :] = 1.0
    elif direction == 1:  # vertical
        kernel[:, center] = 1.0
    elif direction == 2:  # main diagonal
        np.fill_diagonal(kernel, 1.0)
    else:  # anti-diagonal
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    kernel /= kernel.sum()

    array = cv2.filter2D(np.asarray(blurred), -1, kernel)
    return Image.fromarray(array)


def jpeg_compression(
    image: Image.Image, severity: int, rng: np.random.Generator
) -> Image.Image:
    r"""Re-encode the image as a low-quality JPEG.

    :param image: Input image
    :param severity: Severity level (1-3)
    :param rng: Unused; accepted for a uniform degradation signature
    :return: Recompressed image
    """
    _check_severity(severity)
    quality = {1: 40, 2: 25, 3: 10}[severity]

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def lighting(
    image: Image.Image, severity: int, rng: np.random.Generator
) -> Image.Image:
    r"""Jitter brightness, contrast, and colour saturation.

    Enhancement factors are drawn uniformly from ranges that widen with
    severity, simulating backlighting, shade, and colour casts.

    :param image: Input image
    :param severity: Severity level (1-3)
    :param rng: Random generator driving the jitter factors
    :return: Adjusted image
    """
    _check_severity(severity)
    delta = {1: 0.15, 2: 0.30, 3: 0.50}[severity]

    result = image.convert("RGB")
    for enhancer_cls in (
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
    ):
        factor = float(rng.uniform(1.0 - delta, 1.0 + delta))
        result = enhancer_cls(result).enhance(factor)
    return result


def background_swap(
    image: Image.Image, severity: int, rng: np.random.Generator
) -> Image.Image:
    r"""Replace the near-white studio background with another surface.

    Catalogue images are studio shots on white sweeps, so background
    pixels are identified with a simple near-white threshold. Severity
    selects the replacement: flat grey (1), uniform noise (2), or a
    stripe texture (3).

    :param image: Input image
    :param severity: Severity level (1-3)
    :param rng: Random generator driving background parameters
    :return: Image with the background replaced
    """
    _check_severity(severity)

    array = np.asarray(image.convert("RGB")).copy()
    height, width = array.shape[:2]
    background_mask = np.all(array >= 240, axis=-1)

    if severity == 1:
        grey = int(rng.integers(60, 180))
        replacement = np.full((height, width, 3), grey, dtype=np.uint8)
    elif severity == 2:
        replacement = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    else:
        stripe_width = max(4, width // 16)
        offset = int(rng.integers(0, stripe_width))
        columns = (np.arange(width) + offset) // stripe_width % 2
        stripes = np.where(columns == 0, 70, 160).astype(np.uint8)
        replacement = np.broadcast_to(
            stripes[np.newaxis, :, np.newaxis], (height, width, 3)
        ).copy()

    array[background_mask] = replacement[background_mask]
    return Image.fromarray(array)


DEGRADATIONS: dict[str, Degradation] = {
    "perspective_warp": perspective_warp,
    "small_scale": small_scale,
    "blur": blur,
    "jpeg_compression": jpeg_compression,
    "lighting": lighting,
    "background_swap": background_swap,
}

# Mild (name, severity) pairs used by catalogue augmentation (WP3).
# Ordered so the first K entries give a balanced mix; K is capped at 8.
AUGMENTATION_RECIPES: tuple[tuple[str, int], ...] = (
    ("perspective_warp", 1),
    ("lighting", 1),
    ("small_scale", 1),
    ("blur", 1),
    ("perspective_warp", 2),
    ("lighting", 2),
    ("jpeg_compression", 1),
    ("small_scale", 2),
)


def apply_degradation(
    name: str, image: Image.Image, severity: int, rng: np.random.Generator
) -> Image.Image:
    r"""Apply a degradation by registry name.

    :param name: Key in :data:`DEGRADATIONS`
    :param image: Input image
    :param severity: Severity level (1-3)
    :param rng: Random generator for the degradation
    :return: Degraded image
    :raises KeyError: If the degradation name is unknown
    """
    if name not in DEGRADATIONS:
        raise KeyError(
            f"Unknown degradation '{name}'. Available: {sorted(DEGRADATIONS)}"
        )
    return DEGRADATIONS[name](image, severity, rng)
