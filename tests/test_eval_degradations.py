"""Tests for the eval degradation functions."""

import numpy as np
import pytest
from PIL import Image

from src.identification.eval.degradations import (
    AUGMENTATION_RECIPES,
    DEGRADATIONS,
    SEVERITIES,
    apply_degradation,
    background_swap,
)


@pytest.fixture
def sample_image() -> Image.Image:
    """A small structured RGB image (gradient + white background band)."""
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    # Gradient block (the "object")
    for y in range(16, 48):
        for x in range(16, 48):
            array[y, x] = (x * 4 % 256, y * 4 % 256, 128)
    # Near-white surround (the "studio sweep")
    array[:16, :] = 250
    array[48:, :] = 250
    array[:, :16] = 250
    array[:, 48:] = 250
    return Image.fromarray(array)


class TestDegradationContracts:
    """Every registered degradation obeys the shared contract."""

    @pytest.mark.parametrize("name", sorted(DEGRADATIONS))
    @pytest.mark.parametrize("severity", SEVERITIES)
    def test_returns_rgb_image_of_same_size(self, name, severity, sample_image):
        rng = np.random.default_rng(123)

        result = apply_degradation(name, sample_image, severity, rng)

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == sample_image.size

    @pytest.mark.parametrize("name", sorted(DEGRADATIONS))
    def test_deterministic_under_fixed_seed(self, name, sample_image):
        first = apply_degradation(name, sample_image, 2, np.random.default_rng(99))
        second = apply_degradation(name, sample_image, 2, np.random.default_rng(99))

        assert np.array_equal(np.array(first), np.array(second))

    @pytest.mark.parametrize("name", sorted(DEGRADATIONS))
    def test_does_not_mutate_input(self, name, sample_image):
        before = np.array(sample_image)

        apply_degradation(name, sample_image, 3, np.random.default_rng(1))

        assert np.array_equal(np.array(sample_image), before)

    @pytest.mark.parametrize("name", sorted(DEGRADATIONS))
    @pytest.mark.parametrize("severity", [0, 4, -1])
    def test_invalid_severity_rejected(self, name, severity, sample_image):
        with pytest.raises(ValueError):
            apply_degradation(name, sample_image, severity, np.random.default_rng(1))

    def test_unknown_name_rejected(self, sample_image):
        with pytest.raises(KeyError):
            apply_degradation("nonsense", sample_image, 1, np.random.default_rng(1))


class TestBackgroundSwap:
    """background_swap replaces only the near-white sweep."""

    def test_replaces_near_white_pixels_only(self, sample_image):
        rng = np.random.default_rng(5)

        result = np.array(background_swap(sample_image, 1, rng))
        original = np.array(sample_image)

        background = np.all(original >= 240, axis=-1)
        # Background changed away from near-white
        assert not np.any(np.all(result[background] >= 240, axis=-1))
        # Object pixels untouched
        assert np.array_equal(result[~background], original[~background])


class TestAugmentationRecipes:
    """The augmentation recipe list backs WP3's K-variant cap."""

    def test_recipes_reference_registered_degradations(self):
        for name, severity in AUGMENTATION_RECIPES:
            assert name in DEGRADATIONS
            assert severity in SEVERITIES

    def test_recipe_count_covers_augment_k_cap(self):
        assert len(AUGMENTATION_RECIPES) == 8
