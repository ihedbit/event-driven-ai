"""
test_image_generator.py — Tests for image_generator.py

These tests verify the image creation logic independently of any SNN
simulation.  They only depend on NumPy, so they run without NEST.

WHY these tests matter:
  The pixel→neuron mapping is the foundation of the whole system.
  If the T shape is generated incorrectly, detection will silently fail.
"""

import numpy as np
import pytest
from image_generator import (
    IMAGE_SIZE,
    create_t_shape_image,
    create_random_image,
    get_t_shape_pixel_indices,
    image_to_flat,
    T_BAR_ROW_START, T_BAR_ROW_END, T_BAR_COL_START, T_BAR_COL_END,
    T_STEM_ROW_START, T_STEM_ROW_END, T_STEM_COL_START, T_STEM_COL_END,
)


# ── Image shape ───────────────────────────────────────────────────────────────

class TestImageShape:
    """Every image must be exactly IMAGE_SIZE × IMAGE_SIZE."""

    def test_t_shape_image_is_20x20(self):
        # The network expects exactly 400 input neurons (20×20).
        img = create_t_shape_image()
        assert img.shape == (IMAGE_SIZE, IMAGE_SIZE), (
            f"T image shape {img.shape} != ({IMAGE_SIZE}, {IMAGE_SIZE})")

    def test_random_image_is_20x20(self):
        img = create_random_image()
        assert img.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_flat_array_has_400_elements(self):
        img = create_t_shape_image()
        flat = image_to_flat(img)
        assert flat.shape == (400,)


# ── Binary pixel values ───────────────────────────────────────────────────────

class TestBinaryValues:
    """All pixel values must be 0 or 1 — nothing else is valid spike input."""

    def test_t_shape_pixels_are_binary(self):
        img = create_t_shape_image()
        unique_values = set(np.unique(img))
        assert unique_values.issubset({0, 1}), (
            f"Non-binary pixels found: {unique_values}")

    def test_random_image_pixels_are_binary(self):
        img = create_random_image(seed=7)
        unique_values = set(np.unique(img))
        assert unique_values.issubset({0, 1})


# ── T-shape pixel positions ───────────────────────────────────────────────────

class TestTShapeGeometry:
    """Verify the exact spatial layout of the T pattern."""

    def test_bar_pixels_are_active(self):
        # The horizontal bar must be fully lit.
        img = create_t_shape_image()
        bar_region = img[T_BAR_ROW_START:T_BAR_ROW_END, T_BAR_COL_START:T_BAR_COL_END]
        assert bar_region.all(), "Some bar pixels are inactive"

    def test_stem_pixels_are_active(self):
        # The vertical stem must be fully lit.
        img = create_t_shape_image()
        stem_region = img[T_STEM_ROW_START:T_STEM_ROW_END, T_STEM_COL_START:T_STEM_COL_END]
        assert stem_region.all(), "Some stem pixels are inactive"

    def test_corners_are_inactive(self):
        # Pixels outside the T pattern must be 0 (these neurons will be silent).
        img = create_t_shape_image()
        assert img[0, 0]  == 0, "Top-left corner should be inactive"
        assert img[19, 0] == 0, "Bottom-left corner should be inactive"
        assert img[19, 19] == 0, "Bottom-right corner should be inactive"

    def test_total_active_pixel_count(self):
        # 16 bar pixels + 15×2 stem pixels = 46 total T pixels.
        img = create_t_shape_image()
        n_bar  = (T_BAR_ROW_END  - T_BAR_ROW_START)  * (T_BAR_COL_END  - T_BAR_COL_START)
        n_stem = (T_STEM_ROW_END - T_STEM_ROW_START) * (T_STEM_COL_END - T_STEM_COL_START)
        expected = n_bar + n_stem
        assert int(img.sum()) == expected, (
            f"Expected {expected} active pixels, got {int(img.sum())}")


# ── Flat pixel index list ─────────────────────────────────────────────────────

class TestPixelIndices:
    """The flat index list drives synaptic weight assignment."""

    def test_indices_count_matches_active_pixels(self):
        indices = get_t_shape_pixel_indices()
        img = create_t_shape_image()
        assert len(indices) == int(img.sum())

    def test_indices_are_within_range(self):
        # Flat indices must be in [0, 399] for a 20×20 image.
        indices = get_t_shape_pixel_indices()
        assert all(0 <= idx < 400 for idx in indices), (
            "Some T-pixel indices are out of [0, 399]")

    def test_indices_are_unique(self):
        indices = get_t_shape_pixel_indices()
        assert len(indices) == len(set(indices)), "Duplicate T-pixel indices found"

    def test_indices_correspond_to_active_pixels(self):
        # Each returned index must point to a pixel with value 1.
        indices = get_t_shape_pixel_indices()
        flat_image = create_t_shape_image().flatten()
        for idx in indices:
            assert flat_image[idx] == 1, (
                f"Index {idx} points to an inactive pixel (value = {flat_image[idx]})")


# ── Random image properties ───────────────────────────────────────────────────

class TestRandomImage:
    """Random image must be reproducible and different from the T image."""

    def test_different_from_t_shape(self):
        t_img  = create_t_shape_image()
        rnd_img = create_random_image(seed=42)
        assert not np.array_equal(t_img, rnd_img), (
            "Random image is identical to the T-shape image — seed issue?")

    def test_reproducible_with_same_seed(self):
        img1 = create_random_image(seed=99)
        img2 = create_random_image(seed=99)
        assert np.array_equal(img1, img2), "Same seed produced different images"

    def test_different_seeds_give_different_images(self):
        img1 = create_random_image(seed=1)
        img2 = create_random_image(seed=2)
        assert not np.array_equal(img1, img2)

    def test_density_is_approximately_correct(self):
        # At density=0.30, roughly 30% of 400 pixels should be active.
        img = create_random_image(seed=42, density=0.30)
        fraction = img.sum() / img.size
        assert 0.15 < fraction < 0.50, (
            f"Active pixel fraction {fraction:.2f} is far from expected 0.30")

    def test_few_t_pixels_active_in_random_image(self):
        # With 30% density the expected overlap is ~14/46 T-pixels.
        # We verify it stays well below the detection threshold (≥32).
        rnd_img = create_random_image(seed=42, density=0.30)
        t_indices = set(get_t_shape_pixel_indices())
        flat = rnd_img.flatten()
        n_t_active = sum(flat[i] for i in t_indices)
        assert n_t_active < 32, (
            f"Random image activates {n_t_active} T-pixels — may trigger false positive")
