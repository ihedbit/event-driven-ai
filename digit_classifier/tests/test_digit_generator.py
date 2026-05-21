"""
test_digit_generator.py — Tests for digit_generator.py

All tests are pure Python / NumPy — no NEST required.

WHY these tests matter:
  The digit templates ARE the learned knowledge of this classifier.
  If segments are wrongly positioned or pixels overlap, the weight
  calibration breaks and classification fails silently.
"""

import numpy as np
import pytest
from digit_generator import (
    IMAGE_SIZE, SEGMENTS, DIGIT_SEGMENTS,
    create_digit_image, create_all_digit_images,
    create_random_image, get_digit_template_indices,
    get_all_template_indices, get_segment_pixel_counts,
)

ALL_DIGITS = list(range(10))


# ── Image shape and type ──────────────────────────────────────────────────────

class TestImageShape:
    def test_every_digit_image_is_20x20(self):
        for d in ALL_DIGITS:
            img = create_digit_image(d)
            assert img.shape == (IMAGE_SIZE, IMAGE_SIZE), (
                f"Digit {d}: shape {img.shape} != ({IMAGE_SIZE},{IMAGE_SIZE})")

    def test_pixel_values_are_binary(self):
        for d in ALL_DIGITS:
            img = create_digit_image(d)
            unique = set(np.unique(img))
            assert unique.issubset({0, 1}), (
                f"Digit {d}: non-binary values {unique}")

    def test_invalid_digit_raises(self):
        with pytest.raises(ValueError):
            create_digit_image(10)
        with pytest.raises(ValueError):
            create_digit_image(-1)


# ── Active pixel counts ───────────────────────────────────────────────────────

class TestActivePixelCounts:
    """Each digit must have a positive, known number of active pixels."""

    # Expected pixel counts based on segment geometry
    EXPECTED = {
        0: 80,   # A+B+C+D+E+F
        1: 24,   # B+C
        2: 72,   # A+B+D+E+G
        3: 72,   # A+B+C+D+G
        4: 52,   # B+C+F+G
        5: 72,   # A+C+D+F+G
        6: 84,   # A+C+D+E+F+G
        7: 40,   # A+B+C
        8: 96,   # all segments
        9: 84,   # A+B+C+D+F+G
    }

    def test_digit_pixel_counts_match_expected(self):
        for d, expected in self.EXPECTED.items():
            img = create_digit_image(d)
            actual = int(img.sum())
            assert actual == expected, (
                f"Digit {d}: {actual} active pixels, expected {expected}")

    def test_digit_8_has_most_pixels(self):
        counts = {d: int(create_digit_image(d).sum()) for d in ALL_DIGITS}
        assert max(counts, key=counts.get) == 8

    def test_digit_1_has_fewest_pixels(self):
        counts = {d: int(create_digit_image(d).sum()) for d in ALL_DIGITS}
        assert min(counts, key=counts.get) == 1


# ── Digit uniqueness ──────────────────────────────────────────────────────────

class TestDigitUniqueness:
    """Every digit must produce a distinct image — otherwise classification
    is impossible even in principle."""

    def test_all_digit_images_are_unique(self):
        images = {d: create_digit_image(d) for d in ALL_DIGITS}
        for i in ALL_DIGITS:
            for j in ALL_DIGITS:
                if i < j:
                    assert not np.array_equal(images[i], images[j]), (
                        f"Digits {i} and {j} produce identical images!")

    def test_no_digit_image_is_all_zeros(self):
        for d in ALL_DIGITS:
            img = create_digit_image(d)
            assert img.sum() > 0, f"Digit {d} image is completely blank"


# ── Segment geometry ──────────────────────────────────────────────────────────

class TestSegmentGeometry:
    """Verify segment definitions produce correct pixel areas."""

    def test_segment_pixel_counts(self):
        # Expected from the docstring: A=D=G=16, B=C=E=F=12
        expected = {"A": 16, "B": 12, "C": 12, "D": 16,
                    "E": 12, "F": 12, "G": 16}
        counts = get_segment_pixel_counts()
        for seg, count in expected.items():
            assert counts[seg] == count, (
                f"Segment {seg}: got {counts[seg]}, expected {count}")

    def test_segments_do_not_overlap(self):
        """No two segments should share pixels."""
        for seg_a, (ra, ca) in SEGMENTS.items():
            for seg_b, (rb, cb) in SEGMENTS.items():
                if seg_a >= seg_b:
                    continue
                # Check row and column overlap
                rows_a = set(range(ra.start, ra.stop))
                rows_b = set(range(rb.start, rb.stop))
                cols_a = set(range(ca.start, ca.stop))
                cols_b = set(range(cb.start, cb.stop))
                row_overlap = rows_a & rows_b
                col_overlap = cols_a & cols_b
                # Segments overlap only if BOTH row and column ranges overlap
                assert not (row_overlap and col_overlap), (
                    f"Segments {seg_a} and {seg_b} overlap at "
                    f"rows {row_overlap}, cols {col_overlap}")

    def test_digit_8_activates_all_segments(self):
        img = create_digit_image(8)
        for seg_name, (row_sl, col_sl) in SEGMENTS.items():
            region = img[row_sl, col_sl]
            assert region.all(), (
                f"Digit 8 should activate segment {seg_name} but doesn't")

    def test_digit_1_activates_only_b_and_c(self):
        img = create_digit_image(1)
        # Only B and C should be lit
        lit_segments = {
            name for name in SEGMENTS
            if img[SEGMENTS[name]].any()
        }
        assert lit_segments == {"B", "C"}, (
            f"Digit 1 should only have B,C lit; got {lit_segments}")


# ── Template indices ──────────────────────────────────────────────────────────

class TestTemplateIndices:
    def test_indices_are_within_range(self):
        for d in ALL_DIGITS:
            indices = get_digit_template_indices(d)
            assert all(0 <= i < 400 for i in indices), (
                f"Digit {d} has out-of-range indices")

    def test_indices_are_unique(self):
        for d in ALL_DIGITS:
            indices = get_digit_template_indices(d)
            assert len(indices) == len(set(indices)), (
                f"Digit {d} has duplicate indices")

    def test_indices_count_matches_active_pixels(self):
        for d in ALL_DIGITS:
            img = create_digit_image(d)
            indices = get_digit_template_indices(d)
            assert len(indices) == int(img.sum()), (
                f"Digit {d}: index count {len(indices)} != "
                f"pixel sum {int(img.sum())}")

    def test_indexed_pixels_are_active(self):
        for d in ALL_DIGITS:
            img = create_digit_image(d).flatten()
            for idx in get_digit_template_indices(d):
                assert img[idx] == 1, (
                    f"Digit {d}: index {idx} points to inactive pixel")

    def test_all_template_indices_returns_all_10_digits(self):
        all_idx = get_all_template_indices()
        assert set(all_idx.keys()) == set(range(10))


# ── Random image ──────────────────────────────────────────────────────────────

class TestRandomImage:
    def test_random_image_is_20x20(self):
        assert create_random_image().shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_random_image_is_binary(self):
        img = create_random_image()
        assert set(np.unique(img)).issubset({0, 1})

    def test_same_seed_reproducible(self):
        assert np.array_equal(create_random_image(seed=7),
                              create_random_image(seed=7))

    def test_different_seeds_differ(self):
        assert not np.array_equal(create_random_image(seed=1),
                                  create_random_image(seed=2))

    def test_random_image_differs_from_all_digits(self):
        rnd = create_random_image(seed=42)
        for d in ALL_DIGITS:
            assert not np.array_equal(rnd, create_digit_image(d)), (
                f"Random image (seed=42) is identical to digit {d}!")
