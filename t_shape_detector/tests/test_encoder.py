"""
test_encoder.py — Tests for encoder.py

These tests verify the pixel-to-spike conversion WITHOUT running a simulation.
We only need NumPy here, no NEST.

KEY PROPERTIES being tested:
  - Active pixels (1)   → exactly one spike time in the output
  - Inactive pixels (0) → empty list (no spikes)
  - Total spike sources  → 400 (one per pixel)
  - Spike times are valid (within simulation window)
"""

import numpy as np
import pytest
from encoder import encode_image_to_spikes, count_active_pixels, get_active_neuron_indices
from image_generator import create_t_shape_image, create_random_image, IMAGE_SIZE


# ── Basic output structure ────────────────────────────────────────────────────

class TestOutputStructure:
    """The encoder must always return exactly 400 sublists."""

    def test_returns_400_sublists_for_t_image(self):
        img = create_t_shape_image()
        result = encode_image_to_spikes(img)
        assert len(result) == 400, (
            f"Expected 400 spike-time lists, got {len(result)}")

    def test_returns_400_sublists_for_random_image(self):
        img = create_random_image()
        result = encode_image_to_spikes(img)
        assert len(result) == 400

    def test_returns_400_sublists_for_all_zeros(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        result = encode_image_to_spikes(img)
        assert len(result) == 400

    def test_returns_400_sublists_for_all_ones(self):
        img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        result = encode_image_to_spikes(img)
        assert len(result) == 400


# ── Active pixels generate spikes ────────────────────────────────────────────

class TestActivePixels:
    """Each active pixel (value=1) must produce exactly one spike."""

    def test_single_active_pixel_produces_one_spike(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        img[5, 7] = 1                         # single active pixel
        flat_idx = 5 * IMAGE_SIZE + 7         # = 107
        result = encode_image_to_spikes(img, spike_time=10.0)
        assert len(result[flat_idx]) == 1, (
            "Active pixel should produce exactly one spike")

    def test_active_pixel_spike_time_matches_parameter(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        img[0, 0] = 1
        for t in [5.0, 10.0, 20.0, 50.0]:
            result = encode_image_to_spikes(img, spike_time=t, sim_time=100.0)
            assert result[0] == [t], f"Spike time should be {t}, got {result[0]}"

    def test_t_shape_image_all_t_pixels_produce_spikes(self):
        img = create_t_shape_image()
        result = encode_image_to_spikes(img)
        flat = img.flatten()
        for idx, (pixel, spikes) in enumerate(zip(flat, result)):
            if pixel == 1:
                assert len(spikes) == 1, (
                    f"Active pixel at index {idx} produced {len(spikes)} spikes, expected 1")


# ── Inactive pixels generate no spikes ───────────────────────────────────────

class TestInactivePixels:
    """Each inactive pixel (value=0) must produce an empty list."""

    def test_all_zeros_image_has_no_spikes(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        result = encode_image_to_spikes(img)
        for i, spikes in enumerate(result):
            assert spikes == [], f"Neuron {i} should be silent but has spikes: {spikes}"

    def test_t_shape_image_inactive_pixels_are_silent(self):
        img = create_t_shape_image()
        result = encode_image_to_spikes(img)
        flat = img.flatten()
        for idx, (pixel, spikes) in enumerate(zip(flat, result)):
            if pixel == 0:
                assert spikes == [], (
                    f"Inactive pixel at index {idx} should be silent, got spikes: {spikes}")

    def test_single_inactive_pixel_in_otherwise_active_image(self):
        img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        img[10, 10] = 0
        flat_idx = 10 * IMAGE_SIZE + 10
        result = encode_image_to_spikes(img)
        assert result[flat_idx] == [], (
            "The single inactive pixel should produce no spikes")


# ── Spike time validity ───────────────────────────────────────────────────────

class TestSpikeTimeValidity:
    """All generated spike times must fall within the simulation window."""

    def test_spike_times_are_positive(self):
        img = create_t_shape_image()
        result = encode_image_to_spikes(img, spike_time=10.0, sim_time=100.0)
        for spikes in result:
            for t in spikes:
                assert t > 0, f"Spike time {t} must be positive"

    def test_spike_times_are_within_sim_window(self):
        sim_time = 100.0
        spike_time = 10.0
        img = create_t_shape_image()
        result = encode_image_to_spikes(img, spike_time=spike_time, sim_time=sim_time)
        for spikes in result:
            for t in spikes:
                assert t < sim_time, (
                    f"Spike time {t} exceeds sim_time={sim_time}")

    def test_invalid_spike_time_raises(self):
        img = create_t_shape_image()
        with pytest.raises(ValueError):
            encode_image_to_spikes(img, spike_time=0.0, sim_time=100.0)
        with pytest.raises(ValueError):
            encode_image_to_spikes(img, spike_time=100.0, sim_time=100.0)
        with pytest.raises(ValueError):
            encode_image_to_spikes(img, spike_time=-5.0, sim_time=100.0)


# ── Count helpers ─────────────────────────────────────────────────────────────

class TestCountHelpers:
    """Helper functions must return correct counts."""

    def test_count_active_pixels_matches_image_sum(self):
        img = create_t_shape_image()
        result = encode_image_to_spikes(img)
        assert count_active_pixels(result) == int(img.sum())

    def test_count_active_pixels_is_zero_for_all_zeros(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        result = encode_image_to_spikes(img)
        assert count_active_pixels(result) == 0

    def test_get_active_indices_length_matches_count(self):
        img = create_random_image(seed=42)
        result = encode_image_to_spikes(img)
        indices = get_active_neuron_indices(result)
        assert len(indices) == count_active_pixels(result)

    def test_get_active_indices_all_have_spikes(self):
        img = create_t_shape_image()
        result = encode_image_to_spikes(img)
        indices = get_active_neuron_indices(result)
        for i in indices:
            assert len(result[i]) > 0, (
                f"Index {i} returned by get_active_neuron_indices but has no spikes")

    def test_pixel_to_flat_index_mapping(self):
        # Specific known pixel: row=2, col=2 → flat index = 2*20+2 = 42
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        img[2, 2] = 1
        result = encode_image_to_spikes(img)
        assert result[42] == [10.0]
        for i, spikes in enumerate(result):
            if i != 42:
                assert spikes == [], f"Only pixel 42 should fire, but neuron {i} does too"
