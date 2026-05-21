"""
test_encoder.py — Tests for encoder.py

Pure Python / NumPy — no NEST required.

KEY PROPERTIES:
  - 400 sublists returned (one per pixel)
  - Active pixel (1) → [spike_time]
  - Inactive pixel (0) → []
  - Spike times within simulation window
"""

import numpy as np
import pytest
from encoder import encode_image_to_spikes, count_active_pixels, get_active_neuron_indices
from digit_generator import create_digit_image, create_random_image, IMAGE_SIZE


class TestOutputStructure:
    def test_returns_400_sublists_for_all_digits(self):
        for d in range(10):
            img = create_digit_image(d)
            result = encode_image_to_spikes(img)
            assert len(result) == 400, (
                f"Digit {d}: expected 400 sublists, got {len(result)}")

    def test_returns_400_sublists_for_blank_image(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        result = encode_image_to_spikes(img)
        assert len(result) == 400

    def test_returns_400_sublists_for_all_ones(self):
        img = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        result = encode_image_to_spikes(img)
        assert len(result) == 400


class TestActivePixels:
    def test_active_pixel_produces_one_spike(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        img[5, 7] = 1
        result = encode_image_to_spikes(img, spike_time=10.0)
        idx = 5 * IMAGE_SIZE + 7
        assert result[idx] == [10.0]

    def test_spike_time_matches_parameter(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        img[0, 0] = 1
        for t in [5.0, 10.0, 25.0, 50.0]:
            result = encode_image_to_spikes(img, spike_time=t, sim_time=100.0)
            assert result[0] == [t]

    def test_all_active_pixels_produce_spikes(self):
        for d in range(10):
            img = create_digit_image(d)
            result = encode_image_to_spikes(img)
            flat = img.flatten()
            for i, (px, spikes) in enumerate(zip(flat, result)):
                if px == 1:
                    assert len(spikes) == 1, (
                        f"Digit {d} pixel {i}: expected 1 spike, got {len(spikes)}")


class TestInactivePixels:
    def test_blank_image_all_silent(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        result = encode_image_to_spikes(img)
        assert all(s == [] for s in result)

    def test_inactive_pixels_produce_no_spikes(self):
        for d in range(10):
            img = create_digit_image(d)
            result = encode_image_to_spikes(img)
            flat = img.flatten()
            for i, (px, spikes) in enumerate(zip(flat, result)):
                if px == 0:
                    assert spikes == [], (
                        f"Digit {d} pixel {i} is inactive but has spikes: {spikes}")


class TestSpikeTimeValidity:
    def test_spike_time_inside_window(self):
        img = create_digit_image(8)
        result = encode_image_to_spikes(img, spike_time=10.0, sim_time=100.0)
        for spikes in result:
            for t in spikes:
                assert 0 < t < 100.0

    def test_invalid_spike_time_raises(self):
        img = create_digit_image(0)
        with pytest.raises(ValueError):
            encode_image_to_spikes(img, spike_time=0.0, sim_time=100.0)
        with pytest.raises(ValueError):
            encode_image_to_spikes(img, spike_time=100.0, sim_time=100.0)
        with pytest.raises(ValueError):
            encode_image_to_spikes(img, spike_time=-1.0, sim_time=100.0)


class TestCountHelpers:
    def test_count_matches_image_sum(self):
        for d in range(10):
            img = create_digit_image(d)
            result = encode_image_to_spikes(img)
            assert count_active_pixels(result) == int(img.sum())

    def test_blank_count_is_zero(self):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        result = encode_image_to_spikes(img)
        assert count_active_pixels(result) == 0

    def test_active_indices_all_have_spikes(self):
        img = create_digit_image(5)
        result = encode_image_to_spikes(img)
        for i in get_active_neuron_indices(result):
            assert result[i] != []
