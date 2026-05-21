"""
test_detection.py — End-to-end detection tests (requires PyNN + NEST).

These are integration tests that run a full simulation and check whether the
output neuron fires (T-shape detected) or stays silent (no T-shape).

All tests use the `skip_if_no_nest` fixture and are automatically skipped
if NEST is not installed.

WHAT WE'RE TESTING:
  The key behavioural guarantee of the whole system:
    "If I show it a T, it fires.  If I show it noise, it doesn't."
"""

import pytest
import numpy as np

from image_generator import (
    create_t_shape_image,
    create_random_image,
    get_t_shape_pixel_indices,
    IMAGE_SIZE,
)


# ── Positive detection: T-shape input ────────────────────────────────────────

class TestTShapeDetection:
    """The output neuron MUST fire when a full T-shape is presented."""

    def test_t_shape_causes_output_spike(self, skip_if_no_nest):
        # Core system requirement: the canonical T-shape triggers detection.
        from network import run_detection
        t_img = create_t_shape_image()
        output_spikes, _, _ = run_detection(t_img)
        assert len(output_spikes) > 0, (
            "T-shape image should cause the output neuron to fire, "
            "but no spikes were recorded")

    def test_detect_t_shape_returns_true(self, skip_if_no_nest):
        from network import detect_t_shape
        t_img = create_t_shape_image()
        result = detect_t_shape(t_img)
        assert result is True, (
            "detect_t_shape() should return True for a T-shape image")

    def test_output_spike_within_sim_window(self, skip_if_no_nest):
        from network import run_detection
        sim_time = 100.0
        t_img = create_t_shape_image()
        output_spikes, _, _ = run_detection(t_img, sim_time=sim_time)
        for t in output_spikes:
            assert 0 < t < sim_time, (
                f"Output spike at {t} ms is outside simulation window (0, {sim_time})")

    def test_output_spike_after_input_spike(self, skip_if_no_nest):
        # The output neuron cannot fire before its inputs arrive (causality).
        # Inputs fire at t=10 ms, so the output should fire after ~11 ms.
        from network import run_detection
        spike_time = 10.0
        t_img = create_t_shape_image()
        output_spikes, _, _ = run_detection(t_img, spike_time=spike_time)
        assert len(output_spikes) > 0
        assert output_spikes[0] > spike_time, (
            f"Output spike at {output_spikes[0]} ms is before input spike at {spike_time} ms")


# ── Negative detection: random noise input ───────────────────────────────────

class TestNoiseNonDetection:
    """The output neuron MUST stay silent for low-density noise."""

    def test_random_image_does_not_trigger_detection(self, skip_if_no_nest):
        from network import run_detection
        # Seed=42, density=30%: activates ~14/46 T-pixels → ΔV ≈ 8.8 mV < 20 mV threshold
        rnd_img = create_random_image(seed=42, density=0.30)
        output_spikes, _, _ = run_detection(rnd_img)
        assert len(output_spikes) == 0, (
            f"Random image should NOT trigger detection, but got spikes at {list(output_spikes)}")

    def test_detect_t_shape_returns_false_for_noise(self, skip_if_no_nest):
        from network import detect_t_shape
        rnd_img = create_random_image(seed=42, density=0.30)
        result = detect_t_shape(rnd_img)
        assert result is False, (
            "detect_t_shape() should return False for random noise")

    def test_all_zeros_image_does_not_fire(self, skip_if_no_nest):
        from network import run_detection
        blank_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        output_spikes, _, _ = run_detection(blank_img)
        assert len(output_spikes) == 0, (
            "A blank image with no spikes should never trigger detection")

    def test_detect_t_shape_returns_false_for_blank(self, skip_if_no_nest):
        from network import detect_t_shape
        blank_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        result = detect_t_shape(blank_img)
        assert result is False


# ── Robustness: partial T shapes ─────────────────────────────────────────────

class TestPartialTShape:
    """Verify detection degrades gracefully with incomplete T patterns."""

    def test_only_bar_does_not_fire(self, skip_if_no_nest):
        """
        A horizontal bar alone (16 T-pixels) gives ΔV ≈ 10 mV < 20 mV threshold.
        """
        from network import detect_t_shape
        from image_generator import (T_BAR_ROW_START, T_BAR_ROW_END,
                                      T_BAR_COL_START, T_BAR_COL_END)
        bar_only = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)
        bar_only[T_BAR_ROW_START:T_BAR_ROW_END, T_BAR_COL_START:T_BAR_COL_END] = 1
        result = detect_t_shape(bar_only)
        assert result is False, (
            "Bar-only image activates too few T-pixels to trigger detection")

    def test_full_t_with_extra_noise_still_fires(self, skip_if_no_nest):
        """
        Adding random noise pixels (weight=0) to a full T should not prevent detection.
        """
        from network import detect_t_shape
        t_img = create_t_shape_image().copy()
        # Add some extra active pixels outside the T pattern
        t_img[15, 1] = 1
        t_img[18, 18] = 1
        t_img[0, 0]  = 1
        result = detect_t_shape(t_img)
        assert result is True, (
            "Full T-shape with extra inactive-weight pixels should still be detected")


# ── Voltage trace sanity ──────────────────────────────────────────────────────

class TestVoltageTrace:
    """Membrane voltage should behave according to LIF dynamics."""

    def test_voltage_starts_at_rest(self, skip_if_no_nest):
        from network import run_detection, LIF_PARAMS
        t_img = create_t_shape_image()
        _, v_times, v_values = run_detection(t_img)
        if len(v_values) == 0:
            pytest.skip("No voltage data returned by simulator")
        v_rest = LIF_PARAMS["v_rest"]
        # First recorded voltage should be at or near rest
        assert abs(v_values[0] - v_rest) < 1.0, (
            f"Initial voltage {v_values[0]:.2f} mV is far from v_rest={v_rest} mV")

    def test_t_shape_voltage_exceeds_threshold_transiently(self, skip_if_no_nest):
        from network import run_detection, LIF_PARAMS
        t_img = create_t_shape_image()
        _, v_times, v_values = run_detection(t_img)
        if len(v_values) == 0:
            pytest.skip("No voltage data returned by simulator")
        # For a firing simulation, voltage must at some point reach or exceed threshold
        # (PyNN resets it immediately, so peak may equal threshold).
        v_thresh = LIF_PARAMS["v_thresh"]
        assert v_values.max() >= v_thresh - 2.0, (
            f"Peak voltage {v_values.max():.1f} mV never approached threshold {v_thresh} mV")

    def test_random_image_voltage_stays_below_threshold(self, skip_if_no_nest):
        from network import run_detection, LIF_PARAMS
        rnd_img = create_random_image(seed=42, density=0.30)
        _, v_times, v_values = run_detection(rnd_img)
        if len(v_values) == 0:
            pytest.skip("No voltage data returned by simulator")
        v_thresh = LIF_PARAMS["v_thresh"]
        assert v_values.max() < v_thresh, (
            f"Random image drove voltage to {v_values.max():.1f} mV, "
            f"which exceeded threshold {v_thresh} mV")
