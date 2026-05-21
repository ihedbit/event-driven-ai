"""
test_prediction.py — End-to-end classification tests (requires PyNN + NEST).

These integration tests run full simulations and verify:
  1. All 10 digit images classify correctly
  2. Random noise does not fire all neurons simultaneously
  3. Each digit fires the correct output neuron with most spikes
  4. Incorrect neurons stay silent (or fire less than the correct one)
  5. Voltage dynamics are physically sensible

All tests use the `skip_if_no_nest` fixture and are skipped automatically
when NEST is not installed.
"""

import pytest
import numpy as np
from digit_generator import create_digit_image, create_random_image
from classifier import classify_image, classify_all_digits, compute_accuracy
from network import LIF_PARAMS


ALL_DIGITS = list(range(10))


# ── Correct digit classification ──────────────────────────────────────────────

class TestCorrectClassification:
    """Core requirement: every clean digit image predicts itself."""

    @pytest.mark.parametrize("digit", ALL_DIGITS)
    def test_digit_classifies_correctly(self, skip_if_no_nest, digit):
        img = create_digit_image(digit)
        result = classify_image(img)
        assert result.predicted_digit == digit, (
            f"Digit {digit}: predicted {result.predicted_digit}, "
            f"confidence {result.confidence}, "
            f"spike counts {result.spike_counts}")

    def test_all_10_digits_100_percent_accuracy(self, skip_if_no_nest):
        results = classify_all_digits()
        accuracy = compute_accuracy(results)
        assert accuracy == 1.0, (
            f"Expected 100% accuracy, got {accuracy*100:.0f}%\n" +
            "\n".join(
                f"  digit {d}: predicted {r.predicted_digit}, "
                f"counts {r.spike_counts}"
                for d, r in results.items()
                if r.predicted_digit != d
            )
        )

    @pytest.mark.parametrize("digit", ALL_DIGITS)
    def test_correct_neuron_fires(self, skip_if_no_nest, digit):
        from network import run_classification
        img = create_digit_image(digit)
        results = run_classification(img)
        correct_count = results["spike_counts"][digit]
        assert correct_count > 0, (
            f"Digit {digit}: correct neuron {digit} never fired "
            f"(spike counts: {results['spike_counts']})")

    @pytest.mark.parametrize("digit", ALL_DIGITS)
    def test_correct_neuron_fires_most(self, skip_if_no_nest, digit):
        from network import run_classification
        img = create_digit_image(digit)
        results = run_classification(img)
        counts = results["spike_counts"]
        assert counts[digit] == max(counts), (
            f"Digit {digit}: correct neuron fires {counts[digit]} times, "
            f"but neuron {np.argmax(counts)} fires {max(counts)} times")


# ── Output spike timing ───────────────────────────────────────────────────────

class TestSpikeTiming:
    def test_output_spike_after_input_spike(self, skip_if_no_nest):
        from network import run_classification
        spike_time = 10.0
        img = create_digit_image(8)   # digit 8 has most pixels → fires strongly
        results = run_classification(img, spike_time=spike_time)
        train = results["output_spike_trains"][8]
        assert len(train) > 0, "Digit 8 should cause output neuron 8 to fire"
        assert train[0] > spike_time, (
            f"Output spike at {train[0]} ms is before input at {spike_time} ms")

    @pytest.mark.parametrize("digit", ALL_DIGITS)
    def test_output_spikes_within_sim_window(self, skip_if_no_nest, digit):
        from network import run_classification
        sim_time = 100.0
        img = create_digit_image(digit)
        results = run_classification(img, sim_time=sim_time)
        for k, train in enumerate(results["output_spike_trains"]):
            for t in train:
                assert 0 < t < sim_time, (
                    f"Digit {digit}, neuron {k}: spike at {t} ms outside "
                    f"simulation window (0, {sim_time})")


# ── Random noise behaviour ────────────────────────────────────────────────────

class TestNoiseBehaviour:
    """Random noise must not fire all neurons simultaneously."""

    def test_noise_does_not_fire_all_neurons(self, skip_if_no_nest):
        from network import run_classification
        rnd = create_random_image(seed=42, density=0.25)
        results = run_classification(rnd)
        n_firing = sum(1 for c in results["spike_counts"] if c > 0)
        # At most a few neurons should fire (not all 10)
        assert n_firing < 10, (
            f"Random noise fired all 10 output neurons: {results['spike_counts']}")

    def test_noise_does_not_produce_clear_winner_every_time(self, skip_if_no_nest):
        """
        With correct weight calibration, random 25%-density noise should
        produce low confidence (few spikes) or stay mostly silent.
        """
        rnd = create_random_image(seed=42, density=0.25)
        result = classify_image(rnd)
        # Confidence should be low for random noise (not a clear trained class)
        # We don't assert silence (random could accidentally match a digit partially)
        # but we do assert the max spike count is bounded
        assert result.confidence <= 3, (
            f"Random noise had unexpectedly high confidence: "
            f"{result.confidence} spikes for digit {result.predicted_digit}. "
            f"Spike counts: {result.spike_counts}")

    def test_noise_simulation_completes_stably(self, skip_if_no_nest):
        """Noisy images must not crash the simulation."""
        from network import run_classification
        rnd = create_random_image(seed=99, density=0.50)
        results = run_classification(rnd)
        assert isinstance(results["spike_counts"], list)
        assert len(results["spike_counts"]) == 10


# ── Voltage trace sanity ──────────────────────────────────────────────────────

class TestVoltageTraces:
    def test_voltage_starts_near_rest(self, skip_if_no_nest):
        from network import run_classification
        img = create_digit_image(0)
        results = run_classification(img)
        v_rest = LIF_PARAMS["v_rest"]
        for k in range(10):
            v = results["voltage_values"][k]
            if len(v) == 0:
                continue
            assert abs(v[0] - v_rest) < 1.5, (
                f"Neuron {k}: initial voltage {v[0]:.2f} mV far from "
                f"v_rest {v_rest} mV")

    def test_correct_neuron_voltage_exceeds_threshold(self, skip_if_no_nest):
        from network import run_classification
        img = create_digit_image(3)
        results = run_classification(img)
        v_thresh = LIF_PARAMS["v_thresh"]
        v = results["voltage_values"][3]
        if len(v) == 0:
            pytest.skip("No voltage data")
        # Must reach or approach threshold (PyNN resets immediately on crossing)
        assert v.max() >= v_thresh - 2.0, (
            f"Neuron 3 peak voltage {v.max():.1f} mV never approached "
            f"threshold {v_thresh} mV")

    @pytest.mark.parametrize("digit", ALL_DIGITS)
    def test_incorrect_neurons_stay_below_threshold(self, skip_if_no_nest, digit):
        from network import run_classification
        img = create_digit_image(digit)
        results = run_classification(img)
        v_thresh = LIF_PARAMS["v_thresh"]
        for k in range(10):
            if k == digit:
                continue   # skip correct neuron
            v = results["voltage_values"][k]
            if len(v) == 0:
                continue
            assert v.max() < v_thresh, (
                f"Digit {digit}: incorrect neuron {k} reached "
                f"{v.max():.1f} mV ≥ threshold {v_thresh} mV")


# ── Simulation reproducibility ────────────────────────────────────────────────

class TestReproducibility:
    def test_same_digit_same_result_twice(self, skip_if_no_nest):
        from network import run_classification
        img = create_digit_image(5)
        r1 = run_classification(img)
        r2 = run_classification(img)
        assert r1["spike_counts"] == r2["spike_counts"], (
            f"Same image gave different spike counts: {r1['spike_counts']} vs "
            f"{r2['spike_counts']}")
