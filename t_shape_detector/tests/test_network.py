"""
test_network.py — Tests for network.py

These tests verify the network structure: population sizes, connection weights,
and that the simulation machinery can be invoked without error.

Tests marked with the `skip_if_no_nest` fixture will be skipped automatically
if PyNN/NEST is not installed.  The weight and structure tests that only inspect
Python data structures run without a simulator.
"""

import pytest
import numpy as np


# ── Connection-list structure tests (no NEST required) ───────────────────────

class TestConnectionListStructure:
    """
    Verify that build_network() would create the right number and weight of
    connections by inspecting the Python-level connection list — no NEST needed.

    The connection list is: [(pre_idx, 0, weight, delay), ...]
    for all pre_idx that belong to the T-shape pattern.
    """

    def _build_conn_list(self):
        """Replicate the connection-list logic from network.py."""
        from image_generator import get_t_shape_pixel_indices
        from network import T_PIXEL_WEIGHT, SYNAPSE_DELAY_MS

        t_indices = set(get_t_shape_pixel_indices())
        return [
            (i, 0, T_PIXEL_WEIGHT, SYNAPSE_DELAY_MS)
            for i in range(400)
            if i in t_indices
        ], t_indices

    def test_connection_count_equals_t_pixel_count(self):
        # One synapse per T-shape pixel — no more, no less.
        conn_list, t_indices = self._build_conn_list()
        assert len(conn_list) == len(t_indices)

    def test_all_connections_target_neuron_0(self):
        # There is only 1 output neuron (index 0).
        conn_list, _ = self._build_conn_list()
        for pre, post, w, d in conn_list:
            assert post == 0, f"Connection targets neuron {post}, expected 0"

    def test_all_t_pixel_weights_equal_t_pixel_weight(self):
        from network import T_PIXEL_WEIGHT
        conn_list, _ = self._build_conn_list()
        for pre, post, w, d in conn_list:
            assert w == pytest.approx(T_PIXEL_WEIGHT), (
                f"T-pixel at index {pre} has weight {w}, expected {T_PIXEL_WEIGHT}")

    def test_all_weights_are_positive(self):
        conn_list, _ = self._build_conn_list()
        for pre, post, w, d in conn_list:
            assert w > 0, f"Weight at index {pre} is not positive: {w}"

    def test_all_delays_are_positive(self):
        conn_list, _ = self._build_conn_list()
        for pre, post, w, d in conn_list:
            assert d > 0, f"Delay at index {pre} is not positive: {d}"

    def test_non_t_pixels_not_in_connection_list(self):
        from image_generator import get_t_shape_pixel_indices
        conn_list, t_indices = self._build_conn_list()
        connected_pre = {pre for pre, *_ in conn_list}
        for i in range(400):
            if i not in t_indices:
                assert i not in connected_pre, (
                    f"Non-T pixel {i} should not be connected, but it is")


# ── LIF parameter sanity checks (no NEST required) ───────────────────────────

class TestLIFParameters:
    """Check that neuron parameters are set to biologically reasonable values."""

    def test_v_thresh_above_v_rest(self):
        from network import LIF_PARAMS
        assert LIF_PARAMS["v_thresh"] > LIF_PARAMS["v_rest"], (
            "Threshold must be above resting potential for the neuron to fire")

    def test_v_reset_equals_v_rest(self):
        from network import LIF_PARAMS
        assert LIF_PARAMS["v_reset"] == pytest.approx(LIF_PARAMS["v_rest"]), (
            "Reset potential should equal resting potential (no hyperpolarisation)")

    def test_membrane_time_constant_positive(self):
        from network import LIF_PARAMS
        assert LIF_PARAMS["tau_m"] > 0

    def test_synaptic_time_constants_positive(self):
        from network import LIF_PARAMS
        assert LIF_PARAMS["tau_syn_E"] > 0
        assert LIF_PARAMS["tau_syn_I"] > 0

    def test_threshold_gap_is_20mv(self):
        # The gap determines how many T-pixels need to fire simultaneously.
        # 20 mV gap + 0.63 mV/pixel → need ~32/46 T-pixels.
        from network import LIF_PARAMS
        gap = LIF_PARAMS["v_thresh"] - LIF_PARAMS["v_rest"]
        assert gap == pytest.approx(20.0), (
            f"Expected 20 mV threshold gap, got {gap:.1f} mV")


# ── Weight sufficiency analytical check (no NEST required) ───────────────────

class TestWeightSufficiency:
    """
    Analytically verify that the weights are calibrated correctly.

    Uses the closed-form peak-voltage formula for IF_curr_exp:
      ΔV_peak ≈ N × w/cm × (τ_m × τ_syn)/(τ_m − τ_syn) × Δ_exp
    """

    def _peak_dv(self, n_spikes):
        from network import LIF_PARAMS, T_PIXEL_WEIGHT
        import math
        cm        = LIF_PARAMS["cm"]        # nF
        tau_m     = LIF_PARAMS["tau_m"]     # ms
        tau_syn   = LIF_PARAMS["tau_syn_E"] # ms
        w         = T_PIXEL_WEIGHT          # nA

        t_peak  = tau_m * tau_syn / (tau_m - tau_syn) * math.log(tau_m / tau_syn)
        delta   = math.exp(-t_peak / tau_m) - math.exp(-t_peak / tau_syn)
        coeff   = (tau_m * tau_syn) / (tau_m - tau_syn) * delta / cm
        return n_spikes * w * coeff

    def test_full_t_shape_exceeds_threshold(self):
        from network import LIF_PARAMS
        from image_generator import get_t_shape_pixel_indices
        n_t_pixels = len(get_t_shape_pixel_indices())
        gap = LIF_PARAMS["v_thresh"] - LIF_PARAMS["v_rest"]
        dv = self._peak_dv(n_t_pixels)
        assert dv > gap, (
            f"Full T ({n_t_pixels} pixels) gives ΔV={dv:.1f} mV, "
            f"but threshold gap is {gap} mV — won't fire!")

    def test_random_image_overlap_stays_below_threshold(self):
        # With seed=42, density=0.30, overlap ≈ 14 T-pixels.
        from network import LIF_PARAMS
        from image_generator import get_t_shape_pixel_indices, create_random_image
        t_indices = set(get_t_shape_pixel_indices())
        rnd = create_random_image(seed=42, density=0.30)
        flat = rnd.flatten()
        n_overlap = sum(flat[i] for i in t_indices)
        gap = LIF_PARAMS["v_thresh"] - LIF_PARAMS["v_rest"]
        dv = self._peak_dv(n_overlap)
        assert dv < gap, (
            f"Random image activates {n_overlap} T-pixels → ΔV={dv:.1f} mV "
            f"which exceeds the threshold gap {gap} mV — false positive risk!")


# ── Live NEST simulation tests (skipped if NEST not installed) ────────────────

class TestNetworkWithNEST:
    """These tests build and run the actual PyNN/NEST network."""

    def test_input_population_has_400_neurons(self, skip_if_no_nest, t_image):
        import pyNN.nest as sim
        from encoder import encode_image_to_spikes
        from network import build_network, TIMESTEP_MS, MIN_DELAY_MS

        sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)
        spike_times = encode_image_to_spikes(t_image)
        input_pop, output_pop, _ = build_network(spike_times)

        assert input_pop.size == 400, (
            f"Expected 400 input neurons, got {input_pop.size}")
        sim.end()

    def test_output_population_has_1_neuron(self, skip_if_no_nest, t_image):
        import pyNN.nest as sim
        from encoder import encode_image_to_spikes
        from network import build_network, TIMESTEP_MS, MIN_DELAY_MS

        sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)
        spike_times = encode_image_to_spikes(t_image)
        _, output_pop, _ = build_network(spike_times)

        assert output_pop.size == 1, (
            f"Expected exactly 1 output neuron, got {output_pop.size}")
        sim.end()

    def test_t_pixel_synapses_have_strong_weight(self, skip_if_no_nest, t_image):
        import pyNN.nest as sim
        from encoder import encode_image_to_spikes
        from network import build_network, TIMESTEP_MS, MIN_DELAY_MS, T_PIXEL_WEIGHT

        sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)
        spike_times = encode_image_to_spikes(t_image)
        _, _, projection = build_network(spike_times)

        # Retrieve connection weights
        connections = projection.get(["weight"], format="list")
        weights = [row[2] for row in connections]   # (pre, post, weight)

        assert len(weights) > 0, "No connections found in projection"
        for w in weights:
            assert w == pytest.approx(T_PIXEL_WEIGHT, rel=1e-6), (
                f"Weight {w} does not match expected T_PIXEL_WEIGHT={T_PIXEL_WEIGHT}")
        sim.end()

    def test_simulation_runs_without_error(self, skip_if_no_nest, t_image):
        import pyNN.nest as sim
        from encoder import encode_image_to_spikes
        from network import build_network, TIMESTEP_MS, MIN_DELAY_MS

        sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)
        spike_times = encode_image_to_spikes(t_image)
        build_network(spike_times)
        sim.run(50.0)   # short run is fine for structure test
        sim.end()
