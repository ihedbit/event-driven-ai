"""
test_network.py — Tests for network.py

Sections:
  A. Connection-list structure tests   — pure Python, no NEST
  B. Analytical weight-sufficiency     — pure Python, no NEST
  C. Live NEST simulation tests        — skipped if NEST not installed

WHY test connection lists without running the simulation?
  Bugs in weight assignment are silent: the network builds and runs, but
  the predictions are wrong.  Testing the connection list catches these
  bugs immediately without needing NEST.
"""

import math
import pytest
import numpy as np
from digit_generator import get_all_template_indices, create_digit_image
from network import (
    build_connection_lists, LIF_PARAMS,
    W_EXC, W_INH, N_INPUT, N_OUTPUT,
    SYNAPSE_DELAY_MS, TIMESTEP_MS, MIN_DELAY_MS,
)


# ── A. Connection-list structure ──────────────────────────────────────────────

class TestConnectionListStructure:
    """Verify the Python-level connection lists before any NEST call."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.template_indices = get_all_template_indices()
        self.exc, self.inh = build_connection_lists(self.template_indices)

    def test_total_excitatory_count(self):
        # Each digit contributes exactly N_k excitatory connections
        expected = sum(len(v) for v in self.template_indices.values())
        assert len(self.exc) == expected, (
            f"Expected {expected} excitatory connections, got {len(self.exc)}")

    def test_total_inhibitory_count(self):
        # Each digit contributes (400 - N_k) inhibitory connections
        expected = sum(N_INPUT - len(v) for v in self.template_indices.values())
        assert len(self.inh) == expected, (
            f"Expected {expected} inhibitory connections, got {len(self.inh)}")

    def test_each_digit_has_400_total_connections(self):
        # Every output neuron should receive from all 400 input neurons
        from collections import defaultdict
        exc_by_post = defaultdict(int)
        inh_by_post = defaultdict(int)
        for pre, post, w, d in self.exc:
            exc_by_post[post] += 1
        for pre, post, w, d in self.inh:
            inh_by_post[post] += 1
        for k in range(N_OUTPUT):
            total = exc_by_post[k] + inh_by_post[k]
            assert total == N_INPUT, (
                f"Digit {k}: {total} connections, expected {N_INPUT}")

    def test_excitatory_weights_normalised(self):
        # w_exc_k = W_EXC / N_k for every connection to output k
        for pre, post, w, delay in self.exc:
            n_k = len(self.template_indices[post])
            expected_w = W_EXC / n_k
            assert w == pytest.approx(expected_w, rel=1e-9), (
                f"Digit {post}: exc weight {w:.6f}, expected {expected_w:.6f}")

    def test_inhibitory_weights_normalised(self):
        for pre, post, w, delay in self.inh:
            n_k = len(self.template_indices[post])
            expected_w = W_INH / (N_INPUT - n_k)
            assert w == pytest.approx(expected_w, rel=1e-9), (
                f"Digit {post}: inh weight {w:.6f}, expected {expected_w:.6f}")

    def test_all_weights_positive(self):
        for pre, post, w, delay in self.exc + self.inh:
            assert w > 0, f"Found non-positive weight {w} for (pre={pre}, post={post})"

    def test_all_delays_equal_synapse_delay(self):
        for pre, post, w, delay in self.exc + self.inh:
            assert delay == pytest.approx(SYNAPSE_DELAY_MS)

    def test_exc_connections_target_template_pixels_only(self):
        for pre, post, w, delay in self.exc:
            assert pre in self.template_indices[post], (
                f"Exc connection pre={pre} is NOT in template of digit {post}")

    def test_inh_connections_target_non_template_pixels_only(self):
        for pre, post, w, delay in self.inh:
            assert pre not in self.template_indices[post], (
                f"Inh connection pre={pre} IS in template of digit {post}")

    def test_pre_post_indices_in_range(self):
        for pre, post, w, delay in self.exc + self.inh:
            assert 0 <= pre < N_INPUT,  f"pre index {pre} out of [0,{N_INPUT})"
            assert 0 <= post < N_OUTPUT, f"post index {post} out of [0,{N_OUTPUT})"


# ── B. Analytical weight-sufficiency check ────────────────────────────────────

class TestWeightSufficiency:
    """
    Analytically verify correct firing / non-firing for every digit pair.

    Formula for peak ΔV when N_exc excitatory and N_inh inhibitory spikes
    arrive simultaneously (all with same tau_syn = 5 ms):

      ΔV = (N_exc × w_exc − N_inh × w_inh) × ΔV_factor
      ΔV_factor ≈ 12.61 mV/nA  (computed for cm=0.25, τ_m=20, τ_syn=5)
    """

    DV_FACTOR: float = 12.61   # mV/nA (see network.py docstring for derivation)

    def _compute_dv(self, n_exc: float, w_exc: float,
                    n_inh: float, w_inh: float) -> float:
        return (n_exc * w_exc - n_inh * w_inh) * self.DV_FACTOR

    def test_correct_digit_always_exceeds_threshold(self):
        """Each digit, shown to its OWN neuron, must always fire."""
        template_indices = get_all_template_indices()
        gap = LIF_PARAMS["v_thresh"] - LIF_PARAMS["v_rest"]
        for k in range(N_OUTPUT):
            n_k   = len(template_indices[k])
            w_exc = W_EXC / n_k
            dv    = self._compute_dv(n_k, w_exc, 0, 0)
            assert dv > gap, (
                f"Digit {k}: correct ΔV = {dv:.2f} mV < threshold gap {gap} mV")

    def test_incorrect_digit_stays_below_threshold_all_pairs(self):
        """
        For every (shown digit k, output neuron j) pair where k ≠ j,
        the peak ΔV must stay below threshold.

        This is the critical test: it validates that the inhibitory
        connections prevent false positives across all 90 off-diagonal pairs.
        """
        template_indices = get_all_template_indices()
        gap = LIF_PARAMS["v_thresh"] - LIF_PARAMS["v_rest"]

        failures = []
        for k in range(N_OUTPUT):          # shown digit
            t_k = set(template_indices[k]) # active pixels
            for j in range(N_OUTPUT):      # output neuron under test
                if j == k:
                    continue
                t_j = set(template_indices[j])
                n_k = len(t_k)
                n_j = len(t_j)

                # How many of digit k's pixels fall in neuron j's template?
                n_exc = len(t_k & t_j)
                w_exc = W_EXC / n_j

                # How many of digit k's pixels fall OUTSIDE neuron j's template?
                n_inh = len(t_k - t_j)
                w_inh = W_INH / (N_INPUT - n_j)

                dv = self._compute_dv(n_exc, w_exc, n_inh, w_inh)
                if dv >= gap:
                    failures.append(
                        f"  Digit {k} → neuron {j}: ΔV={dv:.2f} mV ≥ {gap} mV")

        assert not failures, (
            "The following (shown digit, wrong neuron) pairs would fire "
            "incorrectly:\n" + "\n".join(failures))


# ── C. Live NEST simulation tests ─────────────────────────────────────────────

class TestNetworkWithNEST:
    """Require a running NEST installation."""

    def test_input_population_has_400_neurons(self, skip_if_no_nest):
        import pyNN.nest as sim
        from encoder import encode_image_to_spikes
        from network import build_network

        sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)
        img = create_digit_image(0)
        spike_times = encode_image_to_spikes(img)
        template_indices = get_all_template_indices()
        input_pop, _, _, _ = build_network(spike_times, template_indices)

        assert input_pop.size == 400
        sim.end()

    def test_output_population_has_10_neurons(self, skip_if_no_nest):
        import pyNN.nest as sim
        from encoder import encode_image_to_spikes
        from network import build_network

        sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)
        img = create_digit_image(0)
        spike_times = encode_image_to_spikes(img)
        template_indices = get_all_template_indices()
        _, output_pops, _, _ = build_network(spike_times, template_indices)

        assert len(output_pops) == 10, f"Expected 10 populations, got {len(output_pops)}"
        for k, pop_k in enumerate(output_pops):
            assert pop_k.size == 1, f"Population {k} has {pop_k.size} neurons, expected 1"
            assert pop_k.label == f"detector_digit_{k}"
        sim.end()

    def test_excitatory_weights_correct_in_nest(self, skip_if_no_nest):
        import pyNN.nest as sim
        from encoder import encode_image_to_spikes
        from network import build_network

        sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)
        img = create_digit_image(3)
        spike_times = encode_image_to_spikes(img)
        template_indices = get_all_template_indices()
        _, _, exc_projs, _ = build_network(spike_times, template_indices)

        assert len(exc_projs) == 10
        for k, proj_k in enumerate(exc_projs):
            conns = proj_k.get(["weight"], format="list")
            for pre, post, w in conns:
                assert w > 0, f"Digit {k}: non-positive excitatory weight: {w}"
        sim.end()

    def test_inhibitory_weights_correct_in_nest(self, skip_if_no_nest):
        import pyNN.nest as sim
        from encoder import encode_image_to_spikes
        from network import build_network

        sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)
        img = create_digit_image(3)
        spike_times = encode_image_to_spikes(img)
        template_indices = get_all_template_indices()
        _, _, _, inh_projs = build_network(spike_times, template_indices)

        assert len(inh_projs) == 10
        for k, proj_k in enumerate(inh_projs):
            conns = proj_k.get(["weight"], format="list")
            for pre, post, w in conns:
                assert w > 0, (
                    f"Digit {k}: inhibitory weight should be positive "
                    f"(sign from receptor_type), got {w}")
        sim.end()

    def test_simulation_runs_without_error(self, skip_if_no_nest):
        import pyNN.nest as sim
        from encoder import encode_image_to_spikes
        from network import build_network

        sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)
        img = create_digit_image(7)
        spike_times = encode_image_to_spikes(img)
        template_indices = get_all_template_indices()
        build_network(spike_times, template_indices)
        sim.run(50.0)
        sim.end()
