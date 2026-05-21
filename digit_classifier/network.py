"""
network.py — PyNN/NEST network for 10-way digit classification.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

  [ 20×20 binary image ]
        ↓  encode_image_to_spikes()
  [ 400 SpikeSourceArray input neurons ]    ← one per pixel
        ↓  excitatory synapses (template pixels)   weight = W_EXC / N_k
        ↓  inhibitory synapses (non-template pixels) weight = W_INH / (400−N_k)
  [ 10 IF_curr_exp output neurons ]         ← one per digit class 0–9
        ↓
  spike counts per neuron → argmax → prediction

═══════════════════════════════════════════════════════════════════════════════
WEIGHT SCHEME: NORMALISED EXCITATORY + INHIBITORY
═══════════════════════════════════════════════════════════════════════════════

Why TWO types of synapses?

  1. Excitatory (template pixels only): pixel i → output k if pixel i is
     part of digit k's template.  Weight = W_EXC / N_k so that a PERFECT
     match always produces the same voltage change regardless of N_k.

  2. Inhibitory (non-template pixels): pixel i → output k if pixel i is
     NOT part of digit k's template.  Weight = W_INH / (400 − N_k) so that
     pixels belonging to one digit but not another SUPPRESS the wrong neurons.

Why this matters — the "subset problem":
  Digit "1" (B,C segments) is a strict subset of digit "7" (A,B,C).
  Without inhibition: showing digit "7" would FULLY activate neuron 1 as well,
  because all of 1's template pixels fire when 7 is shown.
  With inhibition: the "A" segment pixel (in 7 but not in 1's template) fires
  and suppresses neuron 1 via the inhibitory synapse → neuron 1 stays silent.

═══════════════════════════════════════════════════════════════════════════════
WEIGHT CALIBRATION (peak-voltage formula for IF_curr_exp)
═══════════════════════════════════════════════════════════════════════════════

  For cm=0.25 nF, τ_m=20 ms, τ_syn=5 ms, all inputs simultaneous:

    ΔV_peak per nA ≈ 12.61 mV/nA     (see t_shape_detector/network.py)

  Correct digit k (N_k template pixels active, 0 non-template):
    ΔV = W_EXC × 12.61 = 2.0 × 12.61 = 25.2 mV  >  threshold gap 23 mV  ✓

  Worst-case incorrect (e.g. digit 7 → neuron 1: 24 exc, 16 inh):
    ΔV = (24×W_EXC/24 − 16×W_INH/376) × 12.61
       = (2.0 − 0.426) × 12.61
       = 19.8 mV  <  23 mV  ✓

  All 10×10 cross-combinations verified in tests/test_network.py.

═══════════════════════════════════════════════════════════════════════════════
SPINNAKER MIGRATION
═══════════════════════════════════════════════════════════════════════════════

  Change ONE import line:
      import pyNN.nest as sim          # current (NEST software)
      import pyNN.spiNNaker as sim     # SpiNNaker neuromorphic chip

  IF_curr_exp, SpikeSourceArray, StaticSynapse, FromListConnector and the
  recording API are all supported on SpiNNaker through sPyNNaker.
"""

from __future__ import annotations
import numpy as np

# ── PyNN import — change to spiNNaker for hardware deployment ─────────────────
try:
    import pyNN.nest as sim          # type: ignore[import]
    PYNN_AVAILABLE = True
except ImportError:
    PYNN_AVAILABLE = False
    sim = None  # type: ignore[assignment]

# ── Neuron model ──────────────────────────────────────────────────────────────

# Leaky Integrate-and-Fire neuron with current-based exponential synapses.
# Identical model for all 10 output neurons — fair competition.
LIF_PARAMS: dict = {
    "cm":         0.25,   # nF  — membrane capacitance
    "tau_m":      20.0,   # ms  — membrane time constant
    "tau_syn_E":   5.0,   # ms  — excitatory PSC decay
    "tau_syn_I":   5.0,   # ms  — inhibitory PSC decay (same shape, opposite sign)
    "v_rest":    -65.0,   # mV  — resting potential
    "v_thresh":  -42.0,   # mV  — firing threshold (23 mV above rest)
    "v_reset":   -65.0,   # mV  — post-spike reset
    "i_offset":    0.0,   # nA  — no background current
}

# Weight calibration constants (nA)
W_EXC: float = 2.0    # base excitatory weight, normalised by N_k per pixel
W_INH: float = 10.0   # base inhibitory weight, normalised by (400−N_k) per pixel

SYNAPSE_DELAY_MS: float = 1.0   # ms — conduction delay for all synapses
TIMESTEP_MS:      float = 0.1   # ms — simulation resolution
MIN_DELAY_MS:     float = 0.1   # ms — minimum synaptic delay

N_INPUT:  int = 400  # pixels (input neurons)
N_OUTPUT: int = 10   # digit classes (output neurons)


# ── Connection-list builder ───────────────────────────────────────────────────

def build_connection_lists(
    template_indices: dict[int, list[int]],
) -> tuple[list[tuple], list[tuple]]:
    """
    Build the excitatory and inhibitory connection lists.

    Returns
    -------
    exc_list : [(pre, post, weight, delay), ...] — template pixels
    inh_list : [(pre, post, weight, delay), ...] — non-template pixels
    """
    exc_list: list[tuple] = []
    inh_list: list[tuple] = []

    for k in range(N_OUTPUT):
        t_set = set(template_indices[k])
        n_k   = len(t_set)
        non_t = [i for i in range(N_INPUT) if i not in t_set]

        w_exc = W_EXC / n_k            # normalised: same ΔV for perfect match
        w_inh = W_INH / (N_INPUT - n_k)  # normalised: same suppression

        for i in t_set:
            exc_list.append((i, k, w_exc, SYNAPSE_DELAY_MS))
        for i in non_t:
            inh_list.append((i, k, w_inh, SYNAPSE_DELAY_MS))

    return exc_list, inh_list


# ── Network builder ───────────────────────────────────────────────────────────

def build_network(
    spike_times: list[list[float]],
    template_indices: dict[int, list[int]],
) -> tuple:
    """
    Construct input and output populations plus both projections.

    Must be called after sim.setup() and before sim.run().

    Parameters
    ----------
    spike_times       : 400-element list of spike-time lists (from encoder).
    template_indices  : {digit: [flat pixel indices]} (from digit_generator).

    Returns
    -------
    (input_pop, output_pop, exc_proj, inh_proj)
    """
    if not PYNN_AVAILABLE:
        raise RuntimeError(
            "PyNN + NEST not installed.  "
            "Install NEST then: pip install PyNN")

    assert len(spike_times) == N_INPUT, (
        f"Expected {N_INPUT} spike-time lists, got {len(spike_times)}")

    # ── Input layer: 400 SpikeSourceArray neurons ─────────────────────────────
    # One neuron per pixel.  Each neuron's spike times come from the encoder.
    try:
        from pyNN.parameters import Sequence  # type: ignore[import]
        spike_param = [Sequence(st) for st in spike_times]
    except (ImportError, AttributeError):
        spike_param = spike_times  # type: ignore[assignment]

    input_pop = sim.Population(
        N_INPUT,
        sim.SpikeSourceArray(spike_times=spike_param),
        label="input_pixels",
    )

    # ── Output layer: 10 SEPARATE LIF neurons (one per digit class) ─────────────
    #
    # Each digit class gets its OWN named Population of exactly 1 LIF neuron.
    # This makes the architecture explicit in the code:
    #
    #   detector_digit_0  ← fires when a "0" is shown
    #   detector_digit_1  ← fires when a "1" is shown
    #   ...
    #   detector_digit_9  ← fires when a "9" is shown
    #
    # Why 10 separate populations instead of one Population(10)?
    #   - Each neuron is a distinct conceptual entity with its own label.
    #   - Data (spikes, voltage) is extracted per-population without index arithmetic.
    #   - Matches the neuromorphic design intention: one detector per class.
    #   - On SpiNNaker, each population maps to a dedicated hardware core.
    output_pops: list = []
    exc_projs:   list = []
    inh_projs:   list = []

    for k in range(N_OUTPUT):
        # One LIF neuron — the detector for digit class k
        pop_k = sim.Population(
            1,
            sim.IF_curr_exp(**LIF_PARAMS),
            label=f"detector_digit_{k}",
        )
        pop_k.record(["spikes", "v"])
        output_pops.append(pop_k)

        # Build per-digit connection lists.
        # post index is always 0 (the single neuron in pop_k).
        t_set = set(template_indices[k])
        n_k   = len(t_set)
        non_t = [i for i in range(N_INPUT) if i not in t_set]

        exc_w    = W_EXC / n_k
        exc_conn = [(i, 0, exc_w, SYNAPSE_DELAY_MS) for i in t_set]

        inh_w    = W_INH / (N_INPUT - n_k)
        inh_conn = [(i, 0, inh_w, SYNAPSE_DELAY_MS) for i in non_t]

        # Excitatory projection: template pixels → detector k
        exc_projs.append(sim.Projection(
            input_pop, pop_k,
            sim.FromListConnector(exc_conn),
            synapse_type=sim.StaticSynapse(),
            receptor_type="excitatory",
        ))

        # Inhibitory projection: non-template pixels → detector k
        # Suppresses false activations when a "larger" digit is shown
        # (e.g. showing "7" must not fire the "1" detector).
        inh_projs.append(sim.Projection(
            input_pop, pop_k,
            sim.FromListConnector(inh_conn),
            synapse_type=sim.StaticSynapse(),
            receptor_type="inhibitory",
        ))

    return input_pop, output_pops, exc_projs, inh_projs


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_classification(
    image: np.ndarray,
    sim_time: float = 100.0,
    spike_time: float = 10.0,
) -> dict:
    """
    Full pipeline: encode → setup → build → run → extract → cleanup.

    Parameters
    ----------
    image      : 20×20 binary NumPy array (one digit or random noise).
    sim_time   : Duration of simulation in ms.
    spike_time : When active pixels fire (ms).

    Returns
    -------
    A results dict with keys:
      - "spike_counts"        : list[int]         spike count per output neuron
      - "output_spike_trains" : list[np.ndarray]  spike times per output neuron
      - "voltage_times"       : np.ndarray         shared time axis (ms)
      - "voltage_values"      : list[np.ndarray]   voltage trace per output neuron
      - "input_spike_times"   : list[list[float]]  spike times per input neuron
    """
    from encoder import encode_image_to_spikes
    from digit_generator import get_all_template_indices

    # 1. Encode image → spike times
    spike_times_list = encode_image_to_spikes(
        image, spike_time=spike_time, sim_time=sim_time)

    # 2. Template indices (needed for synaptic weight computation)
    template_indices = get_all_template_indices()

    # 3. Initialise NEST kernel (fresh state for each simulation)
    sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)

    # 4. Build the PyNN network (returns 10 separate output populations)
    _input_pop, output_pops, _exc_projs, _inh_projs = build_network(
        spike_times_list, template_indices)

    # 5. Run the simulation
    sim.run(sim_time)

    # 6. Extract results — one get_data() call per output population
    #    Each population has exactly 1 neuron, so indices are always [0].
    spike_counts       : list[int]         = []
    output_spike_trains: list[np.ndarray]  = []
    voltage_values     : list[np.ndarray]  = []
    voltage_times      : np.ndarray        = np.array([])

    for k, pop_k in enumerate(output_pops):
        data_k  = pop_k.get_data()
        seg_k   = data_k.segments[0]

        # Spike train of the single neuron in this population
        train = seg_k.spiketrains[0].magnitude
        spike_counts.append(len(train))
        output_spike_trains.append(train)

        # Membrane voltage trace
        if seg_k.analogsignals:
            v_sig = seg_k.analogsignals[0]
            if k == 0:
                voltage_times = v_sig.times.rescale("ms").magnitude
            # Shape is (n_timepoints, 1) — the single neuron in this population
            voltage_values.append(v_sig.magnitude[:, 0])
        else:
            voltage_values.append(np.array([]))

    # 7. Clean up NEST state
    sim.end()

    return {
        "spike_counts":         spike_counts,
        "output_spike_trains":  output_spike_trains,
        "voltage_times":        voltage_times,
        "voltage_values":       voltage_values,
        "input_spike_times":    spike_times_list,
    }
