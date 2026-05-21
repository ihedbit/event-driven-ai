"""
network.py — PyNN/NEST network for T-shape spike detection.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

   [ 20×20 binary image ]
          ↓  encode_image_to_spikes()
   [ 400 SpikeSourceArray neurons ]   ← one neuron per pixel
          ↓  synapses (weight depends on T-shape membership)
   [ 1 IF_curr_exp (LIF) neuron ]     ← the "T-shape detector"
          ↓
   fires?  →  True  (T-shape detected)
   silent? →  False (no T-shape)

═══════════════════════════════════════════════════════════════════════════════
HOW THE DETECTOR WORKS (weighted coincidence detection)
═══════════════════════════════════════════════════════════════════════════════

  1. Only pixels that BELONG to the T shape are wired to the output neuron.
     All other pixels have no connection (weight = 0).

  2. Each T-pixel synapse has weight T_PIXEL_WEIGHT = 0.05 nA.

  3. When a full T image arrives, all 46 T-shape pixels fire simultaneously
     at t = 10 ms.  The 46 coincident post-synaptic currents sum to ~29 mV
     of depolarisation — enough to cross the 20 mV gap to threshold.

  4. A random image activates only ~14 of the 46 T-pixels on average
     (30 % density × 46 pixels).  The depolarisation is ~8.8 mV, which stays
     below threshold → no spike → "not a T".

═══════════════════════════════════════════════════════════════════════════════
WEIGHT CALCULATION (for transparency)
═══════════════════════════════════════════════════════════════════════════════

  For IF_curr_exp with cm=0.25 nF, tau_m=20 ms, tau_syn_E=5 ms:

    ΔV_peak per synapse ≈ (w / cm) × (τ_m × τ_syn) / (τ_m − τ_syn) × Δ_exp
    where Δ_exp = exp(−t_peak/τ_m) − exp(−t_peak/τ_syn)
          t_peak = τ_m × τ_syn / (τ_m − τ_syn) × ln(τ_m / τ_syn) ≈ 9.24 ms

    ΔV_peak per synapse ≈ 0.05 / 0.25 × 6.667 × 0.473 ≈ 0.63 mV

    46 simultaneous spikes → ΔV ≈ 46 × 0.63 ≈ 29 mV  (fires, threshold = 20 mV)
    14 simultaneous spikes → ΔV ≈ 14 × 0.63 ≈ 8.8 mV  (silent)

═══════════════════════════════════════════════════════════════════════════════
SPINNAKER MIGRATION NOTE
═══════════════════════════════════════════════════════════════════════════════

  To run this network on SpiNNaker hardware, change only ONE import line:

      # NEST (local simulation)
      import pyNN.nest as sim

      # SpiNNaker hardware
      import pyNN.spiNNaker as sim

  Everything else — neuron parameters, projections, recording — is identical.
  This is the key promise of the PyNN abstraction layer.
"""

from __future__ import annotations
import numpy as np

# ── PyNN import (NEST backend) ────────────────────────────────────────────────
# Change this single line to migrate to SpiNNaker:
#   import pyNN.spiNNaker as sim
try:
    import pyNN.nest as sim          # type: ignore[import]
    PYNN_AVAILABLE = True
except ImportError:
    PYNN_AVAILABLE = False
    sim = None  # type: ignore[assignment]

# ── Neuron model parameters ───────────────────────────────────────────────────

# Leaky Integrate-and-Fire neuron (current-based exponential synapses).
# All voltages in mV, times in ms, currents in nA, capacitance in nF.
LIF_PARAMS: dict = {
    "cm":         0.25,   # membrane capacitance — controls how fast V changes
    "tau_m":      20.0,   # membrane time constant — controls voltage decay
    "tau_syn_E":   5.0,   # decay of excitatory post-synaptic current
    "tau_syn_I":   5.0,   # decay of inhibitory post-synaptic current
    "v_rest":    -65.0,   # resting potential (neuron starts here)
    "v_thresh":  -45.0,   # firing threshold — 20 mV above rest
    "v_reset":   -65.0,   # reset after each spike (back to rest)
    "i_offset":    0.0,   # no constant background injection
}

# Synapse weights
# T-shape pixels excite the output neuron; all others are unconnected.
T_PIXEL_WEIGHT:   float = 0.05   # nA — excitatory; see calculation above
SYNAPSE_DELAY_MS: float = 1.0    # ms — conduction delay (must be ≥ min_delay)

# ── Simulation settings ───────────────────────────────────────────────────────

TIMESTEP_MS: float = 0.1   # simulation resolution (ms)
MIN_DELAY_MS: float = 0.1  # minimum synaptic delay (must be ≥ timestep)


# ── Network builder ───────────────────────────────────────────────────────────

def build_network(
    spike_times: list[list[float]],
) -> tuple:
    """
    Create the input and output populations plus the weighted projection.

    Must be called AFTER sim.setup() and BEFORE sim.run().

    Parameters
    ----------
    spike_times : list of 400 sublists — output of encoder.encode_image_to_spikes().

    Returns
    -------
    (input_pop, output_pop, projection)
    """
    if not PYNN_AVAILABLE:
        raise RuntimeError(
            "PyNN with NEST backend is not installed.  "
            "Run: pip install pyNN  and install NEST separately."
        )

    assert len(spike_times) == 400, (
        f"Expected 400 spike-time lists (one per pixel), got {len(spike_times)}")

    # ── Input layer ───────────────────────────────────────────────────────────
    # 400 SpikeSourceArray neurons — one for every pixel in the 20×20 image.
    # Each neuron fires at the times in its spike_times sublist.
    # Active pixels fire once at t=10 ms; inactive pixels never fire.
    #
    # WHY SpikeSourceArray?
    # In a real neuromorphic pipeline, spikes would arrive from a sensor
    # (e.g. Dynamic Vision Sensor camera).  Here we inject them from a
    # pre-computed list — same abstraction, simpler setup.
    try:
        # PyNN ≥ 0.10: Sequence wrapper for per-neuron spike times
        from pyNN.parameters import Sequence  # type: ignore[import]
        spike_param = [Sequence(st) for st in spike_times]
    except (ImportError, AttributeError):
        # PyNN 0.9.x: plain Python lists also accepted
        spike_param = spike_times  # type: ignore[assignment]

    input_pop = sim.Population(
        400,
        sim.SpikeSourceArray(spike_times=spike_param),
        label="input_pixels",
    )

    # ── Output layer ──────────────────────────────────────────────────────────
    # A single Leaky Integrate-and-Fire (LIF) neuron.
    #
    # The LIF model is the workhorse of computational neuroscience and
    # neuromorphic hardware.  It integrates incoming currents and fires a
    # spike when the membrane voltage reaches v_thresh, then resets to v_reset.
    #
    # IF_curr_exp = Integrate-and-Fire with Current-based Exponential synapses:
    #   - "Current-based" means synaptic input is a current (nA), not a
    #     conductance.  Simpler and hardware-friendly (used on SpiNNaker).
    #   - "Exponential" means the synaptic current decays as exp(−t/τ_syn).
    output_pop = sim.Population(
        1,
        sim.IF_curr_exp(**LIF_PARAMS),
        label="t_shape_detector",
    )

    # Record spikes and membrane voltage for the output neuron
    output_pop.record(["spikes", "v"])

    # ── Synaptic projection ───────────────────────────────────────────────────
    # For each T-shape pixel index, connect input neuron i → output neuron 0
    # with a strong excitatory weight.  Non-T pixels are simply not connected.
    #
    # This implements a "template matching" detector:
    #   if the input pattern overlaps heavily with the stored T template,
    #   the output neuron fires.
    from image_generator import get_t_shape_pixel_indices
    t_indices = set(get_t_shape_pixel_indices())

    conn_list: list[tuple] = [
        (i, 0, T_PIXEL_WEIGHT, SYNAPSE_DELAY_MS)
        for i in range(400)
        if i in t_indices
    ]

    projection = sim.Projection(
        input_pop,
        output_pop,
        sim.FromListConnector(conn_list),   # explicit (pre, post, w, d) tuples
        synapse_type=sim.StaticSynapse(),   # fixed weights (no plasticity)
        receptor_type="excitatory",
    )

    return input_pop, output_pop, projection


# ── High-level simulation runner ─────────────────────────────────────────────

def run_detection(
    image: np.ndarray,
    sim_time: float = 100.0,
    spike_time: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline: encode image → build network → simulate → extract results.

    Parameters
    ----------
    image     : 20×20 binary NumPy array.
    sim_time  : How long to simulate (ms).  Default 100 ms.
    spike_time: When active pixels fire (ms).  Default 10 ms.

    Returns
    -------
    (output_spike_times, voltage_times, voltage_values)
    - output_spike_times : 1-D array of spike times in ms (empty if silent).
    - voltage_times      : 1-D array of time steps in ms.
    - voltage_values     : 1-D array of membrane voltages in mV.
    """
    from encoder import encode_image_to_spikes

    # 1. Encode image pixels to spike-time lists
    spike_times_list = encode_image_to_spikes(image, spike_time=spike_time,
                                              sim_time=sim_time)

    # 2. Initialise the NEST kernel (resets any previous simulation state)
    sim.setup(timestep=TIMESTEP_MS, min_delay=MIN_DELAY_MS)

    # 3. Build network (creates populations and synapses)
    _input_pop, output_pop, _projection = build_network(spike_times_list)

    # 4. Run the simulation
    sim.run(sim_time)

    # 5. Extract recorded data (Neo format)
    data = output_pop.get_data()
    segment = data.segments[0]

    # Spike times for the single output neuron
    output_spike_times: np.ndarray = segment.spiketrains[0].magnitude  # ms

    # Membrane voltage trace
    if segment.analogsignals:
        v_signal = segment.analogsignals[0]
        voltage_times  = v_signal.times.rescale("ms").magnitude          # ms
        voltage_values = v_signal.magnitude[:, 0]                        # mV
    else:
        voltage_times  = np.array([])
        voltage_values = np.array([])

    # 6. Clean up NEST state (required before calling sim.setup() again)
    sim.end()

    return output_spike_times, voltage_times, voltage_values


def detect_t_shape(image: np.ndarray, sim_time: float = 100.0) -> bool:
    """
    Return True if the image triggers at least one spike in the output neuron.

    This is the top-level detection function used by main.py and tests.
    """
    output_spikes, _, _ = run_detection(image, sim_time=sim_time)
    return len(output_spikes) > 0
