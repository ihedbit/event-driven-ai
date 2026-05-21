"""
main.py — Entry point for the T-shape SNN detector.

Runs two complete simulations and saves visualisations to outputs/:

  1. T-shape input   → output neuron should FIRE (detection = True)
  2. Random noise    → output neuron should be SILENT (detection = False)

Usage
-----
  cd t_shape_detector
  python main.py

Requirements
------------
  pip install -r requirements.txt
  (NEST must be installed separately — see README.md)
"""

from __future__ import annotations
import sys
from pathlib import Path

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import pyNN.nest  # noqa: F401
except ImportError:
    print(
        "\n[ERROR] PyNN with NEST backend is not installed.\n"
        "  1. Install NEST:  https://nest-simulator.readthedocs.io/en/stable/installation/\n"
        "  2. Install PyNN:  pip install PyNN\n"
        "Then re-run: python main.py\n"
    )
    sys.exit(1)

# ── Local imports ─────────────────────────────────────────────────────────────
from image_generator import create_t_shape_image, create_random_image, get_t_shape_pixel_indices
from encoder          import encode_image_to_spikes, count_active_pixels
from network          import run_detection, detect_t_shape, LIF_PARAMS
from visualization    import plot_summary, plot_input_image, plot_spike_raster, plot_output_spikes, plot_membrane_voltage


# ── Simulation parameters ─────────────────────────────────────────────────────

SIM_TIME_MS  = 100.0   # total simulation time
SPIKE_TIME_MS = 10.0   # when active pixels fire their spike


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_and_visualise(image, label: str, sim_time: float, spike_time: float):
    """Run one simulation and save all plots."""
    print(f"\n{'='*60}")
    print(f"  Simulation: {label.replace('_', ' ').upper()}")
    print(f"{'='*60}")

    # Encode image → spike times
    spike_times_list = encode_image_to_spikes(image, spike_time=spike_time,
                                              sim_time=sim_time)
    n_active = count_active_pixels(spike_times_list)
    t_indices = set(get_t_shape_pixel_indices())
    n_t_active = sum(1 for i, st in enumerate(spike_times_list)
                     if len(st) > 0 and i in t_indices)

    print(f"  Active input neurons : {n_active} / 400")
    print(f"  T-pixels firing      : {n_t_active} / 46")
    print(f"  Running PyNN/NEST simulation ({sim_time} ms) …")

    # Run NEST simulation
    output_spikes, v_times, v_values = run_detection(
        image, sim_time=sim_time, spike_time=spike_time)

    detected = len(output_spikes) > 0
    print(f"  Output spikes        : {list(output_spikes)}")
    print(f"  Detection result     : {'T-SHAPE DETECTED ✓' if detected else 'No detection ✗'}")

    # Save individual plots
    print("\n  Saving plots …")
    plot_input_image(image,
                     title=f"Input Image — {label.replace('_', ' ').title()}",
                     filename=f"input_{label}.png")
    plot_spike_raster(spike_times_list, sim_time=sim_time,
                      title=f"Spike Raster — {label.replace('_', ' ').title()}",
                      filename=f"raster_{label}.png")
    plot_output_spikes(output_spikes, sim_time=sim_time,
                       title=f"Output Spikes — {label.replace('_', ' ').title()}",
                       filename=f"output_{label}.png")
    plot_membrane_voltage(v_times, v_values, output_spikes,
                          title=f"Membrane Voltage — {label.replace('_', ' ').title()}",
                          filename=f"voltage_{label}.png")

    # Save combined summary
    plot_summary(image, spike_times_list, output_spikes, v_times, v_values,
                 label=label)

    return detected


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  SNN T-Shape Detector  |  PyNN + NEST")
    print("="*60)
    print(f"\n  Network architecture:")
    print(f"    Input  : 400 SpikeSourceArray neurons (one per pixel)")
    print(f"    Output : 1 IF_curr_exp (LIF) neuron")
    print(f"    T-pixel weight  : 0.05 nA (excitatory)")
    print(f"    Non-T pixels    : unconnected (weight = 0)")
    print(f"    Threshold       : {LIF_PARAMS['v_thresh']} mV  "
          f"({abs(LIF_PARAMS['v_thresh'] - LIF_PARAMS['v_rest'])} mV above rest)")

    # ── Test 1: T-shape input ─────────────────────────────────────────────────
    t_image  = create_t_shape_image()
    t_result = run_and_visualise(t_image, label="t_shape",
                                  sim_time=SIM_TIME_MS, spike_time=SPIKE_TIME_MS)

    # ── Test 2: Random noise input ────────────────────────────────────────────
    rnd_image  = create_random_image(seed=42, density=0.30)
    rnd_result = run_and_visualise(rnd_image, label="random_noise",
                                    sim_time=SIM_TIME_MS, spike_time=SPIKE_TIME_MS)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    print(f"  T-shape image   → Detection: {'PASS ✓' if t_result else 'FAIL ✗'}")
    print(f"  Random noise    → Detection: {'FAIL ✗' if not rnd_result else 'FALSE POSITIVE ✗'}")

    both_correct = t_result and not rnd_result
    print(f"\n  Overall: {'ALL TESTS PASSED ✓' if both_correct else 'SOME TESTS FAILED ✗'}")
    print(f"\n  Plots saved to: {Path(__file__).parent / 'outputs'}/")
    print()

    return 0 if both_correct else 1


if __name__ == "__main__":
    sys.exit(main())
