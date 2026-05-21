"""
main.py — Entry point for the SNN digit classifier.

Runs a complete classification of all 10 digits (0–9) plus a random-noise
control, then saves visualisations to outputs/.

Usage
-----
  cd digit_classifier
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
        "  1. Install NEST:  https://nest-simulator.readthedocs.io\n"
        "  2. Install PyNN:  pip install PyNN\n"
    )
    sys.exit(1)

# ── Local imports ─────────────────────────────────────────────────────────────
from digit_generator  import create_digit_image, create_random_image
from network          import run_classification, LIF_PARAMS, W_EXC, W_INH
from classifier       import classify_spike_counts, compute_accuracy
from visualization    import (plot_digit_image, plot_input_raster,
                               plot_output_activity, plot_membrane_voltages,
                               plot_prediction, plot_summary,
                               plot_all_digits_gallery, plot_accuracy_matrix)

SIM_TIME_MS  = 100.0
SPIKE_TIME_MS = 10.0


# ── Helper ────────────────────────────────────────────────────────────────────

def classify_and_visualise(image, true_digit, label: str) -> "ClassificationResult":
    """Run one simulation and save all plots."""
    from classifier import ClassificationResult

    print(f"\n{'─'*60}")
    print(f"  Classifying: {label.upper()}")
    print(f"{'─'*60}")

    results = run_classification(image, sim_time=SIM_TIME_MS,
                                 spike_time=SPIKE_TIME_MS)

    cr = classify_spike_counts(results["spike_counts"])

    print(f"  Spike counts : {results['spike_counts']}")
    print(f"  Prediction   : {cr.predicted_digit}   "
          f"({'✓ CORRECT' if cr.predicted_digit == true_digit else '✗ WRONG'})")
    print(f"  Confidence   : {cr.confidence} spike(s)")

    # Individual plots
    plot_digit_image(image, digit=true_digit,
                     filename=f"input_{label}.png")
    plot_input_raster(results["input_spike_times"], digit=true_digit,
                      sim_time=SIM_TIME_MS, filename=f"raster_{label}.png")
    plot_output_activity(results["spike_counts"], true_digit=true_digit,
                         filename=f"activity_{label}.png")
    plot_membrane_voltages(results["voltage_times"], results["voltage_values"],
                           results["spike_counts"], true_digit=true_digit,
                           filename=f"voltages_{label}.png")
    plot_prediction(results["spike_counts"], true_digit=true_digit,
                    filename=f"prediction_{label}.png")

    # Summary dashboard
    plot_summary(image, results["input_spike_times"], results["spike_counts"],
                 results["voltage_times"], results["voltage_values"],
                 true_digit=true_digit, label=label)

    return cr


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("\n" + "="*60)
    print("  SNN Digit Classifier  |  PyNN + NEST")
    print("="*60)
    print(f"\n  Architecture:")
    print(f"    Input   : 400 SpikeSourceArray neurons (one per pixel)")
    print(f"    Output  : 10 IF_curr_exp (LIF) neurons (one per digit)")
    print(f"    Weights : W_exc = {W_EXC} nA (normalised)  |  "
          f"W_inh = {W_INH} nA (normalised)")
    print(f"    Threshold gap : "
          f"{LIF_PARAMS['v_thresh'] - LIF_PARAMS['v_rest']:.0f} mV")

    # Save gallery of all digit templates
    print("\n  Generating digit gallery …")
    plot_all_digits_gallery()

    # Classify all 10 digits and store results
    all_results = {}
    for d in range(10):
        img = create_digit_image(d)
        cr  = classify_and_visualise(img, true_digit=d, label=f"digit_{d}")
        all_results[d] = cr

    # Random noise control
    rnd_img = create_random_image(seed=42, density=0.25)
    rnd_cr  = classify_and_visualise(rnd_img, true_digit=None, label="random")

    # Accuracy summary
    accuracy = compute_accuracy(all_results)
    plot_accuracy_matrix(all_results)

    print("\n" + "="*60)
    print("  FINAL RESULTS")
    print("="*60)
    for d in range(10):
        cr = all_results[d]
        mark = "✓" if cr.predicted_digit == d else "✗"
        wrong = f" (predicted {cr.predicted_digit})" if cr.predicted_digit != d else ""
        print(f"  Digit {d}: {mark}{wrong}  [{cr.confidence} spike(s)]")

    print(f"\n  Accuracy : {sum(cr.predicted_digit == d for d, cr in all_results.items())}"
          f"/10  ({accuracy * 100:.0f}%)")
    print(f"  Random   : predicted={rnd_cr.predicted_digit}  "
          f"(expect low confidence)\n")
    print(f"  Plots saved to: {Path(__file__).parent / 'outputs'}/")
    print()

    return 0 if accuracy == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
