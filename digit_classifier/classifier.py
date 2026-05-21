"""
classifier.py — Winner-Take-All classification over output neuron spike counts.

═══════════════════════════════════════════════════════════════════════════════
HOW SPIKE-BASED CLASSIFICATION WORKS
═══════════════════════════════════════════════════════════════════════════════

After the simulation ends, each of the 10 output neurons has accumulated some
number of spikes.  The classifier reads these counts and picks the winner:

    prediction = argmax(spike_counts)

This is called "Winner-Take-All" (WTA) — the neuron that fired the MOST times
is declared the winning class.

WHY spike counts and not just "did it fire once?"

  Because even with the inhibitory connections, a borderline image might cause
  one wrong neuron to fire a single spike while the correct one fires several.
  Using spike COUNT rather than threshold crossing is more robust.

HOW THIS DIFFERS FROM SOFTMAX IN DEEP LEARNING

  In an ANN:  softmax(logits) → probability vector → argmax
  In this SNN: LIF dynamics → spike counts → argmax

  Both are "winner takes all at the end," but SNNs get there through:
    * Temporal integration (voltage accumulates over time)
    * Threshold-based all-or-nothing firing
    * Spike timing rather than floating-point activations

═══════════════════════════════════════════════════════════════════════════════
HOW TO EXTEND THIS TO A LEARNED CLASSIFIER
═══════════════════════════════════════════════════════════════════════════════

  The current weights are hand-crafted (template matching).
  A next step would be STDP (Spike-Timing-Dependent Plasticity): if a
  presynaptic spike consistently arrives just before a postsynaptic spike,
  the synapse strengthens.  Over many training examples, the network learns
  the correct weights without backpropagation.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Full result of one classification run.

    Attributes
    ----------
    predicted_digit : int or None — the winning digit class (None if silent).
    confidence      : int         — spike count of the winning neuron.
    spike_counts    : list[int]   — spike count for every output neuron 0–9.
    is_confident    : bool        — True if exactly one neuron fired.
    all_silent      : bool        — True if no output neuron fired at all.
    """
    predicted_digit: int | None
    confidence:      int
    spike_counts:    list[int]
    is_confident:    bool = field(init=False)
    all_silent:      bool = field(init=False)

    def __post_init__(self):
        self.all_silent    = (max(self.spike_counts) == 0)
        n_firing           = sum(1 for c in self.spike_counts if c > 0)
        self.is_confident  = (n_firing == 1)

    def __repr__(self) -> str:
        bar = "".join(
            f"  [{k}] {'█' * c:10s} {c}\n"
            for k, c in enumerate(self.spike_counts)
        )
        pred = self.predicted_digit
        return (
            f"ClassificationResult:\n"
            f"  Predicted digit : {pred}\n"
            f"  Confidence      : {self.confidence} spike(s)\n"
            f"  All silent      : {self.all_silent}\n"
            f"  Spike counts:\n{bar}"
        )


# ── Core classification functions ─────────────────────────────────────────────

def get_prediction(spike_counts: list[int]) -> tuple[int | None, int]:
    """
    Winner-Take-All: return (predicted_digit, confidence).

    Parameters
    ----------
    spike_counts : list of 10 ints — spikes per output neuron.

    Returns
    -------
    (predicted_digit, confidence)
    - predicted_digit: argmax of spike_counts, or None if all zeros.
    - confidence     : spike count of the winning neuron.
    """
    if len(spike_counts) != 10:
        raise ValueError(
            f"Expected 10 spike counts (one per digit), got {len(spike_counts)}")

    max_count = max(spike_counts)
    if max_count == 0:
        return None, 0

    predicted = int(np.argmax(spike_counts))
    return predicted, int(max_count)


def classify_spike_counts(spike_counts: list[int]) -> ClassificationResult:
    """
    Wrap get_prediction() in a richer result object.
    """
    predicted, confidence = get_prediction(spike_counts)
    return ClassificationResult(
        predicted_digit=predicted,
        confidence=confidence,
        spike_counts=spike_counts,
    )


def classify_image(image, sim_time: float = 100.0) -> ClassificationResult:
    """
    End-to-end: run simulation and return ClassificationResult.

    Parameters
    ----------
    image    : 20×20 binary NumPy array.
    sim_time : Simulation duration in ms.
    """
    from network import run_classification
    results = run_classification(image, sim_time=sim_time)
    return classify_spike_counts(results["spike_counts"])


def classify_all_digits(sim_time: float = 100.0) -> dict[int, ClassificationResult]:
    """
    Run classification for all 10 digit images and return results.
    Used by main.py and test_prediction.py.
    """
    from digit_generator import create_digit_image
    return {
        d: classify_image(create_digit_image(d), sim_time=sim_time)
        for d in range(10)
    }


def compute_accuracy(results: dict[int, ClassificationResult]) -> float:
    """
    Return classification accuracy as a fraction (0.0 – 1.0).
    """
    correct = sum(
        1 for d, r in results.items()
        if r.predicted_digit == d
    )
    return correct / len(results)


# ── Module self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test WTA logic without simulation
    test_cases = [
        ([0, 0, 0, 5, 0, 0, 0, 0, 0, 0], 3, 5),
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], None, 0),
        ([1, 1, 1, 1, 1, 1, 1, 1, 1, 2], 9, 2),
    ]
    print("WTA logic tests:")
    for counts, expected_pred, expected_conf in test_cases:
        pred, conf = get_prediction(counts)
        status = "✓" if pred == expected_pred and conf == expected_conf else "✗"
        print(f"  {status}  counts={counts} → pred={pred}, conf={conf}")
