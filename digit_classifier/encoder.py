"""
encoder.py — Pixel-to-spike encoding for the digit classifier.

Identical in structure to the T-shape detector's encoder.  Kept as a
standalone module so the digit_classifier example is fully self-contained.

═══════════════════════════════════════════════════════════════════════════════
ENCODING SCHEME: Direct Binary Encoding
═══════════════════════════════════════════════════════════════════════════════

  pixel = 1  (active)   →  spike at t = spike_time ms
  pixel = 0  (inactive) →  no spike (empty list)

Each of the 400 input neurons corresponds to one pixel.  The set of neurons
that fire encodes WHICH pixels are on, and the single common fire time encodes
WHEN the "image frame" arrived.

This is the simplest possible encoding.  It models the output of a binary
event camera (Dynamic Vision Sensor): a pixel that transitions from dark to
bright fires exactly one event.

═══════════════════════════════════════════════════════════════════════════════
WHY NOT RATE CODING FOR NOW?
═══════════════════════════════════════════════════════════════════════════════

In rate coding, a pixel's intensity determines how MANY spikes it fires per
second.  Rate coding works well when pixels have continuous values (0–255) and
when you want graded responses.  For binary images (0 or 1), direct encoding
is cleaner and faster.

Rate coding is listed as an extension in the README.
"""

from __future__ import annotations
import numpy as np


def encode_image_to_spikes(
    image: np.ndarray,
    spike_time: float = 10.0,
    sim_time: float = 100.0,
) -> list[list[float]]:
    """
    Convert a 20×20 binary image to per-neuron spike-time lists.

    Parameters
    ----------
    image      : 20×20 NumPy array (values 0 or 1).
    spike_time : ms at which active pixels emit their spike. Default 10 ms.
    sim_time   : simulation duration (ms) — used only for validation.

    Returns
    -------
    List of 400 sublists.  Each sublist is [spike_time] or [].
    PyNN's SpikeSourceArray accepts this directly.
    """
    if not (0 < spike_time < sim_time):
        raise ValueError(
            f"spike_time {spike_time} ms must be in (0, {sim_time}) ms")

    flat = image.flatten()
    return [[float(spike_time)] if v == 1 else [] for v in flat]


def count_active_pixels(spike_times: list[list[float]]) -> int:
    """Return how many neurons have at least one scheduled spike."""
    return sum(1 for st in spike_times if st)


def get_active_neuron_indices(spike_times: list[list[float]]) -> list[int]:
    """Return indices of neurons that will fire at least once."""
    return [i for i, st in enumerate(spike_times) if st]


# ── Module self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from digit_generator import create_digit_image, create_random_image

    for digit in range(10):
        img = create_digit_image(digit)
        spikes = encode_image_to_spikes(img)
        print(f"  digit {digit}: {count_active_pixels(spikes)}/400 neurons fire")

    rnd = create_random_image()
    rnd_spikes = encode_image_to_spikes(rnd)
    print(f"  random   : {count_active_pixels(rnd_spikes)}/400 neurons fire")
