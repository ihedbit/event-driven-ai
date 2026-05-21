"""
encoder.py — Pixel-to-spike encoding for the T-shape detector.

═══════════════════════════════════════════════════════════════════════════════
WHY DO WE NEED AN ENCODER?
═══════════════════════════════════════════════════════════════════════════════

Traditional deep-learning networks feed raw floating-point pixel values into
matrix multiplications.  SNNs communicate exclusively through discrete events
called *spikes* (action potentials).

An *encoder* converts a pixel value (0 or 1 here) into a spike-time pattern
that can drive a PyNN SpikeSourceArray neuron:

    pixel = 1  →  spike at t = spike_time ms   (active: fire once)
    pixel = 0  →  (empty)                       (silent: never fire)

This is called "direct encoding" or "binary encoding" — the simplest possible
scheme.  The spike itself carries all the information; no spike means "nothing
to report."

═══════════════════════════════════════════════════════════════════════════════
ALTERNATIVE ENCODING SCHEMES (for later exploration)
═══════════════════════════════════════════════════════════════════════════════

  • Rate coding      – fire N spikes proportional to pixel brightness (0–255).
  • Time-to-first-spike (TTFS) – bright pixels fire earlier than dim ones.
  • Poisson coding   – spike train whose rate matches pixel intensity.

For a binary image (0 or 1) the direct scheme is sufficient and easiest to
reason about.
"""

from __future__ import annotations
import numpy as np


# ── Public API ────────────────────────────────────────────────────────────────

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
    spike_time : Time (ms) at which active pixels fire their single spike.
                 Must be within [0, sim_time).
    sim_time   : Total simulation duration (ms) — used for validation only.

    Returns
    -------
    A list of 400 sublists.  Each sublist is either [spike_time] (active) or
    [] (silent).  PyNN's SpikeSourceArray accepts this format directly.

    Example
    -------
    >>> import numpy as np
    >>> from encoder import encode_image_to_spikes
    >>> img = np.array([[1, 0], [0, 1]])   # 2×2 toy image
    >>> encode_image_to_spikes(img, spike_time=5.0, sim_time=20.0)
    [[5.0], [], [], [5.0]]
    """
    if spike_time <= 0 or spike_time >= sim_time:
        raise ValueError(
            f"spike_time ({spike_time} ms) must be in (0, sim_time={sim_time}) ms")

    flat = image.flatten()
    spike_times: list[list[float]] = []

    for pixel_value in flat:
        if pixel_value == 1:
            # Active pixel → exactly one spike at spike_time
            spike_times.append([float(spike_time)])
        else:
            # Inactive pixel → neuron stays silent for the whole simulation
            spike_times.append([])

    return spike_times


def count_active_pixels(spike_times: list[list[float]]) -> int:
    """Return how many neurons have at least one spike scheduled."""
    return sum(1 for st in spike_times if len(st) > 0)


def get_active_neuron_indices(spike_times: list[list[float]]) -> list[int]:
    """Return the indices of neurons that will fire at least once."""
    return [i for i, st in enumerate(spike_times) if len(st) > 0]


# ── Module self-test (run as script) ─────────────────────────────────────────

if __name__ == "__main__":
    from image_generator import create_t_shape_image, create_random_image

    t_img   = create_t_shape_image()
    rnd_img = create_random_image()

    t_spikes  = encode_image_to_spikes(t_img)
    rnd_spikes = encode_image_to_spikes(rnd_img)

    print(f"T-shape  : {count_active_pixels(t_spikes)}/400 neurons fire at t=10 ms")
    print(f"Random   : {count_active_pixels(rnd_spikes)}/400 neurons fire at t=10 ms")

    first_five_active = get_active_neuron_indices(t_spikes)[:5]
    print(f"First 5 active neuron indices (T-shape): {first_five_active}")
