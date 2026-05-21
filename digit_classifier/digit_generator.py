"""
digit_generator.py — Synthetic 20×20 binary digit images (0–9).

═══════════════════════════════════════════════════════════════════════════════
DESIGN: 7-SEGMENT DISPLAY
═══════════════════════════════════════════════════════════════════════════════

Each digit is built from seven named segments — the same logic used in pocket
calculators and digital clocks.  This is intentional:

  * The segment representation is clean and deterministic.
  * It is easy to reason about overlaps between digits.
  * It mirrors how digit-recognition hardware (e.g., DVS cameras reading LCD
    panels) would produce binary event maps.

Segment layout on a 20×20 grid (each # is a 2-pixel-wide or 2-row-tall bar):

    col  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
    row
     0   . . . . . . . . . . . . . . . . . . . .
     1   . . . . . . A A A A A A A A . . . . . .  ← segment A (top bar)
     2   . . . . . . A A A A A A A A . . . . . .
     3   . . . . . F F . . . . . . B B . . . . .  ← F (upper-left), B (upper-right)
     4   . . . . . F F . . . . . . B B . . . . .
     5   . . . . . F F . . . . . . B B . . . . .
     6   . . . . . F F . . . . . . B B . . . . .
     7   . . . . . F F . . . . . . B B . . . . .
     8   . . . . . F F . . . . . . B B . . . . .
     9   . . . . . . G G G G G G G G . . . . . .  ← segment G (middle bar)
    10   . . . . . . G G G G G G G G . . . . . .
    11   . . . . . E E . . . . . . C C . . . . .  ← E (lower-left), C (lower-right)
    12   . . . . . E E . . . . . . C C . . . . .
    13   . . . . . E E . . . . . . C C . . . . .
    14   . . . . . E E . . . . . . C C . . . . .
    15   . . . . . E E . . . . . . C C . . . . .
    16   . . . . . E E . . . . . . C C . . . . .
    17   . . . . . . D D D D D D D D . . . . . .  ← segment D (bottom bar)
    18   . . . . . . D D D D D D D D . . . . . .
    19   . . . . . . . . . . . . . . . . . . . .

Segment pixel counts (used for weight normalisation in network.py):
  A=16, B=12, C=12, D=16, E=12, F=12, G=16  → total possible = 96

Digit → active segment mapping (standard 7-segment encoding):
  0 → A,B,C,D,E,F      (80 px)
  1 → B,C               (24 px)
  2 → A,B,D,E,G         (72 px)
  3 → A,B,C,D,G         (72 px)
  4 → B,C,F,G           (52 px)
  5 → A,C,D,F,G         (72 px)
  6 → A,C,D,E,F,G       (84 px)
  7 → A,B,C             (40 px)
  8 → A,B,C,D,E,F,G     (96 px)
  9 → A,B,C,D,F,G       (84 px)

═══════════════════════════════════════════════════════════════════════════════
WHY NORMALIZATION MATTERS
═══════════════════════════════════════════════════════════════════════════════

Digit "1" has only 24 pixels while digit "8" has 96.  If all template pixels
used the SAME synapse weight, showing "8" would produce 4× more current in
output neuron 8 than showing "1" in neuron 1.  By normalising each weight as
  w_k = W_BASE / N_k
every digit, when shown to its OWN output neuron, produces exactly the same
peak voltage change — a fair comparison.
"""

from __future__ import annotations
import numpy as np

IMAGE_SIZE = 20  # pixels per side

# ── Segment geometry (slice objects index into the 20×20 grid) ────────────────
# Each slice is (row_slice, col_slice)
SEGMENTS: dict[str, tuple[slice, slice]] = {
    "A": (slice(1,  3),  slice(6, 14)),   # top horizontal bar       — 2×8 = 16 px
    "B": (slice(3,  9),  slice(13, 15)),  # upper-right vertical     — 6×2 = 12 px
    "C": (slice(11, 17), slice(13, 15)),  # lower-right vertical     — 6×2 = 12 px
    "D": (slice(17, 19), slice(6, 14)),   # bottom horizontal bar    — 2×8 = 16 px
    "E": (slice(11, 17), slice(5, 7)),    # lower-left vertical      — 6×2 = 12 px
    "F": (slice(3,  9),  slice(5, 7)),    # upper-left vertical      — 6×2 = 12 px
    "G": (slice(9,  11), slice(6, 14)),   # middle horizontal bar    — 2×8 = 16 px
}

# Standard 7-segment encoding
DIGIT_SEGMENTS: dict[int, list[str]] = {
    0: ["A", "B", "C", "D", "E", "F"],
    1: ["B", "C"],
    2: ["A", "B", "D", "E", "G"],
    3: ["A", "B", "C", "D", "G"],
    4: ["B", "C", "F", "G"],
    5: ["A", "C", "D", "F", "G"],
    6: ["A", "C", "D", "E", "F", "G"],
    7: ["A", "B", "C"],
    8: ["A", "B", "C", "D", "E", "F", "G"],
    9: ["A", "B", "C", "D", "F", "G"],
}


# ── Public API ────────────────────────────────────────────────────────────────

def create_digit_image(digit: int) -> np.ndarray:
    """
    Return a 20×20 binary NumPy array showing the given digit (0–9).

    Active pixels (1) represent the lit segments of the 7-segment display.
    """
    if digit not in DIGIT_SEGMENTS:
        raise ValueError(f"Digit must be 0–9, got {digit}")
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
    for seg_name in DIGIT_SEGMENTS[digit]:
        row_sl, col_sl = SEGMENTS[seg_name]
        image[row_sl, col_sl] = 1
    return image


def create_all_digit_images() -> dict[int, np.ndarray]:
    """Return a dict {digit: 20×20 image} for all 10 digits."""
    return {d: create_digit_image(d) for d in range(10)}


def create_random_image(seed: int = 42, density: float = 0.25) -> np.ndarray:
    """
    Return a 20×20 random binary image.

    With density=0.25, ~25% of pixels are active — fewer than any digit
    template, making it unlikely to strongly activate any single output neuron.
    """
    rng = np.random.RandomState(seed)
    flat = rng.choice([0, 1], size=IMAGE_SIZE * IMAGE_SIZE,
                      p=[1.0 - density, density])
    return flat.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.int32)


def get_digit_template_indices(digit: int) -> list[int]:
    """
    Return the flat (1-D) pixel indices that belong to `digit`'s template.

    Flat index for (row, col) = row × IMAGE_SIZE + col.
    These are the pixels that will receive excitatory connections to output
    neuron `digit` in the network.
    """
    return list(np.where(create_digit_image(digit).flatten() == 1)[0])


def get_all_template_indices() -> dict[int, list[int]]:
    """Return {digit: [flat indices]} for all 10 digits."""
    return {d: get_digit_template_indices(d) for d in range(10)}


def get_segment_pixel_counts() -> dict[str, int]:
    """Return the pixel count for each named segment."""
    counts = {}
    for name, (row_sl, col_sl) in SEGMENTS.items():
        n_rows = row_sl.stop - row_sl.start
        n_cols = col_sl.stop - col_sl.start
        counts[name] = n_rows * n_cols
    return counts


def image_to_flat(image: np.ndarray) -> np.ndarray:
    """Flatten a 20×20 image to a 400-element 1-D array."""
    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE)
    return image.flatten()


# ── Module self-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_imgs = create_all_digit_images()
    seg_counts = get_segment_pixel_counts()

    print("Segment pixel counts:")
    for seg, n in sorted(seg_counts.items()):
        print(f"  {seg}: {n} px")

    print("\nDigit template sizes (N_k):")
    for d in range(10):
        n = int(all_imgs[d].sum())
        print(f"  digit {d}: {n} active pixels  ({', '.join(DIGIT_SEGMENTS[d])})")

    print("\nDigit images (# = active pixel):")
    for d in range(10):
        print(f"\n  Digit {d}:")
        for row in all_imgs[d]:
            print("  " + "".join("#" if p else "." for p in row))
