"""
image_generator.py — Binary image creation for the T-shape detector.

In a Spiking Neural Network (SNN), sensory input is typically a 2-D pixel grid
where each pixel maps 1-to-1 to one input neuron.  A pixel value of 1 means
"this neuron will fire a spike"; 0 means "this neuron stays silent."

This module produces:
  - A canonical 20×20 T-shape (the positive class)
  - A 20×20 random-noise image (the negative class)

The T shape is defined explicitly so we can also extract the flat indices of
its pixels — those indices determine which input→output synapses receive a
strong weight in the network.
"""

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_SIZE = 20  # images are IMAGE_SIZE × IMAGE_SIZE pixels

# T shape geometry (in pixel coordinates, 0-indexed)
T_BAR_ROW_START  = 2          # top horizontal bar starts at this row
T_BAR_ROW_END    = 3          # exclusive: bar occupies rows [2, 3)
T_BAR_COL_START  = 2          # bar left edge
T_BAR_COL_END    = 18         # bar right edge (exclusive) → 16 columns

T_STEM_ROW_START = 3          # stem continues from where the bar ends
T_STEM_ROW_END   = 18         # exclusive: rows [3, 18) → 15 rows
T_STEM_COL_START = 9          # center-left column of the 2-pixel-wide stem
T_STEM_COL_END   = 11         # exclusive → cols 9 and 10


# ── Public functions ──────────────────────────────────────────────────────────

def create_t_shape_image() -> np.ndarray:
    """
    Return a 20×20 binary NumPy array containing only the T shape.

    Layout (each '#' is a pixel with value 1):

        col  0123456789012345678901
        row
         0   ....................
         1   ....................
         2   ..################..   ← horizontal bar (16 pixels)
         3   .........##.........   ← top of stem
         4   .........##.........
         ...
        17   .........##.........   ← bottom of stem
        18   ....................
        19   ....................

    Total active pixels: 16 (bar) + 15×2 (stem) = 46 pixels.
    """
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)

    # Horizontal bar
    image[T_BAR_ROW_START:T_BAR_ROW_END, T_BAR_COL_START:T_BAR_COL_END] = 1

    # Vertical stem
    image[T_STEM_ROW_START:T_STEM_ROW_END, T_STEM_COL_START:T_STEM_COL_END] = 1

    return image


def create_random_image(seed: int = 42, density: float = 0.30) -> np.ndarray:
    """
    Return a 20×20 binary image with randomly scattered active pixels.

    Parameters
    ----------
    seed    : Random seed for reproducibility (default 42).
    density : Fraction of pixels that are active, default 0.30 (30 %).

    With density=0.30 and 46 T-shape pixels in the image, only ~14 T-pixels
    will be active on average — well below the detection threshold.
    """
    rng = np.random.RandomState(seed)
    flat = rng.choice([0, 1], size=IMAGE_SIZE * IMAGE_SIZE,
                      p=[1.0 - density, density])
    return flat.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.int32)


def get_t_shape_pixel_indices() -> list[int]:
    """
    Return the flat (1-D) indices of every T-shape pixel in a 20×20 image.

    A 2-D pixel at (row, col) maps to flat index = row * IMAGE_SIZE + col.
    These indices are used by the network to assign strong synaptic weights
    to the input neurons that represent T-shape positions.
    """
    t_image = create_t_shape_image()
    return list(np.where(t_image.flatten() == 1)[0])


def image_to_flat(image: np.ndarray) -> np.ndarray:
    """Flatten a 20×20 image to a 400-element 1-D array."""
    assert image.shape == (IMAGE_SIZE, IMAGE_SIZE), (
        f"Expected {IMAGE_SIZE}×{IMAGE_SIZE} image, got {image.shape}")
    return image.flatten()


# ── Module self-test (run as script) ─────────────────────────────────────────

if __name__ == "__main__":
    t_img   = create_t_shape_image()
    rnd_img = create_random_image()

    t_idx   = get_t_shape_pixel_indices()
    overlap = int(np.sum(rnd_img.flatten()[t_idx]))

    print(f"T-shape  : {int(t_img.sum())} active pixels")
    print(f"Random   : {int(rnd_img.sum())} active pixels")
    print(f"T-pixels active in random image: {overlap}  "
          f"(need ≥32 to fire, so detection = {overlap >= 32})")
    print("\nT-shape image (# = active):")
    for row in t_img:
        print("".join("#" if p else "." for p in row))
