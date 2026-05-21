"""
visualization.py — Plotting utilities for the T-shape SNN detector.

Generates four figures per simulation run:

  1. Input image (binary pixel grid)
  2. Spike raster (which input neurons fired and when)
  3. Output spike times (detected event timeline)
  4. Output membrane voltage trace (shows integration and firing)

All plots are saved to the outputs/ directory relative to this file.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; works without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Directory for saved figures (created automatically if needed)
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Colour palette ────────────────────────────────────────────────────────────
ACTIVE_COLOR   = "#2196F3"    # blue for active / spiking
INACTIVE_COLOR = "#E0E0E0"    # light grey for inactive / silent
T_PIXEL_COLOR  = "#FF5722"    # orange highlight for T-shape pixels
VOLTAGE_COLOR  = "#4CAF50"    # green for membrane voltage
THRESH_COLOR   = "#F44336"    # red for threshold line


# ── Figure 1: Input image ─────────────────────────────────────────────────────

def plot_input_image(
    image: np.ndarray,
    title: str = "Input Image",
    filename: str = "input_image.png",
    t_shape_overlay: bool = True,
) -> Path:
    """
    Display the 20×20 binary image as a pixel grid.

    Active pixels (1) are shown in blue; inactive (0) in light grey.
    When t_shape_overlay=True, T-shape positions are outlined in orange so
    you can see which pixels carry the strong synaptic weights.
    """
    from image_generator import create_t_shape_image

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#FAFAFA")

    rows, cols = image.shape
    t_mask = create_t_shape_image() if t_shape_overlay else None

    for r in range(rows):
        for c in range(cols):
            # Choose fill colour based on pixel value
            facecolor = ACTIVE_COLOR if image[r, c] == 1 else INACTIVE_COLOR
            rect = plt.Rectangle([c, rows - r - 1], 1, 1,
                                  facecolor=facecolor, edgecolor="white",
                                  linewidth=0.4)
            ax.add_patch(rect)

            # Orange border on T-shape positions
            if t_shape_overlay and t_mask is not None and t_mask[r, c] == 1:
                outline = plt.Rectangle([c, rows - r - 1], 1, 1,
                                        facecolor="none",
                                        edgecolor=T_PIXEL_COLOR,
                                        linewidth=1.5)
                ax.add_patch(outline)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Legend
    legend_handles = [
        mpatches.Patch(color=ACTIVE_COLOR,   label="Active pixel (spike)"),
        mpatches.Patch(color=INACTIVE_COLOR, label="Inactive pixel (silent)"),
    ]
    if t_shape_overlay:
        legend_handles.append(
            mpatches.Patch(facecolor="none",
                           edgecolor=T_PIXEL_COLOR,
                           linewidth=2,
                           label="T-shape position (strong synapse)")
        )
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
              bbox_to_anchor=(0, -0.12), ncol=1, frameon=False)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 2: Spike raster plot ───────────────────────────────────────────────

def plot_spike_raster(
    spike_times_list: list[list[float]],
    sim_time: float = 100.0,
    title: str = "Input Spike Raster",
    filename: str = "spike_raster.png",
) -> Path:
    """
    Raster plot showing which input neurons fired and when.

    In event-driven computing, the raster plot is the primary visualisation:
    it shows information as *events in time*, not as continuous signals.
    Neurons that never fire produce no marks — this is the "sparse" nature
    of SNN computation.
    """
    from image_generator import get_t_shape_pixel_indices

    t_indices = set(get_t_shape_pixel_indices())
    active_neurons = [(i, st[0]) for i, st in enumerate(spike_times_list)
                      if len(st) > 0]

    fig, ax = plt.subplots(figsize=(10, 6))

    for neuron_idx, spike_t in active_neurons:
        color = T_PIXEL_COLOR if neuron_idx in t_indices else ACTIVE_COLOR
        ax.plot(spike_t, neuron_idx, "|", markersize=6,
                color=color, alpha=0.7, markeredgewidth=1.2)

    # Highlight T-shape pixel range as a shaded band
    if t_indices:
        t_min, t_max = min(t_indices), max(t_indices)
        ax.axhspan(t_min - 0.5, t_max + 0.5, alpha=0.06,
                   color=T_PIXEL_COLOR, label="T-shape neuron range")

    ax.set_xlim(0, sim_time)
    ax.set_ylim(-1, 401)
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Neuron index (pixel)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    n_active = len(active_neurons)
    n_t_active = sum(1 for i, _ in active_neurons if i in t_indices)
    ax.text(0.02, 0.97,
            f"Active neurons: {n_active}/400\nT-pixels firing: {n_t_active}/46",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    legend_handles = [
        mpatches.Patch(color=T_PIXEL_COLOR, label=f"T-pixel spike ({n_t_active})"),
        mpatches.Patch(color=ACTIVE_COLOR,  label=f"Other spike ({n_active - n_t_active})"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 3: Output spike timeline ──────────────────────────────────────────

def plot_output_spikes(
    output_spike_times: np.ndarray,
    sim_time: float = 100.0,
    title: str = "Output Neuron Spikes",
    filename: str = "output_spikes.png",
) -> Path:
    """
    Show when (if ever) the output (detector) neuron fired.

    A spike here means "T-shape detected".  No spikes = no detection.
    """
    fig, ax = plt.subplots(figsize=(8, 2.5))

    detected = len(output_spike_times) > 0

    if detected:
        ax.eventplot(output_spike_times, lineoffsets=0.5, linelengths=0.8,
                     colors=T_PIXEL_COLOR, linewidths=2.5)
        for t in output_spike_times:
            ax.annotate(f"{t:.1f} ms", xy=(t, 0.5), xytext=(t, 0.75),
                        ha="center", fontsize=9, color=T_PIXEL_COLOR,
                        arrowprops=dict(arrowstyle="-", color=T_PIXEL_COLOR,
                                        lw=1.5))
        result_text = f"T-SHAPE DETECTED  ({len(output_spike_times)} spike(s))"
        result_color = T_PIXEL_COLOR
    else:
        result_text = "NO DETECTION  (output neuron silent)"
        result_color = "#9E9E9E"

    ax.text(sim_time * 0.5, 0.15, result_text, ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=result_color)

    ax.set_xlim(0, sim_time)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines[["left", "right", "top"]].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 4: Membrane voltage trace ─────────────────────────────────────────

def plot_membrane_voltage(
    voltage_times: np.ndarray,
    voltage_values: np.ndarray,
    output_spike_times: np.ndarray,
    lif_params: dict | None = None,
    title: str = "Output Neuron Membrane Voltage",
    filename: str = "membrane_voltage.png",
) -> Path:
    """
    Plot the membrane voltage of the output (detector) neuron over time.

    This is the most educational plot:
      - You can see the neuron integrating incoming spike currents.
      - When V crosses v_thresh, a spike is emitted and V is reset.
      - When the input is silent, V drifts back to v_rest passively.

    This is the essence of the Leaky Integrate-and-Fire model.
    """
    if lif_params is None:
        from network import LIF_PARAMS
        lif_params = LIF_PARAMS

    v_rest   = lif_params.get("v_rest",   -65.0)
    v_thresh = lif_params.get("v_thresh", -45.0)

    fig, ax = plt.subplots(figsize=(10, 4))

    if len(voltage_times) > 0:
        ax.plot(voltage_times, voltage_values,
                color=VOLTAGE_COLOR, linewidth=1.5, label="V_m (membrane voltage)")
    else:
        ax.text(0.5, 0.5, "No voltage data recorded",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#9E9E9E")

    # Threshold line
    ax.axhline(v_thresh, linestyle="--", color=THRESH_COLOR,
               linewidth=1.2, label=f"Threshold ({v_thresh} mV)")

    # Rest line
    ax.axhline(v_rest, linestyle=":", color="#607D8B",
               linewidth=1.0, label=f"Rest ({v_rest} mV)", alpha=0.7)

    # Mark each output spike
    for t_spike in output_spike_times:
        ax.axvline(t_spike, linestyle="-", color=T_PIXEL_COLOR,
                   linewidth=1.5, alpha=0.6)
        ax.annotate("spike", xy=(t_spike, v_thresh),
                    xytext=(t_spike + 1, v_thresh + 3),
                    fontsize=8, color=T_PIXEL_COLOR)

    sim_time = float(voltage_times[-1]) if len(voltage_times) > 0 else 100.0
    ax.set_xlim(0, sim_time)
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Membrane voltage (mV)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Combined summary figure ───────────────────────────────────────────────────

def plot_summary(
    image: np.ndarray,
    spike_times_list: list[list[float]],
    output_spike_times: np.ndarray,
    voltage_times: np.ndarray,
    voltage_values: np.ndarray,
    label: str = "t_shape",
) -> Path:
    """Save all four sub-plots into one combined 2×2 figure."""
    from image_generator import create_t_shape_image, get_t_shape_pixel_indices

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"SNN T-Shape Detector — {label.replace('_', ' ').title()}",
                 fontsize=16, fontweight="bold", y=0.98)

    # ── Panel 1: input image ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 1)
    t_mask = create_t_shape_image()
    t_indices_set = set(get_t_shape_pixel_indices())
    rows, cols = image.shape
    for r in range(rows):
        for c in range(cols):
            fc = ACTIVE_COLOR if image[r, c] == 1 else INACTIVE_COLOR
            ax1.add_patch(plt.Rectangle([c, rows - r - 1], 1, 1,
                                         facecolor=fc, edgecolor="white",
                                         linewidth=0.3))
            if t_mask[r, c] == 1:
                ax1.add_patch(plt.Rectangle([c, rows - r - 1], 1, 1,
                                             facecolor="none",
                                             edgecolor=T_PIXEL_COLOR,
                                             linewidth=1.2))
    ax1.set_xlim(0, cols); ax1.set_ylim(0, rows)
    ax1.set_aspect("equal"); ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title("Input Image", fontweight="bold")

    # ── Panel 2: raster ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 2)
    sim_time = 100.0
    for i, st in enumerate(spike_times_list):
        if st:
            color = T_PIXEL_COLOR if i in t_indices_set else ACTIVE_COLOR
            ax2.plot(st[0], i, "|", markersize=5,
                     color=color, alpha=0.6, markeredgewidth=1.0)
    ax2.set_xlim(0, sim_time); ax2.set_ylim(-1, 401)
    ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Neuron index")
    ax2.set_title("Input Raster", fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.3)

    # ── Panel 3: membrane voltage ─────────────────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 3)
    from network import LIF_PARAMS
    v_thresh = LIF_PARAMS["v_thresh"]
    v_rest   = LIF_PARAMS["v_rest"]
    if len(voltage_times) > 0:
        ax3.plot(voltage_times, voltage_values, color=VOLTAGE_COLOR, lw=1.5)
    ax3.axhline(v_thresh, ls="--", color=THRESH_COLOR, lw=1.2,
                label=f"Threshold ({v_thresh} mV)")
    ax3.axhline(v_rest,   ls=":",  color="#607D8B",    lw=1.0,
                label=f"Rest ({v_rest} mV)", alpha=0.7)
    for t_s in output_spike_times:
        ax3.axvline(t_s, color=T_PIXEL_COLOR, lw=1.5, alpha=0.6)
    ax3.set_xlabel("Time (ms)"); ax3.set_ylabel("V_m (mV)")
    ax3.set_title("Membrane Voltage", fontweight="bold")
    ax3.legend(fontsize=8); ax3.grid(ls="--", alpha=0.3)

    # ── Panel 4: detection result ─────────────────────────────────────────────
    ax4 = fig.add_subplot(2, 2, 4)
    detected = len(output_spike_times) > 0
    result_text = "T-SHAPE DETECTED" if detected else "NO T-SHAPE"
    result_color = T_PIXEL_COLOR if detected else "#9E9E9E"
    emoji = "✓" if detected else "✗"
    ax4.text(0.5, 0.55, emoji, ha="center", va="center",
             fontsize=60, color=result_color, transform=ax4.transAxes)
    ax4.text(0.5, 0.25, result_text, ha="center", va="center",
             fontsize=16, fontweight="bold", color=result_color,
             transform=ax4.transAxes)
    spike_desc = (f"{len(output_spike_times)} output spike(s) at "
                  f"{list(np.round(output_spike_times, 1))} ms"
                  if detected else "Output neuron stayed silent")
    ax4.text(0.5, 0.12, spike_desc, ha="center", va="center",
             fontsize=9, color="#555", transform=ax4.transAxes)
    ax4.set_title("Detection Result", fontweight="bold")
    ax4.axis("off")

    plt.tight_layout()
    path = OUTPUT_DIR / f"summary_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path
