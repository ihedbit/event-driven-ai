"""
visualization.py — Plotting utilities for the digit classifier.

Generates up to six figure types per simulation run:

  1. Digit image grid         — 20×20 pixel view of the input
  2. Input spike raster       — which of 400 neurons fired and when
  3. Output activity bar chart— spike count per digit class (0–9)
  4. Membrane voltage traces  — LIF dynamics for each output neuron
  5. Prediction display       — winner announcement with confidence
  6. Summary 2×3 dashboard   — all five in one combined figure
  7. All-digits gallery       — 10 digit templates in a single figure

All figures are saved to outputs/ directory relative to this file.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
DIGIT_COLORS = [
    "#E53935", "#8E24AA", "#1E88E5", "#00ACC1",
    "#43A047", "#F4511E", "#6D4C41", "#757575",
    "#FFB300", "#3949AB",
]
INACTIVE_COLOR  = "#ECEFF1"
VOLTAGE_COLOR   = "#4CAF50"
THRESH_COLOR    = "#F44336"
REST_COLOR      = "#607D8B"


# ── Helper: draw a 20×20 pixel grid on an existing Axes ──────────────────────

def _draw_digit_grid(ax, image, digit_color="#2196F3", title="", label_segments=False):
    rows, cols = image.shape
    for r in range(rows):
        for c in range(cols):
            fc = digit_color if image[r, c] == 1 else INACTIVE_COLOR
            ax.add_patch(plt.Rectangle(
                [c, rows - r - 1], 1, 1,
                facecolor=fc, edgecolor="white", linewidth=0.3))
    ax.set_xlim(0, cols); ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontweight="bold")


# ── Figure 1: Input digit image ───────────────────────────────────────────────

def plot_digit_image(
    image: np.ndarray,
    digit: int | None = None,
    filename: str = "input_image.png",
) -> Path:
    """Save a standalone 20×20 pixel grid plot."""
    title = f"Input Image — Digit {digit}" if digit is not None else "Input Image"
    color = DIGIT_COLORS[digit] if digit is not None else "#2196F3"

    fig, ax = plt.subplots(figsize=(5, 5))
    _draw_digit_grid(ax, image, digit_color=color, title=title)

    n_active = int(image.sum())
    ax.text(0.5, -0.04, f"{n_active}/400 active pixels",
            transform=ax.transAxes, ha="center", fontsize=9, color="#555")
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 2: Input spike raster ─────────────────────────────────────────────

def plot_input_raster(
    spike_times_list: list[list[float]],
    digit: int | None = None,
    sim_time: float = 100.0,
    filename: str = "raster.png",
) -> Path:
    """Raster showing which of 400 input neurons fired and when."""
    fig, ax = plt.subplots(figsize=(10, 5))
    color = DIGIT_COLORS[digit] if digit is not None else "#2196F3"

    for i, st in enumerate(spike_times_list):
        if st:
            ax.plot(st[0], i, "|", markersize=5,
                    color=color, alpha=0.7, markeredgewidth=1.0)

    n_active = sum(1 for st in spike_times_list if st)
    title = (f"Input Spike Raster — Digit {digit}"
             if digit is not None else "Input Spike Raster")

    ax.set_xlim(0, sim_time); ax.set_ylim(-1, 401)
    ax.set_xlabel("Time (ms)", fontsize=11)
    ax.set_ylabel("Neuron index (pixel)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.text(0.02, 0.97, f"{n_active}/400 neurons fire",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 3: Output activity bar chart ──────────────────────────────────────

def plot_output_activity(
    spike_counts: list[int],
    true_digit: int | None = None,
    filename: str = "output_activity.png",
) -> Path:
    """
    Bar chart showing spike count per output neuron.

    The winning neuron is highlighted.  If a ground-truth digit is provided,
    correct vs incorrect classification is colour-coded.
    """
    pred_digit = int(np.argmax(spike_counts)) if max(spike_counts) > 0 else None

    fig, ax = plt.subplots(figsize=(10, 4))
    digits = list(range(10))
    bar_colors = []
    for k in digits:
        if k == pred_digit:
            bar_colors.append(DIGIT_COLORS[k] if k == true_digit else "#F44336")
        elif k == true_digit and spike_counts[k] > 0:
            bar_colors.append("#FF9800")  # true digit but not winner
        else:
            bar_colors.append("#B0BEC5")

    bars = ax.bar(digits, spike_counts, color=bar_colors, edgecolor="white",
                  linewidth=0.8, zorder=3)

    for bar, count in zip(bars, spike_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, count + 0.05,
                    str(count), ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits], fontsize=12)
    ax.set_xlabel("Digit class", fontsize=11)
    ax.set_ylabel("Output spike count", fontsize=11)
    title = (f"Output Neuron Activity  |  True digit: {true_digit}"
             if true_digit is not None else "Output Neuron Activity")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_ylim(0, max(spike_counts) + 1 if max(spike_counts) > 0 else 2)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 4: Membrane voltage traces ────────────────────────────────────────

def plot_membrane_voltages(
    voltage_times: np.ndarray,
    voltage_values: list[np.ndarray],
    spike_counts: list[int],
    true_digit: int | None = None,
    filename: str = "voltages.png",
) -> Path:
    """
    Plot LIF membrane voltage trace for each output neuron.

    Shows how each neuron integrates the incoming spike currents.  Neurons
    whose templates match the input rise to threshold; others stay near rest.
    """
    from network import LIF_PARAMS
    v_thresh = LIF_PARAMS["v_thresh"]
    v_rest   = LIF_PARAMS["v_rest"]

    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    sim_time  = float(voltage_times[-1]) if len(voltage_times) > 0 else 100.0

    for k in range(10):
        ax = axes_flat[k]
        color = DIGIT_COLORS[k]

        if len(voltage_values[k]) > 0:
            ax.plot(voltage_times, voltage_values[k],
                    color=color, linewidth=1.5, label=f"V_m neuron {k}")

        ax.axhline(v_thresh, ls="--", color=THRESH_COLOR, lw=1.0, alpha=0.8)
        ax.axhline(v_rest,   ls=":",  color=REST_COLOR,   lw=0.8, alpha=0.6)

        label = f"Neuron {k} — {spike_counts[k]} spike(s)"
        if k == true_digit:
            label += "  ★ correct"
        ax.set_title(label, fontsize=8, fontweight="bold", color=color)
        ax.set_xlim(0, sim_time)
        ax.grid(ls="--", alpha=0.25)
        if k % 2 == 0:
            ax.set_ylabel("mV", fontsize=8)

    for ax in axes_flat[-2:]:
        ax.set_xlabel("Time (ms)", fontsize=9)

    title = (f"Membrane Voltages — True digit: {true_digit}"
             if true_digit is not None else "Membrane Voltages")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 5: Prediction display ─────────────────────────────────────────────

def plot_prediction(
    spike_counts: list[int],
    true_digit: int | None = None,
    filename: str = "prediction.png",
) -> Path:
    """Large "result card" showing the winner and per-class bar chart."""
    pred_digit = int(np.argmax(spike_counts)) if max(spike_counts) > 0 else None
    is_correct = (pred_digit == true_digit) if true_digit is not None else None

    fig = plt.figure(figsize=(12, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 2], figure=fig)

    # Left panel — prediction result card
    ax_pred = fig.add_subplot(gs[0])
    ax_pred.axis("off")
    if pred_digit is not None:
        pred_color = (DIGIT_COLORS[pred_digit]
                      if is_correct is not False else "#F44336")
        ax_pred.text(0.5, 0.65, str(pred_digit),
                     ha="center", va="center", fontsize=80,
                     fontweight="bold", color=pred_color,
                     transform=ax_pred.transAxes)
        status = ""
        if is_correct is True:
            status = "✓ Correct"
        elif is_correct is False:
            status = f"✗ Wrong  (true: {true_digit})"
        ax_pred.text(0.5, 0.28, status, ha="center", va="center",
                     fontsize=14, color=pred_color,
                     transform=ax_pred.transAxes, fontweight="bold")
        ax_pred.text(0.5, 0.16, f"Confidence: {spike_counts[pred_digit]} spike(s)",
                     ha="center", va="center", fontsize=11, color="#555",
                     transform=ax_pred.transAxes)
    else:
        ax_pred.text(0.5, 0.5, "No\nPrediction", ha="center", va="center",
                     fontsize=20, color="#9E9E9E",
                     transform=ax_pred.transAxes)
    ax_pred.set_title("Predicted digit", fontweight="bold", fontsize=12)

    # Right panel — bar chart
    ax_bar = fig.add_subplot(gs[1])
    bar_colors = [
        DIGIT_COLORS[k] if k == pred_digit else "#B0BEC5"
        for k in range(10)
    ]
    ax_bar.bar(range(10), spike_counts, color=bar_colors,
               edgecolor="white", linewidth=0.8, zorder=3)
    ax_bar.set_xticks(range(10))
    ax_bar.set_xticklabels([str(d) for d in range(10)], fontsize=12)
    ax_bar.set_xlabel("Digit class", fontsize=11)
    ax_bar.set_ylabel("Spike count", fontsize=11)
    ax_bar.set_title("Output neuron spike counts", fontweight="bold")
    ax_bar.grid(axis="y", ls="--", alpha=0.4, zorder=0)

    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 6: All-digits gallery ─────────────────────────────────────────────

def plot_all_digits_gallery(filename: str = "all_digits.png") -> Path:
    """Save all 10 digit templates in a 2×5 grid."""
    from digit_generator import create_digit_image

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for d, ax in enumerate(axes.flatten()):
        img = create_digit_image(d)
        _draw_digit_grid(ax, img, digit_color=DIGIT_COLORS[d],
                         title=f"Digit {d}  ({int(img.sum())} px)")

    fig.suptitle("Digit Templates (7-Segment Style)", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 7: Summary dashboard ──────────────────────────────────────────────

def plot_summary(
    image: np.ndarray,
    spike_times_list: list[list[float]],
    spike_counts: list[int],
    voltage_times: np.ndarray,
    voltage_values: list[np.ndarray],
    true_digit: int | None = None,
    label: str = "digit",
) -> Path:
    """All key plots combined into one 2×3 dashboard figure."""
    from network import LIF_PARAMS

    pred_digit = int(np.argmax(spike_counts)) if max(spike_counts) > 0 else None
    color = DIGIT_COLORS[pred_digit] if pred_digit is not None else "#607D8B"
    v_thresh  = LIF_PARAMS["v_thresh"]
    v_rest    = LIF_PARAMS["v_rest"]
    sim_time  = float(voltage_times[-1]) if len(voltage_times) > 0 else 100.0

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: input image ──────────────────────────────────────────────────
    ax_img = fig.add_subplot(gs[0, 0])
    img_color = DIGIT_COLORS[true_digit] if true_digit is not None else "#2196F3"
    _draw_digit_grid(ax_img, image, digit_color=img_color,
                     title=f"Input  (digit {true_digit})"
                     if true_digit is not None else "Input Image")

    # ── Panel B: spike raster ─────────────────────────────────────────────────
    ax_raster = fig.add_subplot(gs[0, 1])
    for i, st in enumerate(spike_times_list):
        if st:
            ax_raster.plot(st[0], i, "|", markersize=4,
                           color=img_color, alpha=0.6, markeredgewidth=0.9)
    ax_raster.set_xlim(0, sim_time); ax_raster.set_ylim(-1, 401)
    ax_raster.set_xlabel("Time (ms)"); ax_raster.set_ylabel("Neuron index")
    ax_raster.set_title("Input Raster", fontweight="bold")
    ax_raster.grid(axis="x", ls="--", alpha=0.3)

    # ── Panel C: output activity ──────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 2])
    bar_colors = [
        DIGIT_COLORS[k] if k == pred_digit else "#B0BEC5"
        for k in range(10)
    ]
    ax_bar.bar(range(10), spike_counts, color=bar_colors,
               edgecolor="white", linewidth=0.6, zorder=3)
    ax_bar.set_xticks(range(10))
    ax_bar.set_xticklabels([str(d) for d in range(10)])
    ax_bar.set_xlabel("Digit class"); ax_bar.set_ylabel("Spike count")
    ax_bar.set_title("Output Activity", fontweight="bold")
    ax_bar.grid(axis="y", ls="--", alpha=0.3, zorder=0)

    # ── Panel D: winning + runner-up voltage traces ───────────────────────────
    ax_v = fig.add_subplot(gs[1, :2])
    # Show top-3 most active neurons for clarity
    top3 = sorted(range(10), key=lambda k: spike_counts[k], reverse=True)[:3]
    for k in top3:
        if len(voltage_values[k]) > 0:
            label_str = f"Neuron {k} ({spike_counts[k]} spikes)"
            ax_v.plot(voltage_times, voltage_values[k],
                      color=DIGIT_COLORS[k], linewidth=1.4,
                      label=label_str)
    ax_v.axhline(v_thresh, ls="--", color=THRESH_COLOR, lw=1.2,
                 label=f"Threshold ({v_thresh} mV)")
    ax_v.axhline(v_rest, ls=":", color=REST_COLOR, lw=0.8, alpha=0.7)
    ax_v.set_xlabel("Time (ms)"); ax_v.set_ylabel("V_m (mV)")
    ax_v.set_title("Membrane Voltages (top-3 neurons)", fontweight="bold")
    ax_v.legend(fontsize=8); ax_v.grid(ls="--", alpha=0.25)

    # ── Panel E: prediction card ──────────────────────────────────────────────
    ax_pred = fig.add_subplot(gs[1, 2])
    ax_pred.axis("off")
    is_correct = (pred_digit == true_digit) if true_digit is not None else None
    pred_color = (color if is_correct is not False else "#F44336")
    if pred_digit is not None:
        ax_pred.text(0.5, 0.60, str(pred_digit), ha="center", va="center",
                     fontsize=72, fontweight="bold", color=pred_color,
                     transform=ax_pred.transAxes)
        status = "✓ Correct" if is_correct else (
                 f"✗ Wrong  (true: {true_digit})" if is_correct is False else "")
        ax_pred.text(0.5, 0.25, status, ha="center",
                     fontsize=13, color=pred_color, fontweight="bold",
                     transform=ax_pred.transAxes)
    else:
        ax_pred.text(0.5, 0.5, "Silent", ha="center", va="center",
                     fontsize=22, color="#9E9E9E",
                     transform=ax_pred.transAxes)
    ax_pred.set_title("Prediction", fontweight="bold")

    fig.suptitle(
        f"SNN Digit Classifier — {'Digit ' + str(true_digit) if true_digit is not None else label}",
        fontsize=15, fontweight="bold")

    path = OUTPUT_DIR / f"summary_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path


# ── Figure 8: Accuracy matrix ─────────────────────────────────────────────────

def plot_accuracy_matrix(
    results: dict,
    filename: str = "accuracy_matrix.png",
) -> Path:
    """
    Show all-digits classification results as a 10×1 verdict strip.

    `results` is {true_digit: ClassificationResult} as returned by
    classifier.classify_all_digits().
    """
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.axis("off")

    for d in range(10):
        r = results[d]
        pred = r.predicted_digit
        correct = (pred == d)
        bg_color = DIGIT_COLORS[d] if correct else "#FFCDD2"
        ax.add_patch(plt.Rectangle([d, 0], 1, 1,
                                    facecolor=bg_color, edgecolor="white",
                                    linewidth=2))
        mark = "✓" if correct else f"→{pred}"
        ax.text(d + 0.5, 0.7, str(d),
                ha="center", va="center", fontsize=18, fontweight="bold",
                color="white" if correct else "#C62828")
        ax.text(d + 0.5, 0.25, mark,
                ha="center", va="center", fontsize=9,
                color="white" if correct else "#C62828")

    n_correct = sum(1 for d, r in results.items() if r.predicted_digit == d)
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.2)
    ax.set_title(f"Classification Results  —  {n_correct}/10 correct  "
                 f"({n_correct * 10:.0f}% accuracy)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
    return path
