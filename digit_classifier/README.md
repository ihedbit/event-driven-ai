# SNN Digit Classifier (0–9)

A minimal, educational **Spiking Neural Network** that classifies handwritten-style
digits 0–9 from a 20×20 binary image using **PyNN** and **NEST**.

---

## What is this?

```
20×20 binary image
       ↓  pixel → spike encoding
400 SpikeSourceArray neurons  (one per pixel)
       ↓  excitatory synapses  (template pixels,     weight = W_EXC / N_k)
       ↓  inhibitory synapses  (non-template pixels, weight = W_INH / (400−N_k))
10  IF_curr_exp (LIF) neurons  (one per digit class 0–9)
       ↓  spike counting
Winner-Take-All → predicted digit
```

This is **not** a deep-learning project.  There is no backpropagation, no gradient
descent, no floating-point activation functions.  The classifier is a set of
biologically inspired template detectors implemented entirely through synaptic
weights and LIF dynamics.

---

## Interactive demo

An interactive canvas runs live in the **Cursor IDE** — try the classifier without
installing NEST.

| Feature | Description |
|---|---|
| **Click pixels** | Toggle any of the 400 pixels on/off |
| **Digit presets** | Buttons 0–9 load the canonical 7-segment template |
| **Random / Clear** | Load a 25%-density noise image or blank the grid |
| **Mini 7-segment display** | Predicted digit rendered as a live 7-segment readout |
| **Activity bar chart** | Peak ΔV for all 10 output neurons with threshold line |
| **Live stats** | Active pixels, firing count, ΔV for the winner |

Open `digit-classifier.canvas.tsx` from the Cursor canvas panel to launch it.

---

## Key concepts illustrated

| Concept | Where to look |
|---|---|
| Pixel → neuron mapping | `digit_generator.py` + `encoder.py` |
| Direct binary spike encoding | `encoder.py` |
| LIF membrane integration | `network.py` (LIF_PARAMS) |
| Normalised excitatory weights | `network.py` (W_EXC / N_k) |
| Inhibitory suppression (subset problem) | `network.py` (W_INH) |
| Winner-Take-All classification | `classifier.py` |
| ANN vs SNN comparison | this README |

---

## Why inhibitory connections?

The 7-segment digit templates have a **subset problem**: digit "1" (segments B,C)
is a strict subset of digit "7" (A,B,C), "3", "9", and others.  Without
inhibition, showing digit "7" would also fully activate output neuron 1.

The fix: every input pixel that is **not** part of digit k's template is connected
to output neuron k with a **negative** (inhibitory) weight.

When digit "7" is shown:
- Segment A fires → inhibits neuron 1 (A is not in digit 1's template)
- Segments B, C fire → excite neuron 1 (they are in the template)
- Net ΔV for neuron 1 = 25.2 − 5.5 = **19.8 mV** < 23 mV threshold ✗ (silent)

When digit "1" is shown:
- Segments B, C fire → excite neuron 1
- No non-template active pixels
- Net ΔV for neuron 1 = 25.2 mV > **23 mV** threshold ✓ (fires!)

All 90 off-diagonal (shown digit, wrong output neuron) combinations are verified
analytically in `tests/test_network.py::TestWeightSufficiency`.

---

## Weight calibration

For `IF_curr_exp` with cm=0.25 nF, τ_m=20 ms, τ_syn=5 ms:

```
ΔV_peak per nA ≈ 12.61 mV/nA

Correct match  (N_k pixels, weight W_EXC/N_k each):
  ΔV = W_EXC × 12.61 = 2.0 × 12.61 = 25.2 mV  >  23 mV threshold ✓

Worst incorrect pair — digit 7 → neuron 1 (24 exc, 16 inh):
  ΔV = (24 × 2.0/24  −  16 × 10.0/376) × 12.61
     = (2.0 − 0.426) × 12.61
     = 19.8 mV  <  23 mV threshold ✓
```

The analytical proof that all 90 wrong pairs stay below threshold is run as part
of the test suite — no NEST needed.

---

## SNN vs ANN: what's different?

| Property | Artificial Neural Network | This SNN |
|---|---|---|
| Neuron output | Continuous float (activation) | Binary spike at a moment in time |
| Computation | Matrix multiply + non-linearity | Voltage integration + threshold |
| "When" matters? | No | Yes — spike timing carries information |
| Energy | Proportional to all neurons | Proportional to **active** neurons only |
| Classification | softmax → argmax | spike counts → argmax |
| Learning | Backpropagation | STDP (extension) |

---

## Digit templates (7-segment style)

```
Digit 0   Digit 1   Digit 2   Digit 3   Digit 4
 ████       ██       ████      ████      ██ ██
██  ██      ██          ██        ██     ██ ██
██  ██      ██      ████      ████      ██████
██  ██      ██     ██            ██         ██
 ████       ██      ████      ████          ██

Digit 5   Digit 6   Digit 7   Digit 8   Digit 9
████      ████      ████      ████      ████
██        ██           ██     ██  ██    ██  ██
████      ████      ██        ████      ████
   ██     ██  ██    ██        ██  ██       ██
████       ████     ██         ████    ████
```

---

## Project layout

```
digit_classifier/
├── digit_generator.py  # 7-segment digit images (0–9) + random image
├── encoder.py          # pixel → spike time lists
├── network.py          # PyNN network (400 in → 10 out) + simulation runner
├── classifier.py       # Winner-Take-All logic, ClassificationResult
├── visualization.py    # 7 plot types saved to outputs/
├── main.py             # classify all 10 digits + generate all plots
├── requirements.txt
└── tests/
    ├── conftest.py            # shared fixtures, NEST skip
    ├── test_digit_generator.py  # 22 pure-Python tests
    ├── test_encoder.py          # 13 pure-Python tests
    ├── test_network.py          # 12 structure/analytical + 5 NEST tests
    ├── test_classifier.py       # 17 pure-Python WTA logic tests
    └── test_prediction.py       # 39 end-to-end NEST tests

# Cursor IDE canvas (outside project directory, no NEST required)
~/.cursor/projects/.../canvases/digit-classifier.canvas.tsx
```

---

## Setup

### 1 — Install NEST

```bash
# Option A: conda (recommended)
conda create -n snn python=3.11
conda activate snn
conda install -c conda-forge nest-simulator

# Option B: apt (Ubuntu/Debian)
sudo apt install nest

# Docs: https://nest-simulator.readthedocs.io
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Verify

```bash
python -c "import pyNN.nest as sim; sim.setup(); sim.end(); print('OK')"
```

---

## Running

```bash
cd digit_classifier

# Classify all 10 digits (runs 11 simulations)
python main.py

# Run tests (65 pass without NEST; 128 pass with NEST)
pytest tests/ -v
```

Expected output from `main.py`:

```
Digit 0: ✓ [1 spike(s)]
Digit 1: ✓ [1 spike(s)]
Digit 2: ✓ [1 spike(s)]
...
Digit 9: ✓ [1 spike(s)]

Accuracy: 10/10 (100%)
```

---

## Outputs

Running `main.py` saves to `outputs/`:

| File pattern | Description |
|---|---|
| `all_digits.png` | Gallery of all 10 digit templates |
| `input_digit_N.png` | 20×20 pixel grid for digit N |
| `raster_digit_N.png` | Which of 400 neurons fired |
| `activity_digit_N.png` | Bar chart of output spike counts |
| `voltages_digit_N.png` | LIF membrane voltage for all 10 neurons |
| `prediction_digit_N.png` | Winner announcement card |
| `summary_digit_N.png` | All panels combined (dashboard) |
| `accuracy_matrix.png` | 10-digit classification strip |

---

## Migration to SpiNNaker

Change **one import line** in `network.py`:

```python
# Software simulation (NEST)
import pyNN.nest as sim

# SpiNNaker neuromorphic hardware
import pyNN.spiNNaker as sim
```

All other code — `IF_curr_exp`, `SpikeSourceArray`, `StaticSynapse`,
`FromListConnector`, recording — is identical on SpiNNaker via `sPyNNaker`.

Additional SpiNNaker steps:
1. `pip install sPyNNaker`
2. Configure `spynnaker.cfg` pointing to your board
3. Keep `sim_time` short (SpiNNaker has real-time execution constraints)

---

## Extensions

| Idea | Difficulty |
|---|---|
| MNIST-style grey-level images + rate coding | Medium |
| Time-To-First-Spike encoding | Easy |
| Lateral inhibition between output neurons | Medium |
| STDP online learning | Hard |
| Shift/noise-robust templates | Medium |
| Multiple prediction epochs (sustained activity) | Easy |
| SpiNNaker board deployment | Medium (hardware access) |
