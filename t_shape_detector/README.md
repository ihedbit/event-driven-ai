# T-Shape SNN Detector

A minimal, educational **Spiking Neural Network** that detects a "T" shape inside
a 20×20 binary image — implemented with **PyNN** and the **NEST** simulator.

---

## What is this?

This project is the simplest possible neuromorphic pattern detector:

```
20×20 binary image
       ↓  pixel → spike encoding
400 SpikeSourceArray neurons  (one per pixel)
       ↓  weighted synapses
 1  IF_curr_exp  (LIF) neuron  ← fires = "T detected"
```

It is **not** a deep-learning accuracy project.  It is a hands-on introduction to:

| Concept | Where you see it |
|---|---|
| Pixel → neuron mapping | `encoder.py` |
| Spike encoding | `encoder.py` |
| LIF neuron model | `network.py` (LIF_PARAMS) |
| Weighted coincidence detection | `network.py` (connection list) |
| Event-driven computation | spike raster plot |
| PyNN abstraction layer | `network.py` import line |
| Path to SpiNNaker | one-line migration below |

---

## Neuromorphic computing in one paragraph

A conventional processor performs matrix multiplications on floating-point arrays
at a fixed clock rate.  A *neuromorphic* processor like Intel Loihi or SpiNNaker
communicates through discrete **spikes** (events).  Neurons that receive no input
consume almost no power.  Computation is **sparse** and **asynchronous** — exactly
like the brain.  PyNN is the hardware-neutral API that lets you write a network
once and run it on NEST (software), SpiNNaker (neuromorphic chip), or BrainScaleS.

---

## Interactive demo

An interactive canvas runs live in the **Cursor IDE** — open it beside the chat
to explore the detector without NEST.

| Feature | Description |
|---|---|
| **Click pixels** | Toggle any of the 400 pixels on/off |
| **Presets** | Load the canonical T-shape, a random-noise image, or a blank canvas |
| **Template overlay** | T-template positions always visible (darker border) |
| **LIF voltage trace** | Simulated membrane voltage at 0.5 ms resolution, spike marker included |
| **Live stats** | T-pixel overlap, estimated peak ΔV, detection threshold, spike time |

Open `t-shape-detector.canvas.tsx` from the Cursor canvas panel to launch it.

---

## Project layout

```
t_shape_detector/
├── image_generator.py   # Creates 20×20 binary images (T shape + random)
├── encoder.py           # Converts pixel values to spike-time lists
├── network.py           # PyNN network: 400 input + 1 LIF output neuron
├── visualization.py     # Four matplotlib figures per simulation run
├── main.py              # Runs two simulations and saves plots to outputs/
├── requirements.txt
└── tests/
    ├── conftest.py              # Shared fixtures, NEST skip logic
    ├── test_image_generator.py  # 18 tests — pure Python, no NEST
    ├── test_encoder.py          # 18 tests — pure Python, no NEST
    ├── test_network.py          # 13 structure tests + 4 NEST live tests
    └── test_detection.py        # 13 end-to-end NEST integration tests

# Cursor IDE canvas (outside project directory, no NEST required)
~/.cursor/projects/.../canvases/t-shape-detector.canvas.tsx
```

---

## Setup

### 1 — Install NEST simulator

NEST is **not** a pip package.  Use one of:

```bash
# Option A: conda (easiest on Linux/macOS)
conda create -n snn python=3.11
conda activate snn
conda install -c conda-forge nest-simulator

# Option B: Ubuntu/Debian apt
sudo apt install nest

# Option C: build from source
# https://nest-simulator.readthedocs.io/en/stable/installation/
```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3 — Verify the installation

```bash
python -c "import pyNN.nest as sim; sim.setup(); sim.end(); print('OK')"
```

---

## Running the simulation

```bash
cd t_shape_detector
python main.py
```

Expected output:

```
============================================================
  SNN T-Shape Detector  |  PyNN + NEST
============================================================

  Network architecture:
    Input  : 400 SpikeSourceArray neurons (one per pixel)
    Output : 1 IF_curr_exp (LIF) neuron
    T-pixel weight  : 0.05 nA (excitatory)
    Non-T pixels    : unconnected (weight = 0)
    Threshold       : -45.0 mV  (20 mV above rest)

============================================================
  Simulation: T SHAPE
============================================================
  Active input neurons : 46 / 400
  T-pixels firing      : 46 / 46
  Running PyNN/NEST simulation (100.0 ms) …
  Output spikes        : [~21.0]
  Detection result     : T-SHAPE DETECTED ✓

============================================================
  Simulation: RANDOM NOISE
============================================================
  Active input neurons : 120 / 400
  T-pixels firing      : 12 / 46
  Running PyNN/NEST simulation (100.0 ms) …
  Output spikes        : []
  Detection result     : No detection ✗

============================================================
  FINAL RESULTS
============================================================
  T-shape image   → Detection: PASS ✓
  Random noise    → Detection: FAIL ✗  (correct — no T-shape)

  Overall: ALL TESTS PASSED ✓

  Plots saved to: .../t_shape_detector/outputs/
```

Five PNG files are written to `outputs/` for each run:

| File | Shows |
|---|---|
| `input_t_shape.png` | 20×20 pixel grid with T highlighted |
| `raster_t_shape.png` | Which of 400 neurons fired and when |
| `output_t_shape.png` | Output neuron spike timeline |
| `voltage_t_shape.png` | Membrane voltage integrating to threshold |
| `summary_t_shape.png` | All four panels in one figure |

---

## Running the tests

```bash
cd t_shape_detector
pytest tests/ -v
```

Without NEST installed:
```
49 passed, 17 skipped   ← NEST tests skip gracefully
```

With NEST installed:
```
66 passed               ← all tests including live simulation tests
```

---

## How detection works — the physics

The output (LIF) neuron integrates incoming synaptic currents.  When the membrane
voltage exceeds the threshold, it fires and resets.

**Weight calibration** (see `network.py` for the derivation):

```
ΔV_peak per synapse ≈ 0.63 mV     (w=0.05 nA, cm=0.25 nF, τ_m=20 ms, τ_syn=5 ms)

Full T-shape  : 46 pixels × 0.63 mV = 29.0 mV  → fires  (gap = 20 mV)  ✓
Random image  : 12 pixels × 0.63 mV =  7.6 mV  → silent                ✓
```

Only the **46 T-shape pixels** have synapses to the output neuron.  All other
pixels have no connection.  This makes the output neuron a *template matcher*:
it counts how much the input pattern overlaps with the stored T template.

---

## Why is this event-driven?

In a rate-coded ANN, every neuron outputs a continuous activation at every
timestep.  In this SNN:

- **Inactive pixels transmit nothing** — no event, no energy.
- **Active pixels transmit exactly one spike** — a single timestamp.
- The output neuron only "wakes up" when it receives enough input events.

On SpiNNaker hardware, this translates directly to zero power for silent neurons.

---

## Migration to SpiNNaker

To run this exact network on SpiNNaker hardware, change **one import line**
in `network.py`:

```python
# Current (NEST software simulation)
import pyNN.nest as sim

# SpiNNaker neuromorphic hardware
import pyNN.spiNNaker as sim
```

Everything else — neuron parameters, projections, recording API — is identical.
This is the key value of the PyNN abstraction layer.

Additional SpiNNaker steps:
1. `pip install sPyNNaker`
2. Provide a `spynnaker.cfg` pointing at your SpiNNaker board
3. Reduce `sim_time` and `num_neurons` if needed for board memory limits

---

## Future extensions

| Idea | Difficulty |
|---|---|
| Rate coding (N spikes proportional to pixel brightness) | Easy |
| Time-to-first-spike encoding (bright → early spike) | Easy |
| Multiple output neurons (detect T, L, +, etc.) | Medium |
| Shifted/rotated T shapes | Medium |
| Inhibitory non-T connections (cleaner noise rejection) | Medium |
| STDP learning (learn the T template from examples) | Hard |
| Run on real SpiNNaker board | Medium (hardware access needed) |
