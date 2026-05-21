# event-driven-ai

**Event-driven AI experiments with Spiking Neural Networks (SNN), PyNN, and neuromorphic computing systems.**

---

## Examples

| # | Directory | Description | Backend |
|---|---|---|---|
| 01 | [`t_shape_detector/`](t_shape_detector/) | Minimal T-shape pattern detector using a single LIF neuron | PyNN + NEST → SpiNNaker |
| 02 | [`digit_classifier/`](digit_classifier/) | 10-class digit classifier (0–9) using normalised excitatory + inhibitory template weights and WTA | PyNN + NEST → SpiNNaker |

---

## What is this repo?

A collection of progressively more complex SNN experiments designed to teach:

- How pixels / sensors map to spike neurons
- How spike encoding works (binary, rate, TTFS)
- How LIF neurons integrate spikes and fire
- How PyNN abstracts over NEST (software) and SpiNNaker (neuromorphic hardware)
- How event-driven computation differs from conventional deep learning

Each example is self-contained with its own README, requirements, and test suite.

---

## Quick start (Example 01)

```bash
# Install NEST (see t_shape_detector/README.md for details)
conda install -c conda-forge nest-simulator

# Install Python dependencies
pip install -r t_shape_detector/requirements.txt

# Run simulation (produces plots in t_shape_detector/outputs/)
python t_shape_detector/main.py

# Run tests
cd t_shape_detector && pytest tests/ -v
```
