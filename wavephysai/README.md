# WavePhysAI 🧠⚡

**Physical Wave Neuromorphic Computing for Full-Body Humanoid Control**

> *Wave interference replaces digital multiply-accumulate operations entirely.*
> *Constructive → EPSP (excitatory). Destructive (π phase) → IPSP (inhibitory).*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-blue)](https://numpy.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Key Performance Targets

| Metric | Value | vs. Digital NPU |
|--------|-------|-----------------|
| Inference complexity | **O(1)** | O(N²) |
| Reflex latency | **< 5 ms** | > 50 ms |
| Synaptic energy | **attojoule** | femtojoule–picojoule |
| Power consumption | **< 1/100×** | baseline |

---

## Architecture

```
[Sensory Input]
      ↓
[Wave Encoder — Phase/Amplitude]
      ↓
[Spinal Cord Wave Field — FDTD]     ← constructive/destructive interference
      ↓
[Ganglion Layer — Threshold Gates]   ← wave + threshold = neuron
      ↓
[Peripheral Nerve Bundles]
   Brachial plexus → Arms
   Sciatic nerve   → Legs
      ↓
[Motor Output — Joint Control]
      ↑
[GR00T Interface — High-level Planning] (async)
```

---

## Repository Structure

```
wavephysai/
│
├── core/
│   ├── wave_field.py          # FDTD 2D wave engine (NumPy + Numba)
│   ├── interference.py        # Constructive/destructive XOR gate
│   └── cavity.py              # Resonant cavity (Fabry-Pérot model)
│
├── simulation/
│   ├── spinal_cord.py         # Spinal cord neural field (Amari equation)
│   ├── cpg.py                 # Phase-Coded Central Pattern Generator
│   ├── ganglion.py            # Ganglion threshold + reflex loop
│   └── plexus.py              # Brachial / sciatic nerve branching
│
├── control/
│   ├── humanoid_mapping.py    # Wave energy → joint angle mapping
│   ├── groot_interface.py     # GR00T async high-level planning bridge
│   ├── wave_params_packet.py  # 32-byte real-time comm protocol
│   └── phase_tracker.py       # Von Mises + Particle Filter
│
├── synapse/
│   ├── hbn_memristor.py       # hBN multilayer memristor model
│   ├── moire_synapse.py       # 23.5° twist moiré quantum synapse
│   └── stdp.py                # STDP / LTP / LTD plasticity rules
│
├── utils/
│   ├── visualize.py           # Wave field + joint angle plots
│   └── metrics.py             # RMSE, SNR, energy, latency
│
├── examples/
│   ├── 01_xor_gate.py         # Wave XOR gate demo
│   ├── 02_spinal_reflex.py    # Spinal cord reflex loop
│   ├── 03_cpg_walking.py      # CPG gait generation
│   ├── 04_arm_control.py      # 3-DOF arm wave control
│   └── 05_moire_synapse.py    # Moiré conductance simulation
│
├── docs/
│   └── equations.tex          # Full LaTeX equation reference
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

```bash
git clone https://github.com/yourname/wavephysai
cd wavephysai
pip install -r requirements.txt

# Run wave XOR gate demo
python examples/01_xor_gate.py

# Run spinal cord reflex
python examples/02_spinal_reflex.py

# Run CPG gait generation
python examples/03_cpg_walking.py
```

---

## Installation

```bash
pip install numpy scipy matplotlib numba torch
# optional: jwave (for differentiable acoustic simulation)
pip install jwave
```

---

## Core Concept: Wave = Computation = Control

```
Wave phenomenon     →   Robotic action
─────────────────────────────────────────
Constructive        →   Muscle contraction (EPSP)
Destructive (π)     →   Inhibition (IPSP)
Phase shift         →   Direction control
Delay               →   Timing
Amplitude           →   Force/torque magnitude
```

---

## Citation

```bibtex
@article{chin2026wavephysai,
  title   = {WavePhysAI: A Wave-Interference Neuromorphic Architecture
             for Full-Body Distributed Humanoid Control},
  author  = {Chin, Chur},
  year    = {2026},
  note    = {Preprint. Dong Eui Medical Center, Busan, Republic of Korea}
}
```

---

## License

MIT License — See [LICENSE](LICENSE)
