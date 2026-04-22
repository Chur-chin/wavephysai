"""
wavephysai/core/interference.py
────────────────────────────────
Wave-Based Interference Logic Gates.

Physics
-------
Output intensity (Eq. 1 in paper):
    I_out = |A·e^{iφ₁} + B·e^{iφ₂}|²

XOR truth table via phase control:
    A=0, B=0  →  no sources       →  I=0   → OUT=0
    A=1, B=0  →  φ₁=0, single     →  I=A²  → OUT=1
    A=0, B=1  →  φ₂=0, single     →  I=B²  → OUT=1
    A=1, B=1  →  φ₁=0, φ₂=π      →  I≈0   → OUT=0  (destructive)

Key insight: π phase shift → destructive interference → IPSP (inhibitory)
             0 phase shift → constructive interference → EPSP (excitatory)
"""

import numpy as np
from .wave_field import WaveField2D


# ─── Analytical interference (no FDTD) ───────────────────────────────────────

def interference_intensity(A: float, phi1: float,
                            B: float, phi2: float) -> float:
    """
    Compute output intensity from two wave sources.

        I = |A·e^{iφ₁} + B·e^{iφ₂}|²

    Parameters
    ----------
    A, B     : amplitudes (≥ 0)
    phi1/2   : phases (radians)

    Returns
    -------
    I_out : float ≥ 0
    """
    phasor = A * np.exp(1j * phi1) + B * np.exp(1j * phi2)
    return float(np.abs(phasor) ** 2)


def wave_xor(a: int, b: int,
             amp: float = 1.0,
             threshold: float = 0.5) -> int:
    """
    Analytical wave XOR gate.

    Parameters
    ----------
    a, b      : binary inputs (0 or 1)
    amp       : source amplitude
    threshold : decision threshold on I_out

    Returns
    -------
    int : 0 or 1
    """
    if a == 0 and b == 0:
        return 0
    elif a == 1 and b == 0:
        I = interference_intensity(amp, 0.0, 0.0, 0.0)
    elif a == 0 and b == 1:
        I = interference_intensity(0.0, 0.0, amp, 0.0)
    else:  # a=1, b=1 → destructive: φ₂ = π
        I = interference_intensity(amp, 0.0, amp, np.pi)
    return int(I > threshold * amp ** 2)


def phase_sweep_xor(n_points: int = 360):
    """
    Sweep phase difference Δφ from 0 to 2π and return I_out.
    Demonstrates destructive (Δφ=π) and constructive (Δφ=0, 2π) interference.

    Returns
    -------
    delta_phi : np.ndarray  shape (n_points,)
    I_out     : np.ndarray  shape (n_points,)
    """
    delta_phi = np.linspace(0, 2 * np.pi, n_points)
    I_out = np.array([
        interference_intensity(1.0, 0.0, 1.0, dp)
        for dp in delta_phi
    ])
    return delta_phi, I_out


# ─── FDTD-based interference (full wave simulation) ──────────────────────────

class WaveXORGate:
    """
    Full FDTD wave XOR gate simulation.

    Two point sources separated on a 2D grid; a probe at the
    intersection reads the interference output.

    Parameters
    ----------
    grid_size   : int   — grid cells per side
    freq        : float — source frequency (Hz)
    c           : float — wave velocity (m/s)
    dx          : float — spatial resolution (m)
    n_steps     : int   — time steps per evaluation
    """

    def __init__(self, grid_size=128, freq=5000.0, c=340.0,
                 dx=5e-4, n_steps=400):
        self.grid_size = grid_size
        self.freq = freq
        self.c = c
        self.dx = dx
        self.n_steps = n_steps

        # Source and probe positions
        N = grid_size
        self.src_A_pos = (N // 4,     N // 2)
        self.src_B_pos = (3 * N // 4, N // 2)
        self.probe_pos = (N // 2,     N // 2)

    def evaluate(self, a: int, b: int) -> dict:
        """
        Run FDTD simulation for given binary inputs.

        Returns
        -------
        dict with keys: 'output', 'I_out', 'field_snapshot'
        """
        wf = WaveField2D(
            Nx=self.grid_size, Ny=self.grid_size,
            dx=self.dx, c=self.c
        )

        # Add sources based on inputs
        if a == 1:
            ia, ja = self.src_A_pos
            wf.add_sinusoidal_source(ia, ja, self.freq, amp=1.0, phase=0.0)
        if b == 1:
            ib, jb = self.src_B_pos
            phase_B = np.pi if (a == 1 and b == 1) else 0.0
            wf.add_sinusoidal_source(ib, jb, self.freq, amp=1.0,
                                     phase=phase_B)

        # Run
        wf.run(self.n_steps, record_every=self.n_steps)

        # Sample probe energy (average over last quarter)
        energies = []
        dt = wf.dt
        T = 1.0 / self.freq
        wf2 = WaveField2D(Nx=self.grid_size, Ny=self.grid_size,
                          dx=self.dx, c=self.c)
        if a == 1:
            ia, ja = self.src_A_pos
            wf2.add_sinusoidal_source(ia, ja, self.freq, amp=1.0, phase=0.0)
        if b == 1:
            ib, jb = self.src_B_pos
            phase_B = np.pi if (a == 1 and b == 1) else 0.0
            wf2.add_sinusoidal_source(ib, jb, self.freq, amp=1.0,
                                      phase=phase_B)
        for _ in range(self.n_steps):
            wf2.step()
        for _ in range(self.n_steps // 4):
            wf2.step()
            pi, pj = self.probe_pos
            energies.append(wf2.energy_at(pi, pj))

        I_out = float(np.mean(energies))
        threshold = 0.1
        output = int(I_out > threshold)

        return {
            "output": output,
            "I_out": I_out,
            "field_snapshot": wf.snapshot(),
            "a": a, "b": b
        }

    def full_truth_table(self):
        """Evaluate all 4 input combinations."""
        results = []
        for a in [0, 1]:
            for b in [0, 1]:
                r = self.evaluate(a, b)
                results.append(r)
        return results


# ─── Multi-ganglion network ───────────────────────────────────────────────────

class GanglionNetwork:
    """
    Network of ganglion threshold nodes reading from a shared wave field.

    Each ganglion:
        O_i = σ(I_i − θ_i)
    where I_i = wave energy at node position, θ_i = threshold,
    σ = sigmoid or Heaviside.

    Biological mapping:
        ganglion → cavity node
        synapse  → impedance barrier (hBN)
        firing   → energy threshold crossing
    """

    def __init__(self, positions, thresholds=None, activation="heaviside"):
        """
        Parameters
        ----------
        positions   : list of (i,j) grid positions
        thresholds  : list of floats (default: 0.05 each)
        activation  : "heaviside" | "sigmoid"
        """
        self.positions = positions
        self.n = len(positions)
        self.thresholds = thresholds if thresholds is not None \
            else [0.05] * self.n
        self.activation = activation

        # Labels for biological plexus mapping
        _labels = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle",
                   "torso_1", "torso_2", "torso_3"]
        self.labels = _labels[:self.n]

    def _activate(self, I, theta):
        if self.activation == "sigmoid":
            return float(1.0 / (1.0 + np.exp(-(I - theta) * 50.0)))
        else:  # heaviside
            return 1.0 if I >= theta else 0.0

    def read(self, wave_field: WaveField2D):
        """
        Read ganglion outputs from current wave field state.

        Returns
        -------
        outputs : dict {label: float}
        """
        outputs = {}
        for k, (pi, pj) in enumerate(self.positions):
            I = wave_field.energy_at(pi, pj)
            O = self._activate(I, self.thresholds[k])
            lbl = self.labels[k] if k < len(self.labels) else f"node_{k}"
            outputs[lbl] = O
        return outputs
