"""
wavephysai/simulation/spinal_cord.py
──────────────────────────────────────
Spinal Cord Wave Field + CPG + Ganglion Reflex Loop.

Models
------
1. Amari-type Neural Field (spinal cord):
       ∂u/∂t = -u + ∫ w(x−x') · σ(u(x',t)) dx' + I_ext(x,t)

2. Phase-Coded CPG (oscillator network):
       dφᵢ/dt = ωᵢ + Σⱼ κᵢⱼ · sin(φⱼ − φᵢ − ψᵢⱼ)   [Kuramoto]

3. Autonomic drift layer:
       φ(t) = φ₀ + ω·t + Δφ_auto(t)

4. Ganglion gate:
       O = σ(I − θ)   where I = ∫ p²(x,t)dt

5. Reflex loop (without brain):
       u[spinal_center] += β · O_ganglion
"""

import numpy as np


# ─── Neural Field (Amari 1D) ─────────────────────────────────────────────────

class SpinalCordField1D:
    """
    1D Amari-type neural field modelling the spinal cord.

    State: u(x,t) — neural activation along the cord (x ∈ [0, L])

    Equation (discretised, Euler):
        u[t+1] = u[t] + dt · (-u[t] + W*σ(u[t]) + I_ext)

    where W is the lateral connectivity kernel (Mexican hat).

    Parameters
    ----------
    N    : number of spatial nodes
    dx   : spatial step (m)
    dt   : time step (s)
    tau  : membrane time constant (s)
    """

    def __init__(self, N=256, dx=1e-3, dt=1e-3, tau=10e-3):
        self.N   = N
        self.dx  = dx
        self.dt  = dt
        self.tau = tau

        self.u   = np.zeros(N)   # activation
        self.t   = 0.0

        # Mexican-hat lateral kernel
        x = np.linspace(-N//2, N//2, N) * dx
        sigma_exc = 5 * dx
        sigma_inh = 15 * dx
        self.W = (1.5 * np.exp(-x**2 / (2*sigma_exc**2))
                  - 0.75 * np.exp(-x**2 / (2*sigma_inh**2)))

        # FFT of kernel for fast convolution
        self._W_fft = np.fft.rfft(np.fft.ifftshift(self.W))

    def _sigma(self, u, theta=0.0, beta=20.0):
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-beta * (u - theta)))

    def step(self, I_ext=None):
        """
        Advance one time step.

        Parameters
        ----------
        I_ext : np.ndarray shape (N,) or None — external input current
        """
        s = self._sigma(self.u)

        # Lateral interaction via FFT convolution
        interaction = np.fft.irfft(np.fft.rfft(s) * self._W_fft,
                                   n=self.N) * self.dx

        if I_ext is None:
            I_ext = np.zeros(self.N)

        du = (1.0 / self.tau) * (-self.u + interaction + I_ext)
        self.u = self.u + self.dt * du
        self.t += self.dt

    def run(self, n_steps, I_ext_func=None):
        """
        Run n_steps.

        Parameters
        ----------
        I_ext_func : callable(t) -> np.ndarray(N) or None

        Returns
        -------
        history : np.ndarray shape (n_steps, N)
        """
        history = np.zeros((n_steps, self.N))
        for k in range(n_steps):
            I_ext = I_ext_func(self.t) if I_ext_func else None
            self.step(I_ext)
            history[k] = self.u.copy()
        return history

    def activation_energy(self, slice_start=None, slice_end=None):
        """Total activation energy in a cord slice."""
        u = self.u[slice_start:slice_end]
        return float(np.sum(u ** 2))


# ─── Phase-Coded CPG (Kuramoto oscillators) ──────────────────────────────────

class WaveCPG:
    """
    Phase-Coded Central Pattern Generator.

    N Kuramoto oscillators with coupling:
        dφᵢ/dt = ωᵢ + Σⱼ κᵢⱼ · sin(φⱼ − φᵢ − ψᵢⱼ)

    For walking/running rhythm:
        - 4 oscillators (Left/Right Hip, Left/Right Knee)
        - Anti-phase coupling between contralateral pairs
        - In-phase coupling between ipsilateral pairs

    Parameters
    ----------
    n_osc  : number of oscillators
    omega  : list of natural frequencies (rad/s)
    kappa  : coupling strength
    psi    : desired phase offsets (rad) [n_osc × n_osc]
    dt     : time step (s)
    """

    def __init__(self, n_osc=4, omega=None, kappa=5.0,
                 psi=None, dt=1e-3):
        self.n    = n_osc
        self.dt   = dt
        self.kappa = kappa

        if omega is None:
            # Walking: ~2 Hz ≈ 12.6 rad/s
            self.omega = np.ones(n_osc) * 2.0 * np.pi * 2.0
        else:
            self.omega = np.asarray(omega, dtype=float)

        # Phase offsets: anti-phase (π) between L/R
        if psi is None:
            self.psi = np.zeros((n_osc, n_osc))
            if n_osc == 4:
                # [LH, RH, LK, RK]
                self.psi[0, 1] = self.psi[1, 0] = np.pi  # LH-RH anti-phase
                self.psi[2, 3] = self.psi[3, 2] = np.pi  # LK-RK anti-phase
        else:
            self.psi = np.asarray(psi)

        self.phi = np.random.uniform(0, 2*np.pi, n_osc)  # initial phases
        self.t   = 0.0

        # Coupling matrix (all-to-all with kappa)
        self.K = kappa * (np.ones((n_osc, n_osc)) - np.eye(n_osc))

        # History
        self.phi_history = []
        self.output_history = []

    def step(self, speed: float = 1.0,
             haptic_reset: np.ndarray = None) -> np.ndarray:
        """
        Advance one CPG time step.

        Parameters
        ----------
        speed : float — scales ω (1.0=walk, 2.0=run, ~3.5=sprint)
        haptic_reset : np.ndarray shape (n_osc,) or None
            Phase reset signal from haptic feedback (Phase Response Curve).
            If provided, φᵢ ← φᵢ + Z(φᵢ)·haptic_reset[i]

        Returns
        -------
        outputs : np.ndarray shape (n_osc,) — sin(φᵢ) ∈ [−1, +1]
        """
        omega_eff = self.omega * speed

        # Kuramoto coupling
        dphi = omega_eff.copy()
        for i in range(self.n):
            for j in range(self.n):
                dphi[i] += self.K[i, j] * np.sin(
                    self.phi[j] - self.phi[i] - self.psi[i, j])

        self.phi = self.phi + self.dt * dphi

        # Phase reset from haptic (PRC: Z(φ) ≈ sin(φ))
        if haptic_reset is not None:
            Z = np.sin(self.phi)  # Phase Response Curve
            self.phi += Z * haptic_reset

        # Wrap to [-π, π]
        self.phi = ((self.phi + np.pi) % (2 * np.pi)) - np.pi
        self.t += self.dt

        outputs = np.sin(self.phi)
        self.phi_history.append(self.phi.copy())
        self.output_history.append(outputs.copy())
        return outputs

    def run(self, n_steps, speed=1.0):
        """
        Run n_steps at constant speed.

        Returns
        -------
        outputs : np.ndarray shape (n_steps, n_osc)
        """
        out = np.zeros((n_steps, self.n))
        for k in range(n_steps):
            out[k] = self.step(speed=speed)
        return out

    def gait_transition(self, n_steps, speed_func):
        """
        Run with time-varying speed (for bifurcation analysis).

        Parameters
        ----------
        speed_func : callable(t) -> float

        Returns
        -------
        outputs, speeds : np.ndarray
        """
        out    = np.zeros((n_steps, self.n))
        speeds = np.zeros(n_steps)
        for k in range(n_steps):
            s = float(speed_func(self.t))
            out[k] = self.step(speed=s)
            speeds[k] = s
        return out, speeds

    @property
    def synchrony_order(self):
        """Kuramoto order parameter R ∈ [0,1]: R=1 → full sync."""
        return float(np.abs(np.mean(np.exp(1j * self.phi))))


# ─── Ganglion Layer ───────────────────────────────────────────────────────────

class GanglionLayer:
    """
    Ganglion threshold gates connecting spinal wave field to motor output.

    Biological model:
        preganglionic  → input
        ganglion node  → threshold decision: O = σ(I − θ)
        postganglionic → gain / delay modulation

    Autonomic drift:
        φ(t) = φ₀ + ω·t + Δφ_auto
        Provides baseline tone independent of reflexes.

    Parameters
    ----------
    n_ganglia : number of ganglion nodes
    thresholds : array of θᵢ
    gains      : array of post-ganglionic gain κᵢ
    delays     : array of delay samples (int) for postganglionic path
    """

    def __init__(self, n_ganglia=6, thresholds=None, gains=None,
                 delays=None, dt=1e-3):
        self.n  = n_ganglia
        self.dt = dt

        self.thresholds = (np.ones(n_ganglia) * 0.05
                           if thresholds is None else np.asarray(thresholds))
        self.gains = (np.ones(n_ganglia)
                      if gains is None else np.asarray(gains))
        self.delays = (np.zeros(n_ganglia, dtype=int)
                       if delays is None else np.asarray(delays, dtype=int))

        max_delay = int(np.max(self.delays)) + 1
        self._buffer = np.zeros((max_delay, n_ganglia))
        self._buf_idx = 0

        # Autonomic drift
        self._auto_phase = np.zeros(n_ganglia)
        self._auto_omega = np.ones(n_ganglia) * 2 * np.pi * 0.1  # 0.1 Hz

        self.t = 0.0

        # Labels
        _bio = ["shoulder_R", "elbow_R", "wrist_R",
                "hip_R", "knee_R", "ankle_R"]
        self.labels = _bio[:n_ganglia]

    def _sigma(self, I, theta):
        return 1.0 / (1.0 + np.exp(-50.0 * (I - theta)))

    def step(self, inputs: np.ndarray,
             reflex_feedback: float = 0.0) -> np.ndarray:
        """
        Process one ganglion step.

        Parameters
        ----------
        inputs : np.ndarray shape (n,) — wave energy at each ganglion
        reflex_feedback : float — feedback signal to inject into field

        Returns
        -------
        outputs : np.ndarray shape (n,) — post-ganglionic signals
        """
        # Autonomic drift (slow phase modulation)
        self._auto_phase += self._auto_omega * self.dt
        auto_mod = 0.02 * np.sin(self._auto_phase)

        # Ganglion gate: O = σ(I + auto_mod − θ)
        raw_out = self._sigma(inputs + auto_mod, self.thresholds)

        # Post-ganglionic gain
        amplified = raw_out * self.gains

        # Delay line
        self._buffer[self._buf_idx % len(self._buffer)] = amplified
        outputs = np.zeros(self.n)
        for i in range(self.n):
            delayed_idx = (self._buf_idx - self.delays[i]) % len(self._buffer)
            outputs[i] = self._buffer[delayed_idx, i]

        self._buf_idx += 1
        self.t += self.dt
        return outputs

    def motor_commands(self, outputs: np.ndarray) -> dict:
        """Map ganglion outputs to labelled joint commands."""
        return {self.labels[i]: float(outputs[i])
                for i in range(self.n)}
