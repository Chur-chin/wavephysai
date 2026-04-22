"""
wavephysai/control/phase_tracker.py
─────────────────────────────────────
Real-Time Phase Drift Compensation for Wave Computer.

Two trackers:
    1. Von Mises Recursive Filter  (fast, GPU-friendly)
    2. Particle Filter / SMC       (robust, non-linear)

Physics / Statistics
────────────────────
State:  θₜ ∈ (−π, π]   (wave phase)
Process:  θₜ = θₜ₋₁ + wₜ,   wₜ ~ N(0, σ_w²)
Observation:  yₜ = A·e^{iθₜ} + nₜ,   nₜ ~ CN(0, σ_r²)

Pilot summary statistic:
    zₜ = Σₖ yₖ,ₜ · ȳₖ   (matched filter output)

Von Mises update (Eq. paper §6.2):
    V_prior = κ_{t|t-1} · e^{iμ_{t|t-1}}
    V_obs   = (2/σ_r²) · zₜ
    V_post  = V_prior + V_obs
    μₜ = ∠V_post,   κₜ = |V_post|

Prediction (moment-matching, §6.3):
    A₁(κ) = I₁(κ)/I₀(κ)
    κ_{t|t-1} = A₁⁻¹(A₁(κ_{t-1}) · exp(−σ_w²/2))

CRLB lower bound (§6.1):
    Var(θ̂) ≥ 1 / (2 · SNR · K)
"""

import numpy as np
from scipy.special import iv as bessel_i   # modified Bessel I_ν


# ─── Utility ──────────────────────────────────────────────────────────────────

def _wrap(theta):
    """Wrap angle to (−π, π]."""
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def _bessel_ratio(kappa):
    """A₁(κ) = I₁(κ) / I₀(κ)."""
    kappa = np.asarray(kappa, dtype=float)
    kappa = np.clip(kappa, 1e-10, 1e6)
    return bessel_i(1, kappa) / bessel_i(0, kappa)


def _inv_bessel_ratio(r, n_iter=50):
    """
    Numerically invert A₁: given r ∈ [0,1), find κ s.t. A₁(κ) = r.
    Uses Newton-Raphson on A₁(κ) − r = 0.
    """
    r = np.asarray(r, dtype=float)
    r = np.clip(r, 1e-8, 1 - 1e-8)

    # Initial guess (Banerjee 2005 approximation)
    kappa = r * (2 - r**2) / (1 - r**2)
    kappa = np.clip(kappa, 1e-6, 1e6)

    for _ in range(n_iter):
        a1   = _bessel_ratio(kappa)
        # Derivative: dA₁/dκ = 1 − A₁² − A₁/κ
        da1  = 1.0 - a1**2 - a1 / (kappa + 1e-15)
        step = (a1 - r) / (da1 + 1e-15)
        kappa = np.clip(kappa - step, 1e-6, 1e6)

    return kappa


# ─── Von Mises Recursive Filter ──────────────────────────────────────────────

class VonMisesFilter:
    """
    Von Mises recursive phase filter with moment-matching prediction.

    Parameters
    ----------
    sigma_r   : float — observation noise std (rad-equivalent)
    sigma_w   : float — process noise std (rad) — phase diffusion speed
    kappa_init: float — initial concentration (1/uncertainty)
    mu_init   : float — initial mean phase (rad)
    """

    def __init__(self, sigma_r: float = 0.3,
                 sigma_w: float = 0.05,
                 kappa_init: float = 0.1,
                 mu_init: float = 0.0):
        self.sigma_r  = sigma_r
        self.sigma_w  = sigma_w
        self.sigma_r2 = sigma_r ** 2
        self.sw2_half = 0.5 * sigma_w ** 2

        self.mu    = float(mu_init)
        self.kappa = float(kappa_init)

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self):
        """Moment-matching Von Mises prediction step."""
        r       = _bessel_ratio(self.kappa) * np.exp(-self.sw2_half)
        r       = float(np.clip(r, 1e-8, 1 - 1e-8))
        self.kappa = float(_inv_bessel_ratio(r))
        # mean unchanged under zero-mean process noise

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, z: complex):
        """
        Update with pilot summary statistic z = Σ yₖ·s̄ₖ.

        Parameters
        ----------
        z : complex — matched filter output
        """
        V_prior = self.kappa * np.exp(1j * self.mu)
        V_obs   = (2.0 / self.sigma_r2) * z
        V_post  = V_prior + V_obs

        self.mu    = float(np.angle(V_post))
        self.kappa = float(np.abs(V_post))

    # ── Full step ─────────────────────────────────────────────────────────────

    def step(self, z: complex):
        """predict + update."""
        self.predict()
        self.update(z)
        return self.mu

    # ── Antiphase pre-emphasis ────────────────────────────────────────────────

    def compensate(self, signal: complex) -> complex:
        """
        Apply antiphase pre-emphasis:
            ψ_pre = ψ · e^{−iη̂}
        where η̂ = estimated drift = self.mu
        """
        return signal * np.exp(-1j * self.mu)


# ─── Particle Filter ─────────────────────────────────────────────────────────

class ParticleFilter:
    """
    Sequential Monte Carlo phase tracker.

    Robust to multi-modal distributions, non-Gaussian noise,
    and phase jumps.

    Parameters
    ----------
    n_particles : int   — number of particles
    sigma_r     : float — observation noise (rad-equivalent)
    sigma_w     : float — process noise (rad)
    """

    def __init__(self, n_particles: int = 500,
                 sigma_r: float = 0.3,
                 sigma_w: float = 0.05):
        self.N        = n_particles
        self.sigma_r  = sigma_r
        self.sigma_r2 = sigma_r ** 2
        self.sigma_w  = sigma_w

        # Particles: uniform init over (−π, π]
        self.particles = np.random.uniform(-np.pi, np.pi, n_particles)
        self.weights   = np.ones(n_particles) / n_particles

    # ── Propagation ──────────────────────────────────────────────────────────

    def propagate(self):
        """Sample process noise and propagate particles."""
        self.particles = _wrap(
            self.particles + np.random.normal(0, self.sigma_w, self.N)
        )

    # ── Weighting ─────────────────────────────────────────────────────────────

    def weight(self, z: complex):
        """
        Update particle weights with log-likelihood.

        Log-weight ∝ Re{ e^{−iθ⁽ⁱ⁾} · z } / (σ_r²/2)
        """
        log_w = (2.0 / self.sigma_r2) * np.real(
            np.exp(-1j * self.particles) * z
        )
        log_w -= log_w.max()           # numerical stability
        w = np.exp(log_w)
        self.weights = w / w.sum()

    # ── Resampling ────────────────────────────────────────────────────────────

    def resample(self, ess_threshold: float = 0.5):
        """Systematic resampling when ESS < threshold × N."""
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < ess_threshold * self.N:
            cumsum = np.cumsum(self.weights)
            cumsum[-1] = 1.0  # fix numerical drift
            u0 = np.random.uniform(0, 1.0 / self.N)
            positions = u0 + np.arange(self.N) / self.N
            indices = np.searchsorted(cumsum, positions)
            indices = np.clip(indices, 0, self.N - 1)
            self.particles = self.particles[indices]
            self.weights   = np.ones(self.N) / self.N

    # ── Estimate ─────────────────────────────────────────────────────────────

    def estimate(self) -> float:
        """Circular mean estimate: μ̂ = ∠ Σ wᵢ e^{iθᵢ}."""
        return float(np.angle(np.dot(self.weights,
                                      np.exp(1j * self.particles))))

    # ── Full step ─────────────────────────────────────────────────────────────

    def step(self, z: complex) -> float:
        """propagate → weight → resample → estimate."""
        self.propagate()
        self.weight(z)
        self.resample()
        return self.estimate()

    def compensate(self, signal: complex) -> complex:
        """Antiphase pre-emphasis using PF estimate."""
        return signal * np.exp(-1j * self.estimate())


# ─── CRLB ────────────────────────────────────────────────────────────────────

def phase_crlb(snr: float, K: int = 1) -> float:
    """
    Cramér–Rao Lower Bound on phase estimation variance.

        Var(θ̂) ≥ 1 / (2 · SNR · K)

    Parameters
    ----------
    snr : float — signal-to-noise ratio (linear, A²/σ_r²)
    K   : int   — number of pilot symbols

    Returns
    -------
    float : minimum variance (rad²)
    """
    return 1.0 / (2.0 * snr * K)


def phase_rmse_crlb(snr_db_array, K: int = 8):
    """Return CRLB RMSE curve over SNR range (dB)."""
    snr_lin = 10.0 ** (np.asarray(snr_db_array) / 10.0)
    return np.sqrt(phase_crlb(snr_lin, K))
