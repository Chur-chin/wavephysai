"""
wavephysai/synapse/moire_synapse.py
─────────────────────────────────────
23.5° Twist Graphene/hBN Moiré Synapse — Quantum Conductance Model.

Physics
-------
Moiré potential (3-fold symmetric, §7.1):
    V(r) = V₀ · Σⱼ₌₁³ cos(Gⱼ·r)
    |G| = (4π/3a) · θ   (a = graphene lattice constant ≈ 0.246 nm)

1D plane-wave Hamiltonian (§7.2):
    H_{G,G'}(k) = ℏ²(k+G)²/(2m*) · δ_{GG'} + V_{G-G'}

Landauer–Büttiker conductance (§7.3):
    G(E_F) = G₀ · T(E_F),   G₀ = 2e²/h
    T(E) ≈ Σᵢ Γᵢ² / [(E−Eᵢ)² + Γᵢ²]   (Lorentzian miniband peaks)

Gate-voltage → Fermi energy (§7.4):
    n(Vg) = C_g(Vg − V₀)/e
    E_F(Vg) = ℏvF √(π|n|) · sgn(n)

Synaptic weight (§7.5):
    w(Vg) = [G(Vg) − G_min] / [G_max − G_min]   ∈ [0, 1]

Update energy (§7.6, ideal gate):
    E_update ≈ ½ C_g ΔVg²
    For C_g ~ 10 aF, ΔVg ~ 0.1V → E ~ 5×10⁻²⁰ J  (attojoule regime)
"""

import numpy as np
from scipy.integrate import quad
from scipy.special  import iv as bessel_i


# ─── Physical constants ───────────────────────────────────────────────────────

HBAR    = 1.0545718e-34   # J·s
E_CHARGE= 1.60218e-19     # C
G0      = 2 * E_CHARGE**2 / (2 * np.pi * HBAR)  # quantum of conductance (S)
KB      = 1.38065e-23     # J/K
VF      = 1e6             # m/s — Dirac fermion velocity in graphene


# ─── Moiré potential ──────────────────────────────────────────────────────────

def moire_potential_1d(x_array: np.ndarray,
                       V0: float = 0.02,
                       L_moire: float = 10e-9) -> np.ndarray:
    """
    1D moiré scalar potential (simplified, single harmonic).

        V(x) = V₀ · cos(G_m · x),   G_m = 2π/L_m

    Parameters
    ----------
    x_array  : positions (m)
    V0       : potential amplitude (eV)
    L_moire  : moiré period (m)  [~10 nm for 1.4° twist, ~0.6 nm for 23.5°]

    Returns
    -------
    V : np.ndarray (eV)
    """
    Gm = 2 * np.pi / L_moire
    return V0 * np.cos(Gm * x_array)


# ─── Miniband calculation (plane-wave diagonalisation) ───────────────────────

class MoireMiniband:
    """
    1D plane-wave miniband calculator for moiré superlattice.

    Computes eigenvalues E_n(k) and constructs DOS + Landauer conductance.

    Parameters
    ----------
    V0      : float — moiré potential amplitude (eV)
    L_moire : float — moiré period (m)
    m_eff   : float — effective mass (kg); default = 0.02 m_e (graphene-like)
    N_G     : int   — number of plane-wave basis vectors (should be odd)
    N_k     : int   — k-points in first Brillouin zone
    Gamma   : float — Lorentzian broadening (eV) for DOS / conductance
    T_K     : float — temperature (K)
    """

    def __init__(self, V0=0.02, L_moire=10e-9,
                 m_eff=None, N_G=51, N_k=101,
                 Gamma=0.005, T_K=300.0):
        self.V0      = V0           # eV
        self.L_moire = L_moire
        self.m_eff   = m_eff if m_eff is not None else 0.02 * 9.109e-31
        self.N_G     = N_G
        self.N_k     = N_k
        self.Gamma   = Gamma        # eV
        self.T_K     = T_K

        self.Gm      = 2 * np.pi / L_moire
        self.E_scale = (HBAR * self.Gm)**2 / (2 * self.m_eff * E_CHARGE)  # eV

        # Reciprocal lattice vectors
        half = N_G // 2
        self.Gvec = np.arange(-half, half + 1) * self.Gm  # m⁻¹

        # k grid (first BZ)
        self.k_grid = np.linspace(-self.Gm / 2, self.Gm / 2, N_k)

        # Run diagonalisation
        self._eigenvalues = None  # shape (N_k, N_G)
        self._dos         = None
        self._E_grid      = None
        self._G_of_E      = None

    def _build_hamiltonian(self, k: float) -> np.ndarray:
        """Build (N_G × N_G) Hamiltonian at k-point."""
        H = np.zeros((self.N_G, self.N_G))
        # Kinetic diagonal
        for i, G in enumerate(self.Gvec):
            H[i, i] = self.E_scale * ((k + G) / self.Gm) ** 2
        # Potential off-diagonal (V_{G-G'} = V₀/2 for |G−G'| = 1)
        for i in range(self.N_G - 1):
            H[i, i+1] = H[i+1, i] = self.V0 / 2.0
        return H

    def compute_bands(self):
        """Diagonalise Hamiltonian for all k-points."""
        evals = np.zeros((self.N_k, self.N_G))
        for ki, k in enumerate(self.k_grid):
            H = self._build_hamiltonian(k)
            evals[ki] = np.linalg.eigvalsh(H)
        self._eigenvalues = np.sort(evals, axis=1)
        return self._eigenvalues

    def compute_dos(self, E_min=-0.2, E_max=0.4, n_pts=500):
        """
        Compute DOS via Lorentzian broadening.

            DOS(E) = (1/N_k) · Σ_{n,k} Γ/π / [(E − E_n(k))² + Γ²]
        """
        if self._eigenvalues is None:
            self.compute_bands()

        self._E_grid = np.linspace(E_min, E_max, n_pts)
        dos = np.zeros(n_pts)
        Gamma = self.Gamma

        for E, erow in zip(self._E_grid,
                           np.broadcast_to(self._E_grid[:, None, None],
                                           (n_pts,) + self._eigenvalues.shape)):
            _ = E  # unused in loop below
        # vectorised
        E_arr = self._E_grid[:, None, None]          # (n_pts, 1, 1)
        Evals = self._eigenvalues[None, :, :]        # (1, N_k, N_G)
        dos = np.mean(
            (Gamma / np.pi) / ((E_arr - Evals)**2 + Gamma**2),
            axis=(1, 2)
        )
        self._dos = dos
        return self._E_grid, dos

    def compute_conductance(self):
        """
        Compute G(E) = G₀ · T(E) where T(E) ∝ normalised DOS.

            G(E) = G₀ · DOS(E) / DOS_max
        """
        if self._dos is None:
            self.compute_dos()
        dos_norm = self._dos / (self._dos.max() + 1e-30)
        self._G_of_E = dos_norm  # normalised (in units of G₀)
        return self._E_grid, self._G_of_E

    def gate_to_ef(self, Vg: float, Cg: float = 1e-2,
                   V0_offset: float = 0.0) -> float:
        """
        Convert gate voltage to Fermi energy (eV).

            E_F = ℏvF √(π|n|) · sgn(n)
            n = Cg(Vg − V0)/e
        """
        n = Cg * (Vg - V0_offset) / E_CHARGE  # m⁻²  (sheet density)
        if n == 0:
            return 0.0
        E_F_J = np.sign(n) * HBAR * VF * np.sqrt(np.pi * abs(n))
        return float(E_F_J / E_CHARGE)  # → eV

    def conductance_at_Vg(self, Vg: float, Cg: float = 1e-2,
                          V0_offset: float = 0.0) -> float:
        """
        Compute normalised conductance G(Vg).

            G(Vg) = ∫ G(E) · (−∂f/∂E) dE

        where f is the Fermi–Dirac distribution.
        """
        if self._G_of_E is None:
            self.compute_conductance()

        E_F = self.gate_to_ef(Vg, Cg, V0_offset)
        kT  = KB * self.T_K / E_CHARGE  # eV

        # Fermi derivative: −∂f/∂E = sech²((E−E_F)/(2kT)) / (4kT)
        x = (self._E_grid - E_F) / (2 * kT + 1e-12)
        fp = (1.0 / (4 * kT)) / (np.cosh(x) ** 2 + 1e-30)

        G = float(np.trapezoid(self._G_of_E * fp, self._E_grid))
        return max(G, 0.0)

    def synaptic_weight(self, Vg: float, Cg: float = 1e-2,
                        Vg_range: tuple = (-3.0, 3.0)) -> float:
        """
        Compute normalised synaptic weight w ∈ [0, 1].

            w(Vg) = [G(Vg) − G_min] / [G_max − G_min]
        """
        G_min = self.conductance_at_Vg(Vg_range[0], Cg)
        G_max = self.conductance_at_Vg(Vg_range[1], Cg)
        G     = self.conductance_at_Vg(Vg, Cg)
        denom = G_max - G_min + 1e-15
        return float(np.clip((G - G_min) / denom, 0.0, 1.0))

    def update_energy(self, delta_Vg: float, Cg: float = 10e-18) -> float:
        """
        Ideal gate-charging energy for a weight update.

            E_update = ½ · Cg · ΔVg²  (Joules)

        Default Cg = 10 aF (atomically thin gate geometry).
        """
        return 0.5 * Cg * delta_Vg ** 2


# ─── hBN Memristor model ──────────────────────────────────────────────────────

class HBNMemristor:
    """
    Simplified hBN multilayer memristor model.

    Stack:  Bottom electrode (Au/Ti 50nm)
            hBN switching layer (CVD, 8–18 layers, ~8–10 nm)
            Top electrode (Ti/Ag 50nm)
            Flexible PET/PI substrate

    Plasticity rules: STDP, LTP, LTD.

    Parameters
    ----------
    n_layers   : int   — number of hBN layers (8–18)
    R_HRS      : float — high resistance state (Ω)
    R_LRS      : float — low resistance state (Ω)
    E_switch   : float — switching energy (J) [default ~attojoule]
    """

    def __init__(self, n_layers=12, R_HRS=1e6, R_LRS=1e3,
                 E_switch=1e-18):
        self.n      = n_layers
        self.R_HRS  = R_HRS
        self.R_LRS  = R_LRS
        self.E_sw   = E_switch

        self.state  = 0.0   # 0 = HRS, 1 = LRS, continuous in [0,1]
        self._ltp_window = 20e-3   # s — LTP time window
        self._ltd_window = 20e-3   # s
        self._last_pre  = -np.inf
        self._last_post = -np.inf

    @property
    def resistance(self):
        """Interpolated resistance between HRS and LRS."""
        return self.R_HRS * (1 - self.state) + self.R_LRS * self.state

    @property
    def conductance(self):
        return 1.0 / (self.resistance + 1e-15)

    def apply_voltage(self, V: float, pulse_width: float = 1e-6):
        """Apply voltage pulse; update state via simple threshold."""
        V_set   = 1.5   # V — set threshold
        V_reset = -1.5  # V — reset threshold
        if V > V_set:
            self.state = min(1.0, self.state + 0.1)
        elif V < V_reset:
            self.state = max(0.0, self.state - 0.1)

    def stdp_update(self, t_pre: float, t_post: float,
                    A_plus=0.05, A_minus=0.05):
        """
        Spike-Timing Dependent Plasticity (STDP).

        Δw = A₊ · exp(−|Δt|/τ₊)  if Δt > 0  (LTP)
           = −A₋ · exp(−|Δt|/τ₋) if Δt < 0  (LTD)
        """
        dt = t_post - t_pre
        if dt > 0:
            dw = A_plus  * np.exp(-dt / self._ltp_window)
        else:
            dw = -A_minus * np.exp(abs(dt) / self._ltd_window)
        self.state = float(np.clip(self.state + dw, 0.0, 1.0))
        return dw

    def energy_per_switch(self):
        """Estimated energy for a full HRS→LRS transition."""
        return self.E_sw
