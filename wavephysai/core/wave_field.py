"""
wavephysai/core/wave_field.py
─────────────────────────────
2D FDTD Wave Engine with Numba JIT acceleration.

Physics
-------
Wave equation (2nd-order PDE):
    ∂²u/∂t² = c² ∇²u − α ∂u/∂t + f(x,y,t)

where:
    u(x,y,t)  : wave pressure / displacement field
    c          : phase velocity (m/s)
    α          : damping coefficient
    f          : external source term

Discretised (leap-frog FDTD):
    u[t+1] = (2 - α·dt) · u[t] - (1 - α·dt) · u[t-1]
           + (c·dt/dx)² · (u[t][i+1,j] + u[t][i-1,j]
                           + u[t][i,j+1] + u[t][i,j-1]
                           - 4·u[t][i,j])
           + dt² · f[i,j,t]

Stability condition (CFL):
    c · dt / dx ≤ 1/√2   (2D)
"""

import numpy as np
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):
        def decorator(fn): return fn
        return decorator
    def prange(n): return range(n)


# ─── Numba-accelerated time step ─────────────────────────────────────────────

@njit(cache=True)
def _step_numba(u_cur, u_prev, c2_dt2_dx2, alpha_dt, Nx, Ny):
    """Single FDTD time step (Numba JIT, interior only)."""
    u_next = np.empty_like(u_cur)
    for i in prange(1, Nx - 1):
        for j in range(1, Ny - 1):
            lap = (u_cur[i+1, j] + u_cur[i-1, j] +
                   u_cur[i, j+1] + u_cur[i, j-1] - 4.0 * u_cur[i, j])
            u_next[i, j] = ((2.0 - alpha_dt) * u_cur[i, j]
                            - (1.0 - alpha_dt) * u_prev[i, j]
                            + c2_dt2_dx2 * lap)
    # Dirichlet boundary conditions (absorbing)
    u_next[0, :] = 0.0
    u_next[-1, :] = 0.0
    u_next[:, 0] = 0.0
    u_next[:, -1] = 0.0
    return u_next


def _step_numpy(u_cur, u_prev, c2_dt2_dx2, alpha_dt):
    """Fallback NumPy step (no Numba)."""
    lap = (np.roll(u_cur, -1, axis=0) + np.roll(u_cur, 1, axis=0) +
           np.roll(u_cur, -1, axis=1) + np.roll(u_cur, 1, axis=1)
           - 4.0 * u_cur)
    u_next = ((2.0 - alpha_dt) * u_cur
              - (1.0 - alpha_dt) * u_prev
              + c2_dt2_dx2 * lap)
    u_next[0, :] = u_next[-1, :] = 0.0
    u_next[:, 0] = u_next[:, -1] = 0.0
    return u_next


# ─── WaveField class ─────────────────────────────────────────────────────────

class WaveField2D:
    """
    2D FDTD wave field for neuromorphic cavity simulation.

    Parameters
    ----------
    Nx, Ny   : grid size (cells)
    dx       : spatial resolution (m)
    dt       : time step (s); auto-computed if None (CFL: dt = 0.5*dx/c)
    c        : wave phase velocity (m/s)  [default: acoustic ~340 m/s]
    alpha    : damping coefficient (1/s)
    use_numba: use Numba JIT acceleration if available
    """

    def __init__(self, Nx=256, Ny=256, dx=1e-3, dt=None, c=340.0,
                 alpha=0.0, use_numba=True):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.c  = c
        self.alpha = alpha

        # CFL-stable dt
        self.dt = dt if dt is not None else 0.45 * dx / (c * np.sqrt(2))

        # Precompute
        self.c2_dt2_dx2 = (c * self.dt / dx) ** 2
        self.alpha_dt   = alpha * self.dt

        # Fields: current / previous
        self.u_cur  = np.zeros((Nx, Ny), dtype=np.float64)
        self.u_prev = np.zeros((Nx, Ny), dtype=np.float64)

        self.t = 0.0
        self.step_count = 0
        self._use_numba = use_numba and NUMBA_AVAILABLE

        # Source list: list of (i, j, func(t) -> float)
        self._sources = []
        # Cavity mask (True = wall / reflector)
        self.cavity_mask = np.zeros((Nx, Ny), dtype=bool)

        # Recorder
        self._history = []

    # ── Sources ──────────────────────────────────────────────────────────────

    def add_source(self, i, j, func):
        """Add a point source at grid (i,j). func(t) -> amplitude."""
        self._sources.append((i, j, func))

    def add_sinusoidal_source(self, i, j, freq, amp=1.0, phase=0.0):
        """Convenience: add a continuous sinusoidal source."""
        self.add_source(i, j, lambda t, f=freq, a=amp, p=phase:
                        a * np.sin(2 * np.pi * f * t + p))

    def add_cavity_rect(self, i0, i1, j0, j1):
        """Add rectangular cavity walls (reflective)."""
        self.cavity_mask[i0:i1, j0:j1] = True

    # ── Time stepping ────────────────────────────────────────────────────────

    def step(self):
        """Advance one FDTD time step."""
        Nx, Ny = self.Nx, self.Ny

        if self._use_numba:
            u_next = _step_numba(self.u_cur, self.u_prev,
                                 self.c2_dt2_dx2, self.alpha_dt, Nx, Ny)
        else:
            u_next = _step_numpy(self.u_cur, self.u_prev,
                                 self.c2_dt2_dx2, self.alpha_dt)

        # Inject sources
        for (i, j, func) in self._sources:
            u_next[i, j] += self.dt ** 2 * func(self.t)

        # Apply cavity (hard walls → zero displacement inside solid)
        u_next[self.cavity_mask] = 0.0

        self.u_prev = self.u_cur
        self.u_cur  = u_next
        self.t += self.dt
        self.step_count += 1

    def run(self, n_steps, record_every=1):
        """
        Run n_steps time steps.

        Returns
        -------
        history : list of 2D arrays (every `record_every` steps)
        """
        self._history = []
        for k in range(n_steps):
            self.step()
            if k % record_every == 0:
                self._history.append(self.u_cur.copy())
        return self._history

    # ── Readout ──────────────────────────────────────────────────────────────

    def energy_at(self, i, j):
        """Instantaneous wave energy (∝ u²) at grid point."""
        return float(self.u_cur[i, j] ** 2)

    def field_energy(self):
        """Total field energy."""
        return float(np.sum(self.u_cur ** 2))

    def snapshot(self):
        """Return a copy of the current field."""
        return self.u_cur.copy()

    # ── Interference helper ──────────────────────────────────────────────────

    def interference_output(self, probe_i, probe_j):
        """
        Read interference output at a probe point.
        Returns signed pressure value (positive = constructive / EPSP,
        negative = destructive / IPSP after thresholding).
        """
        return float(self.u_cur[probe_i, probe_j])

    def reset(self):
        """Reset field to zero."""
        self.u_cur[:]  = 0.0
        self.u_prev[:] = 0.0
        self.t = 0.0
        self.step_count = 0
        self._history = []
