"""
examples/04_arm_control.py
───────────────────────────
Full 3-DOF Humanoid Arm Wave Control Demo.

Pipeline:
    CPG → wave params → FDTD field → ganglion readout → joint angles

Architecture:
    [WaveCPG] → [WaveParameterMapper] → [WaveField2D] → [GanglionLayer]
             → [JointAngleDecoder] → Joint outputs

Run:
    python examples/04_arm_control.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from simulation.spinal_cord import WaveCPG, GanglionLayer
from core.wave_field import WaveField2D
from control.humanoid_mapping import WaveParameterMapper, JointAngleDecoder


# ─── Parameters ──────────────────────────────────────────────────────────────

DT         = 1e-3      # s
N_STEPS    = 500       # total time steps
SPEED      = 1.0       # walking speed
GRID       = 64        # wave field grid
FREQ_BASE  = 50.0      # Hz — wave source frequency
C_WAVE     = 340.0     # m/s
DX         = 5e-3      # m

# Arm joints (3 DOF example)
JOINT_LABELS = ["shoulder_flex", "elbow_flex", "wrist_flex"]
N_JOINTS     = 3


# ─── Build subsystems ─────────────────────────────────────────────────────────

cpg = WaveCPG(n_osc=N_JOINTS,
              omega=[2*np.pi*2.0] * N_JOINTS,
              kappa=4.0, dt=DT)

mapper = WaveParameterMapper(n_nodes=N_JOINTS,
                              node_positions=np.array([0.1, 0.3, 0.5]))

joint_decoder = JointAngleDecoder(joint_labels=JOINT_LABELS,
                                   k_joint=8.0)

# Ganglion positions on wave field
N = GRID
ganglion_pos = [(N//4, N//4), (N//2, N//4), (3*N//4, N//4)]
ganglion = GanglionLayer(n_ganglia=N_JOINTS,
                          thresholds=[0.01]*N_JOINTS,
                          gains=[1.5, 1.2, 1.0],
                          dt=DT)


# ─── Storage ──────────────────────────────────────────────────────────────────

joint_history    = {j: [] for j in JOINT_LABELS}
ganglion_history = []
cpg_history      = []
energy_history   = []


# ─── Main loop ────────────────────────────────────────────────────────────────

print("Running WavePhysAI Arm Control Demo...")
print(f"  Steps: {N_STEPS}, dt={DT}s, grid={GRID}×{GRID}")

wf = WaveField2D(Nx=GRID, Ny=GRID, dx=DX, c=C_WAVE)

for step_idx in range(N_STEPS):
    t = step_idx * DT

    # 1. CPG oscillator step
    cpg_out = cpg.step(speed=SPEED)
    cpg_history.append(cpg_out.copy())

    # 2. Map CPG output → wave parameters
    v_ref  = SPEED * 0.5  # approximate speed
    tau_n  = np.abs(cpg_out) * 2.0  # torque proportional to activation
    params = mapper.map(v_ref=v_ref, tau_n=tau_n, dt=DT)

    # 3. Update wave field sources from CPG phases
    wf = WaveField2D(Nx=GRID, Ny=GRID, dx=DX, c=C_WAVE)
    for j in range(N_JOINTS):
        row = GRID // (N_JOINTS + 1) * (j + 1)
        wf.add_sinusoidal_source(
            row, GRID // 2,
            freq=float(params["f"][j]),
            amp=float(params["A"][j]),
            phase=float(cpg.phi[j])   # use CPG phase directly
        )

    # Advance field a few steps per control step (inner loop)
    n_inner = max(1, int(DT / wf.dt))
    for _ in range(n_inner):
        wf.step()

    # 4. Read ganglion outputs
    energies = np.array([wf.energy_at(pi, pj)
                         for pi, pj in ganglion_pos])
    g_out = ganglion.step(inputs=energies)
    ganglion_history.append(g_out.copy())
    energy_history.append(energies.copy())

    # 5. Accumulate → joint angles
    joint_decoder.accumulate(energies, dt=DT)
    if step_idx % 20 == 19:   # decode every 20 steps
        angles = joint_decoder.decode()
        for j in JOINT_LABELS:
            joint_history[j].append(angles[j])

# ─── Results ──────────────────────────────────────────────────────────────────

t_axis    = np.arange(N_STEPS) * DT
t_joints  = np.arange(len(joint_history[JOINT_LABELS[0]])) * DT * 20

cpg_arr   = np.array(cpg_history)
gang_arr  = np.array(ganglion_history)
e_arr     = np.array(energy_history)

print("\n── Results ──────────────────────────────────")
print(f"  CPG synchrony order R = {cpg.synchrony_order:.3f}")
print(f"  Mean energy (shoulder): {e_arr[:,0].mean():.4f}")
print(f"  Final joint angles:")
for j in JOINT_LABELS:
    if joint_history[j]:
        print(f"    {j:20s}: {np.degrees(joint_history[j][-1]):+.1f}°")

# ─── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(12, 9))
fig.suptitle("WavePhysAI — 3-DOF Arm Control via Wave Interference",
             fontsize=14, fontweight="bold")

# Panel 1: CPG oscillators
ax = axes[0]
colors = ["#00D4FF", "#7C3AED", "#10B981"]
for j, (lbl, c) in enumerate(zip(JOINT_LABELS, colors)):
    ax.plot(t_axis, cpg_arr[:, j], color=c, lw=1.5,
            label=f"CPG φ: {lbl}")
ax.set_ylabel("CPG output (sin φ)")
ax.set_title("Phase-Coded CPG Oscillators")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, N_STEPS * DT])

# Panel 2: Ganglion outputs (reflex activity)
ax = axes[1]
for j, (lbl, c) in enumerate(zip(JOINT_LABELS, colors)):
    ax.plot(t_axis, gang_arr[:, j], color=c, lw=1.2, alpha=0.8,
            label=f"Ganglion: {lbl}")
ax.set_ylabel("Ganglion output O = σ(I−θ)")
ax.set_title("Ganglion Threshold Gates (Reflex Layer)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, N_STEPS * DT])

# Panel 3: Joint angles
ax = axes[2]
for j, (lbl, c) in enumerate(zip(JOINT_LABELS, colors)):
    if joint_history[lbl]:
        ax.plot(t_joints, np.degrees(joint_history[lbl]),
                color=c, lw=2.0, marker='o', ms=3,
                label=f"{lbl}")
ax.set_ylabel("Joint angle (degrees)")
ax.set_xlabel("Time (s)")
ax.set_title("θᵢ = k · ∫p²(xᵢ,t)dt  →  Joint Angles")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, t_joints[-1] if len(t_joints) > 0 else 1])

plt.tight_layout()
plt.savefig("arm_control_demo.png", dpi=150, bbox_inches="tight")
print("\n  Plot saved: arm_control_demo.png")
plt.show()
