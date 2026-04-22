"""
wavephysai/control/humanoid_mapping.py
────────────────────────────────────────
Wave Energy → Joint Angle / Torque Mapping.

Physics
-------
Joint angle from wave energy (Eq. paper §4.3):
    θᵢ = k · ∫ p²(xᵢ, t) dt

Mapping laws (§5.2):
    Frequency:       f_n(t) = f₀ + k_f · ‖v_ref(t)‖
    Amplitude:       A_n(t) = clip(A_min, A₀ + k_A · τ_n(t), A_max)
    Phase offset:    φ_n(t) = φ₀ + k_φ · κ(t) · d_n
    Prop. velocity:  v_p(t) = k_v · ‖v_ref‖ / (α + ‖v_ref‖)

Humanoid joint labels (7 per arm, 7 per leg):
    Arm:  shoulder_flex, shoulder_abduct, shoulder_rot,
          elbow_flex, forearm_rot, wrist_flex, wrist_dev
    Leg:  hip_flex, hip_abduct, hip_rot,
          knee_flex, ankle_flex, ankle_inv, toe_flex
"""

import struct
import time
import numpy as np


# ─── Default mapping coefficients ────────────────────────────────────────────

DEFAULT_COEFFS = {
    "f0":    1.0,    # Hz — baseline CPG frequency
    "k_f":   0.4,    # Hz·s/m — frequency per unit speed
    "A0":    0.02,   # m — baseline amplitude
    "k_A":   0.004,  # m/N — amplitude per unit torque demand
    "A_min": 0.005,  # m
    "A_max": 0.06,   # m
    "phi0":  0.0,    # rad — baseline phase
    "k_phi": 0.5,    # rad·m — phase per curvature × distance
    "k_v":   1.2,    # m/s — velocity saturation numerator
    "alpha": 0.5,    # m/s — velocity saturation denominator
}


# ─── WaveParamsPacket (32-byte binary protocol) ───────────────────────────────

PACKET_FORMAT = ">BBHQIffff"
PACKET_SIZE   = struct.calcsize(PACKET_FORMAT)  # should be 32

MSG_TYPE_WAVE_PARAMS = 1
MSG_TYPE_SAFETY_STOP = 2
PROTOCOL_VERSION     = 1


def encode_wave_params_packet(node_id: int, A: float, f: float,
                               v_p: float, phi: float,
                               t_valid_ms: int = 200,
                               seq: int = 0) -> bytes:
    """
    Encode a 32-byte WaveParamsPacket.

    Layout (Big-Endian):
        Offset  0  (1B)  : Protocol version  (uint8)
        Offset  1  (1B)  : Message type      (uint8 = 1)
        Offset  2  (2B)  : Sequence number   (uint16)
        Offset  4  (8B)  : Timestamp ns      (uint64)
        Offset 12  (4B)  : t_valid_ms        (uint32)
        Offset 16  (4B)  : A (amplitude, m)  (float32)
        Offset 20  (4B)  : f (frequency, Hz) (float32)
        Offset 24  (4B)  : v_p (velocity)    (float32)
        Offset 28  (4B)  : phi (phase, rad)  (float32)
        Total = 32 bytes
    """
    ts_ns = int(time.time_ns())
    return struct.pack(PACKET_FORMAT,
                       PROTOCOL_VERSION,
                       MSG_TYPE_WAVE_PARAMS,
                       seq & 0xFFFF,
                       ts_ns,
                       t_valid_ms,
                       np.float32(A),
                       np.float32(f),
                       np.float32(v_p),
                       np.float32(phi))


def decode_wave_params_packet(data: bytes) -> dict:
    """Decode a 32-byte WaveParamsPacket."""
    assert len(data) == PACKET_SIZE, \
        f"Expected {PACKET_SIZE} bytes, got {len(data)}"
    (ver, msg_type, seq, ts_ns, t_valid_ms,
     A, f, v_p, phi) = struct.unpack(PACKET_FORMAT, data)
    return {
        "protocol_version": ver,
        "msg_type": msg_type,
        "seq": seq,
        "timestamp_ns": ts_ns,
        "t_valid_ms": t_valid_ms,
        "A": float(A),
        "f": float(f),
        "v_p": float(v_p),
        "phi": float(phi),
    }


def encode_safety_stop(seq: int = 0) -> bytes:
    """Encode an 8-byte SafetyStop packet."""
    ts_ms = int(time.time() * 1000) & 0xFFFFFFFF
    return struct.pack(">BBHI",
                       PROTOCOL_VERSION,
                       MSG_TYPE_SAFETY_STOP,
                       seq & 0xFFFF,
                       ts_ms)


# ─── Wave Parameter Mapper ────────────────────────────────────────────────────

class WaveParameterMapper:
    """
    Maps high-level trajectory reference to per-node wave parameters.

    Implements Mapping Layer Block D from the GR00T hybrid architecture:
        x*_ref(t) → {A_n(t), f_n(t), v_p_n(t), φ_n(t)}

    Parameters
    ----------
    n_nodes : number of distributed cavity nodes
    node_positions : array (n_nodes,) — lateral distance from spine (m)
    coeffs  : dict — mapping coefficients (see DEFAULT_COEFFS)
    """

    def __init__(self, n_nodes: int = 12,
                 node_positions: np.ndarray = None,
                 coeffs: dict = None):
        self.n = n_nodes
        self.d = (node_positions if node_positions is not None
                  else np.linspace(0, 0.5, n_nodes))
        self.c = {**DEFAULT_COEFFS, **(coeffs or {})}

        self._v_p_prev = np.zeros(n_nodes)
        self._smooth_tau = 0.1  # s

    def map(self, v_ref: float, tau_n: np.ndarray = None,
            kappa: float = 0.0, dt: float = 1e-3) -> dict:
        """
        Compute wave parameters for all nodes.

        Parameters
        ----------
        v_ref  : float — reference speed magnitude (m/s)
        tau_n  : np.ndarray (n,) — torque demand per node (N·m)
        kappa  : float — path curvature (1/m)
        dt     : float — time step (s)

        Returns
        -------
        params : dict with keys 'A', 'f', 'v_p', 'phi'
                 each np.ndarray shape (n,)
        """
        c = self.c
        if tau_n is None:
            tau_n = np.zeros(self.n)

        # Frequency
        f = np.full(self.n, c["f0"] + c["k_f"] * abs(v_ref))

        # Amplitude
        A = np.clip(c["A0"] + c["k_A"] * tau_n, c["A_min"], c["A_max"])

        # Phase offset
        phi = np.full(self.n, c["phi0"]) + c["k_phi"] * kappa * self.d

        # Propagation velocity (smooth saturation)
        v_p_target = c["k_v"] * abs(v_ref) / (c["alpha"] + abs(v_ref))
        alpha_smooth = dt / (self._smooth_tau + dt)
        v_p = self._v_p_prev + alpha_smooth * (v_p_target - self._v_p_prev)
        self._v_p_prev = v_p.copy()

        return {"A": A, "f": f, "v_p": v_p, "phi": phi}

    def encode_all(self, params: dict,
                   seq_base: int = 0) -> list:
        """
        Encode all node parameters as 32-byte packets.

        Returns
        -------
        packets : list of bytes (one per node)
        """
        packets = []
        for i in range(self.n):
            pkt = encode_wave_params_packet(
                node_id=i,
                A=float(params["A"][i]),
                f=float(params["f"][i]),
                v_p=float(params["v_p"][i]),
                phi=float(params["phi"][i]),
                seq=seq_base + i
            )
            packets.append(pkt)
        return packets


# ─── Joint angle decoder ──────────────────────────────────────────────────────

class JointAngleDecoder:
    """
    Convert wave energy at ganglion nodes to joint angles.

        θᵢ = k_joint · ∫ p²(xᵢ, t) dt

    Parameters
    ----------
    joint_labels : list of str
    k_joint      : float or array — energy-to-angle gain (rad / J)
    max_angles   : array — joint range limits (rad)
    """

    HUMANOID_JOINTS = {
        "right_arm": ["shoulder_flex", "shoulder_abduct", "shoulder_rot",
                      "elbow_flex", "forearm_rot", "wrist_flex", "wrist_dev"],
        "left_arm":  ["shoulder_flex_L", "shoulder_abduct_L",
                      "shoulder_rot_L", "elbow_flex_L",
                      "forearm_rot_L", "wrist_flex_L", "wrist_dev_L"],
        "right_leg": ["hip_flex", "hip_abduct", "hip_rot",
                      "knee_flex", "ankle_flex", "ankle_inv", "toe_flex"],
        "left_leg":  ["hip_flex_L", "hip_abduct_L", "hip_rot_L",
                      "knee_flex_L", "ankle_flex_L", "ankle_inv_L",
                      "toe_flex_L"],
    }

    def __init__(self, joint_labels=None, k_joint=5.0,
                 max_angles=None):
        if joint_labels is None:
            joint_labels = (self.HUMANOID_JOINTS["right_arm"] +
                            self.HUMANOID_JOINTS["right_leg"])
        self.labels = joint_labels
        self.n = len(joint_labels)
        self.k = (np.full(self.n, k_joint)
                  if np.isscalar(k_joint)
                  else np.asarray(k_joint))
        self.max_angles = (np.full(self.n, np.pi / 2)
                           if max_angles is None
                           else np.asarray(max_angles))
        self._energy_accum = np.zeros(self.n)

    def accumulate(self, energies: np.ndarray, dt: float = 1e-3):
        """Accumulate wave energy."""
        self._energy_accum += energies * dt

    def decode(self) -> dict:
        """
        Compute joint angles from accumulated energy.

        Returns
        -------
        joints : dict {label: angle_rad}
        """
        angles = np.clip(self.k * self._energy_accum,
                         -self.max_angles, self.max_angles)
        self._energy_accum[:] = 0.0  # reset after decode
        return {self.labels[i]: float(angles[i]) for i in range(self.n)}
