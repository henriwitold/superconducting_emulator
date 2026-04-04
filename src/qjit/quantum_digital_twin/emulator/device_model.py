# emulator/device_model.py
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum

@dataclass
class Qubit:
    """Single-qubit physical parameters.
    - frequency, anharmonicity: coherent dynamics (Hamiltonian)
    - T1, T2: decoherent processes
    - p1: probability of being in the excited state when initializing qubit
    - confusion_matrix: 2x2 readout assignment matrix [[P(0|0), P(0|1)], [P(1|0), P(1|1)]]
    """
    id: int
    frequency: float                       # Hz (calibrated control frequency)
    anharmonicity: float                   # Hz 
    T1: float                              # s
    T2: float                              # s
    p1: float                              # excited-state population after reset (state-prep)
    confusion_matrix: List[List[float]]    # 2x2 readout confusion matrix
    spam_error: float = 0.0                # Optional lumped SPAM term if you want a single scalar in addition to the matrix
    measurement_duration: float = 1500e-9  # s

    # ---- Frequency drift bookkeeping ----
    true_frequency: Optional[float] = None     # Hz, physical current frequency (post-drift)
    frequency_drift: float = 0.0               # Hz, true_frequency - calibrated frequency
    last_calibration_time: float = 0.0         # device.current_time when last calibrated
    drift_std: float = 500.0  # Hz/sqrt(s)

    def __post_init__(self):
        """
        Initialize true_frequency to calibrated frequency if not provided,
        and keep frequency_drift consistent.
        """
        if self.true_frequency is None:
            self.true_frequency = self.frequency
        self.frequency_drift = self.true_frequency - self.frequency

@dataclass
class Coupling:
    """Pairwise coupling between two qubits (physical edge).
    - qubit_pair: ordered or canonical tuple (i, j)
    - J: effective coupling strength in Hz
    - detuning: f_i - f_j (Hz); updated from qubit frequencies
    """
    qubit_pair: Tuple[int, int]
    J: float                            # Hz
    detuning: float                     # Hz (f_i - f_j)
    connectivity_type: str = "chain"    # 'star', 'lattice', 'chain', etc.

class GateType(str, Enum):
    ONE_Q = "1Q"
    TWO_Q = "2Q"

@dataclass
class GateDefinition:
    """Native gate instance bound to specific qubit(s).
    - type: 1Q or 2Q
    - qubits: (q,) for 1Q or (q_control, q_target) for 2Q
    - fidelity: effective F1 (1Q) or F2 (2Q), often fitted/empirical
    - error_rate: stochastic error probability (if you use it separately)
    """
    name: str
    type: GateType
    duration: float                     # s
    fidelity: float                     # F1 (1Q) or F2 (2Q) effective fidelity
    qubits: Tuple[int, ...]             # (q,) or (q0, q1)
    error_rate: float = 0.0             # optional separate error prob
    is_virtual: bool = False            # True for Rz gates
    parameters: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.is_virtual:
            self.fidelity = 1.0

@dataclass
class DeviceModel:
    """Complete device description with qubits, couplings, and gate instances."""
    qubits: Dict[int, Qubit]
    couplings: Dict[Tuple[int, int], Coupling]   # keyed by (i, j) canonical order
    gate_library: List[GateDefinition]           # one entry per physical gate instance
    topology: str = "star"                       # e.g., 'star', 'lattice', 'chain'
    central_qubit: Optional[int] = None  
    uses_active_reset: bool = True               # vs. passive thermal relaxation

    # ---- Global device time & drift ----
    current_time: float = 0.0                    # in seconds, or arbitrary time units

    # ---- Global hardware specs (used for drift-aware fidelity) ----
    cz_bandwidth: float = 1e6             # ~30 MHz interaction window

    # ---- Lookups ----------------------------------------------------
    def get_qubit(self, qid: int) -> Qubit:
        return self.qubits[qid]

    def get_coupling(self, q0: int, q1: int) -> Optional[Coupling]:
        key = (q0, q1) if (q0, q1) in self.couplings else (q1, q0)
        return self.couplings.get(key, None)

    def get_gates_for_qubits(self, qubits: Tuple[int, ...]) -> List[GateDefinition]:
        return [g for g in self.gate_library if g.qubits == qubits]

    # ---- Utilities --------------------------------------------------
    def update_detunings_from_frequencies(self) -> None:
        """
        Recompute detuning f_i - f_j for all couplings from current qubit frequencies.
        We use true_frequency if available (including drift), otherwise fall back to the calibrated frequency field.
        """
        for (i, j), coup in self.couplings.items():
            qi = self.qubits[i]
            qj = self.qubits[j]
            fi = qi.true_frequency if qi.true_frequency is not None else qi.frequency
            fj = qj.true_frequency if qj.true_frequency is not None else qj.frequency
            coup.detuning = fi - fj

    def add_gate(self, gate: GateDefinition) -> None:
        self.gate_library.append(gate)

    
    def step_frequency_drift(
        self,
        dt: float,
        std_dev: float = None,
    ) -> None:
        """
        Apply Gaussian frequency drift using a Wiener-process model:
            Δf = σ * sqrt(dt) * N(0,1)
        If std_dev is None, use each qubit's own q.drift_std.
        """

        if dt < 0:
            raise ValueError("dt must be non-negative")

        self.current_time += dt

        for q in self.qubits.values():

            # default per-qubit σ
            sigma = q.drift_std if std_dev is None else std_dev

            if q.true_frequency is None:
                q.true_frequency = q.frequency

            drift_step = np.random.normal(
                loc=0.0,
                scale=sigma * np.sqrt(dt)
            )

            q.true_frequency += drift_step
            q.frequency_drift = q.true_frequency - q.frequency

        self.update_detunings_from_frequencies()


    # ------------------------------------------------------------------
    # Drift-aware single-qubit rotation angle
    # ------------------------------------------------------------------
    def effective_single_qubit_angle(
        self,
        qid: int,
        theta_nominal: float,
        gate_duration: float,
    ) -> float:
        """
        Compute drift-aware effective rotation angle for a resonant 1Q drive.

        Model (simplified):
        - Drive amplitude is calibrated for on-resonance Rabi rate Ω_cal.
        - Frequency drift introduces detuning Δ = 2π Δf.
        - Actual Rabi frequency becomes Ω_eff = sqrt(Ω_cal^2 + Δ^2).
        - Rotation angle θ ∝ Ω * t, so:

              θ_eff = θ_nominal * (Ω_eff / Ω_cal)

        This captures coherent under/over-rotation from detuning.
        We ignore axis tilt and keep rotation about nominal X/Y axis.
        """
        q = self.qubits[qid]

        # Detuning from drift (Hz → rad/s)
        delta_f = q.frequency_drift  # Hz
        Delta = 2 * np.pi * delta_f  # rad/s

        if gate_duration <= 0:
            # Virtual or zero-duration gate: no angle renormalization
            return theta_nominal

        # On-resonance Rabi frequency used for calibration
        # Qiskit native gate_duration is usually for a π/2 rotation,
        # but here we only need a consistent Ω_cal, not the exact pulse details.
        Omega_cal = np.pi / (2 * gate_duration)  # rad/s

        # Off-resonant Rabi frequency
        Omega_eff = np.sqrt(Omega_cal ** 2 + Delta ** 2)

        # Scale factor for the angle
        scale = Omega_eff / Omega_cal

        theta_eff = theta_nominal * scale

        # Optional: safety clip for insane drift
        # keep within, say, [-4π, 4π] to avoid numerical craziness
        max_angle = 4 * np.pi
        theta_eff = float(np.clip(theta_eff, -max_angle, max_angle))

        return theta_eff


    # ------------------------------------------------------------------
    # Drift-aware single-qubit fidelity
    # ------------------------------------------------------------------
    def effective_single_qubit_fidelity(
        self,
        qid: int,
        F_nominal: float,
        gate_duration: float,
    ) -> float:
        """
        Compute drift-aware effective 1Q fidelity.

        Physics
        -------
        Detuning Δf = true_frequency - calibrated_frequency causes
        off-resonance driving. For small detuning, the average gate
        fidelity is approximately:

            F_eff ≈ F_nominal * (1 - (Δf / Ω)^2)
        Qiskit native single-qubit pulse = π/2 rotation
        A π rotation is twice the area → Ω = π / (2 * duration)
        where Ω ≈ π / (2 * gate_duration) is the Rabi frequency.

        We clip the correction to keep F_eff within [0, 1].
        """

        q = self.qubits[qid]

        # detuning from drift
        delta_f = abs(q.frequency_drift)  # Hz
        Delta = 2 * np.pi * delta_f

        # Rabi frequency for a π pulse
        # If gate_duration is for π/2 (Qiskit-native), then π pulse = 2 * gate_duration
        Omega = np.pi / (2 * gate_duration)   

        correction = (Delta / Omega) ** 2

        F_eff = F_nominal * (1.0 - correction)

        # Safe clip
        return float(np.clip(F_eff, 0.0, 1.0))


    # ------------------------------------------------------------------
    # Drift-aware coherent CZ phase
    # ------------------------------------------------------------------
    def effective_cz_phase(
        self,
        q0: int,
        q1: int,
        nominal_phase: float,
        gate_duration: float,
    ) -> float:
        """
        Compute an effective CZ conditional phase φ_eff under frequency drift.

        Idea (simple model)
        -------------------
        Let
            Δ_cal  = f0_cal - f1_cal
            Δ_true = f0_true - f1_true
            δ      = Δ_true - Δ_cal      (detuning mismatch, Hz)

        We define a dimensionless mismatch:
            ε = δ / Ω_CZ

        where Ω_CZ ~ self.cz_bandwidth (tunable, Hz).

        Then we model the CZ phase error as
            φ_eff = φ_nominal * (1 + ε)

        so small detuning mismatch ⇒ small relative phase error.

        This is a crude but controllable model: large drift_std_hz or long dt
        → large ε → visibly wrong CZ phase.
        """

        qi = self.qubits[q0]
        qj = self.qubits[q1]

        # calibrated and true detunings
        delta_cal = qi.frequency - qj.frequency
        delta_true = qi.true_frequency - qj.true_frequency
        mismatch = delta_true - delta_cal  # Hz

        # interaction bandwidth for CZ (Hz)
        Omega_CZ = getattr(self, "cz_bandwidth", 20e6)  # fallback 20 MHz

        # dimensionless mismatch
        epsilon = mismatch / Omega_CZ

        # linear phase error model
        phi_err = epsilon * nominal_phase
        phi_eff = nominal_phase + phi_err

        return phi_eff


    # ------------------------------------------------------------------
    # Drift-aware two-qubit fidelity (CZ)
    # ------------------------------------------------------------------
    def effective_two_qubit_fidelity(
        self,
        q0: int,
        q1: int,
        F_nominal: float,
    ) -> float:
        """
        Compute drift-aware effective 2Q (CZ) fidelity.

        Physics
        -------
        CZ gates depend on detuning Δ_cal = f0_cal - f1_cal.

        True detuning:
            Δ_true = f0_true - f1_true

        Misalignment:
            δ = Δ_true - Δ_cal

        Empirical fidelity penalty (quadratic approx):
            F_eff ≈ F_nominal * (1 - γ * (δ / Ω_CZ)^2)

        where Ω_CZ (~ 20–40 MHz) is the CZ “interaction bandwidth”.

        γ ~ 1 is a tunable severity parameter.
        """

        qi = self.qubits[q0]
        qj = self.qubits[q1]

        # calibrated and true detunings
        delta_cal = qi.frequency - qj.frequency
        delta_true = qi.true_frequency - qj.true_frequency
        mismatch = delta_true - delta_cal    # Hz

        # CZ bandwidth (tunable)
        Omega_CZ = self.cz_bandwidth

        # severity constant
        gamma = 1.0

        correction = gamma * (mismatch / Omega_CZ) ** 2

        F_eff = F_nominal * (1.0 - correction)

        return float(np.clip(F_eff, 0.0, 1.0))
    
    # ------------------------------------------------------------------
    # Phase accumulated during idle due to frequency drift
    # ------------------------------------------------------------------
    
    def get_idle_phase(self, qid: int, idle_time: float) -> float:
        """
        Compute the coherent phase accumulated during idle:
            φ = 2π * Δf * idle_time
        where Δf = true_frequency - calibrated_frequency.
        The phase φ is in radians.
        """
        q = self.qubits[qid]
        delta_f = q.frequency_drift  # Hz
        phi = 2 * np.pi * delta_f * idle_time
        return phi

