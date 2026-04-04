# emulator/noise_channels.py

from typing import List
import numpy as np

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

def compose_kraus(kraus_A: List[np.ndarray], kraus_B: List[np.ndarray]) -> List[np.ndarray]:
    """
    Compose two Kraus maps:  ρ ↦ A(B(ρ)).
    If B has {B_j} and A has {A_i}, the composed set is {A_i B_j}.
    """
    return [A @ B for A in kraus_A for B in kraus_B]

def kron_n(ops: List[np.ndarray]) -> np.ndarray:
    """Compute tensor product of a list of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

# ---------------------------------------------------------------------
# Physics channels (single-qubit)
#   1) Reset: Generalized Amplitude Damping (GAD) to thermal state (p1)
#   2) Amplitude damping for T1 (standard AD → ground)
#   3) Dephasing (phase-flip model)
# ---------------------------------------------------------------------

def reset_kraus(p1: float, gamma: float = 1.0) -> List[np.ndarray]:
    """
    Generalized Amplitude Damping (GAD) channel used for reset to a thermal state.
      - p1: excited-state population after reset (thermal equilibrium)
      - gamma in [0,1]: damping strength (1 → full reset, 0 → no reset)
    Returns 4 Kraus operators.
    """
    p1 = np.clip(p1, 0.0, 1.0)
    gamma = np.clip(gamma, 0.0, 1.0)

    K0 = np.sqrt(1 - p1) * np.array([[1.0, 0.0],
                                     [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    K1 = np.sqrt(1 - p1) * np.array([[0.0, np.sqrt(gamma)],
                                     [0.0, 0.0]], dtype=complex)
    K2 = np.sqrt(p1) * np.array([[np.sqrt(1.0 - gamma), 0.0],
                                 [0.0, 1.0]], dtype=complex)
    K3 = np.sqrt(p1) * np.array([[0.0, 0.0],
                                 [np.sqrt(gamma), 0.0]], dtype=complex)
    return [K0, K1, K2, K3]


def amplitude_damping_gamma(gamma: float) -> List[np.ndarray]:
    """
    Standard amplitude damping (T1 relaxation → |0⟩).
      - gamma in [0,1]: gamma = 1 - exp(-dt/T1)
    Returns 2 Kraus operators.
    """
    gamma = np.clip(gamma, 0.0, 1.0)
    K0 = np.array([[1.0, 0.0],
                   [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    K1 = np.array([[0.0, np.sqrt(gamma)],
                   [0.0, 0.0]], dtype=complex)
    return [K0, K1]


def dephasing_phase_flip(delta: float) -> List[np.ndarray]:
    """
    Phase-flip dephasing channel:
       ρ ↦ (1-δ) ρ + δ Z ρ Z, with δ ∈ [0, 0.5].
    Implemented via Kraus: { sqrt(1-δ) I, sqrt(δ) Z }.
    """
    # allow any δ ∈ [0, 1], though for pure dephasing a common param is δ∈[0,0.5]
    delta = np.clip(delta, 0.0, 1.0)
    K0 = np.sqrt(1.0 - delta) * I2
    K1 = np.sqrt(delta) * Z
    return [K0, K1]


def exp_decay_map(dt: float, T1: float, T2: float) -> List[np.ndarray]:
    """
    Exponential decay during an interval dt:
      First T1 amplitude damping with γ_T1 = 1 - e^{-dt/T1},
      then T2 dephasing with δ_T2 = (1 - e^{-dt/T2})/2.

    Returns a composed Kraus set for the sequential map Deph ∘ AD.
    """
    # guard against T1/T2 = 0
    if T1 <= 0 or T2 <= 0 or dt <= 0:
        return [I2]

    gamma_T1 = 1.0 - np.exp(-dt / T1)
    delta_T2 = 0.5 * (1.0 - np.exp(-dt / T2))

    K_ad   = amplitude_damping_gamma(gamma_T1)
    K_deph = dephasing_phase_flip(delta_T2)

    # Compose as Deph( AD(ρ) )  → Kraus set { D_i * A_j }
    return compose_kraus(K_deph, K_ad)

# ---------------------------------------------------------------------
# Readout confusion matrix
# ---------------------------------------------------------------------

def apply_confusion_matrix(bit: int, C: List[List[float]]) -> int:
    """
    Apply a 2x2 readout confusion matrix C to a measured bit.
      C = [[P(0|0), P(0|1)],
           [P(1|0), P(1|1)]]
    """
    assert bit in (0, 1), "bit must be 0 or 1"
    C = np.asarray(C, dtype=float)
    assert C.shape == (2, 2), "Confusion matrix must be 2x2"

    # Probabilities for reported outcomes given the true bit
    # Column = true, Row = reported
    if bit == 0:
        p0 = C[0, 0]
        p1 = C[1, 0]
    else:
        p0 = C[0, 1]
        p1 = C[1, 1]

    # normalize 
    s = p0 + p1
    if s <= 0:
        # fallback to identity if ill-defined
        p0, p1 = 1.0, 0.0
    else:
        p0, p1 = p0 / s, p1 / s

    return 0 if np.random.random() < p0 else 1

# ---------------------------------------------------------------------
# One- and two-qubit effective gate noise
#   (from fitted average fidelities F1, F2)
# ---------------------------------------------------------------------

def single_qubit_gate_noise(F1: float) -> List[np.ndarray]:
    """
    Effective 1Q gate dephasing from average gate fidelity:
      δ1 = 1.5 * (1 - F1), δ1 ∈ [0, 0.5]
    Returns Kraus for phase-flip with δ1.
    """
    F1 = np.clip(F1, 0.0, 1.0)
    delta1 = 1.5 * (1.0 - F1)
    # Clamp to [0, 0.5] to keep a valid dephasing parameter
    delta1 = np.clip(delta1, 0.0, 0.5)
    return dephasing_phase_flip(delta1)


def two_qubit_dephasing_delta2(delta2: float) -> List[np.ndarray]:
    """
    Two-qubit dephasing as in the paper:
      ρ ↦ (1-δ2) ρ + (δ2/3) (Z⊗I ρ Z⊗I + I⊗Z ρ I⊗Z + Z⊗Z ρ Z⊗Z)
    Implementable via Kraus:
      { sqrt(1-δ2) I⊗I, sqrt(δ2/3) Z⊗I, sqrt(δ2/3) I⊗Z, sqrt(δ2/3) Z⊗Z }.
    """
    delta2 = np.clip(delta2, 0.0, 0.75)  # paper bounds mentioned δ2 ∈ [0, 3/4]
    K0 = np.sqrt(1.0 - delta2) * kron_n([I2, I2])
    w  = np.sqrt(delta2 / 3.0) if delta2 > 0 else 0.0
    K1 = w * kron_n([Z, I2])
    K2 = w * kron_n([I2, Z])
    K3 = w * kron_n([Z, Z])
    return [K0, K1, K2, K3]


def two_qubit_gate_noise_from_F2(F2: float) -> List[np.ndarray]:
    """
    Convenience: convert average two-qubit gate fidelity F2 to δ2,
      δ2 = (5/4) * (1 - F2), then build the dephasing Kraus set.
    """
    F2 = np.clip(F2, 0.0, 1.0)
    delta2 = 1.25 * (1.0 - F2)
    # clamp to valid range
    delta2 = np.clip(delta2, 0.0, 0.75)
    return two_qubit_dephasing_delta2(delta2)

# ---------------------------------------------------------------------
# ZZ crosstalk (always-on coupling) as a unitary
# ---------------------------------------------------------------------

def beta_from_J_delta_alpha(J: float, delta: float, alpha_u: float, alpha_v: float) -> float:
    """
    β = J² * (1/(Δ - α_u) - 1/(Δ - α_v))
    with safeguards for small denominators.

    If denominators are nearly equal (|Δ - α_u| ≈ |Δ - α_v|),
    the expression tends to zero — we set β = 0 to avoid numerical blowup.
    """
    eps = 1e-9  # safety offset in Hz

    denom_u = delta - alpha_u
    denom_v = delta - alpha_v

    # Avoid division by zero
    if abs(denom_u) < eps or abs(denom_v) < eps or abs(denom_u - denom_v) < eps:
        return 0.0

    return J**2 * (1.0 / denom_u - 1.0 / denom_v)


def crosstalk_ZZ_unitary(beta: float, duration: float) -> np.ndarray:
    """
    U = exp(-i * beta * duration * Z⊗Z)
      Z⊗Z has eigenvalues {+1, -1, -1, +1} on {|00>,|01>,|10>,|11>}.
      Therefore U is diagonal with phases {e^{-iθ}, e^{+iθ}, e^{+iθ}, e^{-iθ}},
      where θ = beta * duration.
    """
    theta = beta * duration
    phase_p = np.exp(-1j * theta)  # for |00>, |11>
    phase_m = np.exp(+1j * theta)  # for |01>, |10>
    return np.diag([phase_p, phase_m, phase_m, phase_p])

