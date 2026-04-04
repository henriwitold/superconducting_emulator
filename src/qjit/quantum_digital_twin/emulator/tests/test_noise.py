# tests/test_noise.py

"""
Test noise channel Kraus operators: trace preservation, CP, and physical behavior.
"""

import numpy as np
import pytest
from qjit.quantum_digital_twin.emulator.noise_channels import (
    reset_kraus,
    amplitude_damping_gamma,
    dephasing_phase_flip,
    exp_decay_map,
    single_qubit_gate_noise,
    two_qubit_gate_noise_from_F2,
    crosstalk_ZZ_unitary,
    beta_from_J_delta_alpha,
)


# ── Helpers ──────────────────────────────────────────────────────

def _is_trace_preserving(kraus_ops, tol=1e-10):
    """Check Σ K†K = I."""
    n = kraus_ops[0].shape[0]
    total = sum(K.conj().T @ K for K in kraus_ops)
    return np.allclose(total, np.eye(n), atol=tol)


def _apply_channel(rho, kraus_ops):
    """Apply Kraus channel to density matrix."""
    return sum(K @ rho @ K.conj().T for K in kraus_ops)


# ── Reset / state-prep ───────────────────────────────────────────

class TestResetKraus:

    def test_operator_count(self):
        ops = reset_kraus(p1=0.048, gamma=1.0)
        assert len(ops) == 4

    def test_trace_preserving(self):
        ops = reset_kraus(p1=0.048, gamma=1.0)
        assert _is_trace_preserving(ops)

    def test_reset_p1_zero_keeps_ground(self):
        rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_out = _apply_channel(rho_0, reset_kraus(p1=0.0, gamma=1.0))
        assert np.abs(rho_out[0, 0] - 1.0) < 1e-10

    def test_reset_p1_one_flips_to_excited(self):
        rho_0 = np.array([[1, 0], [0, 0]], dtype=complex)
        rho_out = _apply_channel(rho_0, reset_kraus(p1=1.0, gamma=1.0))
        assert np.abs(rho_out[1, 1] - 1.0) < 1e-10


# ── T1/T2 decay ──────────────────────────────────────────────────

class TestDecayKraus:

    def test_trace_preserving(self):
        ops = exp_decay_map(dt=32e-9, T1=40e-6, T2=18e-6)
        assert _is_trace_preserving(ops)

    def test_zero_dt_gives_identity(self):
        ops = exp_decay_map(dt=0, T1=40e-6, T2=18e-6)
        assert len(ops) == 1
        assert np.allclose(ops[0], np.eye(2))


# ── Gate noise ────────────────────────────────────────────────────

class TestGateNoise:

    def test_1q_noise_trace_preserving(self):
        ops = single_qubit_gate_noise(F1=0.996)
        assert len(ops) == 2
        assert _is_trace_preserving(ops)

    def test_2q_noise_trace_preserving(self):
        ops = two_qubit_gate_noise_from_F2(F2=0.987)
        assert len(ops) == 4
        assert _is_trace_preserving(ops)

    def test_perfect_fidelity_is_identity(self):
        ops = single_qubit_gate_noise(F1=1.0)
        rho = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        rho_out = _apply_channel(rho, ops)
        assert np.allclose(rho, rho_out, atol=1e-10)


# ── Dephasing ─────────────────────────────────────────────────────

class TestDephasing:

    def test_preserves_populations(self):
        rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        ops = dephasing_phase_flip(delta=0.5)
        rho_out = _apply_channel(rho_plus, ops)
        assert np.allclose(np.diag(rho_plus), np.diag(rho_out))

    def test_reduces_coherence(self):
        rho_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        ops = dephasing_phase_flip(delta=0.5)
        rho_out = _apply_channel(rho_plus, ops)
        assert np.abs(rho_out[0, 1]) < np.abs(rho_plus[0, 1])


# ── ZZ crosstalk ──────────────────────────────────────────────────

class TestCrosstalkZZ:

    def test_unitary(self):
        beta = beta_from_J_delta_alpha(
            J=15e3, delta=1e9, alpha_u=-200e6, alpha_v=-200e6
        )
        U = crosstalk_ZZ_unitary(beta, duration=45e-9)
        assert U.shape == (4, 4)
        assert np.allclose(U.conj().T @ U, np.eye(4))

    def test_diagonal(self):
        beta = beta_from_J_delta_alpha(
            J=15e3, delta=1e9, alpha_u=-200e6, alpha_v=-200e6
        )
        U = crosstalk_ZZ_unitary(beta, duration=45e-9)
        off_diag = U - np.diag(np.diag(U))
        assert np.allclose(off_diag, 0)

    def test_beta_zero_when_equal_anharmonicities_and_denominators(self):
        """β → 0 when denominators nearly cancel."""
        beta = beta_from_J_delta_alpha(
            J=15e3, delta=200e6, alpha_u=-200e6, alpha_v=-200e6
        )
        # With delta == |alpha|, denominators are degenerate → β = 0 (safeguard)
        assert beta == 0.0
