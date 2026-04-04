# tests/test_drift.py

"""
Drift validation: Wiener-process statistics, coherent misrotation,
CZ fidelity degradation, and reproducibility under fixed seed.
"""

import numpy as np
import pytest
from qjit.quantum_digital_twin.emulator.qpu_config import create_paper_qpu


class TestDriftStatistics:

    def test_wiener_process_scaling(self):
        """Cumulative drift after N steps stays within plausible ±5σ range."""
        np.random.seed(123)
        device = create_paper_qpu()
        q = device.get_qubit(0)
        f0 = q.frequency
        sigma = q.drift_std

        q.true_frequency = f0
        steps, dt = 5000, 1.0
        for _ in range(steps):
            device.step_frequency_drift(dt=dt)

        drift = q.true_frequency - f0
        expected_std = sigma * np.sqrt(steps * dt)
        assert abs(drift) < 5 * expected_std, (
            f"Drift {drift:.0f} Hz outside ±5σ ({5*expected_std:.0f} Hz)"
        )


class TestCoherentMisrotation:

    def test_misrotation_increases_with_drift(self):
        """Positive and negative frequency drift both increase |θ_eff − θ_nom|."""
        device = create_paper_qpu()
        qid, theta_nom, dur = 0, np.pi / 2, 32e-9

        errors = []
        for df in [0.0, +1e6, -1e6]:
            device.qubits[qid].frequency_drift = df
            theta_eff = device.effective_single_qubit_angle(qid, theta_nom, dur)
            errors.append(abs(theta_eff - theta_nom))

        assert errors[1] > errors[0], "Positive drift should increase misrotation"
        assert errors[2] > errors[0], "Negative drift should increase misrotation"


class TestCZDriftFidelity:

    def test_cz_fidelity_decreases(self):
        """1 MHz drift on Q0 should reduce CZ(0,2) fidelity."""
        device = create_paper_qpu()
        q0, q1 = 0, 2
        gate = next(
            g for g in device.gate_library
            if g.name == "CZ" and set(g.qubits) == {q0, q1}
        )
        F_nom = gate.fidelity

        F_baseline = device.effective_two_qubit_fidelity(q0, q1, F_nom)

        device.qubits[q0].true_frequency += 1e6
        device.update_detunings_from_frequencies()
        F_drifted = device.effective_two_qubit_fidelity(q0, q1, F_nom)

        assert F_drifted < F_baseline, (
            f"CZ fidelity should drop: {F_baseline:.4f} → {F_drifted:.4f}"
        )

    def test_cz_phase_shifts(self):
        """1 MHz drift should shift the effective CZ phase away from π."""
        device = create_paper_qpu()
        q0, q1 = 0, 2
        gate = next(
            g for g in device.gate_library
            if g.name == "CZ" and set(g.qubits) == {q0, q1}
        )

        phi_baseline = device.effective_cz_phase(
            q0, q1, nominal_phase=np.pi, gate_duration=gate.duration
        )

        device.qubits[q0].true_frequency += 1e6
        device.update_detunings_from_frequencies()
        phi_drifted = device.effective_cz_phase(
            q0, q1, nominal_phase=np.pi, gate_duration=gate.duration
        )

        assert abs(phi_drifted - np.pi) > abs(phi_baseline - np.pi), (
            "CZ phase should shift further from π under drift"
        )


class TestDriftReproducibility:

    def test_deterministic_under_seed(self):
        np.random.seed(99)
        freqs_a = [q.frequency for q in create_paper_qpu().qubits.values()]

        np.random.seed(99)
        freqs_b = [q.frequency for q in create_paper_qpu().qubits.values()]

        assert freqs_a == freqs_b
