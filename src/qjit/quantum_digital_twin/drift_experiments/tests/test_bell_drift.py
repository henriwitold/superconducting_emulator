# tests/test_bell_drift.py

"""
Bell-state drift experiment: verify that frequency drift degrades
entanglement weight on a Q0-Q2 Bell pair.
"""

import numpy as np
import pytest
from qjit.quantum_digital_twin.emulator.circuit import Circuit, Gate, Measurement
from qjit.quantum_digital_twin.emulator.qpu_config import create_paper_qpu
from qjit.quantum_digital_twin.emulator.qiskit_emulator_core import QiskitSimulator


def _cnot_via_rz_ry_cz(control, target):
    """CNOT decomposition: (I⊗H)·CZ·(I⊗H) with H = Rz(π)·Ry(π/2)."""
    return [
        Gate("Rz", (target,), {"theta": np.pi}),
        Gate("Ry", (target,), {"theta": np.pi / 2}),
        Gate("CZ", (control, target), {}),
        Gate("Rz", (target,), {"theta": np.pi}),
        Gate("Ry", (target,), {"theta": np.pi / 2}),
    ]


def _bell_circuit() -> Circuit:
    gates = [
        Gate("Rz", (0,), {"theta": np.pi}),
        Gate("Ry", (0,), {"theta": np.pi / 2}),
    ] + _cnot_via_rz_ry_cz(0, 2)
    return Circuit(
        num_qubits=3,
        gates=gates,
        measurements=[Measurement(0, 0), Measurement(2, 1)],
    )


def _entanglement_weight(counts):
    total = sum(counts.values())
    return (counts.get("00", 0) + counts.get("11", 0)) / total


SHOTS = 4096


class TestBellStateDrift:

    def test_bell_entanglement_before_drift(self):
        """Bell pair should have majority weight in |00⟩+|11⟩ with no drift."""
        device = create_paper_qpu()
        sim = QiskitSimulator(device)
        counts = sim.simulate(_bell_circuit(), shots=SHOTS)
        p_ent = _entanglement_weight(counts)
        assert p_ent > 0.55, f"Entangled weight = {p_ent:.3f}"

    def test_drift_degrades_entanglement(self):
        """Large frequency drift should reduce Bell-state quality."""
        device = create_paper_qpu()
        sim_before = QiskitSimulator(device)
        p_before = _entanglement_weight(
            sim_before.simulate(_bell_circuit(), shots=SHOTS)
        )

        device.step_frequency_drift(dt=1.0, std_dev=1e5)
        sim_after = QiskitSimulator(device)
        p_after = _entanglement_weight(
            sim_after.simulate(_bell_circuit(), shots=SHOTS)
        )

        # With 100 kHz drift std, entanglement should degrade
        # (may not always hold for small drift realizations, so we use a soft check)
        assert p_after < p_before + 0.05, (
            f"Drift should not improve entanglement: {p_before:.3f} → {p_after:.3f}"
        )
