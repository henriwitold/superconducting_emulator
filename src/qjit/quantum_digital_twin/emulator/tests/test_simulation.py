# tests/test_simulation.py

"""
Full simulation pipeline tests: SPAM, bit flips, entanglement, consistency,
crosstalk, and virtual-RZ behavior.

These tests run actual Qiskit Aer simulations, so they are slower than
the unit tests in test_device / test_noise / etc.
"""

import numpy as np
import pytest
from qjit.quantum_digital_twin.emulator.qpu_config import create_paper_qpu
from qjit.quantum_digital_twin.emulator.qiskit_emulator_core import QiskitSimulator
from qjit.quantum_digital_twin.emulator.circuit import Circuit, Gate, Measurement
from qjit.quantum_digital_twin.emulator.scheduling import (
    find_gate_definition,
    schedule_circuit_alap,
)


# ── Helpers ──────────────────────────────────────────────────────

def _cnot_via_rz_ry_cz(control, target):
    """CNOT decomposition: (I⊗H)·CZ·(I⊗H) with H = Rz(π)·Ry(π/2)."""
    return [
        Gate("Rz", (target,), {"theta": np.pi}),
        Gate("Ry", (target,), {"theta": np.pi / 2}),
        Gate("CZ", (control, target), {}),
        Gate("Rz", (target,), {"theta": np.pi}),
        Gate("Ry", (target,), {"theta": np.pi / 2}),
    ]


SHOTS = 10_000


# ── SPAM and single-gate tests ───────────────────────────────────

class TestSPAM:

    def test_identity_circuit_spam(self, paper_simulator):
        """No gates → mostly |00⟩, errors from state-prep + readout only."""
        circuit = Circuit(
            num_qubits=2, gates=[],
            measurements=[Measurement(0, 0), Measurement(2, 1)],
        )
        counts = paper_simulator.simulate(circuit, shots=SHOTS)
        total = sum(counts.values())
        prob_00 = counts.get("00", 0) / total

        # Theory: (1-p1)² × (1-readout_err)² ≈ 0.85
        assert 0.80 < prob_00 < 0.90, f"P(00) = {prob_00:.3f}, expected ~0.85"

    def test_bit_flip_rx_pi(self, paper_simulator):
        """Rx(π) on Q0 → mostly |10⟩."""
        circuit = Circuit(
            num_qubits=2,
            gates=[Gate("Rx", (0,), {"theta": np.pi})],
            measurements=[Measurement(0, 0), Measurement(1, 1)],
        )
        counts = paper_simulator.simulate(circuit, shots=SHOTS)
        total = sum(counts.values())
        prob_10 = counts.get("10", 0) / total
        assert 0.80 < prob_10 < 0.99, f"P(10) = {prob_10:.3f}"


# ── Entanglement tests ───────────────────────────────────────────

class TestEntanglement:

    def test_bell_state(self, paper_simulator):
        """Bell pair on Q0-Q2 (valid star-topology link)."""
        gates = [
            Gate("Rz", (0,), {"theta": np.pi}),
            Gate("Ry", (0,), {"theta": np.pi / 2}),
        ] + _cnot_via_rz_ry_cz(0, 2)

        circuit = Circuit(
            num_qubits=3, gates=gates,
            measurements=[Measurement(0, 0), Measurement(2, 1)],
        )
        counts = paper_simulator.simulate(circuit, shots=SHOTS)
        total = sum(counts.values())
        p_ent = (counts.get("00", 0) + counts.get("11", 0)) / total
        assert p_ent > 0.55, f"Entangled weight = {p_ent:.3f}"

    def test_ghz_3qubit(self, paper_simulator):
        """GHZ via hub Q2: CNOT(2→0), CNOT(2→1)."""
        gates = [
            Gate("Rz", (2,), {"theta": np.pi}),
            Gate("Ry", (2,), {"theta": np.pi / 2}),
        ]
        gates += _cnot_via_rz_ry_cz(2, 0)
        gates += _cnot_via_rz_ry_cz(2, 1)

        circuit = Circuit(
            num_qubits=3, gates=gates,
            measurements=[Measurement(i, i) for i in range(3)],
        )
        counts = paper_simulator.simulate(circuit, shots=SHOTS)
        total = sum(counts.values())
        p_ghz = (counts.get("000", 0) + counts.get("111", 0)) / total
        assert p_ghz > 0.45, f"GHZ weight = {p_ghz:.3f}"


# ── Statistical consistency ───────────────────────────────────────

class TestConsistency:

    def test_repeated_runs_consistent(self, paper_simulator):
        """Same circuit 5× should give P(0) within shot-noise bounds."""
        circuit = Circuit(
            num_qubits=2,
            gates=[Gate("Ry", (0,), {"theta": np.pi / 2})],
            measurements=[Measurement(0, 0)],
        )
        results = []
        for _ in range(5):
            counts = paper_simulator.simulate(circuit, shots=1000)
            results.append(counts.get("0", 0) / 1000)

        assert np.std(results) < 0.05, f"Std = {np.std(results):.3f}"


# ── Crosstalk and virtual-RZ behavior ────────────────────────────

class TestCrosstalkAndVirtualRZ:
    """Validate that:
    1. ZZ crosstalk reduces ⟨XX⟩ coherence when real time passes.
    2. Virtual RZ has non-zero scheduling duration but zero physical effect.
    """

    N_IDLE = 30
    SHOTS = 4096

    @staticmethod
    def _xx_expectation(counts):
        total = sum(counts.values())
        p00 = counts.get("00", 0) / total
        p11 = counts.get("11", 0) / total
        p01 = counts.get("01", 0) / total
        p10 = counts.get("10", 0) / total
        return (p00 + p11) - (p01 + p10)

    def _idle_circuit(self, n):
        gates = [
            Gate("Ry", (0,), {"theta": np.pi / 2}),
            Gate("Ry", (2,), {"theta": np.pi / 2}),
        ]
        for _ in range(n):
            gates.append(Gate("Ry", (0,), {"theta": 0.0}))
            gates.append(Gate("Ry", (2,), {"theta": 0.0}))
        gates.append(Gate("Ry", (0,), {"theta": -np.pi / 2}))
        gates.append(Gate("Ry", (2,), {"theta": -np.pi / 2}))
        return Circuit(
            num_qubits=3, gates=gates,
            measurements=[Measurement(0, 0), Measurement(2, 1)],
        )

    def test_crosstalk_reduces_xx(self):
        """⟨XX⟩ should drop when ZZ coupling is active."""
        device_no = create_paper_qpu()
        device_no.couplings[(0, 2)].J = 0.0
        sim_no = QiskitSimulator(device_no)

        device_yes = create_paper_qpu()
        sim_yes = QiskitSimulator(device_yes)

        circ = self._idle_circuit(self.N_IDLE)
        xx_no = self._xx_expectation(sim_no.simulate(circ, shots=self.SHOTS))
        xx_yes = self._xx_expectation(sim_yes.simulate(circ, shots=self.SHOTS))

        assert xx_yes < xx_no - 0.05, (
            f"⟨XX⟩ with crosstalk ({xx_yes:.3f}) should be lower than "
            f"without ({xx_no:.3f})"
        )

    def test_rz_scheduling_nonzero_duration(self, paper_device):
        """RZ must get non-zero scheduled duration for correct layer alignment."""
        circuit = Circuit(
            num_qubits=3,
            gates=[
                Gate("Rz", (0,), {"theta": np.pi}),
                Gate("Rz", (2,), {"theta": np.pi}),
            ],
            measurements=[Measurement(0, 0), Measurement(2, 1)],
        )
        layers = schedule_circuit_alap(circuit, paper_device)
        durations = [
            instr.duration
            for layer in layers
            for instr in layer.instructions.values()
        ]
        assert any(d > 0 for d in durations), "RZ should have non-zero scheduling duration"

    def test_rz_no_physical_effect(self, paper_device):
        """RZ-only circuit should produce same statistics as identity."""
        id_circuit = Circuit(
            num_qubits=3, gates=[],
            measurements=[Measurement(0, 0), Measurement(2, 1)],
        )
        rz_circuit = Circuit(
            num_qubits=3,
            gates=[
                Gate("Rz", (0,), {"theta": np.pi}),
                Gate("Rz", (2,), {"theta": np.pi}),
            ],
            measurements=[Measurement(0, 0), Measurement(2, 1)],
        )
        sim = QiskitSimulator(paper_device)
        shots = self.SHOTS

        def p_q0_zero(counts):
            total = sum(counts.values())
            return (counts.get("00", 0) + counts.get("01", 0)) / total

        p_id = p_q0_zero(sim.simulate(id_circuit, shots=shots))
        p_rz = p_q0_zero(sim.simulate(rz_circuit, shots=shots))

        assert abs(p_id - p_rz) < 0.03, (
            f"Virtual RZ changed P(q0=0) by {abs(p_id - p_rz):.3f}"
        )
