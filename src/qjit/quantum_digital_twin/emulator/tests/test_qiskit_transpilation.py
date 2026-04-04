# tests/test_qiskit_transpilation.py

"""
Test Qiskit circuit transpilation: noise instruction insertion and structure.
"""

import numpy as np
import pytest
from qjit.quantum_digital_twin.emulator.circuit import Circuit, Gate, Measurement
from qjit.quantum_digital_twin.emulator.scheduling import schedule_circuit_alap


class TestSimulatorCreation:

    def test_num_qubits(self, paper_simulator):
        assert paper_simulator.num_qubits == 5

    def test_backend_exists(self, paper_simulator):
        assert paper_simulator.simulator is not None


class TestTranspilation:

    def _build(self, simulator, circuit):
        layers = schedule_circuit_alap(circuit, simulator.device)
        return simulator._build_noisy_circuit(circuit, layers)

    def test_empty_circuit(self, paper_simulator):
        circuit = Circuit(
            num_qubits=2, gates=[],
            measurements=[Measurement(0, 0), Measurement(1, 1)],
        )
        qc = self._build(paper_simulator, circuit)
        assert qc.num_qubits == 5
        assert "measure" in qc.count_ops()

    def test_single_gate_has_rx(self, paper_simulator):
        circuit = Circuit(
            num_qubits=2,
            gates=[Gate("Rx", (0,), {"theta": np.pi / 2})],
            measurements=[Measurement(0, 0)],
        )
        qc = self._build(paper_simulator, circuit)
        assert "rx" in qc.count_ops()

    def test_bell_circuit_has_cz(self, paper_simulator):
        circuit = Circuit(
            num_qubits=2,
            gates=[
                Gate("Ry", (0,), {"theta": np.pi / 2}),
                Gate("CZ", (0, 2), {}),
            ],
            measurements=[Measurement(0, 0), Measurement(1, 1)],
        )
        qc = self._build(paper_simulator, circuit)
        ops = qc.count_ops()
        assert "ry" in ops
        assert "cz" in ops

    def test_noise_instructions_present(self, paper_simulator):
        circuit = Circuit(
            num_qubits=2,
            gates=[Gate("Rx", (0,), {"theta": np.pi / 2})],
            measurements=[Measurement(0, 0)],
        )
        qc = self._build(paper_simulator, circuit)
        num_kraus = sum(
            1 for instr in qc.data if instr.operation.name == "kraus"
        )
        num_gates = sum(
            1 for instr in qc.data
            if instr.operation.name in ("rx", "ry", "rz", "cz")
        )
        assert num_kraus > 0, "Should have Kraus noise instructions"
        assert num_gates > 0, "Should have quantum gates"
