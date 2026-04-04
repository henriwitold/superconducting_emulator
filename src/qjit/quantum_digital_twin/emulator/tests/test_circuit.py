# tests/test_circuit.py

"""
Test circuit dataclass creation: gates, measurements, and composite circuits.
"""

import numpy as np
import pytest
from qjit.quantum_digital_twin.emulator.circuit import Circuit, Gate, Measurement


class TestGateCreation:

    def test_single_qubit_gate(self):
        g = Gate("Rx", (0,), {"theta": np.pi / 2})
        assert g.name == "Rx"
        assert g.qubits == (0,)
        assert g.params["theta"] == np.pi / 2

    def test_two_qubit_gate(self):
        g = Gate("CZ", (0, 1), {})
        assert g.name == "CZ"
        assert g.qubits == (0, 1)
        assert g.params == {}


class TestMeasurementCreation:

    def test_basic_measurement(self):
        m = Measurement(qubit=0, classical_bit=0)
        assert m.qubit == 0
        assert m.classical_bit == 0


class TestCircuitCreation:

    def test_identity_circuit(self):
        c = Circuit(num_qubits=2, gates=[], measurements=[
            Measurement(0, 0), Measurement(1, 1),
        ])
        assert c.num_qubits == 2
        assert len(c.gates) == 0
        assert len(c.measurements) == 2

    def test_bell_state_circuit(self):
        c = Circuit(
            num_qubits=2,
            gates=[
                Gate("Ry", (0,), {"theta": np.pi / 2}),
                Gate("CZ", (0, 1), {}),
            ],
            measurements=[Measurement(0, 0), Measurement(1, 1)],
        )
        assert len(c.gates) == 2
        assert c.gates[0].name == "Ry"
        assert c.gates[1].name == "CZ"

    def test_ghz_circuit(self):
        c = Circuit(
            num_qubits=3,
            gates=[
                Gate("Ry", (0,), {"theta": np.pi / 2}),
                Gate("Ry", (1,), {"theta": np.pi / 2}),
                Gate("Ry", (2,), {"theta": np.pi / 2}),
                Gate("CZ", (0, 2), {}),
                Gate("CZ", (1, 2), {}),
            ],
            measurements=[Measurement(i, i) for i in range(3)],
        )
        assert c.num_qubits == 3
        assert len(c.gates) == 5
