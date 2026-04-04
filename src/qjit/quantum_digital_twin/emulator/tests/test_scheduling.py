# tests/test_scheduling.py

"""
Test ALAP scheduling: layer grouping, idle times, parallel/sequential logic.
"""

import numpy as np
import pytest
from qjit.quantum_digital_twin.emulator.circuit import Circuit, Gate, Measurement
from qjit.quantum_digital_twin.emulator.scheduling import schedule_circuit_alap


class TestALAPScheduling:

    def test_empty_circuit(self, paper_device):
        circuit = Circuit(num_qubits=2, gates=[], measurements=[])
        layers = schedule_circuit_alap(circuit, paper_device)
        assert len(layers) == 0

    def test_single_gate_one_layer(self, paper_device):
        circuit = Circuit(
            num_qubits=2,
            gates=[Gate("Rx", (0,), {"theta": np.pi / 2})],
            measurements=[],
        )
        layers = schedule_circuit_alap(circuit, paper_device)
        assert len(layers) == 1
        assert 0 in layers[0].instructions
        assert layers[0].instructions[0].gate.name == "Rx"

    def test_parallel_gates_same_layer(self, paper_device):
        circuit = Circuit(
            num_qubits=3,
            gates=[
                Gate("Rx", (0,), {"theta": np.pi / 2}),
                Gate("Ry", (1,), {"theta": np.pi / 2}),
            ],
            measurements=[],
        )
        layers = schedule_circuit_alap(circuit, paper_device)
        assert len(layers) == 1
        assert 0 in layers[0].instructions
        assert 1 in layers[0].instructions

    def test_sequential_gates_different_layers(self, paper_device):
        circuit = Circuit(
            num_qubits=2,
            gates=[
                Gate("Rx", (0,), {"theta": np.pi / 2}),
                Gate("Ry", (0,), {"theta": np.pi}),
            ],
            measurements=[],
        )
        layers = schedule_circuit_alap(circuit, paper_device)
        assert len(layers) == 2

    def test_two_qubit_gate_creates_dependency(self, paper_device):
        circuit = Circuit(
            num_qubits=3,
            gates=[
                Gate("Rx", (0,), {"theta": np.pi / 2}),
                Gate("CZ", (0, 2), {}),
            ],
            measurements=[],
        )
        layers = schedule_circuit_alap(circuit, paper_device)
        assert len(layers) == 2


class TestIdleTimes:
    """Idle time (Δε_ji) calculation when gates have different durations."""

    def test_mixed_duration_idle_times(self, paper_device):
        """Rx (32 ns) + CZ (45 ns) in same layer → Q0 idles 13 ns."""
        circuit = Circuit(
            num_qubits=3,
            gates=[
                Gate("Rx", (0,), {"theta": np.pi / 2}),   # 32 ns
                Gate("CZ", (1, 2), {}),                     # 45 ns
            ],
            measurements=[],
        )
        layers = schedule_circuit_alap(circuit, paper_device)
        layer = layers[0]

        assert abs(layer.idle_times[0] - 13e-9) < 1e-10, "Q0 should idle 13 ns"
        assert layer.idle_times[1] == 0, "Q1 in CZ — not idle"
        assert layer.idle_times[2] == 0, "Q2 in CZ — not idle"

    def test_idle_qubit_gets_full_layer(self, paper_device):
        circuit = Circuit(
            num_qubits=2,
            gates=[Gate("Rx", (0,), {"theta": np.pi / 2})],
            measurements=[],
        )
        layers = schedule_circuit_alap(circuit, paper_device)
        assert layers[0].idle_times[1] == layers[0].layer_duration
