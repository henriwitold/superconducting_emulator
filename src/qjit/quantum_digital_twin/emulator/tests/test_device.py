# tests/test_device.py

"""
Test device creation, qubit parameters, couplings, and gate library.
"""

import numpy as np
import pytest
from qjit.quantum_digital_twin.emulator.qpu_config import create_paper_qpu, create_steane_qpu
from qjit.quantum_digital_twin.emulator.device_model import DeviceModel, GateType


class TestDeviceCreation:
    """Verify that QPU factory functions produce valid devices."""

    def test_paper_qpu_structure(self, paper_device):
        assert isinstance(paper_device, DeviceModel)
        assert len(paper_device.qubits) == 5
        assert paper_device.topology == "star"
        assert paper_device.central_qubit == 2

    def test_steane_qpu_structure(self):
        device = create_steane_qpu(noise_scale=1.0)
        assert len(device.qubits) == 13
        assert device.topology == "steane"


class TestQubitParameters:
    """Validate qubit parameters against paper specifications (Section IV)."""

    def test_frequency_range(self, paper_device):
        for qid, q in paper_device.qubits.items():
            assert 4.5e9 <= q.frequency <= 6.5e9, f"Q{qid} frequency out of range"

    def test_t1_range(self, paper_device):
        for qid, q in paper_device.qubits.items():
            assert 30e-6 <= q.T1 <= 50e-6, f"Q{qid} T1 out of range"

    def test_t2_range(self, paper_device):
        for qid, q in paper_device.qubits.items():
            assert 10e-6 <= q.T2 <= 25e-6, f"Q{qid} T2 out of range"

    def test_anharmonicity(self, paper_device):
        for qid, q in paper_device.qubits.items():
            assert abs(q.anharmonicity) > 150e6, f"Q{qid} anharmonicity too small"

    def test_true_frequency_initialized(self, paper_device):
        for qid, q in paper_device.qubits.items():
            assert q.true_frequency == q.frequency, (
                f"Q{qid} true_frequency should match calibrated frequency at init"
            )


class TestCouplings:
    """Verify star topology: Q2 is center, connected to Q0, Q1, Q3, Q4."""

    def test_coupling_count(self, paper_device):
        assert len(paper_device.couplings) == 4

    def test_all_couplings_include_center(self, paper_device):
        for qi, qj in paper_device.couplings:
            assert 2 in (qi, qj), f"Coupling ({qi},{qj}) should include center Q2"

    def test_connectivity_type(self, paper_device):
        for coupling in paper_device.couplings.values():
            assert coupling.connectivity_type == "star"


class TestGateLibrary:
    """Check that the native gate set is complete and correct."""

    def test_single_qubit_gate_count(self, paper_device):
        one_q = [g for g in paper_device.gate_library if g.type == GateType.ONE_Q]
        assert len(one_q) == 15  # Rx, Ry, Rz × 5 qubits

    def test_two_qubit_gate_count(self, paper_device):
        two_q = [g for g in paper_device.gate_library if g.type == GateType.TWO_Q]
        assert len(two_q) == 6  # CZ forward+reverse for 3 active pairs

    def test_rx_duration_32ns(self, paper_device):
        rx = next(g for g in paper_device.gate_library if g.name == "Rx")
        assert abs(rx.duration - 32e-9) < 1e-10

    def test_rx_fidelity(self, paper_device):
        rx = next(g for g in paper_device.gate_library if g.name == "Rx")
        assert rx.fidelity >= 0.99

    def test_cz_duration_45ns(self, paper_device):
        cz = next(g for g in paper_device.gate_library if g.type == GateType.TWO_Q)
        assert abs(cz.duration - 45e-9) < 1e-10

    def test_cz_fidelity(self, paper_device):
        cz = next(g for g in paper_device.gate_library if g.type == GateType.TWO_Q)
        assert cz.fidelity >= 0.9

    def test_rz_is_virtual(self, paper_device):
        rz = next(g for g in paper_device.gate_library if g.name == "Rz")
        assert rz.is_virtual
        assert rz.duration == 0.0
        assert rz.fidelity == 1.0


class TestDeviceInfo:
    """Test the simulator's device info utility."""

    def test_device_info_fields(self, paper_simulator):
        info = paper_simulator.get_device_info()
        assert info["num_qubits"] == 5
        assert info["topology"] == "star"
        assert info["num_couplings"] == 4