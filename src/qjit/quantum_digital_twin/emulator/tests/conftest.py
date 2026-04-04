# tests/conftest.py

"""
Shared pytest fixtures for the quantum digital twin test suite.
"""

import pytest
from qjit.quantum_digital_twin.emulator.qpu_config import create_paper_qpu, create_steane_qpu
from qjit.quantum_digital_twin.emulator.qiskit_emulator_core import QiskitSimulator
from qjit.quantum_digital_twin.emulator.device_model import DeviceModel


@pytest.fixture
def paper_device() -> DeviceModel:
    """5-qubit star-topology QPU from paper (Section IV)."""
    return create_paper_qpu()


@pytest.fixture
def steane_device() -> DeviceModel:
    """13-qubit Steane-code QPU with full noise."""
    return create_steane_qpu(noise_scale=1.0)


@pytest.fixture
def paper_simulator(paper_device) -> QiskitSimulator:
    """Qiskit Aer simulator backed by the paper QPU."""
    return QiskitSimulator(paper_device)
