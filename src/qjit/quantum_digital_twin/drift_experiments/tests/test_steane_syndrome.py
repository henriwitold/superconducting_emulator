# tests/test_steane_syndrome.py

"""
Steane syndrome extraction: verify clean syndrome and X-error detection.
These tests use Qiskit transpilation + ideal Aer simulation (no custom noise),
so they validate circuit correctness independently of the emulator noise model.
"""

import pytest
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from qjit.quantum_digital_twin.drift_experiments.steane_drift_snapshots import (
    SteaneDriftExperiment,
)
from qjit.quantum_digital_twin.drift_experiments.syndrome_extraction import (
    build_steane_circuit,
    build_steane_circuit_with_physical_X,
)


def _run_steane_ideal(inject_x: bool, shots: int = 5000):
    """Run Steane syndrome circuit on ideal Aer simulator, return syndrome histogram."""
    exp = SteaneDriftExperiment(drift_std_hz=0.0)
    device = exp.device

    qc, _ = (
        build_steane_circuit_with_physical_X() if inject_x else build_steane_circuit()
    )

    edges = list(device.couplings.keys())
    qc = transpile(
        qc,
        basis_gates=["rx", "ry", "rz", "cz", "reset", "measure"],
        coupling_map=CouplingMap(edges),
        initial_layout=list(range(len(device.qubits))),
        optimization_level=1,
    )

    counts = AerSimulator().run(qc, shots=shots).result().get_counts()

    syndromes = {}
    for bitstring, count in counts.items():
        s = bitstring[-6:]
        syndromes[s] = syndromes.get(s, 0) + count

    return syndromes, shots


class TestSteaneSyndromeClean:

    def test_trivial_syndrome_dominates(self):
        """Without errors, syndrome 000000 should dominate."""
        syndromes, shots = _run_steane_ideal(inject_x=False)
        p0 = syndromes.get("000000", 0) / shots
        assert p0 > 0.90, f"P(000000) = {p0:.3f}, expected > 0.90"


class TestSteaneSyndromeWithError:

    def test_physical_x_triggers_nontrivial_syndrome(self):
        """Physical X on q0 should produce non-trivial Z-stabilizer firings."""
        syndromes, shots = _run_steane_ideal(inject_x=True)
        p0 = syndromes.get("000000", 0) / shots
        assert p0 < 0.50, (
            f"P(000000) = {p0:.3f} with X error — should be <0.50"
        )
