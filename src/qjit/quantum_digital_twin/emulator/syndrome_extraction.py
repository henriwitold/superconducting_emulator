# src/qjit/quantum_digital_twin/emulator/syndrome_extraction.py

import numpy as np
from qiskit import QuantumRegister, ClassicalRegister
from ...experiments.error_counts.ecc.steane import (
    SteaneCodeCircuit,
    append_multi_steane_syndrome_extraction,
)
from .circuit import Circuit, Gate, Measurement


def build_steane_syndrome_only_circuit():
    """
    Prepare |0_L> and perform one full syndrome extraction.
    No injected logical errors.
    """
    circuit = SteaneCodeCircuit(logical_qubit_count=1)
    qc = circuit.physical_quantum_circuit  # qubits 0–6 = data
    ancilla = QuantumRegister(6, "ancilla")
    qc.add_register(ancilla)

    circuit.encode()  # prepare logical |0_L>

    # --- Syndrome extraction round ---
    syndrome = ClassicalRegister(6, "syndrome")
    qc.add_register(syndrome)
    append_multi_steane_syndrome_extraction(qc, qc.qubits[:7], ancilla[:])
    for i in range(6):
        qc.measure(ancilla[i], syndrome[i])
        qc.reset(ancilla[i])

    # --- Optional: final data measurement ---
    data_out = ClassicalRegister(7, "data")
    qc.add_register(data_out)
    for i in range(7):
        qc.measure(qc.qubits[i], data_out[i])

    params = {
        "num_blocks": 1,
        "syndrome_measurement_rounds": 1,
        "block_size": 7,
        "ancilla_block_size": 6,
        "num_data_bits": 7,
    }

    return qc, params


# ============================================================
# Conversion: Qiskit → Internal representation
# ============================================================

def from_qiskit_to_internal(qc):
    """
    Convert a Qiskit QuantumCircuit into our internal Circuit dataclass.

    - Keeps only gates supported by the emulator (Rx, Ry, Rz, CZ, Reset)
    - Translates CX → H(target)·CZ·H(target) using H ≈ Rx(pi)·Ry(pi/2)
    - Extracts rotation parameters as dicts for scheduling
    - Drops barriers/delays (purely visual or timing markers)
    """
    gates = []
    measurements = []

    for instr, qargs, cargs in qc.data:
        name = instr.name.lower()

        # Skip purely visual ops
        if name in {"barrier", "delay"}:
            continue

        # --- Measurements ---
        if name == "measure":
            qid = qc.find_bit(qargs[0]).index
            cid = qc.find_bit(cargs[0]).index
            measurements.append(Measurement(qubit=qid, classical_bit=cid))
            continue

        # --- Determine qubits ---
        qubits = tuple(qc.find_bit(q).index for q in qargs)

        # --- CX → CZ rewrite ---
        if name == "cx":
            ctrl, tgt = qubits
            gates.append(Gate(name="Rx", qubits=(tgt,), params={"theta": np.pi}))
            gates.append(Gate(name="Ry", qubits=(tgt,), params={"theta": np.pi / 2}))
            gates.append(Gate(name="CZ", qubits=(ctrl, tgt), params={}))
            gates.append(Gate(name="Ry", qubits=(tgt,), params={"theta": -np.pi / 2}))
            gates.append(Gate(name="Rx", qubits=(tgt,), params={"theta": -np.pi}))
            continue

        # --- Supported native rotations ---
        if name in {"rx", "ry", "rz"}:
            theta = float(instr.params[0]) if instr.params else 0.0
            gates.append(Gate(name=name.capitalize(), qubits=qubits, params={"theta": theta}))
            continue

        # --- Direct CZ passthrough ---
        if name == "cz":
            gates.append(Gate(name="CZ", qubits=qubits, params={}))
            continue

        # --- Reset passthrough ---
        if name == "reset":
            # single-qubit reset gate
            gates.append(Gate(name="Reset", qubits=qubits, params={}))
            continue

        # Unrecognized gate → warning
        print(f"[WARN] Unsupported gate '{name}' ignored.")

    return Circuit(num_qubits=qc.num_qubits, gates=gates, measurements=measurements)
