# qjit/quantum_digital_twin/drift_experiments/steane_drift_syndrome.py

import time
from typing import Dict, Tuple

from qiskit import transpile
from qiskit.transpiler import CouplingMap

from .steane_drift_snapshots import SteaneDriftExperiment
from .syndrome_extraction import (
    build_steane_circuit,
    build_steane_circuit_with_physical_X,
    from_qiskit_to_internal,
)
from ..emulator.qiskit_emulator_core import QiskitSimulator
from ..emulator.device_model import DeviceModel


# ─── Reusable syndrome extraction on any device ───────────────────

# Cache the transpiled circuit so we don't rebuild it every call
_CACHED_CIRCUIT = None
_CACHED_EDGES = None


def run_syndrome_on_device(
    device: DeviceModel,
    shots: int = 200,
    verbose: bool = False,
) -> Dict[str, int]:
    """
    Run Steane syndrome extraction on a given device, return 6-bit syndrome counts.

    This is the core reusable function used by the calibration loop.
    It builds the circuit once and caches it for subsequent calls.

    Parameters
    ----------
    device : DeviceModel
        The device to simulate (can be drifted, corrected, etc.)
    shots : int
        Number of measurement shots.
    verbose : bool
        Print timing info.

    Returns
    -------
    syndrome_counts : dict
        Maps 6-bit syndrome strings (e.g. '000000') to counts.
        Bit ordering: (a0,a1,a2,a3,a4,a5) in logical Steane order.
        Bits 0,1,2 = Z-type stabilizers (detect X errors)
        Bits 3,4,5 = X-type stabilizers (detect Z errors / drift)
    """
    global _CACHED_CIRCUIT, _CACHED_EDGES

    edges = list(device.couplings.keys())

    # Rebuild circuit only if device topology changed
    if _CACHED_CIRCUIT is None or _CACHED_EDGES != edges:
        qc, _ = build_steane_circuit()
        cmap = CouplingMap(edges)
        layout = list(range(len(device.qubits)))

        qcT = transpile(
            qc,
            basis_gates=["rx", "ry", "rz", "cz", "reset", "measure"],
            coupling_map=cmap,
            initial_layout=layout,
            optimization_level=1,
        )

        _CACHED_CIRCUIT = from_qiskit_to_internal(qcT)
        _CACHED_EDGES = edges

    # Simulate
    sim = QiskitSimulator(device)
    if verbose:
        t0 = time.time()

    counts = sim.simulate(_CACHED_CIRCUIT, shots=shots)

    if verbose:
        print(f"[SIM TIME] {time.time() - t0:.3f} s")

    # Reduce to 6-bit syndromes
    return _extract_syndromes(counts)


def _extract_syndromes(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Reduce 13-bit measurement outcomes to 6-bit syndrome strings.

    Input bitstrings: full_bits[0]=q12 ... full_bits[5]=q7 (ancilla)
    Output: syndrome in logical Steane order
        (a0,a1,a2,a3,a4,a5) = (q7,q8,q9,q10,q11,q12)
    """
    syndrome_counts = {}
    for full_bits, count in counts.items():
        anc = full_bits[:6]

        # Convert Qiskit format [q12 q11 q10 q9 q8 q7]
        # into logical Steane order [q7 q8 q9 q10 q11 q12]
        syndrome = anc[5] + anc[4] + anc[3] + anc[2] + anc[1] + anc[0]

        syndrome_counts[syndrome] = syndrome_counts.get(syndrome, 0) + count

    return syndrome_counts


# ─── Original standalone experiment (kept for backward compat) ────

def run_steane_syndrome_drift_shots(dt=1.0, drift_std_hz=200.0, shots=100):
    """
    Run Steane syndrome extraction before and after drift.
    Original standalone function — creates its own experiment object.

    For use in the calibration loop, use run_syndrome_on_device() instead.
    """

    # 1) INITIALIZE DEVICE + SNAPSHOT A
    exp = SteaneDriftExperiment(drift_std_hz=drift_std_hz)
    device_A = exp.device
    print("\n--- SNAPSHOT A (before drift) ---")
    exp.history[-1][2].pretty_print()

    # 2) SIMULATE BEFORE DRIFT
    print("\n=== BEFORE DRIFT ===")
    syndA = run_syndrome_on_device(device_A, shots=shots, verbose=True)

    print("\n--- SYNDROMES BEFORE DRIFT ---")
    for s, count in sorted(syndA.items()):
        print(f"syndrome {s} : {count}   (a0,a1,a2,a3,a4,a5 = {s[0]} {s[1]} {s[2]} {s[3]} {s[4]} {s[5]})")
    print(f"P(000000) before drift = {syndA.get('000000', 0) / shots:.4f}")

    # 3) APPLY DRIFT
    print("\n--- APPLYING DRIFT ---")
    exp.step_drift(dt)
    device_B = exp.device
    exp.history[-1][2].pretty_print()

    # 4) SIMULATE AFTER DRIFT
    print("\n=== AFTER DRIFT ===")
    syndB = run_syndrome_on_device(device_B, shots=shots, verbose=True)

    print("\n--- SYNDROMES AFTER DRIFT ---")
    for s, count in sorted(syndB.items()):
        print(f"syndrome {s} : {count}   (a0,a1,a2,a3,a4,a5 = {s[0]} {s[1]} {s[2]} {s[3]} {s[4]} {s[5]})")
    print(f"P(000000) after drift = {syndB.get('000000', 0) / shots:.4f}")

    print(
        "\nΔP(000000) = ",
        (syndB.get("000000", 0) / shots) - (syndA.get("000000", 0) / shots),
    )

    return syndA, syndB


if __name__ == "__main__":
    run_steane_syndrome_drift_shots(
        dt=7200.0,
        drift_std_hz=300.0,
        shots=100,
    )











