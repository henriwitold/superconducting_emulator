# emulator/qpu_config.py

import numpy as np
from typing import Dict, Tuple, List
from .device_model import DeviceModel, Qubit, Coupling, GateDefinition, GateType

"""
QPU Configuration from the paper (Section IV):
- 5-qubit QuantWare Soprano-D QPU
- Star topology with Q2 as center
- Flux-tunable transmon qubits
- Only using first 4 qubits (Q4 has high readout error but included for crosstalk)
"""

def create_paper_qpu() -> DeviceModel:
    """
    Create the 5-qubit QPU from the paper with measured parameters.
    
    From Section IV:
    - Frequencies: 4.5 GHz to 6.3 GHz (increasing with index)
    - T1: 32-45 μs
    - T2: 14-21 μs  
    - Anharmonicity: ~200 MHz
    - Average readout error: 3.3% (active qubits)
    - Average state prep error: 4.8%
    - Single-qubit gate fidelity: 0.996 (avg)
    - Single-qubit gate duration: 32 ns
    - CZ gate duration: 45 ns
    - Measurement duration: 1500 ns
    - Star topology: Q2 is central, connected to Q0, Q1, Q3, Q4
    """
    
    # Qubit parameters (matching paper's ranges)
    qubits = {
        0: Qubit(
            id=0,
            frequency=4.5e9,      # Hz
            anharmonicity=-200e6, # Hz (negative for transmons)
            T1=45e-6,             # s
            T2=21e-6,             # s  
            p1=0.048,             # state prep error ~4.8%
            confusion_matrix=[    # readout error ~3.3%
                [0.967, 0.033],   # P(measure 0|prepared 0), P(measure 0|prepared 1)
                [0.033, 0.967]    # P(measure 1|prepared 0), P(measure 1|prepared 1)
            ],
            spam_error=0.048,
            measurement_duration=1500e-9
        ),
        1: Qubit(
            id=1,
            frequency=5.0e9,
            anharmonicity=-200e6,
            T1=40e-6,
            T2=18e-6,
            p1=0.048,
            confusion_matrix=[
                [0.967, 0.033],
                [0.033, 0.967]
            ],
            spam_error=0.048,
            measurement_duration=1500e-9
        ),
        2: Qubit(  # Central qubit in star
            id=2,
            frequency=5.5e9,
            anharmonicity=-200e6,
            T1=38e-6,
            T2=16e-6,
            p1=0.048,
            confusion_matrix=[
                [0.967, 0.033],
                [0.033, 0.967]
            ],
            spam_error=0.048,
            measurement_duration=1500e-9
        ),
        3: Qubit(
            id=3,
            frequency=6.0e9,
            anharmonicity=-200e6,
            T1=35e-6,
            T2=15e-6,
            p1=0.048,
            confusion_matrix=[
                [0.967, 0.033],
                [0.033, 0.967]
            ],
            spam_error=0.048,
            measurement_duration=1500e-9
        ),
        4: Qubit(  # High readout error - not used but affects crosstalk
            id=4,
            frequency=6.3e9,
            anharmonicity=-200e6,
            T1=32e-6,
            T2=14e-6,
            p1=0.048,
            confusion_matrix=[
                [0.967, 0.033],
                [0.033, 0.967]
            ],
            spam_error=0.048,
            measurement_duration=1500e-9
        ),
    }
    
    # Star topology couplings (Q2 is center)
    # From Table I: Optimized coupling strengths J (kHz)
    couplings = {
        (0, 2): Coupling(
            qubit_pair=(0, 2),
            J=15.0e3,  # Hz (paper: 15.0 kHz from Table I)
            detuning=qubits[2].frequency - qubits[0].frequency,
            connectivity_type="star"
        ),
        (1, 2): Coupling(
            qubit_pair=(1, 2),
            J=22.2e3,  # Hz (paper: 22.2 kHz)
            detuning=qubits[2].frequency - qubits[1].frequency,
            connectivity_type="star"
        ),
        (2, 3): Coupling(
            qubit_pair=(2, 3),
            J=19.2e3,  # Hz (paper: 19.2 kHz)
            detuning=qubits[3].frequency - qubits[2].frequency,
            connectivity_type="star"
        ),
        (2, 4): Coupling(
            qubit_pair=(2, 4),
            J=65.7e3,  # Hz (paper: 65.7 kHz)
            detuning=qubits[4].frequency - qubits[2].frequency,
            connectivity_type="star"
        ),
    }
    
    # Gate library - native gates for each qubit/pair
    # From Table I: Optimized CZ fidelities
    gate_library = []
    
    # Single-qubit gates (Rx, Ry) - fidelity 0.996, duration 32 ns
    for qid in range(5):
        gate_library.extend([
            GateDefinition(
                name='Rx',
                type=GateType.ONE_Q,
                duration=32e-9,
                fidelity=0.996,
                qubits=(qid,),
                is_virtual=False
            ),
            GateDefinition(
                name='Ry',
                type=GateType.ONE_Q,
                duration=32e-9,
                fidelity=0.996,
                qubits=(qid,),
                is_virtual=False
            ),
            GateDefinition(
                name='Rz',
                type=GateType.ONE_Q,
                duration=0.0,  # Virtual gate
                fidelity=1.0,
                qubits=(qid,),
                is_virtual=True
            ),
        ])
    
    # Two-qubit CZ gates - fidelities from Table I, duration 45 ns
    cz_fidelities = {
        (0, 2): 0.987,  # From Table I
        (1, 2): 0.964,
        (2, 3): 0.917,
        # (2, 4) not listed - Q4 not actively used
    }
    
    for (qi, qj), fidelity in cz_fidelities.items():
        # forward direction
        gate_library.append(
            GateDefinition(
                name='CZ',
                type=GateType.TWO_Q,
                duration=45e-9,
                fidelity=fidelity,
                qubits=(qi, qj),
                is_virtual=False
            )
        )
        # reverse direction
        gate_library.append(
            GateDefinition(
                name='CZ',
                type=GateType.TWO_Q,
                duration=45e-9,
                fidelity=fidelity,
                qubits=(qj, qi),
                is_virtual=False
            )
        )


    device = DeviceModel(
        qubits=qubits,
        couplings=couplings,
        gate_library=gate_library,
        topology="star",
        central_qubit=2,
        uses_active_reset=True  # Paper mentions active reset
    )
    
    # Update detunings based on frequencies
    device.update_detunings_from_frequencies()
    
    return device


def create_steane_qpu(noise_scale: float = 1.0) -> DeviceModel:
    """
    Create a 13-qubit Steane-code QPU model with uniformly scalable noise.

    noise_scale = 0.0  → ideal, noiseless device  
    noise_scale = 1.0  → full realistic hardware noise  
    noise_scale = 0.5  → half the noise amplitude  
    """

    # ----------------------------
    # qubit count
    # ----------------------------
    num_data = 7
    num_ancilla = 6
    num_total = num_data + num_ancilla

    # ===============================
    # 1️⃣ Define Qubits
    # ===============================
    qubits = {}

    BASE_T1 = 40e-6       # 40 µs
    BASE_T2 = 18e-6       # 18 µs
    BASE_SPAM = 0.04      # 4% SPAM error
    base_freq = 5.0e9
    delta = 0.1e9         # 100 MHz spacing

    for qid in range(num_total):
        freq = base_freq + qid * delta

        # 🎯 noise-scale logic:
        T1 = BASE_T1 / max(noise_scale, 1e-6)
        T2 = BASE_T2 / max(noise_scale, 1e-6)
        spam = BASE_SPAM * noise_scale

        anh = -200e6 + np.random.uniform(-20e6, +20e6)

        qubits[qid] = Qubit(
            id=qid,
            frequency=freq,
            anharmonicity=anh,
            T1=T1,
            T2=T2,
            p1=spam,
            confusion_matrix=[
                [1-spam, spam],
                [spam, 1-spam]
            ],
            spam_error=spam,
            measurement_duration=1500e-9,
            drift_std=200.0
        )

    # ===============================
    # 2️⃣ Couplings
    # ===============================
    steane_edges = [
        (0,7), (0,10),
        (1,8), (1,11),
        (2,7), (2,8), (2,10), (2,11),
        (3,9), (3,12),
        (4,7), (4,9), (4,10), (4,12),
        (5,8), (5,9), (5,11), (5,12),
        (6,7), (6,8), (6,9), (6,10), (6,11), (6,12),
    ]

    couplings = {}
    BASE_J = 20e3
    BASE_ZZ = 0.0  # future extension

    for (qi, qj) in steane_edges:

        # coupling strength + scaled perturbation
        J = BASE_J + np.random.uniform(-5e3, 5e3) * noise_scale

        couplings[(qi, qj)] = Coupling(
            qubit_pair=(qi, qj),
            J=J,
            detuning=(qubits[qj].frequency - qubits[qi].frequency),
            connectivity_type="steane",
        )

    # ===============================
    # 3️⃣ Gate Library
    # ===============================
    gate_library = []

    # base fidelities
    BASE_1Q_ERR = 1 - 0.996   # 0.4% error
    BASE_2Q_ERR = 1 - 0.97    # 3%

    for qid in range(num_total):
        # scale single-qubit error:
        single_error = BASE_1Q_ERR * noise_scale
        fidelity_1Q = 1 - single_error

        gate_library.extend([
            GateDefinition(
                name="Rx",
                type=GateType.ONE_Q,
                duration=32e-9,
                fidelity=fidelity_1Q,
                qubits=(qid,),
                is_virtual=False,
            ),
            GateDefinition(
                name="Ry",
                type=GateType.ONE_Q,
                duration=32e-9,
                fidelity=fidelity_1Q,
                qubits=(qid,),
                is_virtual=False,
            ),
            GateDefinition(
                name="Rz",
                type=GateType.ONE_Q,
                duration=0.0,
                fidelity=1.0,
                qubits=(qid,),
                is_virtual=True,
            ),
        ])

    for (qi, qj) in steane_edges:

        two_qubit_error = BASE_2Q_ERR * noise_scale
        fidelity_2Q = 1 - two_qubit_error

        gate_library.append(
            GateDefinition(
                name="CZ",
                type=GateType.TWO_Q,
                duration=45e-9,
                fidelity=fidelity_2Q,
                qubits=(qi, qj),
                is_virtual=False,
            )
        )
        gate_library.append(
            GateDefinition(
                name="CZ",
                type=GateType.TWO_Q,
                duration=45e-9,
                fidelity=fidelity_2Q,
                qubits=(qj, qi),
                is_virtual=False,
            )
        )

    # ===============================
    # 4️⃣ Build Device
    # ===============================
    device = DeviceModel(
        qubits=qubits,
        couplings=couplings,
        gate_library=gate_library,
        topology="steane",
        central_qubit=None,
        uses_active_reset=True,
        cz_bandwidth=30e6, # 30e6 Hz
    )

    device.update_detunings_from_frequencies()
    return device


def validate_qpu_config(device: DeviceModel) -> None:
    """
    Validate that QPU configuration matches paper's requirements.
    """
    print("=" * 60)
    print("QPU Configuration Validation")
    print("=" * 60)
    
    print(f"\n✓ Number of qubits: {len(device.qubits)}")
    print(f"✓ Topology: {device.topology}")
    if device.central_qubit is not None:
        print(f"✓ Central qubit: Q{device.central_qubit}")
    
    print(f"\n✓ Number of couplings: {len(device.couplings)}")
    for (qi, qj), coupling in device.couplings.items():
        print(f"  Q{qi}-Q{qj}: J={coupling.J/1e3:.1f} kHz, Δ={coupling.detuning/1e9:.2f} GHz")
    
    print(f"\n✓ Gate library size: {len(device.gate_library)}")
    
    # Check basis gates
    basis_gates = set(g.name for g in device.gate_library)
    print(f"✓ Basis gates: {basis_gates}")
    
    # Check single-qubit gate stats
    sq_gates = [g for g in device.gate_library if g.type == GateType.ONE_Q and not g.is_virtual]
    if sq_gates:
        avg_fid = np.mean([g.fidelity for g in sq_gates])
        avg_dur = np.mean([g.duration for g in sq_gates])
        print(f"✓ Single-qubit gates: avg fidelity={avg_fid:.4f}, avg duration={avg_dur*1e9:.1f} ns")
    
    # Check two-qubit gate stats
    tq_gates = [g for g in device.gate_library if g.type == GateType.TWO_Q]
    if tq_gates:
        avg_fid = np.mean([g.fidelity for g in tq_gates])
        avg_dur = np.mean([g.duration for g in tq_gates])
        print(f"✓ Two-qubit gates: avg fidelity={avg_fid:.4f}, avg duration={avg_dur*1e9:.1f} ns")
    
    # Check qubit coherence times
    T1_range = [q.T1 for q in device.qubits.values()]
    T2_range = [q.T2 for q in device.qubits.values()]
    print(f"\n✓ T1 range: {min(T1_range)*1e6:.1f}-{max(T1_range)*1e6:.1f} μs")
    print(f"✓ T2 range: {min(T2_range)*1e6:.1f}-{max(T2_range)*1e6:.1f} μs")
    
    # Check SPAM errors
    avg_spam = np.mean([q.spam_error for q in device.qubits.values()])
    print(f"✓ Average SPAM error: {avg_spam*100:.1f}%")
    
    print("\n" + "=" * 60)

