# emulator/qiskit_emulator_core.py

"""
Qiskit Aer density matrix simulator with custom noise model.

Implements the paper's approach (Section III):
- Custom noise transpilation into Qiskit circuit instructions
- Qiskit Aer density matrix simulation for fast, validated execution

Flow:
  Circuit → Schedule (ALAP) → Transpile (add noise instructions) 
  → Qiskit Aer simulation → Counts
"""

from typing import Dict
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Kraus
from qiskit_aer import AerSimulator

from .device_model import DeviceModel
from .circuit import Circuit, Gate, Layer
from .scheduling import schedule_circuit_alap, find_gate_definition
from .noise_channels import (
    reset_kraus,
    exp_decay_map,
    single_qubit_gate_noise,
    two_qubit_gate_noise_from_F2,
    crosstalk_ZZ_unitary,
    beta_from_J_delta_alpha,
    apply_confusion_matrix
)


class QiskitSimulator:
    """
    Density matrix simulator using Qiskit Aer.
    
    Transpiles our custom noise model into Qiskit circuit instructions,
    then uses Qiskit's optimized C++ density matrix backend.
    
    Example:
        device = create_paper_qpu()
        simulator = QiskitSimulator(device)
        counts = simulator.simulate(circuit, shots=10000)
    """
    
    def __init__(self, device: DeviceModel):
        self.device = device
        self.num_qubits = len(device.qubits)
    
        # Create Qiskit Aer simulator with density matrix method
        self.simulator = AerSimulator(
            method='matrix_product_state', # 'density_matrix', 'matrix_product_state'
            matrix_product_state_truncation_threshold=1e-2,  # or 1e-2
        )

    
    def simulate(self, circuit: Circuit, shots: int = 1024) -> Dict[str, int]:
        """
        Simulate noisy quantum circuit.
        
        Args:
            circuit: Circuit to simulate
            shots: Number of measurement shots
            
        Returns:
            Dict mapping bitstrings to counts: {'0000': 450, '1111': 574}
        """
        # 1. Schedule circuit into layers (ALAP)
        layers = schedule_circuit_alap(circuit, self.device)
        
        # 2. Build Qiskit circuit with noise instructions
        qc = self._build_noisy_circuit(circuit, layers)
        
        # 3. Simulate with Qiskit Aer
        result = self.simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        
        # 4. Apply readout errors manually
        noisy_counts = self._apply_readout_errors(counts, circuit.measurements)

        # 5. Reverse bitstring order for human-readable convention (Q0→leftmost) (qiskit has little_edian convention so Q0 is rightmost)
        reversed_counts = {bitstring[::-1]: count for bitstring, count in noisy_counts.items()}
        
        return reversed_counts
    
    
    def _build_noisy_circuit(self, circuit: Circuit, layers: list) -> QuantumCircuit:
        """
        Build Qiskit circuit with all noise instructions inserted.

        Structure per paper (Section III):
        1. State preparation noise (before first layer)
        2. For each layer:
        - Gates with instruction-based noise
        - Idle decay (T1/T2) for idle qubits
        - Crosstalk (always-on ZZ) between coupled qubits
        3. Measurements
        """
        qc = QuantumCircuit(self.num_qubits, self.num_qubits)

        # 1. State preparation error (layer-based)
        self._add_state_prep_noise(qc)

        # # Optional: debugging output of scheduled layers
        # for layer_idx, layer in enumerate(layers):
        #     print(f"\nLayer {layer_idx}:")
        #     for qid, instr in sorted(layer.instructions.items()):
        #         print(f"  Q{qid}: {instr.gate.name} {instr.gate.qubits} (dur={instr.duration*1e9:.1f} ns)")

        # Mark virtual-only layers
        for layer in layers:
            layer.is_virtual_layer = all(
                find_gate_definition(instr.gate, self.device).is_virtual
                for instr in layer.instructions.values()
            )

        # 2. Process each layer
        for layer in layers:
            # Apply gates with instruction-based noise
            for qid, instr in layer.instructions.items():
                self._add_gate_with_noise(qc, instr.gate)

            # Apply idle decay (layer-based)
            self._add_idle_decay(qc, layer)

            # Apply crosstalk (layer-based)
            self._add_crosstalk(qc, layer)

        # 3. Add measurements
        for meas in circuit.measurements:
            qc.measure(meas.qubit, meas.classical_bit)

        return qc

    
    def _add_state_prep_noise(self, qc: QuantumCircuit) -> None:
        """Add state preparation error to each qubit."""
        for qid in range(self.num_qubits):
            qubit = self.device.get_qubit(qid)
            kraus_ops = reset_kraus(p1=qubit.p1, gamma=1.0)
            qc.append(Kraus(kraus_ops), [qid])
    

    def _add_gate_with_noise(self, qc: QuantumCircuit, gate: Gate) -> None:
        """
        Add gate + instruction-based noise.
        
        Process:
        1. Add ideal gate (rx, ry, rz, cz, reset)
        2. Immediately add noise instruction after gate
        """
        # Handle Reset before looking up gate_def (no entry in gate_library)
        if gate.name.lower() == 'reset':
            qubit = gate.qubits[0]
            p1 = self.device.get_qubit(qubit).p1
            kraus_ops = reset_kraus(p1=p1, gamma=1.0)
            qc.append(Kraus(kraus_ops), [qubit])
            return

        # Normal gates use gate_def
        gate_def = find_gate_definition(gate, self.device)
        if gate_def is None:
            raise ValueError(f"Gate {gate.name} not found")

        # Virtual RZ gate
        if gate_def.is_virtual:
            theta = gate.params.get('theta', 0.0)
            qc.rz(theta, gate.qubits[0])
            # virtual gate has no physical drift or decoherence ---
            # (do NOT use device.effective_… functions)
            return

        # --- RX ---
        if gate.name == 'Rx':
            theta_nom = gate.params.get('theta', 0.0)
            theta_eff = self.device.effective_single_qubit_angle(qid=gate.qubits[0], theta_nominal=theta_nom, gate_duration=gate_def.duration) # coherent misrotation from drift
            qc.rx(theta_eff, gate.qubits[0])
            F_eff = self.device.effective_single_qubit_fidelity(qid=gate.qubits[0], F_nominal=gate_def.fidelity, gate_duration=gate_def.duration)
            kraus_ops = single_qubit_gate_noise(F_eff)
            qc.append(Kraus(kraus_ops), [gate.qubits[0]])

        # --- RY ---
        elif gate.name == 'Ry':
            theta_nom = gate.params.get('theta', 0.0)
            theta_eff = self.device.effective_single_qubit_angle(qid=gate.qubits[0], theta_nominal=theta_nom, gate_duration=gate_def.duration)
            qc.ry(theta_eff, gate.qubits[0])
            F_eff = self.device.effective_single_qubit_fidelity(qid=gate.qubits[0], F_nominal=gate_def.fidelity, gate_duration=gate_def.duration)
            kraus_ops = single_qubit_gate_noise(F_eff)
            qc.append(Kraus(kraus_ops), [gate.qubits[0]])

        # --- CZ ---
        elif gate.name == 'CZ':
            q0, q1 = gate.qubits

            # 1) Ideal calibrated CZ
            qc.cz(q0, q1)

            # 2) Coherent miscalibration due to frequency drift
            phi_nom = np.pi   # ideal phase
            phi_eff = self.device.effective_cz_phase(
                q0=q0, q1=q1, nominal_phase=phi_nom, gate_duration=gate_def.duration
            )

            # Only the **extra** phase beyond ideal
            delta_phi = phi_eff - phi_nom

            cz_matrix = np.diag([
                1.0,
                1.0,
                1.0,
                np.exp(1j * delta_phi),
            ])
            cz_drift_gate = UnitaryGate(cz_matrix, label='CZ_drift')
            qc.append(cz_drift_gate, [q0, q1])

            # 3) Decoherent penalty (Kraus)
            F_eff = self.device.effective_two_qubit_fidelity(q0, q1, gate_def.fidelity)
            kraus_ops = two_qubit_gate_noise_from_F2(F_eff)
            qc.append(Kraus(kraus_ops), [q0, q1])

        else:
            raise ValueError(f"Unknown gate: {gate.name}")

    
    def _add_idle_decay(self, qc: QuantumCircuit, layer: Layer) -> None:
        """
        Add T1/T2 decay for idle time in this layer.
        Mixed-layer logic:
        - If the entire layer is virtual (is_virtual_layer), skip completely.
        - For each qubit:
            * If virtual gate → idle = full layer duration
            * Else idle = layer idle time computed from scheduler
        For each qubit with idle_time > 0, add exponential decay noise.
        Uses Δε_ji (idle time) from layer, not gate duration.
        """
        # Layer has only virtual gates → no real time passes
        if layer.is_virtual_layer:
            return

        for qid in range(self.num_qubits):
            
            # Base idle time from ALAP scheduler
            idle_time = layer.idle_times.get(qid, layer.layer_duration)

            # Check if this qubit's gate is virtual
            instr = layer.get_instruction(qid)
            if instr is not None:
                instr_gate_def = find_gate_definition(instr.gate, self.device)
                if instr_gate_def is not None and instr_gate_def.is_virtual:
                    # This qubit's only gate is virtual
                    # ALAP scheduler gave Rz a fake duration (e.g., 1 ns)
                    # Add that back to idle time:
                    idle_time += instr.duration

            if idle_time > 0:
                qubit = self.device.get_qubit(qid)              
                # Coherent phase buildup during idle due to drift
                φ = self.device.get_idle_phase(qid, idle_time)
                if abs(φ) > 1e-12:
                    qc.rz(φ, qid)
                # Incoherent amplitude & phase decay (T1/T2)
                kraus_ops = exp_decay_map(idle_time, qubit.T1, qubit.T2)
                qc.append(Kraus(kraus_ops), [qid])
    

    def _add_crosstalk(self, qc: QuantumCircuit, layer: Layer) -> None:
        """
        Add always-on ZZ crosstalk *only for real physical time*.
        Virtual RZ gates must contribute **zero** duration.
        """
        # If entire layer is virtual → no crosstalk
        if layer.is_virtual_layer:
            return

        # 2) Process each coupling
        for (qi, qj), coupling in self.device.couplings.items():

            instr_i = layer.get_instruction(qi)
            instr_j = layer.get_instruction(qj)

            # If gate exists but is virtual RZ → treat as duration 0
            if instr_i:
                gate_def_i = find_gate_definition(instr_i.gate, self.device)
                if gate_def_i and gate_def_i.is_virtual:
                    dur_i = 0.0
                else:
                    dur_i = instr_i.duration
            else:
                dur_i = 0.0

            if instr_j:
                gate_def_j = find_gate_definition(instr_j.gate, self.device)
                if gate_def_j and gate_def_j.is_virtual:
                    dur_j = 0.0
                else:
                    dur_j = instr_j.duration
            else:
                dur_j = 0.0

            # Layer duration = max durations of real gates
            duration = max(dur_i, dur_j)

            # If no real gate active → no crosstalk for this pair
            if duration <= 0:
                continue

            # Compute β
            qubit_i = self.device.get_qubit(qi)
            qubit_j = self.device.get_qubit(qj)
            beta = beta_from_J_delta_alpha(
                J=coupling.J,
                delta=coupling.detuning,
                alpha_u=qubit_i.anharmonicity,
                alpha_v=qubit_j.anharmonicity
            )

            # Add ZZ
            U_zz = crosstalk_ZZ_unitary(beta, duration)
            qc.append(UnitaryGate(U_zz, label='ZZ_crosstalk'), [qi, qj])

    # More realistic applying of crosstalk to all qubits even idle-idle interactions (which is much smaller than gate-gate and gate-idle interactions) but much longer to run so not worth it.
    # def _add_crosstalk(self, qc: QuantumCircuit, layer: Layer) -> None:
    #     """
    #     Add always-on ZZ crosstalk to all physically coupled qubits,
    #     including idle qubits, using full layer duration.
    #     """

    #     # If entire layer is virtual → no crosstalk
    #     if layer.is_virtual_layer:
    #         return

    #     # Crosstalk is applied for full layer time
    #     layer_time = layer.layer_duration

    #     for (qi, qj), coupling in self.device.couplings.items():

    #         # Compute beta
    #         qubit_i = self.device.get_qubit(qi)
    #         qubit_j = self.device.get_qubit(qj)

    #         beta = beta_from_J_delta_alpha(
    #             J=coupling.J,
    #             delta=coupling.detuning,
    #             alpha_u=qubit_i.anharmonicity,
    #             alpha_v=qubit_j.anharmonicity
    #         )

    #         # Build ZZ unitary
    #         U_zz = crosstalk_ZZ_unitary(beta, layer_time)

    #         qc.append(UnitaryGate(U_zz, label='ZZ_crosstalk'), [qi, qj])


    def _apply_readout_errors(self, counts: Dict[str, int], measurements: list) -> Dict[str, int]:
        """
        Apply readout confusion matrices to measurement results.
        
        This is done shot-by-shot to properly handle correlations.
        """
        noisy_counts = {}
        
        # Process each shot
        for bitstring, count in counts.items():
            for _ in range(count):
                # Apply confusion matrix to each measured qubit
                noisy_bits = []
                for meas in measurements:
                    qubit_id = meas.qubit
                    bit_idx = meas.classical_bit
                    
                    # Get the bit value for this qubit
                    # Bitstring is in Qiskit format (right-to-left)
                    measured_bit = int(bitstring[-(bit_idx + 1)])
                    
                    # Apply confusion matrix
                    confusion_matrix = self.device.get_qubit(qubit_id).confusion_matrix
                    noisy_bit = apply_confusion_matrix(measured_bit, confusion_matrix)
                    
                    noisy_bits.append(str(noisy_bit))
                
                # Build noisy bitstring (Qiskit format: right-to-left)
                noisy_bitstring = ''.join(reversed(noisy_bits))
                
                # Add to counts
                noisy_counts[noisy_bitstring] = noisy_counts.get(noisy_bitstring, 0) + 1
        
        return noisy_counts
    
    def get_device_info(self) -> Dict:
        """Get summary of device parameters."""
        qubits = self.device.qubits
        T1_values = [q.T1 * 1e6 for q in qubits.values()]  # μs
        T2_values = [q.T2 * 1e6 for q in qubits.values()]
        
        return {
            'num_qubits': len(qubits),
            'topology': self.device.topology,
            'T1_range_us': (min(T1_values), max(T1_values)),
            'T2_range_us': (min(T2_values), max(T2_values)),
            'num_gates': len(self.device.gate_library),
            'num_couplings': len(self.device.couplings),
        }