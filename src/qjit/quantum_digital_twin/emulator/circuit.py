# emulator/circuit.py

"""
Circuit representation and data structures.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class Gate:
    """A gate operation in the circuit."""
    name: str                    # e.g., 'Rx', 'Ry', 'Rz', 'CZ'
    qubits: Tuple[int, ...]      # affected qubit indices
    params: Dict[str, float]     # e.g., {'theta': pi/2} for rotations
    
@dataclass
class Measurement:
    """A measurement operation."""
    qubit: int
    classical_bit: int  # which classical bit to store result

@dataclass  
class Circuit:
    """A quantum circuit with gates and measurements."""
    num_qubits: int
    gates: List[Gate]
    measurements: List[Measurement]

@dataclass
class LayerInstruction:
    """An instruction within a layer with timing info."""
    gate: Gate
    start_time: float  # when it starts in this layer
    duration: float    # how long it takes
    
@dataclass
class Layer:
    """A circuit layer with parallel instructions."""
    instructions: Dict[int, LayerInstruction]  # qubit_id -> instruction
    layer_duration: float                       # max duration in layer
    idle_times: Dict[int, float]                # qubit_id -> idle time
    
    def get_instruction(self, qid: int) -> Optional[LayerInstruction]:
        """
        Return the LayerInstruction acting on this qubit.

        If the qubit is not directly listed in `instructions`
        (for example, if it is the *partner* qubit in a two-qubit gate
        that was stored only under the control qubit),
        this will still return that same LayerInstruction object.

        This makes both qubits in a 2Q gate "see" their shared instruction,
        which is useful for crosstalk and timing logic.
        """
        # Direct match
        instr = self.instructions.get(qid)
        if instr:
            return instr

        # Partner match: check if this qubit participates in any stored 2Q gate
        for ins in self.instructions.values():
            if qid in ins.gate.qubits:
                return ins

        return None
   