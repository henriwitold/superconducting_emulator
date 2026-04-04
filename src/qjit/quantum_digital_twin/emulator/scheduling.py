# emulator/scheduling.py

from typing import List, Dict, Optional
from collections import defaultdict

from .circuit import Gate, Circuit, Layer, LayerInstruction
from .device_model import DeviceModel, GateDefinition

def schedule_circuit_alap(circuit: Circuit, device: DeviceModel) -> List[Layer]:
    """
    Simple ALAP scheduler: places gates as late as possible.
    
    Algorithm:
    1. Track when each qubit will be needed next (backward pass)
    2. Place each gate as late as possible before it's needed
    3. Group into layers
    """
    if not circuit.gates:
        return []
    
    # Get gate durations
    durations = [get_gate_duration(g, device) for g in circuit.gates]
    
    # Backward pass: compute latest start time for each gate
    latest_start = _compute_alap_times(circuit.gates, durations, len(device.qubits))
    
    # Group gates by start time into layers
    return _create_layers(circuit.gates, latest_start, durations, len(device.qubits))


def get_gate_duration(gate: Gate, device: DeviceModel) -> float:
    """Get duration for a gate from device model."""

    # Reset/measure are instantaneous
    if gate.name.lower() in {"reset", "measure"}:
        return 0.0

    # Force RZ to have a tiny duration to enforce correct scheduling
    # Even though RZ is virtual in noise modeling
    if gate.name == "Rz":
        return 1e-9   # anything >0 is fine (1 ns, 0.1 ns, etc.)

    # All other gates use device-defined durations
    gate_def = find_gate_definition(gate, device)
    if gate_def is None:
        raise ValueError(f"No GateDefinition for {gate.name} on qubits {gate.qubits}")
    
    return gate_def.duration



def find_gate_definition(gate: Gate, device: DeviceModel) -> Optional[GateDefinition]:
    """Find the GateDefinition matching this gate operation.
    Accept reversed qubit order for symmetric 2Q gates like CZ.
    """
    # Exact match first
    for gdef in device.gate_library:
        if gdef.name == gate.name and gdef.qubits == gate.qubits:
            return gdef

    # If not found, allow reversed lookup for symmetric two-qubit gates
    if len(gate.qubits) == 2 and gate.name in ("CZ",):
        q0, q1 = gate.qubits
        for gdef in device.gate_library:
            if gdef.name == gate.name and gdef.qubits == (q1, q0):
                return gdef

    return None


def _compute_alap_times(gates: List[Gate], durations: List[float], num_qubits: int) -> List[float]:
    """
    Compute ALAP start times via backward pass.
    Each gate starts as late as possible before next gate on same qubit(s).

    Key behavior:
    - If a gate has no future constraints AND does not share qubits with any
      later gate, we are free to place it at time 0 (parallel with other such gates).
    - Otherwise, we push it to the current circuit depth so dependencies stack.
    """
    n = len(gates)
    latest = [0.0] * n

    # Track latest time each qubit must start being used
    qubit_needed_at = defaultdict(lambda: float('inf'))

    # Backward pass
    for i in range(n - 1, -1, -1):
        gate = gates[i]

        # Gate must end before next gate on these qubits needs them
        must_end_by = min(qubit_needed_at[q] for q in gate.qubits)

        if must_end_by == float('inf'):
            # No future constraint on these qubits.
            # Check if this gate is completely independent of all later gates
            # (i.e. no qubit overlap). If so, we can put it at t=0 and
            # let it run in parallel with other independent last-layer gates.
            if all(not (set(gate.qubits) & set(g2.qubits)) for g2 in gates[i+1:]):
                latest[i] = 0.0
            else:
                # Shares qubits with some later gate → push to current depth
                circuit_depth = max(
                    (latest[j] + durations[j] for j in range(i + 1, n)),
                    default=0.0
                )
                latest[i] = circuit_depth
        else:
            # There *is* a future constraint: must finish before `must_end_by`
            latest[i] = must_end_by - durations[i]

        # Update when these qubits are needed
        for q in gate.qubits:
            qubit_needed_at[q] = latest[i]

    # Shift all times so the earliest gate starts at t=0
    min_time = min(latest) if latest else 0.0
    return [t - min_time for t in latest]


def _create_layers(
    gates: List[Gate], 
    start_times: List[float], 
    durations: List[float],
    num_qubits: int
) -> List[Layer]:
    """Group gates with same start time into layers."""
    
    # Group by start time
    time_groups: Dict[float, List[tuple]] = defaultdict(list)
    for i, gate in enumerate(gates):
        # Quantize/round the start time to avoid floating point mismatch
        st = round(start_times[i], 12)   # 12 decimals is safe for ns precision
        time_groups[st].append((gate, durations[i]))

                
    # Create layers
    layers = []
    for start_time in sorted(time_groups.keys()):
        gate_list = time_groups[start_time]
        
        instructions = {}
        layer_dur = 0.0

        # Track which qubits are busy (and for how long) in this layer
        busy_durations: Dict[int, float] = {}
        
        for gate, dur in gate_list:
            # If the gate truly has duration 0 (e.g., reset or measure),
            # we do NOT add it as a timed LayerInstruction because it does
            # not occupy scheduling time or block a qubit.
            #
            # NOTE:
            #  - RZ IS NOT skipped here. RZ now has a small non-zero duration
            #    specifically so that ALAP scheduling preserves the correct
            #    order of operations (Rz→Ry→CZ→Rz→Ry).
            #
            #  - Only literal 0-duration gates are skipped.
            if dur == 0.0:
                continue

            # Only assign once for two-qubit gates to avoid duplicate execution
            if len(gate.qubits) == 2:
                q0, q1 = gate.qubits
                instructions[q0] = LayerInstruction(
                    gate=gate,
                    start_time=start_time,
                    duration=dur
                )
                # Mark BOTH qubits as busy for idle-time accounting
                busy_durations[q0] = dur
                busy_durations[q1] = dur
            else:
                # normal single-qubit case
                q = gate.qubits[0]
                instructions[q] = LayerInstruction(
                    gate=gate,
                    start_time=start_time,
                    duration=dur
                )
                busy_durations[q] = dur

            layer_dur = max(layer_dur, dur)
        
        # Compute idle times
        idle_times = {}
        for q in range(num_qubits):
            if q in busy_durations:
                idle_times[q] = layer_dur - busy_durations[q]
            else:
                idle_times[q] = layer_dur
        
        layers.append(Layer(
            instructions=instructions,
            layer_duration=layer_dur,
            idle_times=idle_times
        ))
    
    return layers