# Quantum Digital Twin — Drift-Aware Continuous Calibration

A drift-aware quantum device emulator with a syndrome-driven continuous calibration loop, built on the Steane [[7,1,3]] error-correcting code.

The system simulates realistic frequency drift on a 13-qubit device, extracts stabilizer syndromes through a Qiskit Aer backend, and runs an accept/reject calibration loop that compensates drift using only syndrome-derived quality proxies — no direct access to physical qubit parameters.

---

## Architecture

The codebase has two layers: a **device emulator** that models noisy quantum hardware at the pulse level, and a **drift experiments** module that uses it to demonstrate continuous calibration.

### Emulator (`emulator/`)

A parametric noise model driven by physically motivated error channels:

- **Device model** (`device_model.py`) — Qubit objects with frequency, T1/T2, SPAM, confusion matrices, and drift bookkeeping. Drift-aware methods compute effective rotation angles, gate fidelities, CZ phases, and idle phases under detuning.
- **Noise channels** (`noise_channels.py`) — Kraus operators for generalized amplitude damping (reset/state-prep), T1 relaxation, T2 dephasing, per-gate depolarizing noise (1Q and 2Q), and always-on ZZ crosstalk unitaries.
- **Scheduling** (`scheduling.py`) — ALAP layer scheduler that groups gates by start time, computes per-qubit idle durations, and handles virtual (Rz) vs. physical gate timing.
- **Circuit representation** (`circuit.py`) — Lightweight dataclasses for gates, measurements, layers, and circuits, independent of Qiskit.
- **QPU configs** (`qpu_config.py`) — Factory functions for a 5-qubit star-topology device (based on QuantWare Soprano-D parameters) and a 13-qubit Steane-code device with tunable noise scaling.
- **Simulator core** (`qiskit_emulator_core.py`) — Orchestrates the full simulation pipeline: ALAP scheduling → noise transpilation (state-prep, gate noise, idle decay, crosstalk) → Qiskit Aer density matrix / MPS execution → readout confusion. All noise is injected as Qiskit circuit instructions (Kraus ops, unitary gates) before a single Aer call.

### Drift Experiments (`drift_experiments/`)

Demonstrates continuous calibration on a Steane-code device under Wiener-process frequency drift:

- **Drift management** (`steane_drift_snapshots.py`) — Controller that maintains device state, applies time-stepped drift, records parameter snapshots, and supports calibration offsets (simulated Ramsey recalibration).
- **Syndrome extraction** (`steane_drift_syndrome.py`, `syndrome_extraction.py`) — Builds a Steane syndrome extraction circuit, transpiles it to the native gate set (Rx, Ry, Rz, CZ), and reduces 13-bit measurement outcomes to 6-bit stabilizer syndromes. Caches the transpiled circuit across calls for performance.
- **Quality proxy** (`calibration_proxies.py`) — Computes the X-stabilizer firing rate as a drift indicator. Under frequency drift, Z errors appear on data qubits and are detected by X-type stabilizers (ancilla bits 3, 4, 5). The proxy Q1 = 1 − (average X-stabilizer rate) serves as the objective for the calibration loop.
- **Calibration loop** (`steane_continuous_calibration.py`) — Accept/reject optimization: applies drift, measures baseline syndrome quality, then proposes random frequency corrections qubit-by-qubit (round-robin). Keeps corrections that improve Q1, reverts those that don't. Tracks parameter trajectories, syndrome rates, and residual detuning throughout.

---

## Key Design Decisions

**Syndrome-only calibration.** The loop never reads true qubit frequencies — it uses only stabilizer syndrome statistics to decide whether a correction helped. This mirrors what real hardware could support via continuous syndrome streaming.

**Physically motivated noise.** Gate errors aren't flat depolarizing channels. Single-qubit rotations pick up coherent over/under-rotation from detuning; CZ gates accumulate phase errors from detuning mismatch; idle periods produce both coherent phase and incoherent T1/T2 decay; ZZ crosstalk is always-on between coupled qubits.

**Drift model.** Frequency drift follows a Wiener process: Δf = σ√(dt) · N(0,1) per qubit per timestep. This produces correlated, cumulative drift that realistically models thermal and charge fluctuations in transmon qubits.

---

## Usage

All commands are run from the project root with `PYTHONPATH=src`:

```bash
# Run the continuous calibration demo
PYTHONPATH=src python -m qjit.quantum_digital_twin.drift_experiments.steane_continuous_calibration

# Run syndrome extraction before/after drift (standalone)
PYTHONPATH=src python -m qjit.quantum_digital_twin.drift_experiments.steane_drift_syndrome

# Run a simple drift snapshot experiment
PYTHONPATH=src python -m qjit.quantum_digital_twin.drift_experiments.steane_drift_snapshots
```

---

## Module Layout

```
quantum_digital_twin/
├── emulator/
│   ├── circuit.py                # Circuit/Gate/Layer dataclasses
│   ├── device_model.py           # Qubit, Coupling, DeviceModel with drift physics
│   ├── noise_channels.py         # Kraus operators and noise primitives
│   ├── qiskit_emulator_core.py   # Qiskit Aer simulation with custom noise injection
│   ├── qpu_config.py             # QPU factory functions (5Q star, 13Q Steane)
│   ├── scheduling.py             # ALAP gate scheduler
│   └── syndrome_extraction.py    # Qiskit → internal circuit conversion
│
└── drift_experiments/
    ├── calibration_proxies.py            # X-stabilizer syndrome quality proxy
    ├── snapshot.py                       # Device parameter snapshots
    ├── steane_continuous_calibration.py   # Accept/reject calibration loop
    ├── steane_drift_snapshots.py         # Drift experiment controller
    ├── steane_drift_syndrome.py          # Syndrome extraction on drifted devices
    └── syndrome_extraction.py            # Steane circuit builder + Qiskit conversion
```

---

## Metrics

- **Q1 (X-stabilizer syndrome rate)** — primary calibration objective; Q1 = 1 means no drift-induced Z errors detected
- **P(000000)** — trivial syndrome probability; higher is better
- **|Δf| per qubit** — residual detuning after calibration vs. before
- **Acceptance rate** — fraction of proposed corrections that improved Q1

---

## Dependencies

- Python 3.10+
- Qiskit, Qiskit Aer
- NumPy
