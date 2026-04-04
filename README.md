# Superconducting Emulator

Drift-aware quantum device emulator with syndrome-driven continuous calibration, built on the Steane [[7,1,3]] code.

## Structure

```
src/qjit/
├── quantum_digital_twin/    ← start here
│   ├── emulator/            # noise model, device model, ALAP scheduler, Qiskit Aer simulation
│   └── drift_experiments/   # Wiener-process drift, syndrome extraction, calibration loop
│
└── experiments/
    └── error_counts/ecc/    # Steane code implementation
```

The core of the project lives in [`quantum_digital_twin/`](src/qjit/quantum_digital_twin/) — see its [README](src/qjit/quantum_digital_twin/README.md) for architecture details, design decisions, and usage.

## Quick start

```bash
uv sync
uv run continuous_calibration
```
