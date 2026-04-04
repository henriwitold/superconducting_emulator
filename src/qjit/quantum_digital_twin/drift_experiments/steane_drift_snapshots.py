# qjit/quantum_digital_twin/drift_experiments/steane_drift_snapshots.py

import numpy as np
from dataclasses import dataclass
from ..emulator.device_model import DeviceModel
from ..emulator.qpu_config import create_steane_qpu
from .snapshot import DeviceSnapshot

# =====================================================================
#  A simple controller for drift snapshots of the Steane device
# =====================================================================

class SteaneDriftExperiment:
    """
    Manage a Steane 13-qubit device under time-dependent drift.

    Responsibilities:
    - Create Steane device
    - Maintain internal time (seconds)
    - Apply drift steps
    - Record snapshot history
    """

    def __init__(self, drift_std_hz: float = None):
        self.device: DeviceModel = create_steane_qpu(noise_scale=0.0)
        self.current_time: float = 0.0
        self.history = []        # stores snapshots
        self.drift_std_hz = drift_std_hz  # global override

        # Record initial snapshot
        self.record_snapshot("Snapshot A (initial)")

    # ------------------------------------------------------------------
    def record_snapshot(self, label: str):
        snap = DeviceSnapshot.from_device(self.device)
        self.history.append((label, self.current_time, snap))

    # ------------------------------------------------------------------
    def step_drift(self, dt: float):
        """
        Apply drift for time interval dt.
        Updates device.true_frequency and device.detunings.
        """
        self.current_time += dt

        self.device.step_frequency_drift(
            dt=dt,
            std_dev=self.drift_std_hz
        )

        self.record_snapshot(f"Snapshot B at t={self.current_time:.2f}")

    # ------------------------------------------------------------------
    def pretty_print_history(self):
        print("\n=======================")
        print("DRIFT SNAPSHOT HISTORY")
        print("=======================")

        for label, t, snap in self.history:
            print(f"\n--- {label} (t={t:.2f} s) ---")
            snap.pretty_print()

    def apply_calibration_offset(self, qubit_id: int, delta_f: float):
        """
        Simulate recalibrating a single qubit's drive frequency.
        
        This reduces the gap between the calibrated frequency 
        (what the electronics think) and the true frequency 
        (what the qubit actually is).
        
        In reality this would mean: run a Ramsey experiment on qubit_id,
        discover it has drifted, update the drive frequency.
        Here we just shift device.qubits[qubit_id].frequency by delta_f.
        """
        q = self.device.qubits[qubit_id]
        q.frequency += delta_f
        q.frequency_drift = q.true_frequency - q.frequency
        self.device.update_detunings_from_frequencies()


# =====================================================================
# Standalone runnable experiment: snapshot A → drift → snapshot B
# =====================================================================

def run_steane_drift_experiment(dt=1.0, drift_std_hz=200.0):
    """
    Simple test:
      - Build Steane QPU
      - Print Snapshot A
      - Apply drift
      - Print Snapshot B
    """

    print("\n=== Steane Drift Snapshot Experiment ===")
    print(f"dt = {dt} s, drift_std = {drift_std_hz} Hz")

    exp = SteaneDriftExperiment(drift_std_hz=drift_std_hz)

    print("\n--- Snapshot A (before drift) ---")
    exp.history[-1][2].pretty_print()

    print("\n=== Applying drift ===")
    exp.step_drift(dt)

    print("\n--- Snapshot B (after drift) ---")
    exp.history[-1][2].pretty_print()

    return exp


# =====================================================================
# Main entry point
# =====================================================================

if __name__ == "__main__":
    run_steane_drift_experiment(
        dt=1.0,
        drift_std_hz=200.0   # Steane QPU default drift strength
    )
