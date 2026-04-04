# qjit/quantum_digital_twin/drift_experiments/snapshot.py

from dataclasses import dataclass
from typing import Dict
from ..emulator.device_model import DeviceModel

@dataclass
class DeviceSnapshot:
    """
    Store only the key parameters we want to track
    between drift events.
    """
    frequencies: Dict[int, float]
    true_frequencies: Dict[int, float]
    frequency_drift: Dict[int, float]

    @staticmethod
    def from_device(device: DeviceModel) -> "DeviceSnapshot":
        return DeviceSnapshot(
            frequencies={qid: q.frequency for qid, q in device.qubits.items()},
            true_frequencies={qid: q.true_frequency for qid, q in device.qubits.items()},
            frequency_drift={qid: q.frequency_drift for qid, q in device.qubits.items()},
        )

    def pretty_print(self):
        print("\n=== DEVICE SNAPSHOT ===")
        for qid in sorted(self.frequencies.keys()):
            f = self.frequencies[qid] / 1e9
            ft = self.true_frequencies[qid] / 1e9
            df = self.frequency_drift[qid]
            print(f"Q{qid}: f_cal={f:.4f} GHz, f_true={ft:.4f} GHz, drift={df:+.1f} Hz")
