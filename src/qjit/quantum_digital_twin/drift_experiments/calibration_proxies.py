# drift_experiments/calibration_proxies.py

"""
Syndrome-derived quality proxy for continuous calibration (Proxy 1 only).

Based on the drift analysis (Sections 7-8, 12):
  - Frequency drift produces Z errors on data qubits
  - Z errors are detected by X-type stabilizers (ancilla bits 3,4,5)

We keep only the simplest and most drift-sensitive metric:
  Proxy 1: X-stabilizer syndrome rate (direct drift indicator)

Convention (from SyndromeTable):
  - Syndrome is a 6-bit string: (a0, a1, a2, a3, a4, a5)
  - Bits 0,1,2 = Z-type stabilizers (detect X errors)
  - Bits 3,4,5 = X-type stabilizers (detect Z errors ← THIS IS WHERE DRIFT SHOWS UP)
"""

from typing import Dict, Tuple


# ─── Stabilizer structure (Steane code) ───────────────────────────

# X-type stabilizer bit indices within the 6-bit syndrome
X_STAB_INDICES = [3, 4, 5]

# Z-type stabilizer bit indices
Z_STAB_INDICES = [0, 1, 2]

# ─── Helper: extract rates from syndrome counts ──────────────────

def _compute_stabilizer_rates(
    syndrome_counts: Dict[str, int],
    total_shots: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute per-stabilizer firing rates from syndrome histogram.

    Parameters
    ----------
    syndrome_counts : dict
        Maps 6-bit syndrome strings (e.g. '000000') to counts.
    total_shots : int
        Total number of shots (sum of all counts).

    Returns
    -------
    x_rates : dict
        {stab_index: firing_rate} for X-type stabilizers (indices 3,4,5)
    z_rates : dict
        {stab_index: firing_rate} for Z-type stabilizers (indices 0,1,2)
    """
    x_firings = {i: 0 for i in X_STAB_INDICES}
    z_firings = {i: 0 for i in Z_STAB_INDICES}

    for syndrome, count in syndrome_counts.items():
        for i in X_STAB_INDICES:
            x_firings[i] += int(syndrome[i]) * count
        for i in Z_STAB_INDICES:
            z_firings[i] += int(syndrome[i]) * count

    if total_shots == 0:
        return (
            {i: 0.0 for i in X_STAB_INDICES},
            {i: 0.0 for i in Z_STAB_INDICES},
        )

    x_rates = {i: x_firings[i] / total_shots for i in X_STAB_INDICES}
    z_rates = {i: z_firings[i] / total_shots for i in Z_STAB_INDICES}

    return x_rates, z_rates


# ─── Proxy 1: X-stabilizer syndrome rate ─────────────────────────

def proxy_x_syndrome_rate(
    syndrome_counts: Dict[str, int],
    total_shots: int,
) -> float:
    """
    Proxy 1: Average X-stabilizer firing rate.

    This is the simplest drift indicator. Under drift, X-type stabilizers
    fire with probability ≈ π² τ²_idle Δf² per qubit in their support.
    A lower rate means less drift.

    Returns
    -------
    Q1 : float
        Quality score in [0, 1]. Q1 = 1 means no X-stabilizer firings
        (perfect, no drift detected). Q1 = 0 means every X-stabilizer
        fires every shot.

    Analogous to Google's surrogate objective C = E[D], but restricted
    to the X-type detectors that carry the drift signal.
    """
    x_rates, _ = _compute_stabilizer_rates(syndrome_counts, total_shots)

    # Average over the 3 X-type stabilizers
    avg_x_rate = sum(x_rates.values()) / len(X_STAB_INDICES)

    Q1 = 1.0 - avg_x_rate
    return Q1
