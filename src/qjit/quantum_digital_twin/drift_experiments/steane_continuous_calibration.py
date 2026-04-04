# qjit/quantum_digital_twin/drift_experiments/steane_continuous_calibration.py

"""
Continuous calibration loop for the Steane code using only Proxy 1.

Architecture (simplified):
  1. Create ONE experiment object and apply drift
  2. Run syndrome extraction → baseline quality via Proxy 1 (X-rate)
  3. Choose a target qubit round-robin
  4. Propose a random frequency offset for that qubit
  5. Re-run syndrome extraction → candidate Proxy 1 quality
  6. Accept/reject: keep offset if Proxy 1 improves, else revert
  7. Repeat
"""

import numpy as np
from typing import Optional, Dict

from .steane_drift_snapshots import SteaneDriftExperiment
from .steane_drift_syndrome import run_syndrome_on_device
from .calibration_proxies import proxy_x_syndrome_rate


def run_continuous_calibration(
    initial_dt: float = 7200.0,
    drift_std_hz: float = 300.0,
    shots: int = 200,
    max_iterations: int = 20,
    correction_scale: float = 5e3,
    noise_scale: float = 0.0,
    seed: Optional[int] = None,
) -> Dict:
    """
    Continuous calibration loop using syndrome-derived proxies.

    Parameters
    ----------
    initial_dt : float
        Time (seconds) of drift before starting calibration.
        2 hours = 7200s gives Δf_rms ≈ drift_std_hz * sqrt(7200).
    drift_std_hz : float
        Wiener process drift rate in Hz/√s.
    shots : int
        Number of syndrome extraction shots per evaluation.
    max_iterations : int
        Number of calibration proposals to try.
    correction_scale : float
        Scale of frequency corrections in Hz. Should be on the order
        of the expected drift magnitude.
    noise_scale : float
        Passed to create_steane_qpu(). 0.0 = no static noise (isolate drift).
        Use > 0 to test calibration under mixed static + drift noise.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict with history and final quality summary.
    """

    rng = np.random.default_rng(seed)

    # ═══════════════════════════════════════════════════════════════
    # Step 1: Create experiment, apply drift
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 65)
    print("   CONTINUOUS CALIBRATION WITH SYNDROME PROXIES")
    print("=" * 65)

    expected_drift = drift_std_hz * np.sqrt(initial_dt)
    print(f"\n  Configuration:")
    print(f"    Drift time:       {initial_dt:.0f} s ({initial_dt/3600:.1f} hours)")
    print(f"    Drift rate:       {drift_std_hz:.0f} Hz/√s")
    print(f"    Expected Δf_rms:  {expected_drift:.0f} Hz")
    print(f"    Shots/eval:       {shots}")
    print(f"    Max iterations:   {max_iterations}")
    print(f"    Correction scale: {correction_scale:.0f} Hz")

    # Create the experiment object — this persists across all iterations
    exp = SteaneDriftExperiment(drift_std_hz=drift_std_hz)

    print("\n--- Snapshot A (before drift) ---")
    exp.history[-1][2].pretty_print()

    # Apply drift
    print(f"\n  Applying {initial_dt:.0f}s of frequency drift...")
    exp.step_drift(initial_dt)

    initial_delta_f = {}
    for qid in range(7):
        q = exp.device.qubits[qid]
        initial_delta_f[qid] = q.true_frequency - q.frequency

    print("\n--- Snapshot B (after drift, before calibration) ---")
    exp.history[-1][2].pretty_print()

    # ═══════════════════════════════════════════════════════════════
    # Step 2: Baseline measurement
    # ═══════════════════════════════════════════════════════════════

    print("\n--- Running baseline syndrome extraction ---")
    synd_baseline = run_syndrome_on_device(exp.device, shots=shots, verbose=True)

    Q_best = proxy_x_syndrome_rate(synd_baseline, shots)
    p_trivial_best = synd_baseline.get('000000', 0) / shots

    print(f"    Baseline Q1 (1 - avg X-rate): {Q_best:.6f}")
    print(f"    P(000000): {p_trivial_best:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Step 3-6: Accept/reject calibration loop
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 65)
    print("   STARTING ACCEPT/REJECT LOOP")
    print("=" * 65)

    # History tracking
    history = {
        'Q1': [Q_best],
        'p_trivial': [p_trivial_best],
        'iterations': [],
    }

    n_accepted = 0

    for iteration in range(max_iterations):

        # ── Choose target qubit round-robin ──
        target_qubit = iteration % 7

        # Random direction and magnitude
        delta_f = rng.choice([-1, 1]) * correction_scale * rng.uniform(0.5, 1.5)

        print(f"\n  Iteration {iteration + 1}/{max_iterations}")
        print(f"    Target: Q{target_qubit} (round-robin)")
        print(f"    Proposed Δf: {delta_f:+.1f} Hz")

        # ── Apply correction to the persistent device ──
        exp.apply_calibration_offset(target_qubit, delta_f)

        # ── Evaluate on corrected device ──
        synd_candidate = run_syndrome_on_device(exp.device, shots=shots)
        Q_candidate = proxy_x_syndrome_rate(synd_candidate, shots)
        p_trivial_candidate = synd_candidate.get('000000', 0) / shots

        accepted = Q_candidate > Q_best

        # ── Accept or reject ──
        if accepted:
            improvement = Q_candidate - Q_best
            print(f"    ✅ ACCEPT — Q1: {Q_best:.6f} → {Q_candidate:.6f} (+{improvement:.6f})")
            print(f"       P(000000): {history['p_trivial'][-1]:.4f} → {p_trivial_candidate:.4f}")

            Q_best = Q_candidate
            p_trivial_best = p_trivial_candidate
            n_accepted += 1

            history['Q1'].append(Q_best)
            history['p_trivial'].append(p_trivial_best)

        else:
            # REJECT: revert the correction
            exp.apply_calibration_offset(target_qubit, -delta_f)

            print(f"    ❌ REJECT — Q1: {Q_candidate:.6f} ≤ {Q_best:.6f}")

            history['Q1'].append(history['Q1'][-1])
            history['p_trivial'].append(history['p_trivial'][-1])

        history['iterations'].append({
            'iteration': iteration,
            'target_qubit': target_qubit,
            'delta_f': delta_f,
            'accepted': accepted,
            'Q_candidate': Q_candidate,
            'Q_best': Q_best,
        })

    # ═══════════════════════════════════════════════════════════════
    # Final report
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 65)
    print("   CALIBRATION COMPLETE")
    print("=" * 65)

    # Final syndrome evaluation
    synd_final = run_syndrome_on_device(exp.device, shots=shots)
    Q_final = proxy_x_syndrome_rate(synd_final, shots)
    p_trivial_final = synd_final.get('000000', 0) / shots

    print(f"\n  Final Q1: {Q_final:.6f}")
    print(f"  Final P(000000): {p_trivial_final:.4f}")

    # Print device state
    print("\n--- Device state after calibration ---")
    initial_abs = []
    final_abs = []
    for qid in range(7):
        q = exp.device.qubits[qid]
        delta_f_final = q.true_frequency - q.frequency
        delta_f_initial = initial_delta_f[qid]
        initial_abs.append(abs(delta_f_initial))
        final_abs.append(abs(delta_f_final))
        print(f"  Q{qid}: f_cal={q.frequency/1e9:.6f} GHz, "
              f"f_true={q.true_frequency/1e9:.6f} GHz, "
              f"drift={q.frequency_drift:+.1f} Hz, "
              f"Δf_initial={delta_f_initial:+.1f} Hz, "
              f"Δf_final={delta_f_final:+.1f} Hz, "
              f"|Δf|: {abs(delta_f_initial):.1f} → {abs(delta_f_final):.1f} Hz")
    print("\n--- Δf summary ---")
    print(f"  Mean |Δf|: {np.mean(initial_abs):.1f} → {np.mean(final_abs):.1f} Hz")
    print(f"  Max  |Δf|: {np.max(initial_abs):.1f} → {np.max(final_abs):.1f} Hz")

    # Summary
    print(f"\n  Summary:")
    print(f"    Iterations:      {max_iterations}")
    print(f"    Accepted:        {n_accepted} ({100*n_accepted/max(max_iterations,1):.0f}%)")
    print(f"    Q1 improvement:  {history['Q1'][0]:.6f} → {Q_best:.6f}")
    print(f"    P(000000):       {history['p_trivial'][0]:.4f} → {history['p_trivial'][-1]:.4f}")

    return {
        'history': history,
        'final_quality': Q_best,
        'n_accepted': n_accepted,
        'experiment': exp,
    }


if __name__ == "__main__":
    result = run_continuous_calibration(
        initial_dt=7200.0,
        drift_std_hz=1000.0,
        shots=300,
        max_iterations=15,
        correction_scale=100e3,
        seed=42,
    )








