#!/usr/bin/env python3
"""
Classical Shadows — Entanglement Detection via MPS
===================================================

Estimates 2nd Rényi entropy S₂ of a subsystem in the transverse-field
Ising model (TFIM) using the classical shadows protocol with MPS.

Classical shadows let you estimate properties of quantum states using
far fewer measurements than full tomography. For an n-qubit system,
full tomography needs 4^n measurements — shadows need only O(poly(n)).

Protocol:
  1. Prepare the state (Trotter evolution of TFIM)
  2. Apply random single-qubit Cliffords
  3. Measure in computational basis (1 shot)
  4. Reconstruct shadow density matrix from the outcome
  5. Repeat M times, average to estimate S₂

Usage:
    python classical_shadows_demo.py                # 6×6 = 36 qubits
    python classical_shadows_demo.py --small        # 4×4 = 16 qubits + exact ED
    python classical_shadows_demo.py --gpu          # GPU-accelerated MPS
"""

import sys
import os
import time

import numpy as np
import maestro

from helpers import (
    Config, get_nn_bonds, site_coords,
    collect_shadow_snapshots, estimate_purity_from_shadows, renyi_s2,
    compute_exact_s2,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────
# Shadow sweep
# ─────────────────────────────────────────────────────────────────────

def shadow_entanglement_sweep(config, subsystem_qubits):
    """
    Sweep Trotter depths, estimating S₂(t) via classical shadows.
    Each depth requires M full MPS simulations (execute, shots=1).
    """
    n = config.n_qubits
    bonds = get_nn_bonds(config.lx, config.ly)

    qx0, qy0 = site_coords(subsystem_qubits[0], config.ly)
    qx1, qy1 = site_coords(subsystem_qubits[1], config.ly)

    print(f"\n  Subsystem: qubits {subsystem_qubits} — "
          f"site ({qx0},{qy0})↔({qx1},{qy1})")
    print(f"  Depths: {config.trotter_depths}  |  "
          f"Shadows/depth: {config.n_shadows}")
    print(f"  Backend: {'GPU' if config.use_gpu else 'CPU'}  |  "
          f"χ = {config.chi_high if config.use_gpu else config.chi_low}\n")

    results = {'depths': [], 'times': [], 's2': [], 'purity': []}

    for depth in config.trotter_depths:
        t_val = depth * config.dt
        t0 = time.time()
        shadows = collect_shadow_snapshots(
            config, depth, bonds, subsystem_qubits, verbose=False
        )
        elapsed = time.time() - t0

        purity_raw = estimate_purity_from_shadows(shadows)
        s2, purity = renyi_s2(purity_raw, config.d_A)

        results['depths'].append(depth)
        results['times'].append(t_val)
        results['s2'].append(s2)
        results['purity'].append(purity)

        print(f"    depth={depth:2d}  (t={t_val:.2f})  "
              f"S₂={s2:.4f}  purity={purity:.6f}  "
              f"[{elapsed:.1f}s, {elapsed/config.n_shadows:.3f}s/snapshot]")

    return results


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────

def plot_entanglement_growth(results, exact, subsystem, config, path):
    """Plot S₂ vs simulation time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(results['times'], results['s2'], 'o--',
            color='#7B1FA2', linewidth=2.5, markersize=8,
            label=f'Classical shadows (Maestro MPS)')

    if exact is not None:
        ax.plot(exact['times'], exact['s2'], '-',
                color='#4CAF50', linewidth=2.5, alpha=0.8,
                label='Exact ED')

    max_s2 = np.log2(config.d_A)
    ax.axhline(y=max_s2, color='gray', linestyle=':', alpha=0.4,
               label=f'Max S₂ = {max_s2:.1f}')

    ax.set_xlabel('Simulation Time t', fontsize=13)
    ax.set_ylabel('2nd-order Rényi Entropy S₂', fontsize=13)
    ax.set_title(f'Entanglement Growth — {config.lx}×{config.ly} TFIM\n'
                 f'Subsystem: qubits {subsystem}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=-0.05)

    info = (f'{config.n_shadows} snapshots/depth\n'
            f'Full tomo: 4^{config.n_qubits} '
            f'≈ {4**config.n_qubits:.1e}\n'
            f'Speedup: ×{4**config.n_qubits / config.n_shadows:.1e}')
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew',
                      edgecolor='gray'))

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def build_config():
    use_gpu = '--gpu' in sys.argv
    small = '--small' in sys.argv

    if small:
        return Config(
            lx=4, ly=4,
            chi_low=16, chi_high=32,
            n_shadows=300,
            trotter_depths=[1, 2, 3, 4, 6, 8, 10],
            use_gpu=use_gpu,
        )
    else:
        return Config(
            lx=6, ly=6,
            chi_low=16, chi_high=64,
            n_shadows=200,
            trotter_depths=[1, 2, 3, 4, 5, 6, 7, 8, 10],
            use_gpu=use_gpu,
        )


if __name__ == '__main__':
    config = build_config()

    print(f"\n{'─'*60}")
    print(f"  CLASSICAL SHADOWS — ENTANGLEMENT DETECTION")
    print(f"{'─'*60}")
    print(f"  System:  {config.lx}×{config.ly} = {config.n_qubits} qubits")
    print(f"  Model:   TFIM  J={config.j_coupling}  h={config.h_field}")
    print(f"  Shadows: {config.n_shadows} per depth")
    print(f"  GPU:     {'Yes' if config.use_gpu else 'No'}")

    total_start = time.time()

    # Pick center subsystem (bulk of the lattice)
    cx, cy = config.lx // 2, config.ly // 2
    q0 = cx * config.ly + cy
    q1 = cx * config.ly + (cy + 1)
    subsystem = [q0, q1]

    # Shadow sweep
    results = shadow_entanglement_sweep(config, subsystem)

    # Exact ED reference (small systems only)
    exact = None
    if config.n_qubits <= 20:
        print(f"\n  Computing exact ED reference...")
        exact = compute_exact_s2(config, subsystem)
        if exact:
            print(f"  ✓ Exact S₂ computed for {len(exact['depths'])} depths")
    else:
        print(f"\n  ℹ Exact ED skipped (n={config.n_qubits} > 20)")
        print(f"  Classical shadows are the ONLY way to estimate S₂ at this scale!")

    # Plot
    plot_path = plot_entanglement_growth(
        results, exact, subsystem, config,
        os.path.join(SCRIPT_DIR, 'entanglement_growth.png'),
    )
    print(f"\n  📊 Saved: {plot_path}")

    # Summary
    total = time.time() - total_start
    print(f"\n{'─'*60}")
    print(f"  DONE ({total:.1f}s)")
    print(f"{'─'*60}")
    print(f"  S₂ range: {min(results['s2']):.3f} → {max(results['s2']):.3f}")
    if exact:
        mae = np.mean([abs(s - e) for s, e in
                       zip(results['s2'], exact['s2'])])
        print(f"  MAE vs exact: {mae:.3f}")
    print()
