#!/usr/bin/env python3
"""
Entanglement Growth via Classical Shadows — Maestro Showcase
============================================================

Demonstrates quantum entanglement detection using classical shadows
with Maestro's multi-backend simulation (PP Scout → MPS Sniper):

  • Act 1: PP Scout — Scans lattice with Pauli Propagator to find the
           most and least entangled subsystems via ⟨ZZ⟩ correlations
  • Act 2: T-gates break Clifford → MPS takeover (Schrödinger picture)
  • Act 3: Adaptive bond dimension handoff (χ_low → χ_high)
  • Act 4: Random Clifford layer for tomographic completeness
  • Act 5: Bitstring sampling (hardware emulation mode)
  • Act 6: Classical shadows sweep — MPS sniper targets hot vs cold

Usage:
    python classical_shadows_demo.py                # 6×6 = 36 qubits
    python classical_shadows_demo.py --gpu           # GPU acceleration
    python classical_shadows_demo.py --small         # 4×4 = 16 qubits + exact ED
"""

import sys
import os
import time

import numpy as np
import maestro
from maestro.circuits import QuantumCircuit

from helpers import (
    Config, get_nn_bonds, build_pauli_observable,
    build_tfim_trotter_circuit, append_random_clifford_layer,
    site_coords, scout_entanglement,
    collect_shadow_snapshots, estimate_purity_from_shadows, renyi_s2,
    compute_exact_s2,
    plot_energy_evolution, plot_scout_comparison, plot_lattice_heatmap,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────

def header(title):
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")


def banner(config):
    gpu_str = 'ENABLED' if config.use_gpu else 'DISABLED (use --gpu)'
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  MAESTRO SHOWCASE: Classical Shadows — PP Scout / MPS Sniper   ║")
    print("║  From Heisenberg to Schrödinger, from Scout to Sniper         ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Lattice: {config.lx}×{config.ly} = {config.n_qubits} qubits"
          f"{'':>{50 - len(str(config.lx)) - len(str(config.ly)) - len(str(config.n_qubits))}}║")
    print(f"║  GPU mode: {gpu_str:{52}s}║")
    print("╚══════════════════════════════════════════════════════════════════╝")


def classify_region(q0, q1, config):
    """Classify a bond as 'bulk', 'edge', or 'corner'."""
    def n_neighbors(q):
        x, y = site_coords(q, config.ly)
        count = 0
        if x > 0: count += 1
        if x < config.lx - 1: count += 1
        if y > 0: count += 1
        if y < config.ly - 1: count += 1
        return count
    n0, n1 = n_neighbors(q0), n_neighbors(q1)
    if n0 == 4 and n1 == 4:
        return "bulk"
    elif n0 <= 2 or n1 <= 2:
        return "corner"
    else:
        return "edge"


# ─────────────────────────────────────────────────────────────────────
# Act 1: PP Scout — Find the hot subsystem
# ─────────────────────────────────────────────────────────────────────

def demonstrate_pp_scout(config):
    """
    Scout phase: Pauli Propagator identifies the most entangled region.

    Runs a single-step TFIM circuit via PP and computes ⟨Z_i Z_j⟩ for all
    bonds. Uses coordination-weighted scoring to rank subsystems.
    """
    header(f"ACT 1: PP SCOUT — Scanning {config.lx}×{config.ly} Lattice "
           f"({config.n_qubits} qubits)")
    print(f"  Backend: PauliPropagator (Heisenberg picture)")
    print(f"  Circuit: 1-step TFIM Trotter")
    print(f"  Purpose: Find where entanglement will concentrate\n")

    t0 = time.time()
    scout = scout_entanglement(config)
    scout_time = time.time() - t0

    # Print top/bottom scored bonds
    scored = scout['scored_bonds']
    print(f"  {'Bond':>12}  {'Site':>14}  {'⟨ZZ⟩':>10}  {'Coord':>5}  "
          f"{'Score':>8}  {'Region'}")
    print(f"  {'─'*12:>12}  {'─'*14:>14}  {'─'*10:>10}  {'─'*5:>5}  "
          f"{'─'*8:>8}  {'─'*6}")
    for bond, zz, score, nn_sum in scored[:5]:
        x0, y0 = site_coords(bond[0], config.ly)
        x1, y1 = site_coords(bond[1], config.ly)
        region = classify_region(bond[0], bond[1], config)
        print(f"  ({bond[0]:2d},{bond[1]:2d})      "
              f"({x0},{y0})↔({x1},{y1})    {zz:+.6f}  {nn_sum:5d}  "
              f"{score:8.5f}  {region}")
    print(f"  {'...':>12}")
    for bond, zz, score, nn_sum in scored[-3:]:
        x0, y0 = site_coords(bond[0], config.ly)
        x1, y1 = site_coords(bond[1], config.ly)
        region = classify_region(bond[0], bond[1], config)
        print(f"  ({bond[0]:2d},{bond[1]:2d})      "
              f"({x0},{y0})↔({x1},{y1})    {zz:+.6f}  {nn_sum:5d}  "
              f"{score:8.5f}  {region}")

    hot = scout['hot_qubits']
    cold = scout['cold_qubits']
    hx0, hy0 = site_coords(hot[0], config.ly)
    hx1, hy1 = site_coords(hot[1], config.ly)
    cx0, cy0 = site_coords(cold[0], config.ly)
    cx1, cy1 = site_coords(cold[1], config.ly)

    print(f"\n  HOT:  qubits {hot}  ({hx0},{hy0})↔({hx1},{hy1})  "
          f"score={scout['hot_score']:.5f}")
    print(f"  COLD: qubits {cold}  ({cx0},{cy0})↔({cx1},{cy1})  "
          f"score={scout['cold_score']:.5f}")
    print(f"\n  Scouted in {scout_time:.3f}s via Pauli Propagator")

    return scout, scout_time


# ─────────────────────────────────────────────────────────────────────
# Act 2: T-gate magic — Forces Schrödinger picture
# ─────────────────────────────────────────────────────────────────────

def demonstrate_tgate_transition(config):
    """
    Inject T-gates to break Clifford structure, forcing MPS takeover.
    T† X T = (X + Y)/√2 → exponential branching in Pauli propagation.
    """
    header("ACT 2: THE BREAKING POINT — Non-Clifford Magic")
    n = config.n_qubits
    bonds = get_nn_bonds(config.lx, config.ly)

    print(f"  System: {config.lx}×{config.ly} TFIM ({n} qubits)")
    print(f"  H = -J Σ ZᵢZⱼ  -  h Σ Xᵢ   (J={config.j_coupling}, h={config.h_field})")

    qc = build_tfim_trotter_circuit(
        n, bonds, config.j_coupling, config.h_field, config.dt, config.n_trotter_steps
    )
    n_t_gates = min(n // 2, 8)
    for q in range(n_t_gates):
        qc.t(q)

    print(f"\n  ⚠ Injected {n_t_gates} T-gates → PP would need 2^{n_t_gates} "
          f"= {2**n_t_gates} Pauli terms")
    print(f"  Maestro switches to Schrödinger picture (MPS)")

    obs_terms = []
    for q1, q2 in bonds:
        obs_terms.append(build_pauli_observable(n, {q1: 'Z', q2: 'Z'}))
    for q in range(n):
        obs_terms.append(build_pauli_observable(n, {q: 'X'}))

    t0 = time.time()
    result = qc.estimate(
        simulator_type=config.simulator_type,
        simulation_type=maestro.SimulationType.MatrixProductState,
        observables=obs_terms,
        max_bond_dimension=config.chi_low,
    )
    mps_time = time.time() - t0

    exp_vals = result['expectation_values']
    n_bonds = len(bonds)
    e_zz = sum(-config.j_coupling * exp_vals[i] for i in range(n_bonds))
    e_x = sum(-config.h_field * exp_vals[n_bonds + i] for i in range(n))

    print(f"\n  Energy (MPS, χ={config.chi_low}): E = {e_zz + e_x:.4f}")
    print(f"  Computed in {mps_time:.3f}s (Schrödinger picture)")
    return e_zz + e_x


# ─────────────────────────────────────────────────────────────────────
# Act 3: Adaptive bond dimension handoff
# ─────────────────────────────────────────────────────────────────────

def time_evolve_with_handoff(config):
    """CPU→GPU bond dimension handoff during TFIM time evolution."""
    header("ACT 3: THE HANDOFF — Adaptive Bond Dimension")
    n = config.n_qubits
    bonds = get_nn_bonds(config.lx, config.ly)

    low_label = f"CPU χ={config.chi_low}"
    high_label = (f"GPU χ={config.chi_high}" if config.use_gpu
                  else f"CPU χ={config.chi_high}")

    print(f"  Time evolution: T = {config.t_total}, "
          f"{config.n_trotter_steps} Trotter steps")
    print(f"  {low_label} → {high_label} when entanglement exceeds threshold\n")

    obs_terms = []
    for q1, q2 in bonds:
        obs_terms.append(build_pauli_observable(n, {q1: 'Z', q2: 'Z'}))
    for q in range(n):
        obs_terms.append(build_pauli_observable(n, {q: 'X'}))

    energies, sim_times, backends = [], [], []
    switched = False
    t0 = time.time()

    for step in range(config.n_trotter_steps + 1):
        t_sim = step * config.dt
        chi = config.chi_low if not switched else config.chi_high
        sim_type = (maestro.SimulatorType.QCSim if not switched
                    else config.simulator_type)
        backend_label = 'low' if not switched else 'high'

        qc = build_tfim_trotter_circuit(
            n, bonds, config.j_coupling, config.h_field, config.dt, step
        )

        if step == 0:
            energy = -n
        else:
            result = qc.estimate(
                simulator_type=sim_type,
                simulation_type=maestro.SimulationType.MatrixProductState,
                observables=obs_terms,
                max_bond_dimension=chi,
            )
            exp_vals = result['expectation_values']
            n_bonds = len(bonds)
            e_zz = sum(-config.j_coupling * exp_vals[i] for i in range(n_bonds))
            e_x = sum(-config.h_field * exp_vals[n_bonds + i] for i in range(n))
            energy = e_zz + e_x

        energies.append(energy)
        sim_times.append(t_sim)
        backends.append(backend_label)

        elapsed = time.time() - t0
        if step % max(1, config.n_trotter_steps // 5) == 0:
            label = low_label if not switched else high_label
            print(f"  t = {t_sim:5.2f}  |  E(t) = {energy:10.4f}  |  "
                  f"{label}  |  {elapsed:.3f}s")

        if not switched and step >= 2:
            if abs(energies[-1] - energies[-2]) > config.entanglement_threshold:
                switched = True
                print(f"\n  ⚡ Handoff: {low_label} → {high_label} at t = {t_sim:.2f}")

    print(f"\n  ✓ Time evolution complete")
    return energies, sim_times, backends


# ─────────────────────────────────────────────────────────────────────
# Acts 4-5: Shadow pipeline demo
# ─────────────────────────────────────────────────────────────────────

def demonstrate_shadow_pipeline(config):
    """Acts 4-5: Random Cliffords + bitstring sampling demo."""
    n = config.n_qubits
    bonds = get_nn_bonds(config.lx, config.ly)

    header("ACT 4: RANDOMIZED MEASUREMENTS — Shadow Basis Rotation")
    qc = build_tfim_trotter_circuit(
        n, bonds, config.j_coupling, config.h_field, config.dt, config.n_trotter_steps
    )
    rng = np.random.default_rng(0)
    labels = append_random_clifford_layer(qc, n, rng)

    print(f"  Random single-qubit Cliffords on {n} qubits")
    print(f"  U†_i Z U_i = P_i — tracked classically, O(1) per qubit")

    header("ACT 5: HARDWARE EMULATION — Bitstring Sampling")
    chi = config.chi_high if config.use_gpu else config.chi_low
    print(f"  estimate() → execute(shots={config.n_shots})")
    print(f"  Backend: {'GPU' if config.use_gpu else 'CPU'} MPS (χ = {chi})")

    qc.measure_all()
    t0 = time.time()
    result = qc.execute(
        simulator_type=config.simulator_type,
        simulation_type=maestro.SimulationType.MatrixProductState,
        shots=config.n_shots,
        max_bond_dimension=chi,
    )
    sample_time = time.time() - t0

    counts = result['counts']
    print(f"\n  ✓ {len(counts)} unique bitstrings in {sample_time:.3f}s")
    return counts


# ─────────────────────────────────────────────────────────────────────
# Act 6: Shadow sweep — hot vs cold subsystem
# ─────────────────────────────────────────────────────────────────────

def shadow_entanglement_sweep(config, subsystem_qubits, label=""):
    """
    Sweep Trotter depths, estimating S₂(t) via classical shadows
    on the specified subsystem.
    """
    n = config.n_qubits
    bonds = get_nn_bonds(config.lx, config.ly)

    qx0, qy0 = site_coords(subsystem_qubits[0], config.ly)
    qx1, qy1 = site_coords(subsystem_qubits[1], config.ly)

    if label:
        print(f"\n  ── {label}: qubits {subsystem_qubits} — "
              f"site ({qx0},{qy0})↔({qx1},{qy1}) ──")
    print(f"  Depths: {config.trotter_depths}  |  "
          f"Shadows/depth: {config.n_shadows}")

    results = {'depths': [], 'times': [], 's2': [], 'purity': []}

    for depth in config.trotter_depths:
        t_val = depth * config.dt
        shadows = collect_shadow_snapshots(
            config, depth, bonds, subsystem_qubits, verbose=False
        )
        purity_raw = estimate_purity_from_shadows(shadows)
        s2, purity = renyi_s2(purity_raw, config.d_A)

        results['depths'].append(depth)
        results['times'].append(t_val)
        results['s2'].append(s2)
        results['purity'].append(purity)
        print(f"    depth={depth:2d} (t={t_val:.2f})  S₂={s2:.4f}  "
              f"purity={purity:.6f}")

    return results


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def build_config():
    """Parse CLI args and return Config."""
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
    banner(config)
    total_start = time.time()

    # ── Act 1: PP Scout — find where entanglement lives ──
    scout, scout_time = demonstrate_pp_scout(config)
    hot_qubits = scout['hot_qubits']
    cold_qubits = scout['cold_qubits']

    # ── Act 2: T-gates break Clifford → MPS ──
    demonstrate_tgate_transition(config)

    # ── Act 3: Adaptive bond dimension handoff ──
    energies, sim_times, backends = time_evolve_with_handoff(config)

    # ── Acts 4-5: Shadow pipeline demo ──
    demonstrate_shadow_pipeline(config)

    # ── Act 6: MPS Sniper — shadow sweep on hot AND cold subsystems ──
    header("ACT 6: MPS SNIPER — Entanglement Growth Sweep")
    print(f"  PP scout selected: HOT={hot_qubits}  COLD={cold_qubits}")
    print(f"  Running shadows on both to validate the scout...\n")

    hot_results = shadow_entanglement_sweep(
        config, hot_qubits, label="🎯 HOT (PP-selected)"
    )
    cold_results = shadow_entanglement_sweep(
        config, cold_qubits, label="❄️  COLD (contrast)"
    )

    # ── Exact ED (if feasible) ──
    header("GENERATING VISUALIZATIONS")
    hot_exact, cold_exact = None, None
    if config.n_qubits <= 20:
        print("  Computing exact ED reference...")
        hot_exact = compute_exact_s2(config, hot_qubits)
        cold_exact = compute_exact_s2(config, cold_qubits)
        print("  ✓ Exact reference computed for both subsystems.")
    else:
        print(f"  ℹ Exact ED skipped (n={config.n_qubits} > 20)")
        print(f"  Classical shadows are the ONLY way to estimate S₂ at this scale!")

    # ── Plots ──
    energy_path = plot_energy_evolution(
        sim_times, energies, backends,
        os.path.join(SCRIPT_DIR, 'energy_evolution.png'),
    )
    print(f"  📊 Saved: {energy_path}")

    comparison_path = plot_scout_comparison(
        hot_results, cold_results,
        hot_exact, cold_exact,
        hot_qubits, cold_qubits,
        config,
        os.path.join(SCRIPT_DIR, 'entanglement_growth.png'),
    )
    print(f"  📊 Saved: {comparison_path}")

    heatmap_path = plot_lattice_heatmap(
        config, scout,
        os.path.join(SCRIPT_DIR, 'lattice_heatmap.png'),
    )
    print(f"  📊 Saved: {heatmap_path}")

    # ── MAE (when exact is available) ──
    if hot_exact:
        hot_mae = np.mean([abs(s - e) for s, e in
                           zip(hot_results['s2'], hot_exact['s2'])])
        cold_mae = np.mean([abs(s - e) for s, e in
                            zip(cold_results['s2'], cold_exact['s2'])])
        print(f"\n  MAE vs exact:  HOT = {hot_mae:.3f}  |  COLD = {cold_mae:.3f}")

    # ── Summary ──
    total_elapsed = time.time() - total_start
    header("SHOWCASE COMPLETE")
    print(f"  System: {config.lx}×{config.ly} = {config.n_qubits} qubits")
    print(f"  GPU: {'Yes' if config.use_gpu else 'No'}")
    print(f"  Total runtime: {total_elapsed:.1f}s\n")
    print(f"  Scout → Sniper pipeline:")
    print(f"    Act 1: PP scouted {config.n_qubits} qubits in {scout_time:.3f}s"
          f" → HOT={hot_qubits}, COLD={cold_qubits}")
    print(f"    Act 2: T-gates broke Clifford → MPS takeover")
    print(f"    Act 3: Adaptive χ handoff (CPU → "
          f"{'GPU' if config.use_gpu else 'high-bond CPU'})")
    print(f"    Act 6: Shadows on HOT vs COLD subsystems "
          f"({config.n_shadows} snapshots each)")

    if hot_exact:
        hot_mean = np.mean(hot_results['s2'])
        cold_mean = np.mean(cold_results['s2'])
        diff = hot_mean - cold_mean
        print(f"\n  Entanglement comparison:")
        print(f"    HOT  avg S₂ = {hot_mean:.3f}  (PP-selected)")
        print(f"    COLD avg S₂ = {cold_mean:.3f}  (weakest correlation)")
        if abs(diff) > 0.1:
            winner = "HOT" if diff > 0 else "COLD"
            print(f"    → Scout {'confirmed' if diff > 0 else 'overridden'}: "
                  f"{winner} has more entanglement (ΔS₂ = {abs(diff):.3f})")
        else:
            print(f"    → Both subsystems show similar entanglement "
                  f"(ΔS₂ = {abs(diff):.3f})")

    print(f"\n  📊 {energy_path}")
    print(f"  📊 {comparison_path}")
    print(f"  📊 {heatmap_path}")
    print(f"\n  PP scouts, MPS targets — Maestro orchestrates it all. 🎼")
