#!/usr/bin/env python3
"""
Rydberg Atom Array Phase Diagram — Maestro Demo
=================================================

Simulates the adiabatic preparation of a Z2-ordered phase in a 1D Rydberg atom
array using Maestro's Matrix Product State (MPS) backend.

The Hamiltonian is:
    H = (Ω/2) Σ Xᵢ  −  Δ Σ nᵢ  +  V Σ nᵢ nᵢ₊₁

where nᵢ = (I − Zᵢ)/2 is the Rydberg excitation number operator.

The simulation sweeps over a grid of (Δ, Ω) values, computing the Z2 staggered
magnetization order parameter at each point to map out the phase diagram.

Output:
    rydberg_phase_diagram.png  —  Phase diagram heatmap

Usage:
    python rydberg_demo.py
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import maestro
from maestro.circuits import QuantumCircuit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_rydberg_circuit(num_atoms, omega, delta, interaction_v, steps, dt):
    """
    Build a QuantumCircuit simulating adiabatic preparation of a Rydberg state.

    Uses first-order Trotterization with a linear ramp schedule:
      - Ω ramps from 0 → omega
      - Δ ramps from −5.0 → delta

    The circuit implements:
      1. ZZ interactions (Rydberg blockade) via CX-Rz-CX on nearest-neighbor pairs
      2. Single-qubit Rx (drive) and Rz (detuning + interaction corrections)

    Args:
        num_atoms: Number of atoms in the 1D chain.
        omega: Final Rabi frequency.
        delta: Final detuning.
        interaction_v: Nearest-neighbor interaction strength V.
        steps: Number of Trotter steps.
        dt: Time step size.

    Returns:
        A QuantumCircuit instance.
    """
    qc = QuantumCircuit()

    delta_start = -5.0

    for s in range(steps):
        t_frac = s / steps

        curr_omega = t_frac * omega
        curr_delta = delta_start + t_frac * (delta - delta_start)

        # ZZ interactions (Rydberg blockade): θ = V · dt / 2
        zz_theta = interaction_v * dt / 2

        # Even bonds
        for i in range(0, num_atoms - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(i + 1, zz_theta)
            qc.cx(i, i + 1)

        # Odd bonds
        for i in range(1, num_atoms - 1, 2):
            qc.cx(i, i + 1)
            qc.rz(i + 1, zz_theta)
            qc.cx(i, i + 1)

        # Single-qubit gates: drive (Rx) and detuning (Rz)
        for i in range(num_atoms):
            qc.rx(i, curr_omega * dt)

            z_angle = curr_delta * dt

            # Interaction corrections (single-Z terms from V)
            neighbors = 0
            if i > 0:
                neighbors += 1
            if i < num_atoms - 1:
                neighbors += 1
            z_angle += neighbors * (-interaction_v * dt / 2)

            qc.rz(i, z_angle)

    return qc


def calculate_order_parameter(z_expects, num_atoms):
    """
    Calculate the Z2 staggered magnetization order parameter.

    O = |Σᵢ (−1)^i ⟨nᵢ⟩| / (N/2)

    where ⟨nᵢ⟩ = (1 − ⟨Zᵢ⟩) / 2.

    Args:
        z_expects: List of ⟨Zᵢ⟩ expectation values.
        num_atoms: Number of atoms.

    Returns:
        Order parameter value in [0, 1].
    """
    stag_mag = 0.0
    for i, z_val in enumerate(z_expects):
        n_val = (1.0 - z_val) / 2.0
        stag_mag += ((-1) ** i) * n_val

    return abs(stag_mag) / (num_atoms / 2)


def build_z_observables(n_qubits):
    """Build per-qubit Z observables: ['ZIII..', 'IZII..', ...]"""
    obs = []
    for i in range(n_qubits):
        pauli = ['I'] * n_qubits
        pauli[i] = 'Z'
        obs.append("".join(pauli))
    return obs


def main():
    print("=" * 65)
    print("  MAESTRO Demo: 64-Atom Rydberg Array — Phase Diagram")
    print("  Simulating Adiabatic Preparation of Z2 Ordered Phase")
    print("=" * 65)

    # ── Configuration ──
    N = 64
    max_bond_dim = 16
    T_total = 3.0
    dt = 0.15
    steps = int(T_total / dt)
    grid_size = 12

    V_interaction = 5.0
    min_delta = -1.0
    max_delta = 4.0
    max_omega = 3.0

    print(f"\n  System Size:  {N} atoms")
    print(f"  Grid:         {grid_size} × {grid_size}")
    print(f"  Trotter:      {steps} steps (dt = {dt})")
    print(f"  Backend:      MPS (max bond dim = {max_bond_dim})")
    print("-" * 65)
    print("  Sweeping phase space (Δ vs Ω)...")
    print("  Legend: [·] Disordered  [▒] Intermediate  [█] Z2 Ordered")
    print("-" * 65)

    deltas = np.linspace(min_delta, max_delta, grid_size)
    omegas = np.linspace(0.1, max_omega, grid_size)
    heatmap_data = np.zeros((grid_size, grid_size))

    observables = build_z_observables(N)

    start_time = time.time()

    for i, omega in enumerate(reversed(omegas)):
        row_label = f"  Ω={omega:4.2f} | "
        row_viz = ""

        for j, delta in enumerate(deltas):
            qc = create_rydberg_circuit(
                N, omega, delta, V_interaction, steps, dt
            )

            try:
                res = qc.estimate(
                    observables=observables,
                    simulator_type=maestro.SimulatorType.QCSim,
                    simulation_type=maestro.SimulationType.MatrixProductState,
                    max_bond_dimension=max_bond_dim,
                )

                if res and 'expectation_values' in res:
                    z_expects = res['expectation_values']
                    op = calculate_order_parameter(z_expects, N)
                    heatmap_data[grid_size - 1 - i, j] = op

                    if op > 0.6:
                        char = "█"
                    elif op > 0.3:
                        char = "▒"
                    else:
                        char = "·"
                else:
                    char = "X"

            except Exception:
                char = "!"

            row_viz += char + " "

        print(row_label + row_viz)

    total_time = time.time() - start_time
    print("-" * 65)
    print(f"  Sweep completed in {total_time:.2f}s")

    # ── Visualization ──
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        extent = [min_delta, max_delta, 0.1, max_omega]

        im = ax.imshow(
            heatmap_data, origin='lower', extent=extent,
            cmap='inferno', aspect='auto'
        )
        plt.colorbar(im, ax=ax, label='Z2 Staggered Magnetization')
        ax.set_xlabel(r'Detuning ($\Delta$)', fontsize=13)
        ax.set_ylabel(r'Rabi Frequency ($\Omega$)', fontsize=13)
        ax.set_title(
            f'Rydberg Atom Array Phase Diagram\n'
            f'Adiabatic Preparation (N={N}, MPS χ={max_bond_dim})',
            fontsize=14
        )

        ax.text(
            max_delta * 0.8, 0.3, "Z2 Phase",
            color='white', ha='center', fontsize=12, fontweight='bold'
        )
        ax.text(
            min_delta + 0.5, 0.3, "Disordered",
            color='white', ha='center', fontsize=11
        )

        filename = os.path.join(SCRIPT_DIR, "rydberg_phase_diagram.png")
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  📊 Phase diagram saved: {filename}")

    except Exception as e:
        print(f"  Could not generate plot: {e}")


if __name__ == "__main__":
    main()
