# Rydberg Atom Array Simulation

Simulates the adiabatic preparation of a Z2-ordered phase in a 1D Rydberg atom array using Maestro's Matrix Product State (MPS) backend.

## Physics Background

Rydberg atom arrays are a leading platform for quantum simulation. Neutral atoms trapped in optical tweezers interact via strong van der Waals interactions when excited to Rydberg states. The system is governed by:

$$H = \frac{\Omega}{2} \sum_i X_i - \Delta \sum_i n_i + V \sum_{\langle i,j \rangle} n_i n_j$$

where:
- **Ω** (Rabi frequency) drives transitions between ground and Rydberg states
- **Δ** (detuning) controls the energy cost of Rydberg excitation
- **V** (interaction strength) implements the Rydberg blockade
- **n_i = (I − Z_i)/2** is the Rydberg excitation number operator

By sweeping Δ from negative (all atoms in ground state) to positive (favoring excitation), with V preventing adjacent excitations, the system undergoes a quantum phase transition into a **Z2-ordered phase** — an alternating pattern of excited and ground-state atoms (|1010...⟩).

## Examples

### 1. Phase Diagram (`rydberg_demo.py`)

Sweeps over a 2D grid of (Δ, Ω) values to map the phase diagram of the Rydberg atom array. At each point, the Z2 staggered magnetization order parameter is computed:

$$\mathcal{O} = \frac{1}{N/2} \left| \sum_i (-1)^i \langle n_i \rangle \right|$$

This reveals two phases:
- **Disordered** (small Δ): All atoms in the ground state, O ≈ 0
- **Z2 Ordered** (large Δ, moderate Ω): Alternating excitation pattern, O → 1

**Maestro features used:**
- `QuantumCircuit` for programmatic circuit construction
- `qc.estimate()` with MPS backend for efficient 64-qubit simulation
- Expectation value computation without sampling noise

```bash
python rydberg_demo.py
```

**Output:** `rydberg_phase_diagram.png`

### 2. Spatial Correlations (`rydberg_correlations.py`)

Demonstrates MPS **sampling mode** to measure the connected correlation function:

$$C(r) = \langle n_i \, n_{i+r} \rangle - \langle n_i \rangle \langle n_{i+r} \rangle$$

Key insight: computing pairwise correlations via `estimate()` would require O(N²) separate observable evaluations. Using `execute()` (sampling), all correlations are extracted from a single set of bitstrings.

**Maestro features used:**
- `qc.execute()` with MPS backend — stochastic bitstring sampling
- MPS samples bitstrings efficiently via sequential conditional probabilities

```bash
python rydberg_correlations.py
```

**Output:** `rydberg_correlations.png`

## Configuration

Both scripts use default parameters tuned for a laptop CPU:
- **N = 64 atoms** (phase diagram), **N = 15** (correlations)
- **MPS bond dimension:** χ = 16–32
- **Trotter steps:** 20 (dt = 0.15)

## Requirements

- `qoro-maestro` Python package
- `numpy`
- `matplotlib`
