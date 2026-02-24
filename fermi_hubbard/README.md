# Fermi-Hubbard Model: Adaptive Simulation Pipeline

Demonstrates Maestro's multi-tier adaptive simulation for the 1D Fermi-Hubbard model — a fundamental model of strongly correlated electron systems.

## Physics Background

The Fermi-Hubbard model describes electrons hopping on a lattice with on-site interactions:

$$H = -t \sum_{\langle i,j \rangle, \sigma} (c^\dagger_{i,\sigma} c_{j,\sigma} + \text{h.c.}) + U \sum_i n_{i,\uparrow} n_{i,\downarrow}$$

where:
- **t** (hopping) controls kinetic energy — electrons tunnel between adjacent sites
- **U** (interaction) controls potential energy — penalty for double occupancy
- **U/t** ratio determines the phase: metallic (small U/t) vs Mott insulator (large U/t)

### Jordan-Wigner Mapping

The fermionic Hamiltonian is mapped to qubits using the Jordan-Wigner transformation:
- Qubits `[0, N)` → spin-up electrons
- Qubits `[N, 2N)` → spin-down electrons

For 1D nearest-neighbor hopping, no Jordan-Wigner strings are needed, giving:
- **Hopping:** `exp(-i dt t (XX + YY) / 2)` per bond
- **Interaction:** `exp(-i dt U/4 (I - Z↑ - Z↓ + Z↑Z↓))` per site

### Domain Wall Quench

The simulation starts from a **domain wall** initial state — left half filled, right half empty. This creates a local quench whose dynamics reveal charge transport. The key insight is that information propagates at a finite **Lieb-Robinson velocity**, creating a causal light cone.

## The 3-Tier Adaptive Pipeline

### Tier 1: Scout (Pauli Propagator)

**Backend:** `PauliPropagator` · **Cost:** O(n), seconds

Runs a Clifford-only proxy circuit on the **full** system to detect which sites have non-trivial dynamics. The Clifford proxy preserves the circuit's connectivity structure while remaining PP-compatible.

**How it works:** Measures `⟨Z_i⟩` for all sites and compares to the initial domain-wall state. Sites where `|⟨Z⟩ - Z_initial| > threshold` are marked "active."

### Tier 2: Sniper (MPS CPU, χ=64)

**Backend:** `MatrixProductState` on CPU

Runs the **real** Trotterized circuit (with non-Clifford Rz gates) only on the active subregion detected by the scout. For a 200-qubit system with a ~40-qubit active region, this is a **5× reduction** in simulation cost.

### Tier 3: Precision (MPS GPU/CPU, χ=256)

**Backend:** `MatrixProductState` with high bond dimension

Re-runs with higher χ for converged results. GPU acceleration provides 10–100× speedup for the O(χ³) tensor contractions that dominate MPS cost.

## Code Structure

| File | Purpose |
|------|---------|
| `model.py` | `FermiHubbardModel` class — circuit construction for Trotter evolution and Clifford scout |
| `fermi_hubbard_demo.py` | 3-tier pipeline: Scout → Sniper → Precision, with visualization |

## Usage

```bash
# Default: 200-qubit system, CPU only
python fermi_hubbard_demo.py

# With GPU precision tier
python fermi_hubbard_demo.py --gpu

# Include scaling sweep across system sizes
python fermi_hubbard_demo.py --scaling
```

## Output

- **`adaptive_hubbard_density.png`** — Particle density profile showing the domain wall spreading, with active vs frozen regions highlighted
- **`adaptive_hubbard_scaling.png`** — Wall-clock time vs system size showing that MPS time is constant (light cone is fixed) while only scout time grows (generated with `--scaling`)

## Key Maestro Features Used

| Feature | API | Where Used |
|---------|-----|------------|
| Pauli Propagator | `SimulationType.PauliPropagator` | Tier 1 (scout) |
| MPS Simulation | `SimulationType.MatrixProductState` | Tiers 2, 3 |
| Bond dimension control | `max_bond_dimension=64/256` | Tier 2 vs 3 |
| GPU acceleration | `SimulatorType.Gpu` | Tier 3 with `--gpu` |
| QuantumCircuit | `maestro.circuits.QuantumCircuit` | All tiers |

## Requirements

- `qoro-maestro` Python package
- `numpy`
- `matplotlib`
