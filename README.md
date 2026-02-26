# Maestro Examples

Example simulations demonstrating quantum simulation capabilities with the [Maestro](https://github.com/QoroQuantum/maestro) high-performance quantum circuit simulator.

## Table of Contents

1. [Rydberg Atom Simulation](#1-rydberg-atom-simulation) — Phase diagram and spatial correlations of a 64-atom array
2. [Classical Shadows](#2-classical-shadows) — Entanglement detection via MPS-based shadow tomography
3. [Fermi-Hubbard Model](#3-fermi-hubbard-model) — Adaptive 3-tier simulation exploiting Lieb-Robinson bounds
4. [Adaptive Bond Dimension](#4-adaptive-bond-dimension) — CPU↔GPU backend switching for time evolution

## Getting Started

```bash
pip install -r requirements.txt
```

Or install the core dependency directly:

```bash
pip install qoro-maestro
```

Each example directory includes its own README with detailed instructions.

## Examples

### 1. [Rydberg Atom Simulation](./rydberg_atom_simulation)

Simulates the adiabatic preparation of a Z2-ordered phase in a 1D Rydberg atom array. Sweeps over detuning and Rabi frequency to map the quantum phase diagram, then measures spatial correlations to confirm long-range crystalline order.

**Key Features:**
- 64-qubit MPS simulation of Rydberg blockade physics
- Phase diagram sweep with Z2 staggered magnetization order parameter
- Spatial correlation measurement via MPS bitstring sampling
- Comparison of `estimate()` (noise-free) vs `execute()` (sampling) modes

**Scripts:**
- `rydberg_demo.py` — Phase diagram sweep → `rydberg_phase_diagram.png`
- `rydberg_correlations.py` — Correlation function → `rydberg_correlations.png`

📓 **[Interactive notebook](./rydberg_atom_simulation/rydberg_atom_simulation.ipynb)** — step-by-step tutorial

---

### 2. [Classical Shadows](./classical_shadows)

Estimates entanglement entropy using the classical shadows protocol ([Huang et al., 2020](https://arxiv.org/abs/2002.08953)) with Maestro's MPS backend. Tracks how the 2nd Rényi entropy $S_2$ grows during Trotter evolution of the transverse-field Ising model.

**Key Features:**
- Classical shadow protocol with random single-qubit Cliffords
- MPS-based state preparation and measurement (`execute(shots=1)`)
- Entanglement growth curves across Trotter depths
- Exact ED reference for small systems (≤20 qubits)

**Scripts:**
- `classical_shadows_demo.py` — Shadow sweep → `entanglement_growth.png`
- `helpers.py` — Reusable library: config, circuits, shadow reconstruction

📓 **[Interactive notebook](./classical_shadows/classical_shadows.ipynb)** — step-by-step tutorial

```bash
# Quick test (4×4 = 16 qubits, ~2 min)
python classical_shadows_demo.py --small

# Full run (6×6 = 36 qubits)
python classical_shadows_demo.py

# With GPU acceleration
python classical_shadows_demo.py --gpu
```

At 36 qubits, full tomography needs $4^{36} \approx 4.7 \times 10^{21}$ measurements — classical shadows use just 200 snapshots.

---

### 3. [Fermi-Hubbard Model](./fermi_hubbard)

Adaptive simulation of the 1D Fermi-Hubbard model — a fundamental model of strongly correlated electrons. Exploits the Lieb-Robinson bound: after a local quench, information propagates at finite speed, so most of the system remains frozen. A 200-qubit system is reduced to ~40 active qubits.

**Key Features:**
- 3-tier adaptive pipeline: PP Scout → MPS Sniper (χ=64) → Precision (χ=256)
- Clifford-only Pauli Propagator for light-cone detection on the full system
- Jordan-Wigner mapping with nearest-neighbor hopping (no JW strings)
- Domain-wall quench dynamics with charge transport visualization
- Scaling sweep demonstrating constant MPS cost regardless of total system size
- GPU acceleration for the precision tier

**Scripts:**
- `fermi_hubbard_demo.py` — Full pipeline → `adaptive_hubbard_density.png`, `adaptive_hubbard_scaling.png`
- `model.py` — `FermiHubbardModel` class for circuit construction

📓 **[Interactive notebook](./fermi_hubbard/fermi_hubbard.ipynb)** — step-by-step tutorial

```bash
# Default run (CPU only)
python fermi_hubbard_demo.py

# With GPU precision tier
python fermi_hubbard_demo.py --gpu

# Include scaling sweep across system sizes
python fermi_hubbard_demo.py --scaling
```

The Lieb-Robinson light cone lets Maestro simulate a 200-qubit system at the cost of ~40 qubits, with GPU acceleration providing ~10× speedup on the precision tier.

---

### 4. [Adaptive Bond Dimension](./adaptive_bond_dimension)

Demonstrates how easy it is to switch between CPU and GPU backends during MPS time evolution. As entanglement grows, the simulation automatically upgrades from low bond dimension (CPU) to high bond dimension (GPU) — with just a single argument change.

**Key Features:**
- Side-by-side comparison: CPU low-χ vs CPU high-χ vs GPU high-χ
- Automatic handoff when entanglement exceeds threshold
- Per-step timing breakdown showing where GPU acceleration pays off
- Trivial backend switching — same API, same code

**Scripts:**
- `adaptive_mps.py` — Full comparison → `adaptive_comparison.png`

📓 **[Interactive notebook](./adaptive_bond_dimension/adaptive_bond_dimension.ipynb)** — step-by-step tutorial

```bash
# CPU only (compare bond dimensions)
python adaptive_mps.py

# With GPU acceleration
python adaptive_mps.py --gpu

# Large system (8×8 = 64 qubits)
python adaptive_mps.py --large --gpu
```

The pitch: Maestro lets you switch backends with one argument. No code rewrite, no separate GPU code paths.

## Maestro Features Demonstrated

| Feature | API | Examples |
|---------|-----|----------|
| Matrix Product State | `SimulationType.MatrixProductState` | All examples |
| Pauli Propagator | `SimulationType.PauliPropagator` | Fermi-Hubbard (Tier 1) |
| Bond dimension control | `max_bond_dimension=χ` | Adaptive Bond Dimension, Fermi-Hubbard |
| Expectation values | `qc.estimate(observables=...)` | Rydberg, Adaptive, Fermi-Hubbard |
| Bitstring sampling | `qc.execute(shots=N)` | Rydberg (correlations), Classical Shadows |
| CPU backend | `SimulatorType.QCSim` | All examples |
| GPU acceleration | `SimulatorType.CuQuantum` | Adaptive Bond Dimension, Fermi-Hubbard |

## License

See [LICENSE](./LICENSE) for details.
