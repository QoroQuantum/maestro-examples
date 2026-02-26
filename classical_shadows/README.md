# Classical Shadows: Entanglement Detection via MPS

## Why This Problem Matters

**Entanglement entropy** quantifies genuine quantum correlations but requires exponentially many measurements via full tomography. **Classical Shadows** ([Huang et al., 2020](https://arxiv.org/abs/2002.08953)) reduce this to $O(\log n / \varepsilon^2)$ samples — estimating properties of quantum states with far fewer measurements than full state tomography.

This example demonstrates the classical shadows protocol using Maestro's MPS backend, estimating the 2nd Rényi entropy $S_2$ of subsystems in a transverse-field Ising model (TFIM) on a 2D lattice.

## Protocol

1. **Prepare** the state via Trotterized time evolution of the TFIM
2. **Apply** random single-qubit Cliffords to each qubit
3. **Measure** in the computational basis (1 shot per snapshot)
4. **Reconstruct** shadow density matrix $\hat{\rho} = \bigotimes_i (3 U_i^\dagger |b_i\rangle\langle b_i| U_i - I)$
5. **Repeat** $M$ times and average to estimate $S_2$

## Code Structure

| File | Purpose |
|------|---------|
| `helpers.py` | Reusable functions: `Config`, circuit builders, shadow collection, purity estimation |
| `classical_shadows_demo.py` | Main script: sweep Trotter depths, estimate $S_2$, compare with exact ED |

## Usage

```bash
# Quick test: 4×4 = 16 qubits + exact ED reference (~2 min)
python classical_shadows_demo.py --small

# Full run: 6×6 = 36 qubits
python classical_shadows_demo.py

# With GPU acceleration
python classical_shadows_demo.py --gpu
```

## Output

- **`entanglement_growth.png`** — $S_2$ vs simulation time, with exact ED reference (when ≤20 qubits)

## Results

The shadow estimates track the exact entanglement growth curve, demonstrating that the protocol correctly captures how entanglement develops during Trotter evolution:

![Entanglement growth via classical shadows](entanglement_growth.png)

At 36 qubits, exact diagonalization is impossible ($2^{36}$ amplitudes). Classical shadows use just 200 snapshots vs $4^{36} \approx 4.7 \times 10^{21}$ full tomography measurements.

### Sample complexity note

Each shadow snapshot is an independent MPS simulation (`execute(shots=1)`). The shadow reconstruction factor of $3^k$ (where $k$ is the subsystem size) introduces variance — individual $S_2$ estimates are noisy, but the qualitative trend is reliable. For higher quantitative accuracy, increase the number of snapshots (e.g., `--n-shadows 1000`).

## Key Maestro Features Used

| Feature | API | Purpose |
|---------|-----|---------|
| MPS Simulation | `SimulationType.MatrixProductState` | State preparation and measurement |
| Bond dimension | `max_bond_dimension=χ` | Controls accuracy vs speed |
| Bitstring sampling | `qc.execute(shots=1)` | Shadow snapshot collection |
| Expectation values | `qc.estimate(observables=...)` | Exact ED reference |
| GPU acceleration | `SimulatorType.CuQuantum` | Optional GPU speedup with `--gpu` |

## Requirements

- `qoro-maestro` Python package
- `numpy`
- `matplotlib`
