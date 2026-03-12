# Rydberg Atom Array Simulation

> 🚀 **Try Maestro GPU mode with a free trial.**
> Sign up at **[maestro.qoroquantum.net](https://maestro.qoroquantum.net)** — no credit card required.

## Why GPU Mode?

A 64-atom Rydberg phase diagram sweep means **144 independent MPS simulations** (12×12 grid). At bond dimension χ=16, that's manageable on CPU. But accurate phase boundaries need higher χ — and MPS cost scales as **O(χ³)**. Doubling χ from 16 to 32 means **8× the runtime per simulation point**. GPU mode makes high-χ sweeps practical.

## What It Does

Simulates the adiabatic preparation of a **Z2-ordered phase** in a 1D Rydberg atom array. Neutral atoms interact via van der Waals blockade — sweeping the detuning Δ drives a quantum phase transition into an alternating excitation pattern (|1010...⟩).

### Phase 1 — Local (CPU)

64 atoms, χ=16, 12×12 parameter grid. Fast enough on a laptop to map the phase diagram and see the Z2 phase emerge.

### Phase 2 — GPU Mode

Scale to higher bond dimension (χ=32–64) for sharper phase boundaries, or increase grid resolution — GPU handles the O(χ³) cost efficiently.

```bash
# Phase 1: CPU (default)
python rydberg_demo.py

# Phase 2: GPU-accelerated
python rydberg_demo.py --gpu
```

## Scripts

| Script | What It Does |
|--------|-------------|
| `rydberg_demo.py` | Phase diagram sweep → `rydberg_phase_diagram.png` |
| `rydberg_correlations.py` | Spatial correlation function → `rydberg_correlations.png` |

📓 **[Interactive notebook](./rydberg_atom_simulation.ipynb)** — step-by-step tutorial

## Configuration

| Parameter | Default |
|-----------|---------|
| Atoms | 64 |
| Bond dim χ | 16 |
| Grid | 12 × 12 (Δ vs Ω) |
| Trotter steps | 20 (dt = 0.15) |
| Interaction V | 5.0 |

## Expected Output

**`rydberg_phase_diagram.png`** — Heatmap showing two phases:
- **Disordered** (small Δ): All atoms in ground state
- **Z2 Ordered** (large Δ, moderate Ω): Alternating excitation pattern

**`rydberg_correlations.png`** — Connected correlation function C(r) from bitstring sampling, confirming long-range crystalline order.

---

👉 **Ready for sharper phase boundaries?** [Start your free GPU trial](https://maestro.qoroquantum.net) and scale up with `--gpu`.
