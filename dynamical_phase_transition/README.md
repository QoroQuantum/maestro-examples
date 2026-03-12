# Dynamical Quantum Phase Transition

> 🚀 **Try Maestro GPU mode with a free trial.**
> Sign up at **[maestro.qoroquantum.net](https://maestro.qoroquantum.net)** — no credit card required.

## Why GPU Mode?

DQPT simulations sweep **multiple quench strengths**, each requiring **40 Trotter steps** of MPS evolution. Crisp Loschmidt echo cusps demand χ=64+ on 80 qubits — that's a 3-quench × 40-step sweep where every step involves O(χ³) tensor contractions. On CPU, the full sweep takes hours. GPU mode makes it practical.

## What It Does

Demonstrates **dynamical quantum phase transitions** after a sudden quench in the 1D transverse-field Ising model. The system starts in |+⟩^⊗N (paramagnetic ground state), then quenches to the ferromagnetic regime (h_f < J). The Loschmidt rate function λ(t) develops **non-analytic cusps** at critical times — dynamical analogs of thermodynamic free-energy singularities.

**Key observables:**
- **Loschmidt rate function λ(t)** — cusp singularities at critical times
- **Transverse magnetization ⟨X(t)⟩** — order parameter oscillations

### Phase 1 — Local (CPU)

30 qubits, χ=32, 3 quench strengths × 40 Trotter steps. See the cusps emerge on CPU. The physics is clear even at modest system size.

### Phase 2 — GPU Mode

80 qubits, χ=64, same 3-quench sweep. Sharper cusps, larger system, reasonable runtime. Includes exact analytic reference curves for validation.

📓 **[Interactive notebook](./dqpt.ipynb)** — step-by-step tutorial

```bash
# Phase 1: 30 qubits, CPU
python dqpt_demo.py

# Phase 2: 80 qubits, GPU
python dqpt_demo.py --large --gpu

# GPU at default size (30 qubits, faster)
python dqpt_demo.py --gpu
```

## Expected Output

**`dqpt_loschmidt.png`** — Two-panel plot:
- **Left:** Loschmidt rate function λ(t) for quench depths h_f = 0.2, 0.5, 0.8. Sharp cusps appear at periodic critical times. Dashed lines show exact analytic predictions.
- **Right:** Average transverse magnetization ⟨X(t)⟩ oscillates and decays, crossing zero near the same critical times.

Deeper quenches (smaller h_f/J) produce more pronounced cusps.

## Configuration

| Parameter | Default | Large |
|-----------|---------|-------|
| Qubits | 30 | 80 |
| Bond dim χ | 32 | 64 |
| Trotter steps | 40 | 40 |
| dt | 0.1 | 0.1 |
| J | 1.0 | 1.0 |
| h_f values | 0.2, 0.5, 0.8 | 0.2, 0.5, 0.8 |

---

👉 **Ready for large-scale quench dynamics?** [Start your free GPU trial](https://maestro.qoroquantum.net) and run with `--gpu`.
