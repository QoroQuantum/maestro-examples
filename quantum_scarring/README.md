# Quantum Many-Body Scarring

> 🚀 **Try Maestro GPU mode with a free trial.**
> Sign up at **[maestro.qoroquantum.net](https://maestro.qoroquantum.net)** — no credit card required.

## Why GPU Mode?

Scarring simulations need **many Trotter steps** to capture the full revival structure — 60+ steps at moderate-to-high bond dimension. A 64-atom chain at χ=64 means **hours on CPU**. GPU mode handles the O(χ³) contractions efficiently, cutting runtime to a fraction — you see the full revival curve without waiting overnight.

## What It Does

Simulates **quantum many-body scarring** in a 1D Rydberg atom chain. Starting from the Néel state |01010…⟩, most quantum systems would thermalize — the staggered magnetization M(t) would decay to zero. But the PXP model has **quantum scars**: special non-thermal eigenstates that cause M(t) to **oscillate persistently**, violating the eigenstate thermalization hypothesis.

**Key observable:** Staggered magnetization M(t) = (1/N) Σᵢ (−1)ⁱ ⟨Zᵢ(t)⟩ — should oscillate instead of decaying.

### Phase 1 — Local (CPU)

32 atoms, χ=32, 60 Trotter steps. See the revival oscillations emerge on CPU. The first few revivals are clear; runtime is manageable.

### Phase 2 — GPU Mode

64 atoms, χ=64, 60 Trotter steps. Capture the full revival structure at higher accuracy on a larger system — GPU handles the heavy tensor contractions.

📓 **[Interactive notebook](./quantum_scarring.ipynb)** — step-by-step tutorial

```bash
# Phase 1: 32 atoms, CPU
python scarring_demo.py

# Phase 2: 64 atoms, GPU
python scarring_demo.py --large --gpu

# Validation: 12 atoms + exact ED reference
python scarring_demo.py --small
```

## Expected Output

**`quantum_scarring.png`** — Staggered magnetization M(t) showing periodic revivals instead of monotonic decay. If running `--small`, includes exact fidelity F(t) = |⟨Z₂|ψ(t)⟩|² from exact diagonalization.

The revival period is T_rev ≈ 4.5 (in units of 1/Ω), **independent of system size** — a hallmark of scarring.

## Configuration

| Parameter | Default | Large |
|-----------|---------|-------|
| Atoms | 32 | 64 |
| Bond dim χ | 32 | 64 |
| Trotter steps | 60 | 60 |
| dt | 0.2 | 0.2 |
| V/Ω | 20 | 20 |

---

👉 **Ready for large-scale scarring simulations?** [Start your free GPU trial](https://maestro.qoroquantum.net) and run with `--gpu`.
