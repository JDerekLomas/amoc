# Emerging ML Approaches

## Differentiable Ocean Models

### Oceananigans.jl (CliMA, Caltech)
- Julia-based GPU ocean model, designed for differentiability
- Global ocean at 488m on 768 A100s; 9.9 sim-years/day at 10km on 68 A100s
- Automatic differentiation via Enzyme.jl
- Hydrostatic + non-hydrostatic
- [GitHub](https://github.com/CliMA/Oceananigans.jl) | [Paper](https://arxiv.org/abs/2309.06662)

### NeuralOGCM (Dec 2025)
- First hybrid ocean GCM: differentiable physics + deep learning
- Learnable parameters + neural net for subgrid correction
- End-to-end gradient-based optimization
- [arXiv 2512.11525](https://arxiv.org/abs/2512.11525)

### NeuralGCM (Google/DeepMind, 2024)
- Differentiable atmospheric GCM with learned physics
- Competitive with ECMWF ensemble at 1-15 day range
- Atmosphere-only (ocean coupling = key next step)
- [Nature 2024](https://www.nature.com/articles/s41586-024-07744-y)

## ML Emulators (fast surrogates)

### Samudra (2025)
- ML emulator of GFDL OM4 using modified ConvNeXt UNet
- Predicts SST, SSH, T, S across full depth
- **150x speedup** on single A100 (~100 sim-years in 1.3 hours)
- Stable for multi-century simulations
- [GRL paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL114318)

### SamudrACE (2025)
- Couples Samudra ocean + ACE atmosphere → fully ML coupled climate
- Captures ENSO dynamics
- [arXiv 2509.12490](https://arxiv.org/html/2509.12490v1)

### ACE2-SOM (2025)
- ML atmosphere + slab ocean model
- Reproduces equilibrium climate sensitivity at 25x less cost
- [Clark et al. 2025](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JH000575)

## Physics-Informed Neural Operators (PINOs)

- Learn PDE solution maps with physics constraints in loss
- Train on coarse data, enforce PDE at higher resolution
- [Original PINO paper](https://arxiv.org/abs/2111.03794)

### FourCastNet (NVIDIA)
- Adaptive Fourier Neural Operators for global weather at 0.25 deg
- Primarily atmospheric, architecture applicable to ocean
- [ResearchGate](https://www.researchgate.net/publication/362153109)

## Relevance to Our Project

Our Wiggum loop is a lightweight version of the "learn-to-correct" paradigm:
- **Samudra**: trains a neural net to emulate the full GCM → we use Gemini to tune a simplified sim
- **NeuralOGCM**: differentiable solver + neural correction → we use physics sim + AI parameter tuning
- **ECCO adjoint**: adjusts parameters to fit observations → our loop does the same but with LLM reasoning instead of gradients

Key difference: we're using LLM reasoning (hypothesize → test → validate) instead of gradient-based optimization. Slower per iteration but can propose structural changes, not just parameter tweaks.

## Sources
- [Nature 2026 — Rewiring climate modeling with ML](https://www.nature.com/articles/s43247-026-03238-z)
