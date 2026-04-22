# Development Timeline

## Pre-Session (Luke Barrington)
- Built v1-v3 iterations of ocean circulation simulator
- v4-physics: 2-layer barotropic vorticity equation with WebGPU compute shaders
- Wind-driven gyres, western boundary currents, ACC, temperature field
- Paleoclimate scenarios (Drake Passage, Panama, Greenland melt)
- Interactive paint tools (SimEarth-style coastline editing)
- Real coastline mask at 1° resolution

## Session: 2026-04-21 (Claude + Derek)

### Phase 0: Wiggum Loop Setup (~1hr)
- Researched Ralph Wiggum loops (AI iteration pattern)
- Built `wiggum-loop.mjs` — 3-agent dialectic (Physicist, Tuner, Validator)
- Gemini 3.1 Flash Lite for cheap parameter tuning ($0.25/1M tokens)
- 4-tier evaluation: Conservation, Structure, Sensitivity, Quantitative
- First run: completely broken (reference data NaN, param injection failed, AMOC parsing wrong)

### Phase 1: Bug Fixes (~1hr)
- Fixed SST JSON key (`sst` not `data`), deep temp key (`temp` not `data`)
- Fixed parameter injection (`eval()` for `let`-scoped vars instead of `window[key]`)
- Fixed AMOC reading (variable directly, not parsing "WEAK"/"STRONG" text)
- Fixed grid geography (western intensification was checking Indian Ocean, not Atlantic)
- Fixed AMOC units (non-dimensional ~0.001, not Sverdrups)
- Result: RMSE measurable (7.9°C), agents can now see real errors

### Phase 2: Physics Code Review (~30min)
- Claude code review identified 5 root causes:
  1. No deep buoyancy forcing (AMOC can't emerge)
  2. Ice-albedo death spiral at |lat|>40° (high latitudes crash to -10°C)
  3. Deep water formation without stratification check
  4. Poisson solver under-converged (60 Jacobi on 360×180)
  5. Missing salinity field (fundamental for AMOC)
- AMOC research (van Westen 2024, Jackson 2025, etc.) confirmed salinity is critical

### Phase 3: Structural Physics Fixes (~30min)
- Added deep buoyancy term: `+αT·∂Td/∂x` in deep vorticity equation
- Ice-albedo: threshold 40°→55°, strength 85%→50%
- Deep water formation: only when `Ts < Td` (surface denser)
- Poisson: 60→200 iterations (later optimized with SOR)
- Result: midlatitude SST within 1°C of observations (40-50°N)

### Phase 4: Salinity Implementation (~1.5hr)
- Full salinity field: surface S + deep Sd with advection, diffusion, vertical exchange
- Stacked GPU buffer layout: T at offset 0, S at offset NX*NY (zero new buffers)
- Linear equation of state: ρ = ρ₀(1 - αΔT + βΔS)
- Density-based buoyancy in both surface and deep vorticity equations
- Density-based deep water formation (cold+salty sinks, not just cold)
- Freshwater forcing correctly reduces salinity (was wrongly cooling temperature)
- Surface salinity restoring to climatology
- CPU fallback mirrors all GPU changes
- **Result: AMOC positive for first time. Freshwater collapses AMOC (Stommel bifurcation live).**

### Phase 5: Compute Optimization (~30min)
- Red-Black SOR Poisson solver replaces Jacobi (~4x faster convergence per iteration)
- Two params buffers (red/black color flag) with separate bind groups
- Iterations: 200 Jacobi → 25 SOR (2.6x fewer dispatches/frame)
- Steps/frame: 50→30
- Total: 5700 → 2220 dispatches/frame

### Phase 6: Real Data (~30min)
- ETOPO1 bathymetry via Open Topo Data API (57,600 ocean + 18,200 land cells)
- Replaces BFS distance-to-coast approximation with real seafloor topography
- Land elevation for terrain rendering
- Observed SST initialization (NOAA OI SST v2, 1991-2020)
- Observed deep temp initialization (WOA23 at 1000m)
- Sim starts from realistic Earth, not blank canvas

### Phase 7: Wind + Ice-Albedo + Scoring (~30min)
- Wind curl: asymmetric `cos(2φ)+0.8sin(φ)` → 3-belt `cos(3φ)` with SH 2x boost
- Ice-albedo: hard cutoff at 55° → gradual ramp 45-65° with latitude weighting
- T1 scoring: binary gate (20% cap) → partial credit (fraction passing × 25%)
- **Result: composite 20% → 74.1%, RMSE 5.7 → 3.8°C**

### Phase 8: UI + Rendering (~30min)
- Salinity view (brown→white→purple colormap)
- Density view (yellow→blue colormap)
- Deep Water Formation slider
- Solar Constant slider
- AMOC numeric display with sign
- Land temperature: same SST colormap, offscreen canvas cache (smooth, fast)
- Coastal heat flux clamp (fixes Agulhas/Florida hotspots)
- On-demand agents: Numerical Analyst, Skeptic, Observational Scientist, Literature

### Phase 9: Documentation + Knowledge (~1hr throughout)
- Knowledge bank: 13 files covering equations, parameters, diagnostics, models, observations, numerical methods, AMOC science, root causes, development timeline
- CLIMATE-MODELS.md: comprehensive survey of MOM6, NEMO, MITgcm, HYCOM, ROMS + emerging ML
- IMPLEMENTATION.md: full audit of architecture, assumptions, bugs, validation
- ROADMAP.md: 6-phase research plan with priorities

## Metrics Summary

| Metric | Start of Day | End of Day |
|--------|-------------|------------|
| RMSE vs NOAA | ∞ (broken) | **3.8°C** |
| Composite score | 20% (gated) | **74.1%** |
| AMOC | Negative | **Positive** |
| Freshwater response | None | **Collapses AMOC** |
| Physics fields | Temperature only | **T + S + ρ(T,S)** |
| Bathymetry | BFS fake | **Real ETOPO1** |
| Poisson solver | Jacobi 60 iter | **Red-Black SOR 25 iter** |
| Wind pattern | Asymmetric 2-term | **3-belt cos(3φ)** |
| Visualization | 8 views | **10 views** |
| Parameters | 7 sliders | **9 sliders** |
| Knowledge | None | **13 files + 3 docs** |
| Gemini cost | — | **~$0.10 total** |
| Code | 3014 lines | **3480 lines** |

## Links
- **Live:** https://amoc-sim.vercel.app/v4-physics/
- **GitHub:** https://github.com/JDerekLomas/amoc
