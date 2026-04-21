# AMOC Simulation Research Roadmap

*Updated 2026-04-21 after salinity implementation + Wiggum loop results*

## Where We Are

A WebGPU 2-layer ocean simulation with:
- Wind-driven gyres, western boundary currents, ACC (correct physics)
- Salinity field with density EOS ρ(T,S) (just added)
- Working AMOC that responds to freshwater forcing (Stommel bifurcation live)
- RMSE 5.7°C vs NOAA observations (down from ∞)
- Real ETOPO1 bathymetry + land elevation
- Observed SST/deep temp initialization
- Red-Black SOR Poisson solver
- 7-agent Wiggum loop with 4-tier evaluation
- Total cost: ~$0.10 in Gemini API calls

## Current Blockers (fix before moving forward)

### 1. Western Boundary Intensification (T2 fail)
**Problem:** Gulf Stream region not significantly faster than NA interior.
**Root cause candidates:**
- SOR at 25 iterations still under-converged for basin-scale modes
- Wind curl pattern may be too weak or wrong spatial structure
- A_visc (viscosity) too high for 1° grid
**Approach:** Compare our wind curl `F = -cos(2φ) + 0.8sin(φ)` against Hellerman-Rosenstein climatology. Try multigrid or more SOR iterations. Reduce A_visc to 1e-4.

### 2. NH 50°N Warm Bias (+14°C)
**Problem:** 50°N shows 23°C, should be 9°C.
**Root cause candidates:**
- Ice-albedo threshold at 55° leaves 40-55° with no ice feedback
- Thermal diffusion too weak to transport heat poleward efficiently
- AMOC may be transporting heat the wrong direction at some latitudes
**Approach:** Add a weaker ice-albedo at 45-55° (gradual onset), check heat transport direction by latitude.

### 3. T1 Gate Too Strict
**Problem:** Composite stuck at 20-36% because one T1 failure caps score.
**Approach:** Allow partial credit in T1 (fraction passing × weight) instead of binary gate.

---

## Phase 1: Polish Current Physics (1-2 days)

- [ ] Fix wind curl pattern — compare to Hellerman-Rosenstein, adjust amplitude/shape
- [ ] Tune ice-albedo — gradual onset from 45° with latitude-dependent strength
- [ ] Increase SOR iterations to 40 (test frame rate impact)
- [ ] Relax T1 gate to partial credit
- [ ] Add salinity visualization mode (button + colormap)
- [ ] Run Wiggum loop with 10 iterations, 180s spinup → target RMSE < 4°C
- [ ] Validate freshwater scenarios against van Westen et al. 2024 qualitatively

## Phase 2: Improve AMOC Realism (1 week)

- [ ] **AMOC diagnostic improvement** — compute as zonal transport integral, not point velocity
- [ ] **Nordic overflow parameterization** — enhanced deep water formation near Greenland-Scotland Ridge
- [ ] **Salinity initialization from observations** — fetch WOA23 salinity at surface + 1000m
- [ ] **Seasonal salinity cycle** — precipitation/evaporation forcing
- [ ] **Surface salinity restoring** — tune restoring timescale against climatology
- [ ] **FovS diagnostic** — compute the freshwater transport indicator from van Westen et al.
- [ ] Run hosing experiments: gradually increase freshwater, find tipping point
- [ ] Compare AMOC collapse dynamics to Stommel box model predictions

## Phase 3: Numerical Optimization (1 week)

- [ ] **Multigrid Poisson solver** — replace SOR with V-cycle multigrid on GPU (10x convergence)
- [ ] **Resolution doubling** — 720×360 (0.5°) with GPU compute
- [ ] **Adaptive time stepping** — CFL-based dt adjustment
- [ ] **Biharmonic viscosity** — more scale-selective than Laplacian
- [ ] **Smagorinsky viscosity** — flow-dependent, auto-adjusting
- [ ] Profile GPU pipeline — identify actual bottleneck (Poisson vs tracer vs vorticity)
- [ ] Consider compute shader workgroup size optimization (8×8 → 16×16?)

## Phase 4: Scientific Validation (2 weeks)

- [ ] **Systematic comparison to NOAA/ERA5** — basin-by-basin SST, not just zonal means
- [ ] **AMOC strength calibration** — convert non-dimensional to pseudo-Sverdrups, compare to RAPID
- [ ] **Heat transport validation** — meridional heat transport vs observed ~1.3 PW at 25°N
- [ ] **Gyre transport validation** — Sverdrup transport integral vs theory
- [ ] **Seasonal cycle validation** — does the sim produce realistic seasonal SST amplitude?
- [ ] **Paleoclimate scenarios** — Drake Passage, Panama effects vs published literature
- [ ] **Sensitivity matrix** — systematic parameter perturbation, document response
- [ ] Write up methodology for peer review / blog post

## Phase 5: Advanced Physics (1 month)

- [ ] **GM-like eddy parameterization** — thickness diffusion to flatten isopycnals
- [ ] **Mixed layer depth** — variable H_surface based on buoyancy forcing
- [ ] **Wind from ERA5** — replace analytical with observed wind stress
- [ ] **Atmospheric energy balance model** — simple 1D atmosphere coupled to ocean
- [ ] **Sea ice model** — thermodynamic ice growth/melt, albedo feedback, ice transport
- [ ] **River runoff** — Amazon, Congo, Mississippi freshwater sources

## Phase 6: ML Integration (ongoing)

- [ ] **Wire knowledge bank into Wiggum agent prompts** — ground agents in published parameter ranges
- [ ] **Train a fast emulator** — neural net trained on sim output for instant predictions
- [ ] **Differentiable physics** — gradient-based parameter optimization via autodiff
- [ ] **Compare to Samudra/NeuralGCM** — how does our simple model compare to ML emulators?
- [ ] **Real-time prediction mode** — use emulator for instant scenario exploration

## Aspirational Goals

- RMSE < 2°C (requires higher resolution + better physics)
- Reproduce AMOC tipping with realistic freshwater forcing timeline
- Live coupling with simple atmospheric model
- Used in educational settings (classroom, museum)
- Published as a methods paper on AI-assisted ocean model development

## Resources

- **Knowledge bank**: `knowledge/` (11 files covering equations, parameters, diagnostics, models, observations)
- **Implementation audit**: `IMPLEMENTATION.md`
- **Climate model survey**: `CLIMATE-MODELS.md`
- **Wiggum loop**: `wiggum-loop.mjs` (1384 lines, 7 agents)
- **Reference data**: NOAA SST, WOA23 deep temp, ETOPO1 bathymetry

## Key Papers to Guide Next Steps

- [van Westen et al. 2024 — AMOC tipping](https://www.science.org/doi/10.1126/sciadv.adk1189) — our freshwater scenarios should reproduce their qualitative findings
- [Jackson et al. 2025 — AMOC never fully collapses](https://www.nature.com/articles/s41586-024-08544-0) — Southern Ocean upwelling floor
- [Portmann et al. 2026 — 51% AMOC weakening by 2100](https://www.science.org/doi/10.1126/sciadv.adx4298) — observational constraint
- [Adcroft et al. 2019 — MOM6/OM4](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001726) — parameter reference
- [Gent 2011 — GM parameterization](https://staff.cgd.ucar.edu/gent/gm20.pdf) — eddy effects
