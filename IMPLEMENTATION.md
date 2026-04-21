# AMOC Simulation — Implementation Audit

*Full documentation of architecture, assumptions, known issues, and validation status*
*Generated 2026-04-21 via Claude code review + Wiggum loop findings*

---

## 1. System Architecture

### 1.1 Simulation (`v4-physics/index.html`, 3014 lines)

Single-file HTML/JS/WGSL application. Two execution paths:

| Component | GPU Path | CPU Fallback |
|-----------|----------|-------------|
| Vorticity timestep | WGSL compute shader | JavaScript loops |
| Poisson solver | Jacobi iteration (200 iters) | SOR (100 iters) |
| Temperature | WGSL compute shader | JavaScript loops |
| Deep layer | WGSL compute shader | JavaScript loops |
| Rendering | WebGPU fragment shader + 2D canvas overlay | 2D canvas only |

### 1.2 Wiggum Loop (`wiggum-loop.mjs`, 1384 lines)

Node.js harness using Playwright to drive the simulation headlessly:

```
Puppeteer (WebGPU) → Sim spinup → Data extraction → Evaluation → Gemini agents → Parameter injection → Repeat
```

**Model:** Gemini 3.1 Flash Lite ($0.25/1M in, $1.50/1M out)
**Typical cost:** ~$0.03 per 5-iteration run
**Budget cap:** Configurable, default $50

### 1.3 Knowledge Bank (`knowledge/`, 11 files)

Organized reference covering equations, parameters, diagnostics, ocean model survey, observations, numerical methods, AMOC science, and lessons learned.

---

## 2. Physics Model

### 2.1 Equations Solved

**Surface vorticity:**
```
dq/dt + J(ψ, q) = curl(τ) − r·ζ + A·∇²ζ − αT·∂T/∂x + F₁(ψd − ψ)
```

**Deep vorticity:**
```
dqd/dt + J(ψd, qd) = −rd·ζd + A·∇²ζd + F₂(ψ − ψd) + αT·∂Td/∂x
```

**Surface temperature:**
```
dT/dt = −J(ψ,T) + S·cos(θz)·αice − (A_olr + B_olr·T) + κ·∇²T − γ(Ts−Td)/H + Q_land
```

**Deep temperature:**
```
dTd/dt = γ(Ts − Td)/Hd + κd·∇²Td
```

### 2.2 Key Assumptions

| Assumption | Physical Basis | Limitation |
|-----------|---------------|------------|
| **Barotropic vorticity equation** | Filters gravity waves, retains geostrophic dynamics | Cannot represent free surface, barotropic tides |
| **Two vertical layers** (100m + 4000m) | Captures first baroclinic mode | No thermocline resolution, no mixed layer depth |
| **No salinity** | Temperature-only density | Cannot represent salt-advection feedback (Stommel bifurcation), haline contribution to AMOC |
| **Linearized OLR** (A + B·T) | Standard energy balance model (Budyko 1969) | No cloud feedback, no water vapor feedback detail |
| **Prescribed wind pattern** | Analytical trades + westerlies | No atmospheric coupling, no NAO/ENSO variability |
| **Ice-albedo at |lat|>55°** | Simplified polar ice feedback | No sea ice dynamics, no ice transport, no seasonal ice edge |
| **Arakawa Jacobian** | Conserves energy + enstrophy | Correct; prevents spectral cascade |
| **Jacobi Poisson solver** | Simple parallel iteration | Slow convergence; 200 iters marginal for 360×180 |
| **Forward Euler time stepping** | Explicit, first-order | Limits timestep; no implicit terms |
| **1° resolution (360×180)** | Standard coarse climate model | Cannot resolve mesoscale eddies, Gulf Stream separation |
| **Freshwater forcing = cooling** | Proxy for density reduction | Physically backwards: freshwater should lighten, not cool |
| **Deep buoyancy: +αT·∂Td/∂x** | Density-driven deep circulation | Added 2026-04-21; sign may need verification |
| **Deep water formation: γ when Ts<Td** | Convective sinking when surface denser | Stratification check added 2026-04-21 |

### 2.3 What the Model CAN Represent

1. Wind-driven subtropical and subpolar gyres (Sverdrup balance)
2. Western boundary current intensification (Stommel/Munk dynamics)
3. Antarctic Circumpolar Current
4. Equator-to-pole temperature gradient with seasonal cycle
5. Two-layer baroclinic coupling
6. Ice-albedo feedback (simplified)
7. Paleoclimate scenarios (Drake Passage, Panama, Greenland melt — qualitative)
8. Response to radiative forcing changes

### 2.4 What the Model CANNOT Represent

1. **Salt-advection feedback** — no salinity field (THE critical AMOC mechanism)
2. **Stommel bifurcation / AMOC tipping** — requires salinity
3. **Mesoscale eddies** — unresolved at 1° (parameterized via viscosity)
4. **Nordic Sea overflows** — no sill hydraulics or entrainment
5. **Isopycnal upwelling** — Southern Ocean AMOC closure mechanism
6. **Realistic mixed layer depth** — fixed 100m surface layer
7. **Atmospheric coupling** — fixed wind pattern
8. **Sea ice dynamics** — only thermodynamic albedo effect
9. **Gulf Stream separation at Cape Hatteras** — needs ~1/10° resolution

---

## 3. Numerical Implementation

### 3.1 Grid

```
360 × 180 cells, 1° resolution
LON: -180° to +180°  (i=0 → dateline, i=180 → prime meridian)
LAT: -80° to +80°    (j=0 → Antarctic, j=179 → Arctic)
```

Conversion: `i = (lon + 180) / 360 × 359`, `j = (lat + 80) / 160 × 179`

### 3.2 Time Stepping

- dt = 5×10⁻⁵ (non-dimensional)
- Steps per frame: 50 (configurable 10-200)
- All terms explicit (Forward Euler)
- CFL constraint: max_vel × dt × nx < 1 → max_vel < 55.6

### 3.3 Poisson Solver

| Path | Method | Iterations | Convergence |
|------|--------|-----------|-------------|
| GPU surface | Jacobi | 200 | Marginal — spectral radius ~0.9999 |
| GPU deep | Jacobi | 60 | Under-converged for strong forcing |
| CPU surface | SOR (optimal ω) | 100 | Better than Jacobi but still limited |
| CPU deep | SOR | 60 | Adequate for weak deep forcing |

**Known issue:** Jacobi on 360×180 needs O(N) = O(360) iterations for full convergence. 200 is better than 60 but still leaves residual noise in basin-scale modes, affecting western boundary current structure.

### 3.4 Clamping

| Field | Min | Max | Purpose |
|-------|-----|-----|---------|
| Surface temp | -10°C | 40°C | Prevent runaway |
| Deep temp | -5°C | 30°C | Prevent runaway |
| Vorticity | -500 | 500 | Numerical stability |

### 3.5 Stability Features

- NaN detection and field reset
- Coastal damping (vorticity × 0.9 at land-adjacent cells)
- GPU readback every 5 frames for monitoring (not every frame)
- Velocity clamping for CFL check

---

## 4. Wiggum Loop Architecture

### 4.1 Agent Team

**Core (every iteration):**

| Agent | Input | Output | Model |
|-------|-------|--------|-------|
| Physicist | 4 screenshots + scorecard + zonal errors + history | 2-3 ranked hypotheses | Gemini 3.1 Flash Lite |
| Tuner | Physicist's recommendation + params + bounds | Parameter changes (max 3, max 30% step) | Gemini 3.1 Flash Lite |
| Validator | Tuner's proposal + Physicist's reasoning | APPROVE / MODIFY / REJECT | Gemini 3.1 Flash Lite |

**On-demand (triggered by conditions):**

| Agent | Trigger | Purpose |
|-------|---------|---------|
| Numerical Analyst | T1 conservation fails | Check for computational artifacts |
| Skeptic | Every 4th iteration | Audit for curve-fitting |
| Observational Scientist | Worst errors at polar latitudes | Assess reference data quality |
| Literature Agent | Any parameter hits bound | Check published model values |
| Claude | Stalled 3 iterations | Structural code review |

### 4.2 Evaluation Tiers

**T1: Conservation (GATE — must pass, else score capped at 20%)**

| Check | Threshold | Status |
|-------|-----------|--------|
| Temperature range | [-15, 40]°C | FAILING (hits -10 clamp) |
| Equator-pole gradient | 10-50°C | PASSING |
| Hemispheric symmetry | diff < 8°C | PASSING |
| Deep < surface | 0 bands inverted | FAILING (high-lat inversion) |
| AMOC positive | > 1e-5 (non-dim) | FAILING (slightly negative) |

**T2: Structure (35% of score, 7 checks)**

| Check | Method | Status |
|-------|--------|--------|
| Western intensification | Gulf Stream region (-80 to -60°, 25-50°N) vs NA interior speed | FAILING |
| Subtropical gyre exists | ψ range in NA subtropics | PASSING |
| ACC eastward flow | Mean zonal velocity at 58°S | PASSING |
| Deep water formation | Polar deep < tropical deep | PASSING |
| Poleward heat transport | Mean v×T at 30°N > 0 | PASSING (post-fix) |
| Atlantic > Pacific at 40°N | Basin mean comparison | FAILING |

**T3: Sensitivity (20% of final, run once on best params)**

| Test | Perturbation | Expected | Status |
|------|-------------|----------|--------|
| Freshwater | freshwaterForcing = 2.0 | AMOC weakens | PASSING (barely) |
| Ice age | globalTempOffset = -8 | Cooling | PASSING |

**T4: Quantitative (35% of score)**

| Metric | Current | Target |
|--------|---------|--------|
| RMSE vs NOAA SST | 7.3-8.1°C | < 3°C for "good" |
| Global mean SST | ~17°C | 14.2°C (obs) |

### 4.3 Known Wiggum Loop Issues

1. **Parameter injection uses `eval()`** — needed because top-level `let` vars aren't on `window`. Works but fragile.
2. **AMOC units are non-dimensional** — agents told this but sometimes propose Sv-scale targets anyway.
3. **Gemini fixates on A_visc** — agents repeatedly propose viscosity changes; may need prompt engineering to force diversity.
4. **Numerical Analyst triggers too often** — flags "CRITICAL" every iteration because Poisson is structurally under-converged. Should distinguish "always-present limitation" from "new problem."
5. **T1 gate is too strict** — one failing check blocks all score improvement, even when T2/T4 improve dramatically.
6. **90s spinup insufficient for deep ocean** — deep layer has 40× thermal inertia; needs 180s+ for equilibration.
7. **Screenshots sent only to Physicist** — Tuner and Validator reason from text only.

---

## 5. Data Pipeline

### 5.1 Reference Data

| File | Key | Grid | Source | Reliability |
|------|-----|------|--------|-------------|
| `sst_global_1deg.json` | `sst` | 360×160 | NOAA OI SST v2 (1991-2020) | HIGH tropics/midlat, LOW polar |
| `deep_temp_1deg.json` | `temp` | 360×160 | WOA23 annual mean 1000m | MEDIUM, sparse in S. Ocean |
| `mask.json` | hex-packed | 360×180 | Land/ocean mask | — |
| `coastlines.json` | GeoJSON | — | Polygon outlines | — |

### 5.2 Data Extraction from Simulation

Zonal means extracted via `page.evaluate()` accessing top-level `let` variables (`temp`, `mask`, `deepTemp`, `psi`, `amocStrength`).

**Known issue:** Variables are `let`-scoped (not on `window`), so parameter injection requires `eval()`. Data extraction works because `page.evaluate` runs in page context where `let` vars are accessible.

---

## 6. Validation History

### 6.1 Run Results

| Run | Date | Iters | Spinup | RMSE | Composite | Key Finding |
|-----|------|-------|--------|------|-----------|-------------|
| v1 (broken) | 2026-04-21 | 10 | 120s | 0.0 | 20% | Reference data NaN, param injection failed |
| v2 (fixed data) | 2026-04-21 | 5 | 120s | 7.9 | 36% | Data works, AMOC still negative |
| v3 (fixed geography) | 2026-04-21 | 5 | 60s | 8.7→7.3 | 20% | Correct basin checks, T2 improved to 67% |
| v4 (physics fixes) | 2026-04-21 | 5 | 90s | 7.3 | 36% | Ice-albedo fix: 40-50°N within 1°C of obs |
| v5 (longer spinup) | 2026-04-21 | 3 | 180s | — | — | Running |

### 6.2 Bugs Found and Fixed

| Bug | Found By | Fix | Impact |
|-----|----------|-----|--------|
| SST JSON key is `sst` not `data` | Manual debug | Fixed loader | Reference data was all NaN |
| Deep temp key is `temp` not `data` | Manual debug | Fixed loader | Deep reference was NaN |
| `window[key]` doesn't work for `let` vars | Manual debug | Use `eval()` | Parameter injection silently failed |
| AMOC display shows "WEAK"/"STRONG" text | Manual debug | Read `amocStrength` variable directly | parseFloat("WEAK") = NaN |
| Grid indices for "western Atlantic" pointed to Indian Ocean | Scale audit | Use `lonToI()`/`latToJ()` | T2 western intensification checked wrong basin |
| AMOC compared to "15-20 Sv" but is non-dimensional | Scale audit | Use non-dim thresholds | AMOC check could never pass |
| Gyre threshold `|ψ|>0.1` too high for actual ψ scale | Scale audit | Use ψ range > 0.005 | Gyre check borderline |
| No deep buoyancy forcing in deep vorticity equation | Claude code review | Added `+αT·∂Td/∂x` term | AMOC has no physical driver |
| Ice-albedo death spiral at |lat|>40° | Claude code review | Raised to 55°, reduced strength | High-lat temps crashed to -10°C |
| Deep water formation without stratification check | Claude code review | Added `Ts < Td` condition | Created inverted stratification |
| Poisson solver under-converged (60 Jacobi iters) | Claude code review | Increased to 200 | Noisy streamfunction |

### 6.3 Outstanding Issues

| Issue | Severity | Root Cause | Potential Fix |
|-------|----------|-----------|---------------|
| AMOC slightly negative | HIGH | Deep buoyancy sign may need tuning; insufficient spinup time | Longer spinup; verify sign convention |
| Deep warmer than surface at 60-70° | HIGH | Residual ice-albedo issues; deep water formation threshold | Fine-tune ice-albedo; adjust γ_deep_form |
| Southern Hemisphere too warm (+12°C at 60°S) | MEDIUM | Ice-albedo fix may have overcorrected SH | Asymmetric ice-albedo or SH-specific tuning |
| Western intensification fails | MEDIUM | Poisson still under-converged; viscosity too high | Multigrid solver; reduce A_visc |
| Atlantic not warmer than Pacific at 40°N | MEDIUM | AMOC not functioning → no extra Atlantic heat | Depends on AMOC fix |
| No salinity field | FUNDAMENTAL | Design limitation | Add S(x,y,t) field, linear EOS ρ(T,S) |
| Freshwater slider cools instead of freshens | FUNDAMENTAL | No salinity | Requires salinity implementation |
| Composite stuck at 20% due to T1 gate | EVALUATION | T1 failures from AMOC/stratification | Fix AMOC; or relax T1 gate slightly |

---

## 7. Cost Tracking

| Run | Gemini Calls | Input Tokens | Output Tokens | Total Cost |
|-----|-------------|-------------|---------------|------------|
| v1 (10 iters) | ~20 | ~150K | ~15K | $0.017 |
| v2 (5 iters) | ~15 | ~120K | ~12K | $0.027 |
| v3 (5 iters) | ~18 | ~130K | ~13K | $0.026 |
| v4 (5 iters) | ~18 | ~130K | ~13K | $0.027 |
| **Total** | **~71** | **~530K** | **~53K** | **~$0.10** |

At current rates, we could run ~500 iterations for $1. The budget cap of $50 allows thousands of iterations.

---

## 8. File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `v4-physics/index.html` | 3014 | Simulation (physics, rendering, UI) |
| `wiggum-loop.mjs` | 1384 | Ralph Wiggum parameter tuning loop |
| `sst_global_1deg.json` | 375KB | NOAA SST reference data |
| `deep_temp_1deg.json` | 400KB | WOA23 deep temperature reference |
| `v4-physics/mask.json` | 16KB | Land/ocean mask |
| `v4-physics/coastlines.json` | 44KB | Coastline polygons |
| `KNOWLEDGE.md` | — | Original flat knowledge file |
| `CLIMATE-MODELS.md` | 400+ | Comprehensive ocean model survey |
| `IMPLEMENTATION.md` | this file | Implementation audit |
| `knowledge/` | 11 files | Organized knowledge bank |
| `screenshots/wiggum/` | ~50 files | Run outputs, screenshots, results |

---

## 9. Architectural Decisions Log

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| Single HTML file for sim | Zero build step, instant deployment | Hard to maintain at 3000+ lines |
| Gemini 3.1 Flash Lite for agents | $0.25/1M in — extremely cheap | Less capable than Sonnet/Opus for physics reasoning |
| 4-tier evaluation | Prevents curve-fitting by requiring structural + conservation checks | T1 gate may be too strict |
| 3 core + 4 on-demand agents | Keeps iteration cost low while having specialists available | On-demand triggers may fire too often |
| eval() for parameter injection | Only way to set `let`-scoped vars from Playwright | Security concern (acceptable for local dev) |
| Jacobi over SOR on GPU | Jacobi is trivially parallelizable on GPU | Converges much slower; 200 iters still marginal |
| Screenshots to Physicist only | Reduces token cost (images are expensive) | Tuner/Validator miss visual context |
| Non-dimensional units | Matches simulation internals | Hard to compare with real-world observations |

---

## 10. Recommendations

### Immediate (fix current issues)
1. Verify deep buoyancy sign convention — run with opposite sign to check
2. Increase spinup to 180-300s for deep equilibration
3. Consider relaxing T1 gate to allow partial credit (not all-or-nothing)
4. Fine-tune ice-albedo for Southern Hemisphere

### Short-term (next sprint)
5. Add salinity field — enables real thermohaline circulation
6. Replace Jacobi with red-black SOR on GPU — much faster convergence
7. Wire knowledge bank context into agent prompts
8. Add AMOC diagnostic as meridional transport integral, not point velocity

### Medium-term (next milestone)
9. Implement GM-like eddy parameterization for 1° resolution
10. Add Nordic Sea overflow parameterization
11. Consider higher resolution (720×360) with GPU compute
12. Validate against ECCO v4 state estimate

### Long-term (vision)
13. Full salinity + equation of state → Stommel bifurcation
14. Differentiable physics → gradient-based optimization instead of LLM tuning
15. Couple with atmospheric energy balance model
16. ML emulator trained on this simulation for real-time prediction
