# SimAMOC: System Documentation

*A real-time, AI-improving ocean circulation simulator*
*Last updated: 2026-04-25*

---

## Vision

SimAMOC is a real-time ocean circulation model that runs in the browser and continuously improves its own physics through AI agents. The core idea: instead of spending years hand-tuning climate model parameters, use AI agents to diagnose errors against observations, propose fixes, and validate them — at ~$0.03 per iteration.

The simulation solves the barotropic vorticity equation on the GPU, producing wind-driven gyres, western boundary currents (Gulf Stream, Kuroshio), the Antarctic Circumpolar Current, and thermohaline overturning circulation from first principles. Users can paint land, run paleoclimate scenarios, and trigger AMOC collapse with a slider.

**Live:** https://amoc-sim.vercel.app/simamoc/
**Repo:** https://github.com/JDerekLomas/amoc
**Leaderboard:** https://amoc-sim.vercel.app/leaderboard/

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Physics Model](#2-physics-model)
3. [Data Pipeline](#3-data-pipeline)
4. [AI Self-Improvement Loop](#4-ai-self-improvement-loop)
5. [Versioning & Competition](#5-versioning--competition)
6. [File Inventory](#6-file-inventory)
7. [Stale Documentation Audit](#7-stale-documentation-audit)
8. [Improvement Notes](#8-improvement-notes)

---

## 1. System Architecture

### 1.1 The Simulation (simamoc/)

A browser-based ocean model split into 8 JS modules sharing global scope. **5,249 lines total.**

| File | Lines | Role in the system |
|------|------:|-----|
| `model.js` | 1,985 | **Physics engine.** All state arrays, ~50 parameters, 5 WGSL compute shader strings, CPU fallback solver, data loading. Zero DOM dependencies. The brain. |
| `gpu-solver.js` | 881 | **GPU compute pipeline.** Buffer allocation, shader compilation, dispatch batching, CPU readback. Translates model.js physics into WebGPU work. |
| `renderer.js` | 1,267 | **Visualization.** 14 view modes, colormaps, GPU render pipeline, land elevation rendering, particle overlays, diagnostic charts. The eyes. |
| `main.js` | 265 | **Orchestrator.** Main loop (`gpuTick`/`cpuTick`), atmosphere sub-stepping between GPU readbacks, cloud field updates, `window.lab` API for automation. |
| `ui.js` | 127 | **Controls.** Slider bindings, 7 paint brush modes, 6 paleoclimate scenarios. |
| `overlay.js` | 70 | **Mobile UI.** Drawer layout, swipe gestures, speed presets (1x/3x/10x/MAX). |
| `index.html` | 297 | **Shell.** Pure HTML+CSS, no logic. Windy-style full-bleed layout. |
| `input-widget.js` | 357 | **Touch widget.** Mobile interaction handling. |

**Load order:** model.js -> gpu-solver.js -> renderer.js -> main.js -> ui.js -> overlay.js

**Execution paths:**

| | GPU Path (default) | CPU Fallback |
|---|---|---|
| Vorticity timestep | WGSL compute shader | JavaScript loops |
| Poisson solver | FFT (exact) via WGSL | SOR (iterative) in JS |
| Temperature/Salinity | WGSL compute shader | JavaScript loops |
| Deep layer | WGSL compute shader | JavaScript loops |
| Atmosphere | CPU (between readbacks) | CPU (every step) |
| Rendering | WebGPU fragment shader + 2D overlay | 2D canvas only |

**Current state (2026-04-25):** Running at 512x160 in CPU+FFT mode. The GPU FFT Poisson solver outputs zeros at this grid size (last commit forces CPU+FFT fallback). AMOC oscillates at 1024x512 due to Poisson/physics Laplacian consistency issues (see handoff).

### 1.2 The Lab API

`window.lab` exposes the simulation to automation scripts via Playwright:

```
lab.step(n)         — advance n steps
lab.diagnostics()   — extract SST, salinity, currents, AMOC, zonal profiles
lab.getParams()     — read all parameters
lab.setParams({})   — inject parameter changes
lab.sweep()         — parameter sweep with scoring
lab.timeSeries()    — run simulation recording metrics over time
lab.benchmark()     — performance measurement
lab.scenario(name)  — trigger paleoclimate scenario
lab.fields()        — extract raw field arrays
lab.reset()         — reinitialize from observations
lab.view(mode)      — change visualization
lab.pause/resume()  — control simulation
```

This API is the critical bridge between the browser simulation and all external tooling (Wiggum loop, tournament, tuning, testing, version submission).

### 1.3 Grid

512x160 cells at ~0.7° resolution. Latitude range -79.5° to +79.5° (excludes polar ice caps). Periodic in longitude. Metric correction: all derivatives include cos(lat) scaling. Coriolis clamped near equator (|lat| < 5°).

---

## 2. Physics Model

### 2.1 Equations

**Surface vorticity** (wind-driven circulation + buoyancy):
```
dq/dt + J(psi, q) = windCurl - r*zeta + A*laplacian(zeta) - alpha_T*dRho/dx + F_couple*(psiDeep - psi)
```

**Deep vorticity** (thermohaline-driven):
```
dqd/dt + J(psiD, qd) = -rD*zetaD + A*laplacian(zetaD) + F_coupleD*(psi - psiD) + alpha_T*dRhoDeep/dx
```

**Surface temperature:**
```
dT/dt = -J(psi,T) - Ekman*gradT + S_solar*cosZenith*iceAlbedo*cloudAlbedo
        - (A_olr + B_olr*T)*(1 - cloudGreenhouse) + kappa*laplacian(T)
        - gamma*(T-Td)/H + landFlux
```

**Salinity:**
```
dS/dt = -J(psi,S) + kappa_sal*laplacian(S) + salRestore*(S_clim - S) + freshwaterForcing
```

**Density** (linear EOS):
```
rho = rho_0 * (1 - alpha*dT + beta*dS)
```

**Atmosphere** (1-layer energy balance, CPU):
```
dT_air/dt = kappa_atm*laplacian(T_air) + gamma_oa*(T_surface - T_air)
```

### 2.2 Physical Processes (see PHYSICS_REGISTRY.md for full detail)

| Process | Status | Key parameter | Data source |
|---------|--------|--------------|-------------|
| Solar radiation | Active | S_solar=6.2 | Analytical |
| Ice-albedo | Active | onset 45°, max 50% | Ad hoc |
| Cloud SW albedo | Active | low 0.35, high 0.20 | MODIS cloud types |
| Cloud LW greenhouse | Active | low 0.03, high 0.12 | MODIS |
| OLR (linearized) | Active | A_olr=1.8, B_olr=0.13 | Budyko 1969 |
| Wind forcing | Active | ERA5 wind stress | NCEP Reanalysis |
| Ekman transport | Active | RMS-matched | Wind stress field |
| Beta effect | Active | beta=1.0 | Exact |
| Friction | Active | r=0.04 | Tuned |
| Viscosity | Active | A_visc=2e-4 | Tuned |
| Buoyancy (T+S) | Active | alpha_T=0.05, beta_S=0.8 | Thermal wind |
| Thermal diffusion | Active | kappa=2.5e-4 | Tuned |
| Salinity restoring | Active | rate=0.005 | WOA23 |
| Two-layer coupling | Active | F_couple_s=0.5, F_couple_d=0.0125 | Ad hoc |
| Variable MLD | Active | Lat-dependent profile | Approximate |
| Atmosphere | Active | kappa_atm=3e-3 | Simple diffusion |
| Land-air coupling | Active | gamma_la=0.01 | Thermal inertia |
| Moisture/latent heat | **Missing** | — | Biggest gap |
| Snow-albedo | **Missing** | — | Data ready (MODIS) |
| P-E salinity flux | **Missing** | — | Precipitation data ready |
| Evaporative cooling | **Missing** | — | Not modeled |

### 2.3 Feedback Loops

The model's behavior emerges from coupled feedbacks, not individual terms:

| Loop | Sign | Modeled? |
|------|------|----------|
| Ice-albedo | Positive | Yes |
| Cloud-SST (convective) | Negative | Yes |
| Cloud-SST (stratocumulus) | Positive | Weak |
| AMOC salt-advection | Positive | Yes |
| Vertical mixing-density | Positive | Yes |
| Temperature-OLR (Planck) | Negative | Yes |
| Atmosphere-ocean coupling | Negative | Yes |
| Water vapor greenhouse | Positive | **No** (biggest missing feedback) |
| Snow-albedo | Positive | **No** (data ready) |
| P-E to salinity | Mixed | **No** |
| Evaporative cooling | Negative | **No** |

### 2.4 What the Model Can and Cannot Do

**Can represent:** Wind-driven gyres, western boundary currents, ACC, equator-to-pole temperature gradient, seasonal cycle, AMOC and its response to freshwater forcing (Stommel bifurcation), ice-albedo feedback, cloud radiative effects, density-driven deep circulation, paleoclimate scenarios.

**Cannot represent:** Mesoscale eddies (need ~0.1°), realistic mixed layer dynamics, atmospheric moisture/latent heat, sea ice dynamics (only thermodynamic albedo), Gulf Stream separation at Cape Hatteras (need ~0.1°), Nordic Sea overflows, tidal mixing, diurnal cycle.

---

## 3. Data Pipeline

### 3.1 Observational Data Used at Runtime

The model loads 10 JSON files on page load (all at 360x160, 1° resolution):

| File | Source | Content | Used for |
|------|--------|---------|----------|
| `sst_global_1deg.json` | NOAA OI SST v2 | 1991-2020 LTM | Init + RMSE scoring |
| `deep_temp_1deg.json` | WOA23 1000m | Climatology | Deep layer init |
| `bathymetry_1deg.json` | ETOPO1 | Seafloor + land elevation | Depth field, land rendering |
| `salinity_1deg.json` | WOA23 surface | 1991-2020 | Salinity restoring target |
| `wind_stress_1deg.json` | NCEP Reanalysis | 1991-2020 LTM tau_x, tau_y | Wind curl + Ekman |
| `albedo_1deg.json` | MODIS MCD43A3 | 2020-2023 mean | Land surface albedo |
| `precipitation_1deg.json` | GPM IMERG | 2015-2023 mean | Land cloud fraction |
| `cloud_fraction_1deg.json` | MODIS MOD08_M3 | 2020-2023 mean | Cloud validation |
| `cloud_types_1deg.json` | MODIS MOD08_M3 | 2020-2023 mean | Low/high cloud validation |
| `mask.json` | Natural Earth 110m | Static | Land/ocean mask |

### 3.2 High-Resolution Data (Available, Not Yet Used)

The `data/` directory contains 20+ fields at 1024x512 resolution, fetched via `fetch-data-hires.py` from Google Earth Engine. Includes monthly climatologies for wind stress and albedo. These are ready for when the model moves to higher resolution.

### 3.3 Time Series (Downloaded, Not Used in Physics)

`earth-data/timeseries/` has 11 CSV files: RAPID AMOC (monthly + 12-month), OSNAP AMOC, CO2 (Mauna Loa), GISTEMP, HadCRUT5, HadSST4, ocean heat content, Arctic/Antarctic sea ice. Available for validation and driving scenarios.

### 3.4 Data Generation Scripts

| Script | Function |
|--------|----------|
| `fetch-data-hires.py` | GEE pipeline for 1024x512 fields (Python, uses SA key) |
| `fetch-real-earth-data.py` | GEE pipeline for 360x160 fields |
| `generate-land-data.mjs` | Generates albedo/precip JSON from lat/lon/elevation |
| `fetch-bathymetry.mjs` | ETOPO1 via Open Topo Data API |
| `scripts/fetch_godas_currents.py` | GODAS ocean current data |

---

## 4. AI Self-Improvement Loop

The system has three mechanisms for AI-driven improvement, operating at different levels:

### 4.1 Wiggum Loop (Parameter Tuning) — `wiggum-loop.mjs`

**1,379 lines.** Runs the simulation headlessly via Playwright, extracts diagnostics, sends to AI agents, injects improved parameters.

**Agent team (core, every iteration):**
- **Physicist:** Sees 4 screenshots + scorecard + zonal error profiles. Produces 2-3 ranked hypotheses.
- **Tuner:** Translates winning hypothesis to parameter changes (max 3 params, max 30% step).
- **Validator:** Checks physical consistency, catches compensating errors.

**On-demand agents (triggered by conditions):**
- **Numerical Analyst** — when T1 conservation fails
- **Skeptic** — every 4th iteration (audits for curve-fitting)
- **Observational Scientist** — when polar errors dominate
- **Literature Agent** — when params hit bounds
- **Claude** — when stalled 3 iterations (structural code review)

**Evaluation tiers:**
- **T1 Conservation** (gate): temperature range, gradient, stratification, AMOC positive
- **T2 Structure** (35%): western intensification, gyre, ACC, deep formation, heat transport, basin asymmetry
- **T3 Sensitivity** (20%): freshwater weakens AMOC, cooling cools ocean
- **T4 Quantitative** (35%): zonal-mean SST RMSE vs NOAA

**Cost:** ~$0.03 per 5-iteration run using Gemini 3.1 Flash Lite.

### 4.2 Tournament (Physics Code Mutation) — `tournament.mjs`

**286 lines.** Spawns N parallel git worktrees, each with a different physics code mutation proposed by Claude CLI. Evaluates all branches against observations, merges the winner.

This is Layer 2 of the self-improvement architecture: instead of just tuning parameters, the AI proposes structural changes to the physics code itself. Pre-defined mutation hypotheses include Ekman heat transport, realistic wind stress curl, deep overturning forcing, and ice-albedo parameterization.

### 4.3 Manual Tuning — `tune.mjs`

**220 lines.** Launches sim with given parameters, spins up, extracts diagnostics and screenshots, computes RMSE. Used for quick manual experiments.

---

## 5. Versioning & Competition

### 5.1 Version Submission — `submit-version.mjs`

Snapshots the current `simamoc/` code into `versions/<author-TIMESTAMP>/`, runs spinup + evaluation, records scores in `versions/scores.json`. Two contributors (Derek and Luke) compete on the same leaderboard.

### 5.2 Leaderboard — `leaderboard/index.html`

Public page showing all submitted versions ranked by composite score and RMSE. Tabs for Leaderboard, Gallery, Scenarios, Science Targets.

### 5.3 Score History

20 submitted versions (4 from Luke, 16 from Derek). Best achieved RMSE: **3.30°C** (360x160, with cos(lat) + Ekman + variable MLD). Composite peaked at **74.1%**.

---

## 6. File Inventory

### Core Simulation
```
simamoc/                    — Deployed simulation (5,249 lines)
  model.js                  — Physics engine (1,985 lines)
  gpu-solver.js             — WebGPU compute (881 lines)
  renderer.js               — Visualization (1,267 lines)
  main.js                   — Main loop + lab API (265 lines)
  ui.js                     — Controls (127 lines)
  overlay.js                — Mobile drawers (70 lines)
  index.html                — HTML/CSS shell (297 lines)
  input-widget.js           — Touch widget (357 lines)
  ARCHITECTURE.md           — Internal architecture doc
  mask.json                 — 360x160 land/ocean mask
  coastlines.json           — Coastline polygons
  equation.html             — Equation explainer page
```

### Automation & Testing
```
wiggum-loop.mjs             — AI parameter tuning loop (1,379 lines)
tournament.mjs              — Physics mutation tournament (286 lines)
tune.mjs                    — Manual tuning runner (220 lines)
submit-version.mjs          — Version submission (320 lines)
wiggum-capture.mjs          — Screenshot capture (196 lines)
test-amoc-collapse.mjs      — AMOC collapse test (147 lines)
test-quantitative.mjs       — Global RMSE validation (166 lines)
test-ireland-sst-compare.mjs — Gulf Stream proxy test (178 lines)
test-ireland-warmth.mjs     — Ireland warming test (216 lines)
test-convergence-local.mjs  — Solver convergence test (107 lines)
test-screenshots.mjs        — Automated screenshots (155 lines)
capture-diagnostics.mjs     — Quick diagnostics (55 lines)
debug-month.mjs             — Monthly climatology debug (37 lines)
```

### Data
```
*_1deg.json (root)          — 10 observation files at 360x160 (loaded by model)
data/                       — 20+ observation files at 1024x512 (future use)
  viewer.html               — Browser-based data inspector
earth-data/                 — Raw CSV downloads + reference PNGs
  timeseries/               — 11 time series CSVs (RAPID, CO2, SST, ice)
```

### Data Generation
```
fetch-data-hires.py         — GEE pipeline, 1024x512 (1,377 lines)
fetch-real-earth-data.py    — GEE pipeline, 360x160 (434 lines)
generate-land-data.mjs      — Albedo/precip from biome classification (338 lines)
fetch-bathymetry.mjs        — ETOPO1 bathymetry (103 lines)
scripts/fetch_godas_currents.py — GODAS current data (~160 lines)
```

### Documentation
```
SYSTEM.md                   — This file (comprehensive system doc)
PHYSICS_REGISTRY.md         — Every physical process with equations, data, status
CLIMATE-MODELS.md           — Survey of MOM6, NEMO, MITgcm, HYCOM, ROMS, ML models
IMPLEMENTATION.md           — Architecture audit (partially stale)
ROADMAP.md                  — 6-phase research plan (partially stale)
KNOWLEDGE.md                — Knowledge bank index (stale)
README.md                   — Project overview (stale)
knowledge/                  — 16 research files
simamoc/ARCHITECTURE.md     — Internal architecture (current)
```

### Other
```
v4-physics/                 — Previous monolithic version (3,849 lines, reference only)
v2/, v3-oscar/, v5-story/   — Earlier explorations
versions/                   — 20 version snapshots + leaderboard data
leaderboard/                — Public competition page
blog/                       — Blog post about building SimAMOC
issues/                     — 2 open bug reports (markdown)
reference-sst.html          — SST reference viewer
```

---

## 7. Stale Documentation Audit

### README.md — STALE

| Issue | Current state | README says |
|-------|--------------|-------------|
| Live demo URL | `/simamoc/` | `/v4-physics/` |
| Grid | 512x160 | "360x180" |
| Solver | FFT Poisson (exact) | "Jacobi iteration (200 iters)" |
| Physics | T+S+density, clouds, atmosphere, Ekman | "Temperature field with seasonal solar heating" |
| Features | 14 view modes, salinity, density, clouds, air temp | Only mentions temperature |
| Root index.html | Redirects to `/v4-physics/` | Should redirect to `/simamoc/` |

### IMPLEMENTATION.md — PARTIALLY STALE

Written 2026-04-21. Accurate for the architecture and physics concepts but many specifics are outdated:
- Line counts wrong (was 3,014 lines in single file; now 5,249 across 8 files in simamoc/)
- Grid is 512x160, not 360x180
- Poisson solver is FFT, not Jacobi/SOR
- Salinity is listed as missing but was added same day (Phase 4)
- Wiggum loop line count wrong (was 1,384; now 1,379)
- Validation history (Section 6) stops at v5 — 20 versions have been submitted since
- Cost tracking (Section 7) shows only $0.10 total — accurate but incomplete
- Section 8 file inventory misses simamoc/ entirely (file split hadn't happened yet)
- RMSE 7.3-8.1°C referenced; current best is 3.30°C

### ROADMAP.md — PARTIALLY STALE

Written 2026-04-21. Many items completed but not checked off:
- "Red-Black SOR Poisson solver" — done, then replaced with FFT
- "Salinity initialization from observations" — done (WOA23)
- "ERA5 wind stress" — done (NCEP Reanalysis wind stress loaded)
- "AMOC timeseries panel" — done
- "Observed cloud fraction view" — done
- Blockers section references old issues (NH 50°N warm bias, T1 gate) that may be resolved or changed
- "Where We Are" section says RMSE 5.7°C; best achieved is 3.30°C

### KNOWLEDGE.md — STALE

Lists 9 knowledge files but 16 exist. Many details (Poisson solver iterations, grid size, AMOC status) reflect the state from April 21, not April 25.

### knowledge/README.md — STALE

Lists 9 files ("01-equations.md" through "09-wiggum-loop.md") but the directory has 16 files (01 through 16). The "How to Use" section is still valid.

### knowledge/14-next-session-plan.md — PARTIALLY STALE

Some items completed (WOA23 salinity, wind stress, AMOC timeseries), others still relevant. The "Quick Start" port number (8780) may not match current setup.

### Root index.html — STALE

Redirects to `/v4-physics/` but the canonical simulation is at `/simamoc/`.

### simamoc/ARCHITECTURE.md — MOSTLY CURRENT

The most up-to-date internal doc. Line counts are slightly off from current (says 1,248 for model.js; actual is 1,985). Grid section says 360x160 but actual is 512x160. Heat transfer diagram and coupling rates appear accurate.

### PHYSICS_REGISTRY.md — CURRENT

The most comprehensive and recent document. Covers all processes, data, feedback loops, known gaps. One minor issue: Section 4g says "GPU FFT solver" with "High" confidence, but the GPU FFT currently outputs zeros (CPU+FFT is the working fallback).

---

## 8. Improvement Notes

### Architecture / Code

1. **Fix the GPU FFT Poisson solver.** Currently forced to CPU+FFT mode because GPU FFT outputs zeros. This is the blocking issue for performance at any grid size.

2. **Resolve the 512x160 grid lock-in.** The FFT solver only works at 512x160. The 1024x512 high-res data pipeline is built but unusable until the solver works at other sizes. Need to either fix the FFT for arbitrary dimensions or implement a fallback multigrid solver.

3. **Unify land temperature calculations.** Three inconsistent land temp calculations exist: `landTempField` (renderer, seasonal + altitude), `50*cosZenith - 20` (GPU shader), and `28 - 0.55*|lat|` (fallback). Should upload `landTempField` to GPU.

4. **GPU atmosphere.** The atmosphere runs on CPU between GPU readbacks. At high speed this means the atmosphere only updates every ~2500 GPU timesteps. Uploading `airTemp` as a GPU buffer and adding air-ocean exchange to the temperature shader would make this physically consistent.

5. **Single-file globals pattern.** All 8 JS files share browser global scope. Works but fragile — any file can accidentally shadow a variable. No module system means no import/export safety. Fine for now, would matter if the codebase grows further.

6. **eval() for parameter injection.** The Wiggum loop uses `eval()` because top-level `let` vars aren't on `window`. Works but fragile. Could switch to a params object on `window.lab`.

### Physics

7. **Moisture is the biggest gap.** No water vapor, no latent heat transport (~30% of real poleward heat transport), no evaporative cooling, no precipitation-salinity coupling. All the data exists (precipitation, evaporation fields downloaded). This is what separates "toy model" from "useful model."

8. **Snow-albedo on land.** MODIS snow cover data is downloaded but not wired. High-latitude land albedo should change seasonally from 0.15 to 0.60. Would improve seasonal cycle amplitude.

9. **Cloud model needs validation.** 7 cloud regimes implemented but NH mid-latitude clouds are too low (model ~0.30 at 50°N, observed ~0.90). The cloud RMSE diagnostic exists but hasn't driven systematic improvement.

10. **AMOC diagnostic.** Currently a point velocity difference at one latitude. Should be a zonally-integrated meridional transport across the Atlantic at multiple latitudes. The AMOC timeseries panel exists but the underlying metric is crude.

### Infrastructure

11. **No automated testing.** All test scripts exist but aren't wired to CI. A GitHub Action running `test-quantitative.mjs` on push would catch regressions.

12. **Version snapshots are full code copies.** Each version in `versions/` is a complete copy of all source files. A git-tag-based approach would save space and make diffs easier.

13. **Blog post exists but isn't linked.** `blog/2026-04-21-building-simamoc.html` documents the initial build session. Not linked from README or the simulation itself.

14. **Issues are markdown files, not GitHub Issues.** The `issues/` directory has 2 markdown files. The project memory references 12+ GitHub issues filed, but they're not cross-linked in the codebase.

### Documentation

15. **Too many overlapping docs.** README, KNOWLEDGE.md, IMPLEMENTATION.md, ROADMAP.md, PHYSICS_REGISTRY.md, simamoc/ARCHITECTURE.md, and 16 knowledge/ files cover overlapping territory. Consolidation needed — this SYSTEM.md aims to be the single source of truth, with PHYSICS_REGISTRY.md as the physics detail reference and ARCHITECTURE.md as the code structure reference.

16. **Root index.html should redirect to /simamoc/.** Currently points to /v4-physics/ which is the old monolithic version.

---

## Development History

Built by **Luke Barrington** (original v1-v4 physics, GPU compute, coastlines, paint tools) and **Derek Lomas** (salinity, AI loop, data pipeline, clouds, atmosphere, documentation).

### Key milestones

| Date | Event | RMSE |
|------|-------|------|
| Pre-Apr 21 | Luke builds v1-v4: barotropic vorticity, WebGPU, gyres, WBCs | -- |
| Apr 21 | Wiggum loop, salinity, ETOPO1 bathymetry, SOR solver | 3.8°C |
| Apr 22 | Model/UI split into simamoc/, land temperature | 3.8°C |
| Apr 23 | cos(lat) correction, FFT Poisson, Ekman transport, variable MLD | 3.3°C |
| Apr 24 | Cloud model (7 regimes), atmosphere layer, wind stress from ERA5 | ~7.2°C* |
| Apr 25 | Physics registry, interaction map, data pipeline expansion | ~7.2°C* |

*RMSE increased when cloud albedo was added without retuning other parameters. The no-cloud leaderboard entry (5.60°C) was tuned for the old physics. Full retuning with clouds is the next step.

### RMSE progression
```
infinity -> 7.9 -> 5.7 -> 3.8 -> 3.3 -> ~7.2 (cloud model added, needs retuning)
```

---

## Quick Start

```bash
cd /Users/dereklomas/lukebarrington/amoc
python3 -m http.server 8765 &
open http://localhost:8765/simamoc/
```

Run tests:
```bash
node test-quantitative.mjs        # RMSE + basin validation
node test-amoc-collapse.mjs       # freshwater hosing test
node wiggum-loop.mjs --iters 5    # AI tuning loop
node tournament.mjs               # physics mutation tournament
node submit-version.mjs           # snapshot + score current version
```
