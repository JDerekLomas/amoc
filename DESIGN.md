# SimAMOC — Design Notes & Project Plan

*A first-principles, browser-native, interactive ocean circulation simulator.*

---

## What it is, and where it sits

SimAMOC is a real-time ocean circulation simulator that runs in the browser. It computes the Atlantic Meridional Overturning Circulation — the global conveyor belt that moves heat from the tropics to the poles — from first principles: wind stress, solar heating, density-driven flow. The goal is to reproduce observed ocean patterns (SST, salinity, currents) well enough that you can break things realistically — open the Panama seaway, melt Greenland, crank CO2 — and see physically meaningful responses.

The niche is genuinely empty. Existing projects tile around it but don't hit the same target:

| Project | First-principles? | Browser-native? | Real-time? | Interactive perturbations? |
|---|---|---|---|---|
| **SimAMOC** | yes | yes | yes | yes |
| PlaSim / ExoPlaSim | yes | no | fast batch | parameter-driven |
| Build Your Own Earth | yes (FOAM) | yes | no (cached runs) | yes (preset scenarios) |
| CLIMLAB | yes | no | varies | yes |
| Oceananigans | yes | no | research-grade | scriptable |
| Samudra / SamudrACE | no (ML emulator) | no | yes | no (training-distribution only) |
| En-ROADS, C-ROADS | no (stock-flow) | yes | yes | yes (policy sliders) |

The honest framing: **SimAMOC is a browser-native PlaSim that you can poke.** Nothing else is that.

### Current status (as of this writing)

- Wind-driven gyres work.
- Temperature patterns are recognizable (RMSE ~1.3-7C depending on version).
- ERA5 observed winds drive circulation on GPU.
- Cloud feedbacks exist but are crude.
- AMOC itself is weak.

### Known physics gaps

- **Atmospheric moisture** — the dominant tropical energy transport. Without it, you can't get tropical heat balance right.
- **Salinity forcing from P-E** — precipitation minus evaporation is the freshwater flux that drives AMOC. Without it, AMOC has no buoyancy-forcing mechanism worth the name.
- **Cloud-radiation closure** — current cloud feedbacks don't match observed OLR or shortwave reflection.
- **Tighter land boundaries** — coastlines and bathymetry need to interact properly with circulation.

The energy balance is currently tuned to produce reasonable SST but the individual fluxes don't match satellite observations. Closing the books on each flux separately is the path to a model that responds correctly to perturbations.

---

## Architecture

The architectural decision that saves the most pain on a project like this: build it as components that don't know about each other, talking through a flux coupler. This is how every serious climate model is organized — CESM has CPL7, GFDL has FMS, MITgcm has its own — and it's not bureaucracy. Different physical subsystems have radically different timescales and natural grids, and you'll want to swap implementations without breaking everything else.

**The discipline that buys you everything: the only thing components share are fluxes at their interfaces.** Ocean gives atmosphere SST and roughness; atmosphere gives ocean wind stress, heat flux, freshwater. Sea ice mediates where present. Nothing else crosses the boundary. Enforce this and "open the Panama seaway" becomes a geometry change in one place, not a refactor of three coupled modules.

```
+----------------------------------------------------------+
|                    Flux Coupler                          |
|         (exchanges + conservation checks)                |
+-----+-----------+-----------+-----------+----------------+
      |           |           |           |
   +--v---+    +--v---+    +--v---+    +--v---+
   |Atmos |    |Ocean |    | Ice  |    | Land |
   |      |    |      |    |      |    |      |
   +------+    +------+    +------+    +------+
      ^           ^           ^           ^
      |           |           |           |
   wind tau    SST, S      albedo,     albedo,
   heat Q    currents     melt/salt    ET, runoff
   freshwater P-E          reject
```

### Core principles

**State is data, physics is code.** Model state — temperatures, salinities, velocities, ice fractions — should live in flat typed arrays you can serialize and snapshot. The physics operates on that state. This makes save/load, perturbation experiments, and diagnostics nearly free.

**Geometry and forcing are configuration, not code.** Continents, bathymetry, CO2, solar constant, orbital parameters — all loadable data. "Open Panama" = swap a continent mask. "Crank CO2" = change a parameter. If any of these require code changes, the boundary is drawn wrong.

**Timescales drive the timestepping hierarchy.** Atmosphere wants minutes, surface ocean hours, deep ocean months, ice sheets decades. For interactive use, pick a coupling timestep and let each component substep internally to remain stable. Target: roughly one simulated month per real second.

**Diagnostics are separate from simulation.** AMOC strength, global mean T, ocean heat content, sea ice extent — derived from state, never stored in it. Visualization can change without touching physics.

**Spin-up and runtime are different modes.** Satellite data isn't really "initial conditions" in the strict sense. It's three different things: validation targets, quasi-static boundary fields (vegetation, fixed land albedo), and starting estimates that you'll relax via spin-up. Plan an offline spin-up that runs faster than real-time to reach equilibrium, then snapshot that state as the starting point for interactive sessions.

### Folder structure

```
src/
  state/          flat typed arrays + grid descriptors
  components/
    atmosphere/   radiation, clouds, dynamics, humidity
    ocean/        surface mixed layer + deep
    ice/          sea ice + land ice
    land/         soil moisture, snow, vegetation
  coupler/        flux exchange, timestepping, conservation
  config/
    geometry/     continents, bathymetry, river routing
    forcing/      solar, CO2, orbital
    parameters/   tunable constants per component
  initial/        satellite ingestion -> state
  diagnostics/    derived quantities (AMOC, OHC, ice extent)
  scenarios/      named perturbation experiments
  viz/            rendering, UI
  workers/        web worker glue
```

Each major component runs in a Web Worker. The coupler in the main thread orchestrates message passing. Keep typed-array transfers, not deep copies, where possible.

---

## Resolution & performance

### Recommendation

**256x128 as default, 360x180 as a "fidelity mode" for screenshots or when paused.** Skip 512x1024 — it's costing you the headroom to add the missing physics, and AMOC structure doesn't need 0.35 deg resolution to look right.

### Why these numbers

Earth is 360 deg around and 180 deg pole to pole — a **2:1 longitude:latitude ratio**. Standard climate grids reflect this: T42 is 128x64, T85 is 256x128, ERA5 is 1440x721. Avoid grids like 128x256 — that inverts the aspect ratio, gives you tall-thin cells with anisotropic numerical diffusion, and bottlenecks your timestep on the dimension that doesn't need fine resolution.

Powers-of-two are nice but not required on modern GPUs. **Divisible by 8, 16, or 32** is what actually matters for workgroup tiling. Both 256x128 and 360x180 qualify. The few-percent edge-workgroup penalty for non-POT is far smaller than the cost of running at the wrong scale.

The resolution that matters physically is the **western boundary current** (Gulf Stream, Kuroshio). At 2 deg you get ~3 cells across the Gulf Stream — visible but not resolved. At 1.4 deg (256x128), ~4-5 cells. At 1 deg (360x180), ~6 cells, which looks markedly better. Going below T42 (~2.8 deg) starts smearing the boundary currents and AMOC visualization suffers.

### Apple Silicon / WebGPU notes

The M3 is genuinely strong silicon for this — base ~10 TFLOPS, Max ~28 TFLOPS, and unified memory means no PCIe penalty for moving fields between compute and rendering. Safari 18+ does WebGPU through Metal 3 directly; Chrome via ANGLE->Metal. You're getting close-to-native compute throughput in the browser.

But raw flops is rarely the bottleneck. In order of how often they bite:

1. **Pressure / elliptic solve.** Iterative Poisson solve for SSH or incompressibility projection. Often 50%+ of frame time. Multigrid helps a lot but is annoying on GPU.
2. **Memory bandwidth, not compute.** Stencil ops on 3D grids are bandwidth-bound. Bandwidth scales much slower than flops on Apple silicon.
3. **GPU<->CPU sync.** If you read back state every frame for visualization, you murder throughput. Render directly from GPU buffers.
4. **Dispatch overhead.** Lots of small kernels = lots of dispatch cost. Fuse where you can.

### Vertical levels

A 3D ocean at 256x128x20 = 655k cells, comparable to your current 2D 512x1024. If you're at 30 levels, that's where the cost sits, not in the horizontal. Most AMOC structure is captured by ~15 well-chosen levels with thin layers near the surface and thicker ones in the abyss. The Samudra paper uses 19 levels at [2.5, 10, 22.5, 40, 65, 105, 165, 250, 375, 550, 775, 1050, 1400, 1850, 2400, 3100, 4000, 5000, 6000] m — a reasonable starting template.

---

## Roadmap

### Phase 1 — Architectural refactor

Move to component + coupler structure. State as flat typed arrays. Geometry as loadable data. Diagnostics layer separated from physics. Web Worker per component. **Drop interactive resolution to 256x128.**

This is foundation work. It doesn't improve the physics but it makes every subsequent improvement cheap.

### Phase 2 — Close the energy balance

Make individual fluxes match observations, not just the total. OLR and cloud effects against satellite data (CERES, MODIS). This is where you stop being SST-tuned and start being physically accountable.

### Phase 3 — Add the missing physics

In order of AMOC impact:

1. **P-E freshwater forcing on salinity** — biggest AMOC lever you don't have. North Atlantic freshening from runoff + precipitation is what controls deep water formation strength.
2. **Atmospheric moisture transport** — dominant tropical energy transport. Without it, the tropics can't shed heat correctly and the whole circulation is biased.
3. **Cloud-radiation closure** — match observed shortwave/longwave by region.
4. **Tighter coastlines and bathymetry** — sills (Denmark Strait, Gibraltar) matter enormously for deep water pathways.

### Phase 4 — Spin-up & snapshot infrastructure

Offline spin-up runner (faster than real-time, headless) that produces equilibrium snapshots. Interactive sessions load a snapshot. Library of snapshots for different boundary configurations (modern, Pliocene, Last Glacial Maximum, aquaplanet, hothouse).

### Phase 5 — Scenario library

Named perturbation experiments with curated parameter sets: open Panama, close Bering, melt Greenland, double CO2, weakened solar, orbital extremes. Each is a config delta on a snapshot. Now the project is genuinely playable.

---

## Resources

### Foundational reading

- **Vallis, *Atmospheric and Oceanic Fluid Dynamics* (2nd ed., 2017)** — the textbook. Chapters on AMOC, thermohaline circulation, and reduced-form models are essential.
- **Stommel (1961), "Thermohaline convection with two stable regimes of flow"** — the original two-box AMOC model. Read this before designing your AMOC diagnostic. It's six pages and explains why salinity feedback is the whole game.
- **Marshall & Schott (1999), "Open-ocean convection: Observations, theory, and models"** — deep water formation physics.
- **Kuhlbrodt et al. (2007), "On the driving processes of the Atlantic meridional overturning circulation"** — what actually sets AMOC strength.

### Most relevant codebases

- **PlaSim** — Hamburg, Fortran, intermediate complexity. Source: https://github.com/Edilbert/PLASIM. Read `puma` (dynamical core) and `plasim/src` for the slab ocean and simplified physics.
- **ExoPlaSim** — Python wrapper around PlaSim, easier to read for the high-level structure: https://github.com/alphaparrot/ExoPlaSim
- **CLIMLAB** (Brian Rose) — pedagogical hierarchy in Python. https://github.com/climlab/climlab. Particularly useful for energy balance and radiative-convective foundations.
- **Oceananigans** — Julia, GPU-accelerated, modern design. https://github.com/CliMA/Oceananigans.jl. Even if you don't read the Julia, the architectural papers (Ramadhan et al. 2020, and the 2025 Ramadhan et al. arXiv:2502.14148) are worth your time.
- **Veros** — pure-Python ocean model. https://github.com/team-ocean/veros. Useful as a reference for "how do you implement a primitive equation ocean without C/Fortran."

### ML emulators (to know what you're competing with, and to know what you can ignore)

- **Samudra** (Dheeshjith et al. 2025, arXiv:2412.03795) — ocean emulator
- **SamudrACE** (arXiv:2509.12490) — coupled ocean+atmosphere emulator

These are 150x faster than physics models but only within the training distribution. They literally cannot do "what if Panama opens." This is your moat.

### Datasets

- **ERA5** — reanalysis, atmospheric forcing. Already using it.
- **CERES** — radiation budget at TOA and surface. Essential for closing your energy balance.
- **GPCP / TRMM / IMERG** — observed precipitation. Needed for P-E validation.
- **WOA (World Ocean Atlas)** — climatological T/S. The standard for ocean state validation.
- **GEBCO** — bathymetry.
- **MODIS** — surface albedo, NDVI, snow cover, land surface T. Already in your visualization.
- **OISST** — observed SST, the validation target.
- **AVISO** — sea surface height altimetry, useful for validating surface circulation.
- **RAPID array** — observational AMOC strength at 26.5N. The single number your model has to get right.

### Browser-native climate / visualization peers

- **Build Your Own Earth** — https://www.buildyourownearth.com (Bristol/Leeds). The closest delivery analog. Look at their UX.
- **En-ROADS** — Climate Interactive's policy slider tool. Different abstraction level but similar user-experience instincts.

### WebGPU / Metal references

- WebGPU spec: https://www.w3.org/TR/webgpu/
- Metal Shading Language guide: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- Codrops, "WebGPU Fluid Simulations" (Feb 2025) — practical patterns from someone who's shipped browser fluid sims.

---

## Open questions

These are the decisions worth making explicitly rather than letting accumulate by default:

1. **Coupling timestep.** What's the master timestep, and which components substep inside it? This sets the entire performance budget and the realism ceiling.
2. **Conservation policy.** At the coupler boundary, energy and water/salt should balance. What drift do you tolerate before flagging? Real climate models error out when drift exceeds threshold.
3. **3D vs layered ocean.** Full 3D primitive equations, or a stack of well-mixed layers (~5-10) coupled by exchange? The latter is dramatically cheaper and captures AMOC structure if you're careful.
4. **Atmospheric model fidelity.** Slab atmosphere (PlaSim-style)? Two-layer (Held-Suarez)? Full primitive equations? This is the largest unsettled architectural question.
5. **What's the validation gold standard?** Pick 3-5 numbers the model has to hit for "calibrated": global mean T, AMOC strength at 26.5N, ENSO-like variability, ITCZ position, sea ice extent. Track them as CI metrics.
