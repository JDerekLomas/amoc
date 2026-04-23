# SimAMOC Architecture

## Model/UI Separation (Phases 1-3 Complete)

The simulation was split from a single 4,187-line `index.html` into separate layers.

### File Structure

```
simamoc/
  model.js          ~1,185 lines  Physics engine (no DOM dependencies)
  gpu-solver.js       ~622 lines  WebGPU compute pipelines, buffers, dispatch
  renderer.js       ~1,042 lines  Canvas rendering, colormaps, GPU render pipeline
  index.html        ~1,287 lines  Main loop, desktop UI, mobile UI, lab API
  input-widget.js                 Touch interaction widget
  mask.json                       360x180 land/ocean mask (hex-encoded)
  coastlines.json                 Coastline polygon data
  equation.html                   Equation guide page
```

### model.js — Physics Core

Zero DOM dependencies. Declares all simulation state as globals, loaded via `<script src="model.js">` before the main script block.

**Contents:**
- **Parameters** (~50 variables): `beta`, `r_friction`, `A_visc`, `windStrength`, `S_solar`, `kappa_diff`, `alpha_T`, `gamma_mix`, `freshwaterForcing`, `globalTempOffset`, etc.
- **Grid state**: `NX`, `NY`, `dx`, `dy`, `invDx`, `invDy`, grid constants (`GPU_NX=360`, `GPU_NY=180`, `LON0=-180`, `LAT0=-80`, etc.)
- **Field arrays**: `psi` (streamfunction), `zeta` (vorticity), `temp`, `deepTemp`, `sal`, `deepSal`, `deepPsi`, `deepZeta`, `depth`, `mask`
- **Data loading**: Fetches `mask.json`, `coastlines.json`, `sst_global_1deg.json`, `deep_temp_1deg.json`, `bathymetry_1deg.json`
- **Mask helpers**: `buildMask()`, `buildMaskU32()`
- **5 WebGPU compute shader strings** (WGSL): `timestepShaderCode`, `poissonShaderCode`, `enforceBCShaderCode`, `deepTimestepShaderCode`, `temperatureShaderCode`
- **Initialization**: `generateDepthField()` (ETOPO1 bathymetry or BFS fallback), `initTemperatureField()` (NOAA/WOA observations), `initStommelSolution()`
- **CPU fallback solver**: `initCPU()`, `cpuTimestep()`, `cpuReset()`, `cpuSolveSOR()`, `cpuSolveDeepSOR()`, `cpuJacobian()`, `cpuLaplacian()`
- **Velocity & particles**: `getVel()`, `initParticles()`, `advectParticles()`, `spawnInOcean()`
- **Stability**: `stabilityCheck()` — CFL check, clamping, NaN detection, emergency damping

### gpu-solver.js — WebGPU Compute Pipeline

Manages all GPU buffer creation, compute pipeline dispatch, and CPU readback. Depends on model.js globals (shader strings, parameters, field arrays). Loaded via `<script src="gpu-solver.js">` after model.js.

**Contents:**
- **Buffer management** (~15 GPU buffers): `gpuPsiBuf`, `gpuZetaBuf`, `gpuTempBuf`, `gpuMaskBuf`, `gpuDepthBuf`, readback buffers, etc.
- **Pipeline creation**: 5 compute pipelines (timestep, Poisson, enforceBC, temperature, deep timestep)
- **Bind groups**: ~20 bind groups including Red-Black SOR variants for surface + deep Poisson solves
- `initWebGPU()` — device init, buffer creation, pipeline creation, field initialization
- `gpuRunSteps(n)` — dispatches n timesteps in a single command encoder (vorticity + temperature + Poisson + deep layer)
- `gpuReadback()` — async map GPU buffers back to CPU arrays (`psi`, `zeta`, `temp`, `sal`, `deepTemp`, `deepSal`, `deepPsi`)
- `gpuReset()` — reinitialize all GPU buffers from model state
- `uploadParams()` — pack 36 simulation parameters into uniform buffer
- `updateGPUBuffersAfterPaint()` — sync CPU-side mask/field changes to GPU after paint tool use
- `rebuildBindGroups()` — recreate all bind groups (called after buffer changes)

### index.html — Rendering & UI

### renderer.js — Canvas Rendering & GPU Render Pipeline

All visualization code. Depends on model.js globals (fields, particles, params) and gpu-solver.js globals (gpuDevice, GPU buffers). Loaded via `<script src="renderer.js">` after gpu-solver.js.

**Contents:**
- **Canvas refs**: `simCanvas`, `ctx`, `W`, `H`, `mapCanvas`, `mapCtx`, `fieldCanvas`
- **Coordinate helpers**: `lonToX()`, `latToY()`
- **Map underlay**: `drawMapUnderlay()` — land elevation coloring from ETOPO1 bathymetry
- **GPU render shaders**: `renderVertexShaderCode`, `renderFragmentShaderCode` (fullscreen quad)
- **GPU render pipeline**: `initGPURenderPipeline()`, `gpuRenderField()` — direct GPU-to-screen field visualization
- **CPU colormaps**: `tempToRGB()`, `psiToRGB()`, `speedToRGB()`, `salToRGB()`, `densityToRGB()`, `depthToRGB()`
- **Land temperature**: `initLandTemp()`, `drawSeasonalLand()` — thermal inertia + altitude lapse rate
- `draw()` — full CPU canvas render (field + land + particles + contours + labels + legend)
- `drawOverlay()` — overlay-only render for GPU mode (particles + labels on transparent 2D canvas)
- `drawProfile()` — velocity profile chart (sidebar)
- `drawRadProfile()` — radiative balance chart (sidebar)

### index.html — Main Loop, UI, Lab API

**Main Loop** (~80 lines):
- `gpuTick()` — GPU path: run steps, readback, render, advect particles
- `cpuTick()` — CPU path: timestep, draw, advect particles
- `updateStats()` — velocity, KE, season, AMOC strength display
- `resetSim()`, `init()`

**Desktop UI** (~350 lines):
- Sidebar slider wiring (wind, friction, viscosity, sim speed, etc.)
- View mode buttons (streamfunction, vorticity, speed, temp, etc.)
- Paint tool with brush size (land, ocean, heat, cold, ice, wind CW/CCW)
- Paleoclimate scenarios (Drake Passage, Panama, Greenland, Ice Age)
- Onboarding overlay

**Mobile UI** (~170 lines):
- Bottom toolbar with 4 tabs (Paint, Params, Science, Help)
- Slide-up drawers with scrim backdrop, swipe-to-close
- Floating HUD (AMOC strength, season)
- Pause FAB
- All interactions delegate to desktop DOM elements

**Lab API** (~300 lines):
- `window.lab` — console API for experimentation
- `lab.step(n)`, `lab.diagnostics()`, `lab.sweep()`, `lab.timeSeries()`
- `lab.getParams()`, `lab.setParams()`, `lab.scenario()`

## How Globals Are Shared

Both `model.js` and `index.html`'s `<script>` block share the browser's global lexical environment. Variables declared with `let`/`const`/`var` at the top level of either file are accessible from both. `model.js` loads first, so its declarations are available when `index.html`'s script runs.

Key shared globals:
- **Model -> UI**: `psi`, `zeta`, `temp`, `mask`, `NX`, `NY`, `showField`, `paused`, `totalSteps`, `simTime`, shader strings
- **UI -> Model**: `W`, `H` (canvas dimensions, referenced by `initCPU()`)
- **Both read/write**: `psi`, `zeta`, `temp`, `deepTemp`, `sal`, `deepSal` (GPU readback writes, rendering reads)

## Heat Transfer Model

Heat flows through three coupled layers: atmosphere, ocean surface, and deep ocean.

```
              SOLAR RADIATION
                    |
                    v  (S_solar * cos_zenith * ice_albedo * cloud_albedo)
         ┌──────────────────────┐
         │     ATMOSPHERE       │
         │     (airTemp)        │
         │                      │  kappa_atm diffusion (Hadley/Ferrel cells)
         │  carries heat across │  smooths temperature meridionally
         │  land and ocean      │
         └───┬──────────┬───────┘
             |          |
        gamma_la    gamma_oa        surface → atmosphere (strong)
        gamma_la    gamma_ao        atmosphere → surface (gentle)
             |          |
         ┌───┴───┐  ┌───┴──────┐
         │ LAND  │  │  OCEAN   │
         │(landT)│←→│  (temp)  │   landHeatK coastal exchange
         │       │  │          │
         └───────┘  └────┬─────┘
                         |
                    gamma_mix / gamma_deep_form
                    (density-driven deep water formation)
                         |
                    ┌────┴─────┐
                    │   DEEP   │
                    │ (deepT)  │   kappa_deep horizontal diffusion
                    └──────────┘
                         |
                         v
                    OLR = A + B*T  (longwave radiation to space)
                    (reduced by cloud greenhouse effect)
```

### Coupling Rates

| Parameter | Value | Direction | Notes |
|-----------|-------|-----------|-------|
| `gamma_oa` | 0.005 | ocean → air | Ocean warms/cools atmosphere above |
| `gamma_ao` | 0.001 | air → ocean | Atmosphere feeds back to SST (gentle — ocean has ~1000x thermal inertia) |
| `gamma_la` | 0.01 | land → air | Land warms/cools atmosphere (faster than ocean — less thermal mass) |
| `landHeatK` | 0.02 | land ↔ coastal ocean | Direct heat exchange at coastlines (GPU shader) |
| `gamma_mix` | 0.001 | surface ↔ deep | Background vertical mixing |
| `gamma_deep_form` | 0.05 | surface → deep | Enhanced when surface denser than deep (cold+salty sinks) |
| `kappa_atm` | 0.003 | horizontal | Atmospheric heat diffusion (represents large-scale circulation) |
| `kappa_diff` | 2.5e-4 | horizontal | Ocean thermal diffusion |
| `kappa_deep` | 2e-5 | horizontal | Deep ocean diffusion (slower) |

### Two-Way Atmosphere Coupling

The atmosphere layer runs on CPU between GPU readback cycles. Each readback:

1. **Air temp update**: Diffusion + exchange with surface below (ocean SST or seasonal land temp)
2. **Feedback to ocean**: `temp += dt * gamma_ao * (airTemp - temp)` for ocean cells
3. **Re-upload**: Corrected SST pushed back to GPU temperature buffer

This enables atmospheric teleconnections — e.g., warm Atlantic air transported over cold Pacific, or El Nino heat affecting remote basins via the atmospheric bridge.

### Cloud Parameterization

Cloud fraction is computed from latitude + SST:
- **Base**: `0.25 + 0.15 * cos(2*lat)` — subtropical stratocumulus
- **Convective**: `0.15 * clamp((SST - 15) / 15)` — warm SST drives convection
- **Polar**: `0.10 * clamp((|lat| - 50) / 30)` — polar stratus

Clouds affect radiation two ways:
- **Shortwave albedo**: `qSolar *= 1 - 0.30 * cloudFrac` (clouds reflect sunlight)
- **Longwave greenhouse**: `OLR *= 1 - 0.08 * cloudFrac * convective_fraction` (high clouds trap heat)

## Planned Separation (Phases 4-6)

| Phase | Extract | From |
|-------|---------|------|
| 4 | `ui-desktop.js` | Sliders, paint tool, scenarios, onboarding |
| 5 | `ui-mobile.js` | Toolbar, drawers, sync logic |
| 6 | `main.js` | Init, main loop, lab API; index.html becomes thin shell |

The end state: `index.html` is ~260 lines of HTML+CSS with `<script type="module" src="main.js">`.
