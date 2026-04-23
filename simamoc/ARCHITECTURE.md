# SimAMOC Architecture

## Model/UI Separation (Phase 1 Complete)

The simulation was split from a single 4,187-line `index.html` into a physics model layer and a presentation layer.

### File Structure

```
simamoc/
  model.js          ~1,050 lines  Physics engine (no DOM dependencies)
  index.html        ~3,016 lines  GPU solver, rendering, desktop UI, mobile UI
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

### index.html — Rendering, GPU Solver, UI

**GPU Solver** (~600 lines):
- `initWebGPU()` — device, buffers, pipelines, bind groups
- `gpuRunSteps()` — timestep + Poisson + temperature in one command encoder
- `gpuReadback()` — async map GPU buffers back to CPU arrays
- `gpuReset()`, `uploadParams()`, `updateGPUBuffersAfterPaint()`
- `rebuildBindGroups()` — Red-Black SOR bind group pairs

**GPU Render Pipeline** (~100 lines):
- WebGPU render shaders (vertex + fragment) for direct-to-screen field visualization
- `initGPURenderPipeline()`, `gpuRenderField()`

**CPU Rendering** (~600 lines):
- Colormap functions: `tempToRGB()`, `psiToRGB()`, `speedToRGB()`, `salToRGB()`, `densityToRGB()`, `depthToRGB()`
- `draw()` — full CPU canvas render (field + land + particles + contours + labels)
- `drawOverlay()` — overlay-only render for GPU mode (particles + labels on transparent canvas)
- `drawSeasonalLand()` — land temperature with thermal inertia
- `drawProfile()`, `drawRadProfile()` — sidebar charts

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

## Planned Separation (Phases 2-6)

| Phase | Extract | From |
|-------|---------|------|
| 2 | `gpu-solver.js` | GPU buffer/pipeline/dispatch code from index.html |
| 3 | `renderer.js` | Colormaps, draw(), drawOverlay(), charts |
| 4 | `ui-desktop.js` | Sliders, paint tool, scenarios, onboarding |
| 5 | `ui-mobile.js` | Toolbar, drawers, sync logic |
| 6 | `main.js` | Init, main loop, lab API; index.html becomes thin shell |

The end state: `index.html` is ~260 lines of HTML+CSS with `<script type="module" src="main.js">`.
