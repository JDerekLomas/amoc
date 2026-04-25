# SimAMOC Architecture

## Model/UI Separation (Complete)

The simulation was split from a single 4,187-line `index.html` into 8 JS modules sharing the browser's global scope.

### File Structure

```
simamoc/
  model.js          ~1,248 lines  Physics engine (no DOM dependencies)
  gpu-solver.js       ~628 lines  WebGPU compute pipelines, buffers, dispatch
  renderer.js       ~1,114 lines  Canvas rendering, colormaps, GPU render pipeline
  main.js             ~197 lines  Main loop, init, lab API (window.lab)
  ui.js               ~100 lines  Control wiring, paint tool, scenarios
  overlay.js           ~70 lines  Reparent sidebar into drawers, FAB, speed presets
  index.html          ~296 lines  Pure HTML+CSS shell (no logic)
  input-widget.js     ~357 lines  Touch interaction widget
  mask.json                       360x160 land/ocean mask (hex-encoded)
  coastlines.json                 Coastline polygon data
  equation.html                   Equation guide page
```

Load order: `model.js` -> `gpu-solver.js` -> `renderer.js` -> `main.js` -> `ui.js` -> `overlay.js`

### model.js -- Physics Core

Zero DOM dependencies. Declares all simulation state as globals.

**Contents:**
- **Parameters** (~50 variables): `beta`, `r_friction`, `A_visc`, `windStrength`, `S_solar`, `kappa_diff`, `alpha_T`, `gamma_mix`, `freshwaterForcing`, `globalTempOffset`, etc.
- **Atmosphere parameters**: `kappa_atm`, `gamma_oa`, `gamma_ao`, `gamma_la`
- **Grid state**: `NX`, `NY`, `dx`, `dy`, `invDx`, `invDy`, grid constants (`GPU_NX=512`, `GPU_NY=160`, `LON0=-180`, `LAT0=-79.5`, etc.)
- **Field arrays**: `psi` (streamfunction), `zeta` (vorticity), `temp`, `deepTemp`, `sal`, `deepSal`, `deepPsi`, `deepZeta`, `depth`, `mask`, `airTemp`, `cloudField`
- **Data loading**: Fetches `mask.json`, `coastlines.json`, `sst_global_1deg.json`, `deep_temp_1deg.json`, `bathymetry_1deg.json`, `salinity_1deg.json`, `wind_stress_1deg.json`, `albedo_1deg.json`, `precipitation_1deg.json`, `cloud_fraction_1deg.json`
- **Mask helpers**: `buildMask()`, `buildMaskU32()`, `buildRemappedFields()` (wires obs data arrays for renderer access)
- **5 WebGPU compute shader strings** (WGSL): `timestepShaderCode`, `poissonShaderCode`, `enforceBCShaderCode`, `deepTimestepShaderCode`, `temperatureShaderCode`
- **5 FFT shader strings** (WGSL): `fftButterflyShaderCode`, `fftBitRevShaderCode`, `fftTridiagShaderCode`, `fftTransposeShaderCode`, `fftScaleMaskShaderCode`
- **FFT Poisson solver**: `fftRadix2()`, `initFFTPoisson()`, `cpuSolveFFT()` — exact spectral solver
- **Initialization**: `generateDepthField()` (ETOPO1 bathymetry or BFS fallback), `initTemperatureField()` (NOAA/WOA observations), `initStommelSolution()`
- **CPU fallback solver**: `initCPU()`, `cpuTimestep()`, `cpuReset()`, `cpuSolveSOR()`, `cpuSolveDeepSOR()` -- includes two-way atmosphere coupling
- **Data resampling**: `resampleToModelGrid()` — bilinear interpolation from 360×160 obs data to model grid
- **Velocity & particles**: `getVel()`, `initParticles()`, `advectParticles()`, `spawnInOcean()`
- **Stability**: `stabilityCheck()` -- CFL check, clamping, NaN detection, emergency damping

### gpu-solver.js -- WebGPU Compute Pipeline

Manages all GPU buffer creation, compute pipeline dispatch, and CPU readback. Depends on model.js globals.

**Contents:**
- **Buffer management**: physics buffers (psi, zeta, temp+sal stacked, deep, mask, depth) + packed data buffers:
  - `gpuEkmanSalBuf`: stacked [u_ek | v_ek | salClimatology] (3×N) — binding 8 in temperature shader
  - `gpuForcingBuf`: stacked [snow | seaIce | evap | precip] (4×N) — binding 9, read_write for dynamic ice
  - `gpuAtmBuf`/`gpuAtmNewBuf`: stacked [airTemp | moisture] (2×N) — binding 10, double-buffered
  - Buffer packing keeps total storage bindings ≤ 10 (hardware limit on Apple GPUs)
- **Pipeline creation**: 5 physics + 1 atmosphere + 5 FFT compute pipelines
- **Bind groups**: ~20 physics + atmosphere bind groups + ~20 FFT bind groups (pre-created per butterfly pass)
- **GPU FFT solver**: `gpuFFTPoissonSolve()` — encodes full FFT+tridiagonal+IFFT into command encoder. Production solver for both surface and deep Poisson.
- `initWebGPU()` -- device init (requests maxStorageBuffersPerShaderStage=10), buffer creation, pipeline creation, field initialization
- `gpuRunSteps(n)` -- dispatches n timesteps: vorticity → temperature → atmosphere → enforceBC → FFT Poisson → deep layer → FFT deep Poisson
- `gpuReadback()` -- async map GPU buffers back to CPU arrays (psi, zeta, temp+sal, deep, atmosphere)
- `gpuReset()` -- reinitialize all GPU buffers from model state
- `uploadParams()` -- pack 40 simulation parameters into uniform buffer (160 bytes)
- `updateGPUBuffersAfterPaint()` -- sync CPU-side mask/field changes to GPU after paint tool use
- `rebuildBindGroups()` -- recreate all bind groups (called after buffer changes)

### renderer.js -- Canvas Rendering & GPU Render Pipeline

All visualization code. Depends on model.js globals and gpu-solver.js globals.

**Contents:**
- **Canvas refs**: `simCanvas`, `ctx`, `W`, `H`, `mapCanvas`, `mapCtx`, `fieldCanvas`
- **Coordinate helpers**: `lonToX()`, `latToY()`
- **Map underlay**: `drawMapUnderlay()` -- land elevation coloring from ETOPO1 bathymetry
- **GPU render shaders**: `renderVertexShaderCode`, `renderFragmentShaderCode` (fullscreen quad)
- **GPU render pipeline**: `initGPURenderPipeline()`, `gpuRenderField()` -- direct GPU-to-screen field visualization
- **CPU colormaps**: `tempToRGB()`, `psiToRGB()`, `speedToRGB()`, `salToRGB()`, `densityToRGB()`, `depthToRGB()`, `cloudFracToRGB()`
- **Land temperature**: `initLandTemp()`, `drawSeasonalLand()` -- thermal inertia + altitude lapse rate
- **View modes**: Temperature, Deep Ocean, Currents, Deep Currents, Speed, Vorticity, Salinity, Density, Bathymetry, Clouds, Air Temp, Particles (overlay)
- `draw()` -- full CPU canvas render (field + land + particles + contours + labels + legend)
- `drawOverlay()` -- overlay-only render for GPU mode (particles + labels on transparent 2D canvas)
- `drawProfile()` -- velocity profile chart
- `drawRadProfile()` -- radiative balance chart

### main.js -- Main Loop, Init, Lab API

Extracted from index.html. Orchestrates the simulation lifecycle.

**Main Loop:**
- `gpuTick()` -- GPU path: run steps, readback, stability check, atmosphere update (CPU-side sub-stepping), cloud field update, render, advect particles
- `cpuTick()` -- CPU path: timestep, draw, advect particles
- `updateStats()` -- velocity, KE, season, AMOC strength display

**Atmosphere**: Now runs fully on GPU via the atmosphere compute shader (diffusion + surface exchange + evaporation + condensation). CPU-side atmosphere loop removed. Readback populates `airTemp` and `moisture` CPU arrays for cloud parameterization.

**Cloud field update**: Recomputes `cloudField` from latitude + SST + atmosphere moisture each readback cycle (CPU-only, for rendering).

**Init:**
- Loads all data files, initializes WebGPU (falls back to CPU), sets up render pipeline, starts main loop
- Saves original mask for scenario reset

**Lab API** (`window.lab`):
- `lab.step(n)`, `lab.diagnostics()`, `lab.sweep()`, `lab.timeSeries()`, `lab.benchmark()`
- `lab.getParams()`, `lab.setParams()`, `lab.scenario()`
- `lab.fields()`, `lab.reset()`, `lab.view()`, `lab.pause()`, `lab.resume()`

### ui.js -- Controls, Paint Tool, Scenarios

Wires DOM elements to model.js globals.

**Controls:** Slider wiring for all parameters (wind, friction, viscosity, speed, year speed, freshwater, CO2/temp offset, deep water formation, solar).

**Paint tool:** 7 brush modes (land, ocean, heat, cold, ice, wind CW/CCW) with adjustable brush size. Mouse + touch support.

**Scenarios:** 6 paleoclimate scenarios (Drake Passage open/close, Panama Seaway toggle, Greenland melting, Ice Age, Present Day reset) with mask manipulation and parameter changes.

### overlay.js -- Drawer UI & Speed Presets

Reparents sidebar control groups into mobile-friendly drawers at runtime:
- View buttons -> top layer bar
- Paint palette -> paint drawer
- Sliders -> controls drawer (grouped: Climate, Ocean, Speed)
- Scenarios -> scenarios drawer
- Physics/charts -> info drawer

Also handles: drawer open/close with scrim, swipe-to-dismiss, pause FAB, speed preset buttons (1x/3x/10x/MAX).

### index.html -- Pure HTML+CSS Shell

No JavaScript logic. Contains:
- Full-bleed layout structure (viewbar, HUD, FAB, speed strip, toolbar, drawers, scrim)
- All control DOM elements (sliders, buttons, paint palette, scenario cards)
- Onboarding overlay
- CSS for Windy-style unified layout (same on mobile and desktop)
- Script tags loading all JS in order

## Grid Resolution

The model grid uses power-of-2 NX for the FFT Poisson solver. Currently NX=512, NY=160 on CPU.

| | Value | Notes |
|---|---|---|
| NX | 512 | Power-of-2 for radix-2 FFT |
| NY | 160 | Matches observation data latitude dimension |
| Lat range | -79.5° to +79.5° | Polar regions excluded (no open-ocean data) |
| Cells | 81,920 | |
| Data files | 360×160 | Bilinearly resampled to model grid at load time |

**Why power-of-2 NX?** The Poisson equation ∇²ψ = ζ is solved via FFT in the periodic x-direction. Radix-2 FFT requires power-of-2 length. NX=512 gives 0.7° longitude resolution — finer than the 1° observation data.

**Metric correction:** cos(lat) scaling on zonal derivatives in the physics operators (Jacobian, viscosity, beta term). The Poisson solver uses the **grid Laplacian** (no cos(lat)) — this is consistent with how ζ is defined in the vorticity equation.

**Canvas:** Display resolution (960×427) is independent of model resolution.

## Poisson Solver

The streamfunction ψ is computed from vorticity ζ via ∇²ψ = ζ every timestep. This is the computational bottleneck.

### FFT + Tridiagonal (CPU and GPU, exact)

1. **Forward FFT** each row of ζ (radix-2, NX=1024 → 10 butterfly passes)
2. **Tridiagonal solve** per Fourier mode (Thomas algorithm, NY=512 per mode)
3. **Inverse FFT** each row to recover ψ

Eigenvalue for mode m: `km² = invDx² × 2 × (cos(2πm/NX) - 1)`.
Tridiagonal diagonal: `b[j] = km² - 2 × invDy²` (grid Laplacian, no cos(lat)).
Boundary: ψ = 0 at j=0 and j=NY-1 (hardcoded in Thomas back-substitution).

**CPU cost:** ~1.4s per solve at 1024×512 (O(NX × NY × log NX)). Float64.
**GPU cost:** ~25 compute passes per solve (1 bit-rev + 10 butterfly + 2 transpose + 1 tridiag + 2 transpose + 1 bit-rev + 10 butterfly + 1 scale+mask). Float32. ~1ms estimated.

**Land mask handling:** The FFT solves on the full rectangle including land cells. ψ is NOT zeroed over land — doing so creates discontinuities at coastlines that corrupt ∇²ψ at adjacent ocean cells. Land ψ values are non-physical but harmless since the vorticity equation skips land cells.

### GPU FFT (production solver)

5 WGSL compute shaders: butterfly, bit-reversal, transpose, tridiagonal, scale+mask. Wired into `gpuRunSteps()` for both surface and deep Poisson solves, replacing the previous red-black SOR which couldn't converge at 1024×512.

**Tridiagonal solver:** One workgroup per Fourier mode (workgroup_size(1), NX dispatches). Forward elimination runs j=1..NY-2 only (interior rows). Back-substitution hardcodes ψ=0 at boundaries j=0 and j=NY-1.

**Validated:** `gpu-test.html` runs the full pipeline standalone with uniform ζ=1, verifies against analytical solution (parabolic ψ). Residual ~4e-4 (float32 precision).

### SOR (deprecated)

Red-Black SOR bind groups still exist in gpu-solver.js but are no longer dispatched. Cannot converge at 1024×512.

## How Globals Are Shared

All JS files share the browser's global lexical environment. Variables declared with `let`/`const`/`var` at the top level of any file are accessible from all others.

Key shared globals:
- **Model -> all**: `psi`, `zeta`, `temp`, `mask`, `airTemp`, `cloudField`, `NX`, `NY`, `showField`, `paused`, `totalSteps`, `simTime`, shader strings
- **Renderer -> UI**: `simCanvas`, `W`, `H`, `draw()`, `drawOverlay()`, `gpuRenderField()`
- **GPU solver -> renderer**: `gpuDevice`, `gpuPsiBuf`, `gpuZetaBuf`, `gpuTempBuf`, `gpuMaskBuf`
- **Both read/write**: `psi`, `zeta`, `temp`, `deepTemp`, `sal`, `deepSal` (GPU readback writes, rendering reads, atmosphere modifies)

## Heat Transfer Model

Heat flows through three coupled layers: atmosphere, ocean surface, and deep ocean.

```
              SOLAR RADIATION
                    |
                    v  (S_solar * cos_zenith * ice_albedo * cloud_albedo)
         +------------------------+
         |     ATMOSPHERE         |
         |     (airTemp)          |
         |                        |  kappa_atm diffusion (Hadley/Ferrel cells)
         |  carries heat across   |  smooths temperature meridionally
         |  land and ocean        |
         +---+----------+---------+
             |          |
        gamma_la    gamma_oa        surface -> atmosphere (strong)
        gamma_la    gamma_ao        atmosphere -> surface (gentle)
             |          |
         +---+---+  +---+------+
         | LAND  |  |  OCEAN   |
         |(landT)|<>|  (temp)  |   landHeatK coastal exchange
         |       |  |          |
         +-------+  +----+----+
                          |
                     gamma_mix / gamma_deep_form
                     (density-driven deep water formation)
                          |
                     +----+-----+
                     |   DEEP   |
                     | (deepT)  |   kappa_deep horizontal diffusion
                     +----------+
                          |
                          v
                     OLR = A + B*T  (longwave radiation to space)
                     (reduced by cloud greenhouse effect)
```

### Coupling Rates

| Parameter | Value | Direction | Notes |
|-----------|-------|-----------|-------|
| `gamma_oa` | 0.005 | ocean -> air | Ocean warms/cools atmosphere above |
| `gamma_ao` | 0.001 | air -> ocean | Atmosphere feeds back to SST (gentle -- ocean has ~1000x thermal inertia) |
| `gamma_la` | 0.01 | land -> air | Land warms/cools atmosphere (faster than ocean -- less thermal mass) |
| `landHeatK` | 0.02 | land <-> coastal ocean | Direct heat exchange at coastlines (GPU shader) |
| `gamma_mix` | 0.001 | surface <-> deep | Background vertical mixing |
| `gamma_deep_form` | 0.05 | surface -> deep | Enhanced when surface denser than deep (cold+salty sinks) |
| `kappa_atm` | 0.003 | horizontal | Atmospheric heat diffusion (represents large-scale circulation) |
| `kappa_diff` | 2.5e-4 | horizontal | Ocean thermal diffusion |
| `kappa_deep` | 2e-5 | horizontal | Deep ocean diffusion (slower) |

### Two-Way Atmosphere Coupling

The atmosphere layer runs on CPU between GPU readback cycles. Each readback:

1. **Air temp update**: Diffusion + exchange with surface below (ocean SST or seasonal land temp from `landTempField`)
2. **Feedback to ocean**: `temp += dt * gamma_ao * (airTemp - temp)` for ocean cells
3. **Re-upload**: Corrected SST pushed back to GPU temperature buffer

This enables atmospheric teleconnections -- e.g., warm Atlantic air transported over cold Pacific, or El Nino heat affecting remote basins via the atmospheric bridge.

### Cloud Parameterization (Regime-Based)

Cloud fraction is computed from six physical regimes (matching GPU shader and stored in `cloudField`):

**Cloud regimes:**
- **ITCZ deep convection**: `0.30 * exp(-((lat - itczLat)/10)^2) * humidity` -- seasonal migration
- **Warm-pool convection**: `0.20 * clamp((SST - 26) / 4)` -- threshold-based tropical convection
- **Subtropical subsidence**: `-0.25 * exp(-((|lat| - 25)/10)^2)` -- Hadley descent clears skies
- **Marine stratocumulus**: `0.30 * LTS * clamp((35 - |lat|) / 20)` -- cold SST + stable air = low clouds
- **Mid-latitude storm track**: `0.25 * ramp(35-65 deg)` -- frontal cloudiness
- **Polar stratus**: `0.12 * clamp((|lat| - 55) / 20)` -- thin persistent low clouds

**Key proxies:**
- `humidity = clamp((SST - 5) / 25)` -- warm ocean = more evaporation
- `LTS = clamp((airTempEst - SST) / 15)` -- lower tropospheric stability (inversion strength)
- `itczLat = 5 * sin(yearPhase)` -- ITCZ migrates seasonally

**Radiative effects (cloud-type dependent):**
- **SW albedo**: low clouds reflect more (0.35) than high convective clouds (0.20)
- **LW greenhouse**: high clouds trap more OLR (0.12) than low clouds (0.03)
- Net: stratocumulus cools strongly, deep convection is nearly radiatively neutral
