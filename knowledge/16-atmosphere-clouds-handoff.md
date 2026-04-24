# Atmosphere & Clouds: Status, Issues, and Future Work

## What We Built (2026-04-23/24)

### Atmosphere Layer
- **1-layer energy balance model** (`airTemp` field, 360x180 grid)
- Diffusion represents large-scale atmospheric circulation (Hadley/Ferrel/polar cells)
- Two-way coupled: ocean→air (`gamma_oa=0.005`) and air→ocean (`gamma_ao=0.001`)
- Land→air coupling (`gamma_la=0.01`) uses seasonal `landTempField` with altitude lapse rate
- Runs on CPU between GPU readback cycles (sub-stepped to avoid instability at high speed)
- After atmosphere update, corrected SST is re-uploaded to GPU

### Cloud Parameterization
- Diagnostic cloud fraction from latitude + SST (not prognostic — no cloud water budget)
- Three components: `cloudBase + convective + polar`
  - Base: `0.25 + 0.15 * cos(2*lat)` — subtropical stratocumulus
  - Convective: `0.15 * clamp((SST-15)/15)` — warm water drives convection
  - Polar: `0.10 * clamp((|lat|-50)/30)` — polar stratus
- **Shortwave**: `qSolar *= 1 - 0.30 * cloudFrac` (albedo)
- **Longwave**: `OLR *= 1 - 0.08 * cloudFrac * convective_frac` (greenhouse, tropics only)
- `cloudField` array computed at readback rate, rendered as a view mode

## Known Issues

### 1. Cloud Albedo Too Strong for Current Parameters
The #1 leaderboard entry (5.60°C RMSE) was tuned WITHOUT clouds. Adding cloud albedo (`-30% * cloudFrac`) drops tropical SST by ~10°C. The parameters need full retuning:
- `S_solar` must increase from 105 to ~120-130 to compensate cloud cooling
- `A_olr` and `B_olr` need rebalancing for the new radiative equilibrium
- **Action**: Run wiggum loop or tournament with cloud physics active

### 2. GPU Shader vs CPU Atmosphere Mismatch
The GPU temperature shader runs the ocean physics (advection, diffusion, solar, OLR, vertical mixing). The atmosphere runs on CPU BETWEEN readback cycles. This means:
- Atmosphere feedback is applied as a correction step, not within the physics timestep
- At high speed (500 steps/frame, readback every 5 frames), the atmosphere only updates every ~2500 timesteps
- The air→ocean feedback (`gamma_ao * (airTemp - temp)`) is applied as a bulk correction, then re-uploaded
- **Risk**: This is physically inconsistent — the GPU doesn't "know" about the atmosphere during its timestep
- **Fix options**:
  a. Upload `airTemp` as a GPU buffer and add air-ocean exchange to the temperature shader
  b. Increase readback frequency when atmosphere is active
  c. Accept the approximation (it's gentle enough that it probably doesn't matter much)

### 3. Land Temperature Inconsistency
Three different land temperature calculations exist:
- **Renderer** (`landTempField`): Seasonal with thermal inertia + altitude lapse rate. Used for visualization and air-land coupling in CPU path
- **GPU shader** (`landT = 50*cosZenith - 20`): Simple radiative equilibrium, no altitude, used for land-ocean coastal exchange
- **Air temp init** (`28 - 0.55*|lat|`): Fallback when `landTempField` not available
- **Fix**: Upload `landTempField` to GPU as a buffer, or compute it in the shader using bathymetry elevation data (already available as `depthField` — negative depth = land elevation)

### 4. No Wind-Driven Atmospheric Circulation
Current atmospheric transport is pure diffusion (`kappa_atm * laplacian(airTemp)`). Real atmosphere has:
- **Hadley cells**: Strong equator→30° meridional transport
- **Ferrel cells**: Mid-latitude mixing
- **Polar cells**: Weak high-latitude transport
- **Western boundary of atmosphere**: Jet streams concentrate heat transport
- **Fix**: Add latitude-dependent advection: `v_atm = v0 * sin(2*lat)` in the atmosphere update, or use a latitude-varying diffusion coefficient

### 5. Cloud-SST Feedback Not Captured
Clouds cool the surface (albedo) AND warm it (greenhouse). Currently:
- Cloud fraction is diagnostic (computed from SST, not evolved)
- If SST drops → less convective cloud → less albedo → more solar → SST rises (negative feedback)
- This feedback IS implicitly present because cloudFrac depends on temp[k]
- But the greenhouse term only activates for convective clouds in tropics, not for all clouds
- **Possible improvement**: Strengthen cloud greenhouse for polar stratus (currently zero for polar clouds)

### 6. No Moisture/Precipitation
The atmosphere has temperature but no moisture. This means:
- No latent heat transport (huge in reality — ~30% of poleward heat transport)
- No precipitation feedback on salinity (freshwater forcing is only from the "Meltwater" slider)
- No evaporative cooling of the ocean surface
- **Future**: Add a moisture field `q(x,y)` with evaporation from warm ocean, condensation at saturation, precipitation that feeds into salinity

## Ideas for Future Work

### Near-term (parameter tuning)
1. **Run wiggum loop** with atmosphere+clouds active to find optimal S_solar, A_olr, B_olr, cloud coefficients
2. **Sweep cloud albedo strength** (0.10 to 0.40) to find the value that minimizes RMSE with other params fixed
3. **Add cloud albedo as a tunable parameter** — currently hardcoded as 0.30 in both GPU shader and CPU

### Medium-term (physics improvements)
4. **Upload airTemp to GPU** — enable air-ocean feedback within the physics timestep
5. **Latitude-dependent atmospheric transport** — stronger Hadley cell transport at tropics, weaker at poles
6. **Prognostic clouds** — evolve cloud water/ice as a field, not just diagnose from SST
7. **Moisture field** — evaporation, transport, condensation, precipitation→salinity coupling
8. **Diurnal cycle** — currently seasons only; adding day/night would affect cloud formation and land temp

### Long-term (structural)
9. **Multi-layer atmosphere** — surface + upper troposphere (needed for jet streams, Hadley cell)
10. **Atmospheric waves** — Rossby waves in the atmosphere affect blocking patterns and teleconnections
11. **Sea ice model** — currently ice is just "cold ocean" (temp < -1.8°C). A real ice model would have ice thickness, albedo feedback, brine rejection, ice drift
12. **Carbon cycle** — CO2 as a prognostic variable, ocean uptake, temperature-dependent dissolution

## Parameter Sensitivity

From the tuning experiments (2026-04-24):

| S_solar | A_olr | B_olr | Cloud albedo | RMSE | Tropical SST | Polar SST | Notes |
|---------|-------|-------|-------------|------|-------------|-----------|-------|
| 105 | 40 | 2.5 | 0.30 | 6.09°C | 18.6°C | -3.5°C | #1 params, new physics |
| 105 | 40 | 2.5 | 0.15 | 7.21°C | 20.6°C | -1.1°C | Halved cloud albedo |
| 120 | 38 | 2.2 | 0.30 | 9.06°C | 26.5°C | 2.9°C | Boosted solar mid |
| 145 | 35 | 2.0 | 0.30 | 14.73°C | 39.9°C | 6.0°C | Way too hot |
| 105 | 40 | 2.5 | 0.00 | 5.60°C | ~27°C | ~-1°C | #1 entry (no clouds) |

The sweet spot for S_solar with 0.30 cloud albedo is probably around 110-115.

## File Locations

- **Atmosphere physics (CPU)**: `model.js` lines 1046-1075 (cpuTimestep atmosphere section)
- **Atmosphere physics (GPU path)**: `main.js` lines 13-48 (gpuTick readback section)
- **Cloud physics (GPU shader)**: `model.js` temperatureShaderCode, search "CLOUD PARAMETERIZATION"
- **Cloud physics (CPU)**: `model.js` cpuTimestep, search "Cloud parameterization"
- **Cloud field computation**: `main.js` lines 50-64 (after atmosphere update)
- **Cloud rendering**: `renderer.js` `cloudFracToRGB()` function
- **Air temp rendering**: `renderer.js` draw() function, `showField === 'airtemp'`
- **Parameters**: `model.js` lines 59-65 (kappa_atm, gamma_oa, gamma_ao, gamma_la)
- **Heat transfer docs**: `simamoc/ARCHITECTURE.md` "Heat Transfer Model" section
