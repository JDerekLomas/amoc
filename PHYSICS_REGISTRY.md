# SimAMOC Physics Registry

Every physical process in the model, its parameters, data sources, validation targets, and known gaps.

## How to read this

Each process has:
- **Equation**: what the code computes
- **Parameters**: tunable numbers (with current values)
- **Data**: observational datasets that inform or validate it
- **Status**: Active / Partial / Missing
- **Confidence**: How well-constrained by theory + data

---

## 1. SOLAR RADIATION

### 1a. Top-of-atmosphere insolation
| | |
|---|---|
| Equation | `qSolar = S_solar * max(0, cosZenith)` where cosZenith = cos(lat)cos(decl) + sin(lat)sin(decl) |
| Parameters | `S_solar = 6.2` (nondimensional, tuned) |
| Data | None (analytical from orbital geometry) |
| Theory | Milankovitch / spherical geometry. cosZenith is exact. S_solar is a free parameter absorbing unit conversion. |
| Status | **Active** |
| Confidence | High for pattern, low for magnitude (S_solar is tuned, not derived) |
| Code | `model.js` temperature shader, line ~817 |

### 1b. Ice-albedo feedback
| | |
|---|---|
| Equation | `if \|lat\| > 45: qSolar *= 1 - 0.50 * iceFrac * latRamp` where iceFrac = smoothstep(SST, -2, 8) |
| Parameters | Max albedo reduction: 0.50. SST window: [-2, 8]°C. Lat onset: 45° |
| Data | MODIS snow cover (`earth-data/snow_cover.csv`) — downloaded, NOT used |
| Theory | Ice/snow reflects 60-80% of solar. Smoothstep transition is ad hoc. |
| Status | **Active** (ocean only) |
| Confidence | Medium — correct sign and location, coefficients are guesses |
| Gap | **No land snow-albedo.** Siberia/Canada winter albedo goes from 0.15 to 0.60. Data exists but isn't wired. |
| Code | `model.js` temperature shader, line ~819-828 |

### 1c. Cloud shortwave albedo
| | |
|---|---|
| Equation | `qSolar *= 1 - cloudAlbedo` where cloudAlbedo = cloudFrac * (0.35*(1-convFrac) + 0.20*convFrac) |
| Parameters | Low cloud albedo: 0.35. High cloud albedo: 0.20. |
| Data | MODIS cloud fraction (`cloud_fraction_1deg.json`, 2020-2023 mean). MODIS low/high cloud types (`cloud_types_1deg.json`). |
| Theory | Twomey (1977). Low maritime clouds: albedo 0.3-0.5. Deep convective: 0.2-0.3. Cirrus: 0.05-0.10. |
| Status | **Active** |
| Confidence | Medium — low cloud albedo is good, but cirrus and deep convection are lumped together |
| Gap | **Cirrus not separated from deep convection.** Thin cirrus has albedo ~0.05 but greenhouse ~0.20 (net warming). Our "high cloud" category averages this with thick convective towers. |
| Code | `model.js` temperature shader, line ~655-657 |

### 1d. Ocean surface albedo
| | |
|---|---|
| Equation | Not modeled — ocean absorbs 100% of post-cloud solar |
| Parameters | None |
| Data | Implicit 0.06 in albedo_1deg.json for ocean cells |
| Theory | Fresnel reflection ~0.06 at high sun, increasing to ~0.40 at low angles. Wind roughness reduces specular reflection. |
| Status | **Missing** |
| Confidence | N/A |
| Gap | 6% of solar reflected by ocean surface is ignored. Small globally but systematic. |

### 1e. Land surface albedo
| | |
|---|---|
| Equation | Used in land temperature calculation (renderer.js), feeds into atmosphere via air-land exchange |
| Parameters | Per-cell from MODIS data |
| Data | `albedo_1deg.json` — MODIS MCD43A3, 2020-2023 annual mean via GEE |
| Theory | Surface reflectivity depends on vegetation, soil, snow, moisture |
| Status | **Active** (annual mean only) |
| Confidence | High for annual mean, missing seasonal variation (snow) |
| Gap | **No seasonal albedo.** Winter snow dramatically increases high-latitude land albedo. Issue #18. |

---

## 2. OUTGOING LONGWAVE RADIATION

### 2a. OLR (Stefan-Boltzmann linearized)
| | |
|---|---|
| Equation | `olr = A_olr - B_olr * globalTempOffset + B_olr * T` |
| Parameters | `A_olr = 1.8`, `B_olr = 0.13` (tuned) |
| Data | CERES EBAF would be the validation target (not downloaded) |
| Theory | Linearization of sigma*T^4 around a reference temperature. A = OLR at T=0, B = climate sensitivity parameter. Real B ~ 2.2 W/m²/K → model's B_olr=0.13 is nondimensional. |
| Status | **Active** |
| Confidence | Medium — functional form is standard (Budyko 1969), coefficients are tuned |

### 2b. Cloud longwave greenhouse
| | |
|---|---|
| Equation | `effectiveOlr = olr * (1 - cloudGreenhouse)` where cloudGreenhouse = cloudFrac * (0.03*(1-convFrac) + 0.12*convFrac) |
| Parameters | Low cloud greenhouse: 0.03. High cloud greenhouse: 0.12. |
| Data | MODIS cloud types (`cloud_types_1deg.json`) for validation |
| Theory | High clouds emit at cold temperatures → trap more OLR. Low clouds are warm → weak greenhouse. Cirrus specifically: very strong greenhouse (0.15-0.25) relative to small albedo. |
| Status | **Active** |
| Confidence | Medium — correct qualitative behavior, but cirrus greenhouse likely underestimated |
| Gap | **Cirrus greenhouse too weak.** Thin cirrus traps ~50 W/m² but reflects only ~10 W/m² (net +40 W/m² warming). Our 0.12 coefficient averages this with thick convective clouds that have nearly neutral net effect. |

---

## 3. CLOUD PARAMETERIZATION

### 3a. ITCZ deep convection
| | |
|---|---|
| Equation | `convCloud = 0.30 * exp(-((lat - itczLat)/10)^2) * humidity` |
| Parameters | Amplitude: 0.30. Width: 10°. ITCZ migration: 5° seasonal. |
| Data | MODIS high cloud fraction (`cloud_types_1deg.json`: high_cloud_fraction_ir) |
| Theory | Deep convection over warm SST > 26°C. ITCZ follows max insolation. |
| Status | **Active** |
| Confidence | Medium |

### 3b. Warm-pool convection
| | |
|---|---|
| Equation | `warmPool = 0.20 * clamp((SST - 26) / 4)` |
| Parameters | Threshold: 26°C. Ramp: 4°C. Amplitude: 0.20. |
| Data | MODIS high cloud fraction |
| Theory | Graham & Barnett (1987) — convection onset at ~26-27°C SST |
| Status | **Active** |
| Confidence | High for threshold, medium for amplitude |

### 3c. Subtropical subsidence
| | |
|---|---|
| Equation | `subsidence = 0.25 * exp(-((|lat| - 25)/10)^2)` — subtracted from cloud fraction |
| Parameters | Center: 25°. Width: 10°. Strength: 0.25. |
| Data | MODIS total cloud fraction (Sahara 0.10 validates this) |
| Theory | Hadley cell descending branch suppresses cloud formation |
| Status | **Active** |
| Confidence | High — Sahara/Arabian desert cloudlessness is well-captured |

### 3d. Marine stratocumulus
| | |
|---|---|
| Equation | `stratocu = 0.30 * LTS * clamp((35 - |lat|) / 20)` where LTS = clamp((airTempEst - SST)/15) |
| Parameters | LTS proxy: airTempEst = 28 - 0.55*|lat|. Amplitude: 0.30. |
| Data | MODIS low cloud fraction (`cloud_types_1deg.json`: low_cloud_fraction). SE Pacific: 0.59 observed. |
| Theory | Klein & Hartmann (1993) — stratocumulus scales with lower tropospheric stability |
| Status | **Active** |
| Confidence | Medium — LTS proxy is crude (should use actual airTemp). Could use real airTemp if uploaded to GPU. |

### 3e. Mid-latitude storm track
| | |
|---|---|
| Equation | `stormTrack = 0.25 * ramp(35-45) * ramp(65-80)` |
| Parameters | Band: 35-80°. Amplitude: 0.25. |
| Data | MODIS total cloud fraction |
| Status | **Active** |
| Confidence | Low — uniform in longitude, but real storm tracks concentrate over western ocean basins |

### 3f. Southern Ocean boundary layer clouds
| | |
|---|---|
| Equation | `soCloud = 0.35 * clamp((|lat| - 45) / 10)` (SH only) |
| Parameters | Onset: 45°S. Amplitude: 0.35. |
| Data | MODIS: SO -60° observed 0.92 total, 0.55 liquid (low). Model gives ~0.65. |
| Theory | Persistent boundary layer clouds from cold ocean + strong baroclinic activity |
| Status | **Active** (added 2026-04-24) |
| Confidence | Low — still underestimates SO cloudiness by ~0.25 |
| Gap | NH has no equivalent term — N Atlantic 50° observed 0.90 but model gives ~0.30 |

### 3g. Polar stratus
| | |
|---|---|
| Equation | `polarCloud = 0.15 * clamp((|lat| - 55) / 15)` |
| Parameters | Onset: 55°. Amplitude: 0.15. |
| Status | **Active** |
| Confidence | Low |

---

## 4. OCEAN CIRCULATION (Vorticity equation)

### 4a. Wind stress curl forcing
| | |
|---|---|
| Equation | `F = windStrength * windCurlField[k]` |
| Parameters | `windStrength = 1.0` (multiplier). Wind field pre-scaled via RMS matching. |
| Data | `wind_stress_1deg.json` — NCEP/NCAR Reanalysis 10m wind, 1991-2020. Tau_x, tau_y, curl. |
| Theory | Sverdrup balance: wind curl drives gyre circulation. |
| Status | **Active** (with RMS-matched scaling) |
| Confidence | Medium — RMS scaling is empirical, not derived from nondimensionalization |

### 4b. Beta effect (planetary vorticity gradient)
| | |
|---|---|
| Equation | `betaV = beta * cos(lat) * dpsi/dx` |
| Parameters | `beta = 1.0` |
| Theory | Coriolis parameter varies with latitude: df/dy = 2*Omega*cos(lat)/R |
| Status | **Active** |
| Confidence | High |

### 4c. Friction
| | |
|---|---|
| Equation | `fric = -r_friction * zeta` |
| Parameters | `r_friction = 0.04` |
| Theory | Linear bottom/lateral friction. Rayleigh damping. |
| Status | **Active** |
| Confidence | Medium — value is tuned for stability, not physically derived |

### 4d. Viscosity
| | |
|---|---|
| Equation | `visc = A_visc * laplacian(zeta)` |
| Parameters | `A_visc = 2e-4` |
| Theory | Laplacian viscosity for numerical stability. Real ocean viscosity is much smaller. |
| Status | **Active** |
| Confidence | Low — purely numerical, not physical. Interacts with cos(lat) metric correction. |
| Gap | Magnitude may need retuning after cos(lat) correction (see issue #25) |

### 4e. Buoyancy (density-driven)
| | |
|---|---|
| Equation | `buoyancy = -dRho/dx * 0.5 * invDx` where dRho = -alpha_T * dT/dx + beta_S * dS/dx |
| Parameters | `alpha_T = 0.05` (thermal expansion), `beta_S = 0.8` (haline contraction) |
| Theory | Thermal wind relation — horizontal density gradients drive vertical shear |
| Status | **Active** |
| Confidence | Medium |

### 4f. Ekman heat transport
| | |
|---|---|
| Equation | `ekmanAdvec = u_ek * dT/dx + v_ek * dT/dy` |
| Parameters | Ekman velocity computed from tau_x, tau_y via `v_e = -tau_x / (rho*f*H_ek)`. RMS-scaled to 0.3 model units. |
| Data | `wind_stress_1deg.json` tau_x, tau_y |
| Theory | Ekman (1905) — wind-driven surface transport perpendicular to wind direction |
| Status | **Active** |
| Confidence | Medium |

### 4g. Poisson solver (streamfunction from vorticity)
| | |
|---|---|
| Equation | `nabla^2(psi) = zeta` with cos(lat) metric correction |
| Parameters | None (exact solve) |
| Theory | Definition of streamfunction for non-divergent flow |
| Status | **Active** (GPU FFT solver) |
| Confidence | High — exact solution via FFT |

---

## 5. TEMPERATURE EQUATION

### 5a. Advection
| | |
|---|---|
| Equation | `advec = J(psi, T) + Ekman` (Arakawa Jacobian + Ekman transport) |
| Parameters | None (follows from circulation) |
| Status | **Active** |
| Confidence | High |

### 5b. Thermal diffusion
| | |
|---|---|
| Equation | `diff = kappa_diff * laplacian(T)` |
| Parameters | `kappa_diff = 2.5e-4` |
| Theory | Parameterizes sub-grid mixing. Real ocean: mesoscale eddies transport heat. |
| Status | **Active** |
| Confidence | Low — value is tuned. Currently too strong (spreads heat to poles faster than radiation can remove it). |
| Gap | Should scale with cos(lat) for physical consistency |

### 5c. Land-ocean heat exchange
| | |
|---|---|
| Equation | `landFlux = landHeatK * (landT - SST) * (nOcean/4)` clamped to [-0.5, 0.5] |
| Parameters | `landHeatK = 0.02` |
| Data | Land temperature from `albedo_1deg.json` + elevation lapse rate |
| Status | **Active** |
| Confidence | Low |

---

## 6. SALINITY

### 6a. Salinity restoring
| | |
|---|---|
| Equation | `salRestore = salRestoringRate * (salClim - S)` |
| Parameters | `salRestoringRate = 0.005`. Climatology from WOA23 or zonal formula fallback. |
| Data | `salinity_1deg.json` — WOA23 surface salinity, 1991-2020 |
| Theory | Newtonian relaxation toward observed climatology (Haney 1971) |
| Status | **Active** |
| Confidence | High for data, medium for restoring rate |

### 6b. Freshwater forcing (meltwater)
| | |
|---|---|
| Equation | `if y > 0.75: fwSal = -freshwaterForcing * 3 * (y-0.75) * 4` |
| Parameters | `freshwaterForcing = 0` (user slider) |
| Theory | Hosing experiments — freshwater into N Atlantic suppresses AMOC |
| Status | **Active** (user-controlled) |
| Confidence | High for mechanism |
| Gap | **No precipitation-evaporation (P-E) salinity flux.** Real ocean gains freshwater from rain and loses from evaporation. We have precipitation data but don't use it for salinity. |

---

## 7. TWO-LAYER VERTICAL STRUCTURE

### 7a. Variable mixed layer depth
| | |
|---|---|
| Equation | `mixedLayerDepth = mldBase + mldACC + mldSubpolar` (latitude-dependent profile) |
| Parameters | Profile parameters (tuned) |
| Theory | MLD varies from ~20m in tropics to ~500m in Southern Ocean deep convection zones |
| Status | **Active** |
| Confidence | Medium |

### 7b. Vertical mixing
| | |
|---|---|
| Equation | `vertExchange = gamma * (T_surface - T_deep) * hasDeep` |
| Parameters | `gamma_mix = 0.001` (background), `gamma_deep_form = 0.05` (enhanced at high lat when surface denser than deep) |
| Theory | Wind mixing + convective overturning when surface water is denser than deep water |
| Status | **Active** |
| Confidence | Medium |

### 7c. Deep layer dynamics
| | |
|---|---|
| Equation | Separate vorticity equation with own Poisson solve, coupled to surface via `F_couple` |
| Parameters | `kappa_deep = 2e-5`, `r_deep = 0.1`, `F_couple_s = 0.5`, `F_couple_d = 0.0125` |
| Status | **Active** |
| Confidence | Low — coupling coefficients are ad hoc |

---

## 8. ATMOSPHERE

### 8a. 1-layer energy balance
| | |
|---|---|
| Equation | `dT_air/dt = kappa_atm * laplacian(T_air) + gamma * (T_surface - T_air)` |
| Parameters | `kappa_atm = 3e-3`, `gamma_oa = 0.005` (ocean-air), `gamma_ao = 0.001` (air-ocean feedback), `gamma_la = 0.01` (land-air) |
| Theory | Simple diffusive atmosphere with surface exchange |
| Status | **Active** (CPU-side, between GPU readback cycles) |
| Confidence | Low — no wind-driven transport, no Hadley/Ferrel cells, no moisture |
| Gap | **No moisture/latent heat.** Real atmosphere carries ~30% of poleward heat via moisture. No evaporation, condensation, or precipitation feedback. |

---

## 9. DATA INVENTORY

### Spatial fields (loaded into model)

| File | Source | Period | Used for | Remapped? |
|------|--------|--------|----------|-----------|
| `bathymetry_1deg.json` | ETOPO1 | Static | Ocean depth, land elevation | Bilinear |
| `sst_global_1deg.json` | NOAA OI SST | 1991-2020 LTM | Init + validation (RMSE target) | Bilinear |
| `deep_temp_1deg.json` | WOA23 1000m | Climatology | Deep layer init + validation | Bilinear |
| `salinity_1deg.json` | WOA23 surface | 1991-2020 | Salinity restoring target | Bilinear |
| `wind_stress_1deg.json` | NCEP Reanalysis | 1991-2020 LTM | Wind curl + Ekman velocity | Bilinear |
| `albedo_1deg.json` | MODIS MCD43A3 | 2020-2023 mean | Land temperature | Via GEE |
| `precipitation_1deg.json` | CHIRPS v2 | 2015-2023 mean | Land cloud fraction | Via GEE |
| `cloud_fraction_1deg.json` | MODIS MOD08_M3 | 2020-2023 mean | Cloud RMSE validation | Via GEE |
| `cloud_types_1deg.json` | MODIS MOD08_M3 | 2020-2023 mean | Low/high cloud validation | Via GEE |

### Spatial fields (downloaded, NOT used)

| File | Source | Could validate/drive |
|------|--------|---------------------|
| `earth-data/snow_cover.csv` | MODIS MOD10C1 | Seasonal land albedo |
| `earth-data/ndvi.csv` | MODIS NDVI | Land thermal inertia, evapotranspiration |
| `earth-data/land_surface_temp.csv` | MODIS LST | Land temperature validation |
| `godas_currents_1deg.json` | GODAS | Current speed/direction validation |

### Time series (downloaded, NOT used in physics)

| File | Source | Period | Potential use |
|------|--------|--------|--------------|
| `rapid_amoc_monthly.csv` | RAPID 26.5N | 2004-present | AMOC validation (shown on chart) |
| `co2_annual_mlo.csv` | Mauna Loa | 1958-present | Drive globalTempOffset |
| `hadcrut5_global_annual.csv` | HadCRUT5 | 1850-present | Global temp validation |
| `hadsst4_global.csv` | HadSST4 | 1850-present | SST trend validation |
| `sea_ice_arctic.csv` | NSIDC | 1978-present | Ice extent validation |
| `ocean_heat_content.csv` | EN4 | 1950-present | Deep warming validation |

---

## 10. KNOWN GAPS (ranked by impact)

### High impact
1. **No moisture/latent heat** — atmosphere has no water vapor. Missing ~30% of poleward heat transport, all precipitation-salinity coupling, and evaporative cooling.
2. **No snow-albedo on land** — data exists (MODIS snow cover), not wired. Affects seasonal cycle amplitude at high latitudes.
3. **Cirrus vs deep convection not separated** — both lumped as "high clouds" with averaged radiative properties. Cirrus warms strongly, deep convection is neutral.
4. **No P-E salinity flux** — ocean gains/loses freshwater from precipitation and evaporation. We have precipitation data but only use it for land clouds.

### Medium impact
5. **No ocean surface albedo** — 6% reflection ignored globally.
6. **NH mid-latitude clouds too low** — model gives ~0.30 at 50°N, observed 0.90.
7. **Thermal diffusion too strong** — kappa_diff spreads heat to poles faster than radiation removes it. Contributes to warm pole bias.
8. **Annual mean data only** — all spatial fields are annual averages. No seasonal precipitation, albedo, or cloud variation.

### Low impact
9. **No diurnal cycle** — seasons only.
10. **Atmosphere is pure diffusion** — no Hadley/Ferrel cells, no jet stream.
11. **No sea ice dynamics** — ice is just "cold ocean," no thickness, drift, or brine rejection.
