# AMOC Simulation — Knowledge Bank

## Physical Concepts

### Barotropic Vorticity Equation
The core equation governing horizontal ocean circulation:
```
dq/dt + J(ψ, q) = curl(τ) − rζ + A∇²ζ
```
- **q = ζ + βy**: potential vorticity (relative + planetary)
- **J(ψ, q)**: Arakawa Jacobian — conserves energy AND enstrophy (critical for stability)
- **curl(τ)**: wind stress curl — the engine driving gyres
- **rζ**: linear bottom friction (Stommel dynamics)
- **A∇²ζ**: lateral viscosity (Munk dynamics)

### Western Boundary Intensification
Why the Gulf Stream and Kuroshio are on the *western* side of ocean basins:
- **Stommel model**: friction balances beta effect → boundary layer width ∝ r/β
- **Munk model**: viscosity balances beta → boundary layer width ∝ (A/β)^(1/3)
- In this sim: both mechanisms operate. Reducing r sharpens Stommel layer; reducing A sharpens Munk layer
- **Key diagnostic**: max speed at western boundary should be 2-5x eastern interior

### AMOC (Atlantic Meridional Overturning Circulation)
Thermohaline-driven deep overturning:
- Warm surface water flows north in the Atlantic
- Cools at high latitudes → becomes dense → sinks
- Returns south as cold deep water
- Transports ~1.3 PW of heat northward
- **In this sim**: parameterized via `alpha_T` (buoyancy coupling) and `gamma_deep_form` (deep water formation rate)
- **AMOC index**: non-dimensional (psi-gradient units), typical range 1e-4 to 1e-2

### Radiative Balance
```
dT/dt = S·cos(θ_z) − (A_olr + B_olr·T) + diffusion + advection
```
- **Equilibrium**: T_eq = (S·cos_z − A_olr) / B_olr
  - Equator (cos_z ≈ 0.9): T ≈ (S×0.9 − A) / B
  - Pole (cos_z ≈ 0.2): T ≈ (S×0.2 − A) / B
- **S_solar**: controls incoming energy (higher = warmer everywhere)
- **A_olr**: OLR offset (higher = colder equilibrium)
- **B_olr**: restoring strength (higher = temperatures pulled harder toward equilibrium)
- **Ice-albedo**: αice reduces solar absorption at |lat| > 40° when T < 0°C → positive feedback

### Two-Layer Ocean
- **Surface** (100m): fast response, wind-driven, holds most kinetic energy
- **Deep** (4000m): 40x thermal inertia, thermohaline-driven, fills slowly from polar source
- **Coupling**: γ_mix (vertical mixing), F_couple (momentum coupling)
- **F_couple_d ≈ (H_surface/H_deep) × F_couple_s** by conservation of momentum

### Antarctic Circumpolar Current (ACC)
- Only ocean current that circles the globe (no land barrier at ~60°S)
- Driven by Southern Hemisphere westerly winds
- Drake Passage (opened ~35 Ma) enables it
- In the sim: check eastward flow at ~58°S

## Numerical Concepts

### Grid
- 360×180 (1° resolution), lon -180° to 180°, lat -80° to 80°
- i=0 → -180° (dateline), i=180 → 0° (prime meridian)
- j=0 → -80° (Antarctic), j=179 → 80° (Arctic)
- Conversion: `i = (lon + 180) / 360 × 359`, `j = (lat + 80) / 160 × 179`

### Stability Constraints
- **CFL**: max_vel × dt × nx < 1 (or equivalently max_vel × dt / dx < 1)
  - With dt=5e-5 and nx=360: max safe velocity ≈ 55
- **Diffusive stability**: κ × dt / dx² < 0.5
- **Clamping**: temp [-10, 40]°C surface, [-5, 30]°C deep, vorticity ±500

### Poisson Solver
- Inverts ∇²ψ = ζ to get streamfunction from vorticity
- Iterative Jacobi: 60 iterations (surface), 20 (deep)
- Under-convergence → noisy velocity field, affects all diagnostics

### Arakawa Jacobian
- Conserves both energy AND enstrophy (unlike simple centered differences)
- Critical for long-term stability — prevents spectral energy cascade to grid scale
- Three-point stencil in both directions

## Modeling Concepts

### Parameter Tuning Philosophy
- Parameters should be **physically interpretable**, not just numerically convenient
- **Compensating errors** (changing S and A together) = curve-fitting, not physics
- Dimensional analysis constrains reasonable ranges:
  - Friction damping time: 1/r ≈ 25 steps (r=0.04)
  - Munk width: (A/β)^(1/3) should be 2-5 grid cells
  - Thermal diffusion scale: √(κ × T_year)

### What the Simulation CAN capture
- Large-scale gyre structure (Sverdrup balance)
- Western boundary current intensification
- Meridional overturning circulation
- Equator-to-pole temperature gradient
- Seasonal cycle
- Ice-albedo feedback
- Response to paleoclimate scenarios

### What the Simulation CANNOT capture
- Mesoscale eddies (need ~0.1° resolution)
- Realistic salinity (only parameterized via freshwater forcing)
- Atmospheric coupling (fixed wind pattern, no weather)
- Realistic mixed layer depth
- Tidal mixing
- Sea ice dynamics (only thermodynamic albedo effect)

### Key Diagnostics (what to measure)
1. **Zonal-mean SST by latitude** — compare to NOAA OI SST
2. **Western boundary current speed ratio** — west/east > 1.5
3. **Meridional heat transport** (v×T at 30°N) — must be poleward
4. **AMOC index** — non-dimensional, should be > 1e-5
5. **Deep stratification** — polar deep < tropical deep
6. **Atlantic-Pacific asymmetry** at 40°N — Atlantic warmer
7. **ACC flow** at 58°S — significant eastward component

## Reference Data

### NOAA OI SST v2 (1991-2020 annual mean)
- File: `sst_global_1deg.json` (key: `sst`, 360×160 array)
- Zonal means: -1.5°C (70°S) to 27.8°C (10°N), global mean 14.2°C
- Most reliable: tropics and NH midlatitudes
- Less reliable: polar regions (sea ice, sparse sampling)

### WOA23 (annual mean, 1000m depth)
- File: `deep_temp_1deg.json` (key: `temp`, 360×160 array)
- Range: -0.3°C (70°N) to 5.3°C (30°N)
- Sparse at poles, interpolated in Southern Ocean

## Wiggum Loop Architecture

### Agent Team (core — every iteration)
1. **Physicist**: sees screenshots + data, generates competing hypotheses
2. **Tuner**: translates winning hypothesis to parameter changes (max 3 params, 30% max step)
3. **Validator**: checks physical consistency, catches compensating errors

### Agent Team (on-demand — triggered by conditions)
4. **Numerical Analyst**: triggered when T1 fails — checks for computational artifacts
5. **Skeptic**: every 4th iteration — audits for curve-fitting and parameter drift
6. **Observational Scientist**: when polar errors dominate — assesses reference data quality
7. **Literature Agent**: when params hit bounds — checks published model values
8. **Claude**: when stalled 3 iterations — structural code review

### Evaluation Tiers
- **T1 Conservation** (gate): temperature range, gradient, stratification, AMOC positive
- **T2 Structure** (35%): western intensification, gyre, ACC, deep formation, heat transport, basin asymmetry
- **T3 Sensitivity** (final only): freshwater weakens AMOC, cooling cools ocean
- **T4 Quantitative** (35%): zonal-mean SST RMSE vs NOAA observations
