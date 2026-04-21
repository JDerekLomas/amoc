# AMOC Wiggum Loop — Claude Escalation (Iteration 5)

## Situation
The Ralph Wiggum loop has run 5 iterations of physics-aligned parameter tuning
using Gemini 3.1 Flash Lite, but has stalled at a composite score of 20.0%.

Parameter tuning alone cannot close the gap. We need your help reviewing the
**physics code** in `v4-physics/index.html` for structural improvements.

## Current Scorecard
- T1 Conservation: FAIL
  - FAIL: temperature_range — [-10.0, 40.0]°C
  - OK: equator_pole_gradient — 22.3°C
  - OK: hemispheric_symmetry — NH=14.3°C, SH=13.3°C, diff=1.0°C
  - FAIL: deep_colder_than_surface — 1/15 bands have deep warmer than surface
  - OK: amoc_positive — AMOC index: 4.30e-3 (non-dim)
- T2 Structure: 83%
  - FAIL: western_intensification — Gulf Stream max: 0.912, NA interior max: 1.071, ratio: 0.9x
  - OK: subtropical_gyre_exists — NA subtrop psi: mean=-0.0311, range=0.0405, cells=1173
  - OK: acc_eastward_flow — Mean zonal flow at 58°S: -0.1486 (360 ocean cells)
  - OK: deep_water_formation — Polar deep: 2.7°C (84 cells), Tropical deep: 4.3°C (278 cells)
  - OK: poleward_heat_transport — Mean v*T at 30°N: 2.0268 (198 cells)
  - OK: atlantic_warmer_than_pacific — 40°N Atlantic: 16.0°C (61), Pacific: 15.9°C (80)
- T4 RMSE: 4.82°C
- AMOC: 0.0 Sv

## Physicist's Last Hypothesis
Failure of western boundary current intensification and excessive poleward heat transport leading to polar temperature inversion.
- H1: The lateral viscosity (A_visc) is too high, effectively 'smearing' the western boundary currents and preventing the formation of tight, high-velocity jets. This leads to inefficient heat transport at the boundaries and forces the model to rely on excessive eddy diffusion, which causes the observed poleward heat transport errors. [high]
- H2: The coupling between the surface and deep layers (gamma_mix) is insufficient to transport heat downward in the tropics, while the deep water formation (gamma_deep_form) is too weak to effectively flush the deep basins with cold polar water, resulting in the deep layer being warmer than the surface in some regions. [medium]
- H3: The bottom friction (r_friction) is too low to properly constrain the barotropic flow, allowing the interior ocean to develop unrealistic circulation patterns that interfere with the formation of the western boundary currents. [low]

## What To Look For
1. Are there physics terms missing from the equations that would fix the structural (T2) failures?
2. Is the boundary condition handling causing artifacts?
3. Is the wind forcing pattern realistic enough?
4. Could the ice-albedo feedback be improved?
5. Is the deep water formation parameterization too simple?

## Screenshots
Review the iteration 5 screenshots in `./screenshots/wiggum/` — temperature, streamfunction, speed, and deep temperature views.

## Parameters (current best)
```json
{
  "S_solar": 100,
  "A_olr": 40,
  "B_olr": 2,
  "kappa_diff": 0.00025,
  "alpha_T": 0.05,
  "r_friction": 0.052,
  "A_visc": 0.00022295,
  "gamma_mix": 0.01,
  "gamma_deep_form": 0.5,
  "kappa_deep": 0.00002,
  "F_couple_s": 0.5,
  "r_deep": 0.1,
  "windStrength": 1
}
```

After reviewing, please make code changes to `v4-physics/index.html` that would
improve the physics, then we'll re-run the Wiggum loop with the updated code.
