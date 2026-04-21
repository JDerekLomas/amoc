# AMOC Wiggum Loop — Claude Escalation (Iteration 3)

## Situation
The Ralph Wiggum loop has run 3 iterations of physics-aligned parameter tuning
using Gemini 3.1 Flash Lite, but has stalled at a composite score of 70.8%.

Parameter tuning alone cannot close the gap. We need your help reviewing the
**physics code** in `v4-physics/index.html` for structural improvements.

## Current Scorecard
- T1 Conservation: FAIL
  - FAIL: temperature_range — [-10.0, 40.0]°C
  - OK: equator_pole_gradient — 30.5°C
  - OK: hemispheric_symmetry — NH=11.2°C, SH=16.0°C, diff=4.8°C
  - FAIL: deep_colder_than_surface — 3/15 bands have deep warmer than surface
  - OK: amoc_positive — AMOC index: 3.26e-3 (non-dim)
- T2 Structure: 67%
  - FAIL: western_intensification — Gulf Stream max: 0.116, NA interior max: 0.094, ratio: 1.2x
  - OK: subtropical_gyre_exists — NA subtrop psi: mean=-0.0030, range=0.0055, cells=1173
  - OK: acc_eastward_flow — Mean zonal flow at 58°S: 0.1207 (360 ocean cells)
  - OK: deep_water_formation — Polar deep: 2.8°C (84 cells), Tropical deep: 4.2°C (278 cells)
  - OK: poleward_heat_transport — Mean v*T at 30°N: 1.9567 (198 cells)
  - FAIL: atlantic_warmer_than_pacific — 40°N Atlantic: 9.0°C (61), Pacific: 20.1°C (80)
- T4 RMSE: 4.58°C
- AMOC: 0.0 Sv

## Physicist's Last Hypothesis
Severe polar temperature bias and deep-layer thermal inversion indicating failure of the thermohaline overturning loop.
- H1: The deep-layer thermal inversion suggests that the coupling coefficient (gamma_mix) is too low to effectively transport heat from the surface to the deep ocean, while the polar temperature bias suggests the deep water formation (gamma_deep_form) is not efficiently exporting cold water away from the poles, leading to local stagnation and excessive cooling. [high]
- H2: The weak western intensification ratio (1.2x) and poor Atlantic-Pacific temperature contrast suggest that the lateral viscosity (A_visc) is too high, dampening the boundary currents and preventing the efficient poleward heat transport required to warm the North Atlantic. [medium]

## What To Look For
1. Are there physics terms missing from the equations that would fix the structural (T2) failures?
2. Is the boundary condition handling causing artifacts?
3. Is the wind forcing pattern realistic enough?
4. Could the ice-albedo feedback be improved?
5. Is the deep water formation parameterization too simple?

## Screenshots
Review the iteration 3 screenshots in `./screenshots/wiggum/` — temperature, streamfunction, speed, and deep temperature views.

## Parameters (current best)
```json
{
  "S_solar": 100,
  "A_olr": 40,
  "B_olr": 2,
  "kappa_diff": 0.00025,
  "alpha_T": 0.05,
  "r_friction": 0.052,
  "A_visc": 0.00065,
  "gamma_mix": 0.0286,
  "gamma_deep_form": 0.65,
  "kappa_deep": 0.00002,
  "F_couple_s": 0.5,
  "r_deep": 0.1,
  "windStrength": 1
}
```

After reviewing, please make code changes to `v4-physics/index.html` that would
improve the physics, then we'll re-run the Wiggum loop with the updated code.
