# AMOC Wiggum Loop — Claude Escalation (Iteration 2)

## Situation
The Ralph Wiggum loop has run 2 iterations of physics-aligned parameter tuning
using Gemini 3.1 Flash Lite, but has stalled at a composite score of 20.0%.

Parameter tuning alone cannot close the gap. We need your help reviewing the
**physics code** in `v4-physics/index.html` for structural improvements.

## Current Scorecard
- T1 Conservation: FAIL
  - FAIL: temperature_range — [-10.0, 40.0]°C
  - FAIL: equator_pole_gradient — 9.8°C
  - FAIL: hemispheric_symmetry — NH=10.2°C, SH=21.4°C, diff=11.2°C
  - FAIL: deep_colder_than_surface — 2/15 bands have deep warmer than surface
  - OK: amoc_positive — AMOC index: 6.56e-3 (non-dim)
- T2 Structure: 67%
  - FAIL: western_intensification — Gulf Stream max: 1.061, NA interior max: 1.049, ratio: 1.0x
  - OK: subtropical_gyre_exists — NA subtrop psi: mean=-0.0329, range=0.0409, cells=1173
  - OK: acc_eastward_flow — Mean zonal flow at 58°S: -0.1646 (360 ocean cells)
  - OK: deep_water_formation — Polar deep: 2.8°C (84 cells), Tropical deep: 4.3°C (278 cells)
  - OK: poleward_heat_transport — Mean v*T at 30°N: 2.0608 (198 cells)
  - FAIL: atlantic_warmer_than_pacific — 40°N Atlantic: 15.9°C (61), Pacific: 23.0°C (80)
- T4 RMSE: 8.78°C
- AMOC: 0.0 Sv

## Physicist's Last Hypothesis
Severe hemispheric asymmetry and failure of western boundary current intensification
- H1: The model lacks sufficient resolution or parameterization of the beta effect (planetary vorticity gradient) relative to the lateral viscosity, preventing the formation of tight, intensified western boundary currents and leading to an unrealistic heat distribution. [high]
- H2: The hemispheric asymmetry suggests an imbalance in the thermohaline coupling or surface forcing, where the Southern Ocean is failing to export cold water effectively, causing deep-layer warming and disrupting the global overturning circulation. [medium]

## What To Look For
1. Are there physics terms missing from the equations that would fix the structural (T2) failures?
2. Is the boundary condition handling causing artifacts?
3. Is the wind forcing pattern realistic enough?
4. Could the ice-albedo feedback be improved?
5. Is the deep water formation parameterization too simple?

## Screenshots
Review the iteration 2 screenshots in `./screenshots/wiggum/` — temperature, streamfunction, speed, and deep temperature views.

## Parameters (current best)
```json
{
  "S_solar": 100,
  "A_olr": 40,
  "B_olr": 2,
  "kappa_diff": 0.00025,
  "alpha_T": 0.05,
  "r_friction": 0.052,
  "A_visc": 0.000245,
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
