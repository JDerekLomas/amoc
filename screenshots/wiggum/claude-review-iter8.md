# AMOC Wiggum Loop — Claude Escalation (Iteration 8)

## Situation
The Ralph Wiggum loop has run 8 iterations of physics-aligned parameter tuning
using Gemini 3.1 Flash Lite, but has stalled at a composite score of 20.0%.

Parameter tuning alone cannot close the gap. We need your help reviewing the
**physics code** in `v4-physics/index.html` for structural improvements.

## Current Scorecard
- T1 Conservation: FAIL
  - FAIL: temperature_range — [-10.0, 40.0]°C
  - FAIL: equator_pole_gradient — -2.6°C
  - FAIL: hemispheric_symmetry — NH=13.8°C, SH=22.7°C, diff=8.9°C
  - FAIL: deep_colder_than_surface — 1/15 bands have deep warmer than surface
  - FAIL: amoc_positive — 0.0 Sv
- T2 Structure: 50%
  - FAIL: western_intensification — West max: 0.07, East max: 4.38, ratio: 0.0x
  - FAIL: subtropical_gyre_exists — Mean psi in NA subtropics: 0.046
  - OK: southern_ocean_flow — Mean zonal flow at 60°S: -0.9146
  - OK: deep_water_formation_signature — Polar deep: 0.6°C, Tropical deep: 2.0°C
- T4 RMSE: 0.00°C
- AMOC: 0.0 Sv

## Physicist's Last Hypothesis
Complete collapse of western intensification and AMOC-driven overturning circulation
- H1: The beta-effect or wind-stress curl coupling is insufficient to overcome the current lateral viscosity (A_visc), preventing the formation of tight western boundary currents and suppressing the poleward heat transport required for AMOC. [high]
- H2: The thermohaline coupling (gamma_deep_form) is too weak to create the density gradient necessary to drive deep water formation, resulting in a stagnant deep ocean and zero AMOC. [medium]
- H3: The bottom friction (r_friction) is too high, acting as a drag on the entire flow field and preventing the development of the high-velocity jets characteristic of western boundary currents. [medium]

## What To Look For
1. Are there physics terms missing from the equations that would fix the structural (T2) failures?
2. Is the boundary condition handling causing artifacts?
3. Is the wind forcing pattern realistic enough?
4. Could the ice-albedo feedback be improved?
5. Is the deep water formation parameterization too simple?

## Screenshots
Review the iteration 8 screenshots in `./screenshots/wiggum/` — temperature, streamfunction, speed, and deep temperature views.

## Parameters (current best)
```json
{
  "S_solar": 100,
  "A_olr": 40,
  "B_olr": 2,
  "kappa_diff": 0.00025,
  "alpha_T": 0.05,
  "r_friction": 0.04,
  "A_visc": 0.00012005,
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
