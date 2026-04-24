# Air Temp view: needs real atmospheric dynamics or clear 'experimental' label

## Problem
The Air Temp view shows a smoothed copy of SST — the atmosphere layer has no real dynamics. Over land it's flat tan from a crude formula. Misleading without caveat.

## Current Implementation
- 1-layer energy balance: `dT_air/dt = kappa_atm * laplacian(T_air) + gamma * (T_surface - T_air)`
- Diagnostic only — no feedback to ocean (removed because it destabilized)
- No wind-driven advection
- No moisture, latent heat, or convection
- Land temperature is `28 - 0.55 * |lat|` (no seasonal cycle, no elevation)

## Options

### Option A: Make it useful (minimal effort)
- Use `landTempField` (already computed for land rendering with altitude lapse rate) as surface boundary over land
- Add wind-driven advection: `u_atm * dT/dx` from thermal wind or prescribed wind field
- Gets realistic continental/maritime contrast and seasonal effects

### Option B: Make it honest
- Rename to "Air Temp (beta)" or add subtitle
- Keep as diagnostic layer

### Option C: Full atmospheric energy balance (long-term, Issue #3)
- Wind-driven moisture transport, latent heat flux, convective adjustment
- Feeds back into ocean (sensible + latent heat flux)
- Required for realistic cloud formation

## Recommendation
Option A first — wire in landTempField + simple advection. 80% of visual improvement, minimal code.
