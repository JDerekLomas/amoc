# AMOC Science 2024-2026: What We're Missing

## The Honest Assessment

**Our simulation is a wind-driven ocean model with thermal coupling, not an AMOC model.** We have the "thermo" but not the "haline." The single most important AMOC mechanism — the salt-advection feedback — requires explicit salinity transport that we completely lack.

## What We Capture Correctly
- Wind-driven gyres, western boundary currents (Gulf Stream, Kuroshio) — this IS the core physics
- Beta effect and western intensification
- ACC through Drake Passage
- ~80% of Gulf Stream transport (wind-driven part)
- Equator-to-pole thermal gradient

## What We Fundamentally Cannot Capture

### 1. No Salinity Field (THE critical omission)
Without salinity:
- No salt-advection feedback (the mechanism behind AMOC tipping)
- No Stommel bifurcation (our freshwater slider is cosmetic — it cools instead of freshening)
- No haline contribution to density (~40% of AMOC driving)
- Cannot represent the FovS early warning signal from van Westen et al.

The simplest model that correctly captures AMOC is the Stommel 2-box with BOTH T and S. We're more spatially complex but less physically correct because we omit S.

### 2. No True Overturning Streamfunction
Real AMOC is Psi(y,z) — zonally integrated. Our psi is horizontal only.

### 3. No Density-Driven Deep Flow
Deep layer driven by passive interfacial friction, not density gradients. Code review confirmed: **deep vorticity equation has no buoyancy forcing term.**

### 4. No Nordic Sea Overflows
Denmark Strait (~3 Sv) and Faroe Bank Channel (~2 Sv) overflows require sill hydraulics. Our grid has land where these sills are but no overflow mechanism.

### 5. No Isopycnal Upwelling
Southern Ocean closes AMOC via adiabatic upwelling along tilted isopycnals. Our 2-layer coupling can't represent this.

## Key Recent Papers

### AMOC Tipping
- **van Westen et al. 2024** — First AMOC tipping in comprehensive ESM (CESM). Physics-based early warning: FovS at 34°S is NEGATIVE in reanalysis → bistable regime. [Science Advances](https://www.science.org/doi/10.1126/sciadv.adk1189)
- **van Westen et al. 2025** — First AMOC collapse in eddying (1/4°) model. Eddies partly stabilize but don't prevent collapse. [GRL](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL114532)
- **Portmann et al. 2026** — Observational constraint: ~51±8% AMOC weakening by 2100. Dominated by South Atlantic salinity biases. [Science Advances](https://www.science.org/doi/10.1126/sciadv.adx4298)

### AMOC Observations
- **RAPID array (26.5°N)**: 1.0 Sv/decade weakening (2004-2023). Mean 16.9 Sv. [rapid.ac.uk](https://rapid.ac.uk/)
- **OSNAP (subpolar)**: Mean 16.5±3.3 Sv. No statistically significant trend 2004-2024. [o-snap.org](https://www.o-snap.org/)

### Southern Ocean Push vs Pull
- **Jackson et al. 2025 (Nature)**: Under extreme forcing, AMOC weakens but NEVER fully collapses to zero. Southern Ocean wind-driven upwelling sustains residual AMOC in ALL 34 models tested. [Nature](https://www.nature.com/articles/s41586-024-08544-0)

### Gulf Stream ≠ AMOC
- Gulf Stream: ~150 Sv total, ~80% wind-driven, ~20% thermohaline
- AMOC: ~17 Sv of that is density-driven overturning
- Wind-driven part barely changes under climate change; thermohaline part weakens substantially
- [2024](https://www.nature.com/articles/s43247-024-01907-5)

## Ranked Improvements We Could Make

1. **Add salinity field** (HIGH) — Two new 2D arrays, linear EOS rho(T,S), salinity advection-diffusion. Enables salt-advection feedback. ~2x computation of temperature (already done for T, replicate for S).

2. **Add deep buoyancy forcing** (HIGH) — Missing term in deep vorticity equation. Without this, AMOC cannot emerge from physics.

3. **Fix AMOC diagnostic** (MEDIUM) — Compute zonal integral of transport difference, not velocity at one point.

4. **Add density-dependent vertical mixing** (MEDIUM) — gamma proportional to density difference between layers.

5. **Parameterize overflow entrainment** (LOW-MEDIUM) — Enhanced deep water formation near Greenland-Scotland Ridge.

## GitHub Resources
- [Qiyu-Song/AMOC-Box-Model](https://github.com/Qiyu-Song/AMOC-Box-Model) — Python Stommel box model
- [cdr30/RapidMoc](https://github.com/cdr30/RapidMoc) — AMOC diagnostics from GCM output
- [niccolozanotti/stommel-model](https://github.com/niccolozanotti/stommel-model) — Stommel with data assimilation
