# Comprehensive Technical Comparison of Ocean Circulation Models

*A reference for building simplified 2-layer barotropic ocean simulations*

---

## 1. Major Ocean General Circulation Models (OGCMs)

### 1.1 MOM6 (GFDL / NOAA)

**Equations solved:** Boussinesq hydrostatic primitive equations in vector-invariant form. Tracer equations (temperature, salinity) in flux form. Hydrostatic pressure gradient via analytic integration. Reference density: rho_0 = 1035 kg/m^3.

**Grid:** Arakawa C-grid. Tripolar grids for global domains. Vertical: Arbitrary Lagrangian-Eulerian (ALE) remapping — hybrid z*/isopycnal/terrain-following. OM4.0 uses 75 vertical levels.

**Resolution:** OM4p25 (~1/4 degree), OM4p5 (~1/2 degree), 1 degree for CMIP. Tracer advection: quasi-third-order PPM with monotonic limiters.

**Key parameterizations:**
- Mesoscale eddies: GM/Redi with interface height diffusion (Ferrari et al. 2010). GM coeff ~600-800 m^2/s at 1 degree.
- Lateral viscosity: Biharmonic Smagorinsky (C_smag ~0.06-0.15). OM4p5 uses Laplacian viscosity as max of Smagorinsky dynamic viscosity (coeff 0.15) and static floor.
- Vertical mixing: KPP boundary layer; ePBL scheme; interior from shear instability, internal wave breaking, double diffusion. Background K_v: 2e-6 m^2/s at equator to 1.15e-5 m^2/s at 60°. Background A_v: 1e-4 m^2/s.
- Bottom drag: Quadratic, C_d = 0.003. 20% of KE lost to bottom drag converts to PE via near-bottom mixing.

**Sources:**
- [MOM6 GitHub](https://github.com/NOAA-GFDL/MOM6)
- [MOM6 ReadTheDocs](https://mom6.readthedocs.io/en/main/api/generated/pages/Specifics.html)
- [GFDL MOM page](https://www.gfdl.noaa.gov/mom-ocean-model/)
- [OM4.0 — Adcroft et al. 2019](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001726)

---

### 1.2 NEMO (Nucleus for European Modelling of the Ocean)

**Equations solved:** Primitive equations (hydrostatic, Boussinesq) for u, v, T, S, SSH.

**Grid:** Arakawa C-grid. ORCA tripolar grid globally. Vertical: full/partial step z, s-coordinate, or hybrid.

**Resolution:** ORCA025 (1/4°), ORCA12 (1/12°), ORCA1 (1°), ORCA2 (2°).

**Key parameterizations:**
- Lateral: Harmonic and biharmonic viscosity/diffusion, rotatable along geopotential or neutral directions.
  - ORCA2: A_h = 4e4 m^2/s poleward of 20°, decreasing to 2e3 m^2/s at equator.
  - ORCA025: Biharmonic isopycnal viscosity ~ -1.5e11 m^4/s.
  - Smagorinsky: A_smag = (C_smag/pi)^2 * L^2 * |D|
- Vertical: TKE closure (default); KPP available.
- Convection: Non-penetrative adjustment, enhanced diffusion (large K_v when N^2 < 0), or TKE.
- Bottom friction: Linear (r = 4e-4 m/s, ~115-day timescale) or quadratic (C_D = 1e-3, with background TKE e_b = 2.5e-3 m^2/s^2). Optional log-layer: C_D = [kappa / ln(0.5*e3/z0)]^2.

**Sources:**
- [NEMO documentation](https://www.nemo-ocean.eu/doc/node4.html)
- [NEMO lateral mixing](https://www.nemo-ocean.eu/doc/node63.html)
- [NEMO bottom friction](https://www.nemo-ocean.eu/doc/node70.html)
- [NEMO framework](https://www.nemo-ocean.eu/framework/components/engines/)

---

### 1.3 POP (Parallel Ocean Program) — CESM

**Equations:** 3D primitive equations (hydrostatic, Boussinesq) in z-coordinates.

**Grid:** Arakawa B-grid. Displaced-pole grid (singularity over Greenland). Standard grids: gx1v6 (~1°, ~0.3° near equator, 60 levels), gx3v7 (~3°). Note: Being replaced by MOM6 in CESM3.

**Key parameterizations:** GM/Redi, KPP, anisotropic viscosity, overflow parameterization for Nordic Sea dense water.

**Sources:**
- [POP Reference Manual](https://www2.cesm.ucar.edu/models/cesm1.0/pop2/doc/sci/POPRefManual.pdf)
- [POP documentation](https://ncar.github.io/POP/doc/build/html/users_guide/introduction.html)

---

### 1.4 MITgcm

**Equations:** Boussinesq Navier-Stokes on rotating sphere. Unique: both hydrostatic AND non-hydrostatic modes.

**Grid:** Finite volume on orthogonal curvilinear grids. Cubed-sphere, lat-lon, Cartesian. Arakawa C-grid. Vertical: z, z*, p, p*. Partial cells for topography.

**Resolution:** O(1 km) regional to 1° global. ECCO v4 at 1°, ECCO2 at 1/6°.

**Key parameterizations:** GM/Redi (default kappa = 500-1000 m^2/s); KPP; GLS TKE; bulk formulae. Supports automatic differentiation via TAF for adjoint generation — enables ECCO state estimation.

**Sources:**
- [MITgcm GitHub](https://github.com/MITgcm/MITgcm)
- [MITgcm overview](https://mitgcm.readthedocs.io/en/latest/overview/overview.html)
- [MITgcm GM/Redi](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/gmredi.html)
- [MITgcm KPP](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/kpp.html)
- [ECCO project](https://ecco-group.org/)

---

### 1.5 HYCOM (HYbrid Coordinate Ocean Model)

**Equations:** Primitive equations with generalized hybrid vertical coordinate — isopycnic in stratified open ocean, z-level in mixed layer, terrain-following sigma in shallow water.

**Grid:** Arakawa C-grid. Global operational at 1/12° (~8 km). Barotropic-baroclinic mode splitting with ~30-40 barotropic sub-steps per baroclinic step.

**Key parameterizations:** KPP default. Data assimilation via NCODA (3D-Var). Used operationally by US Navy.

**Sources:**
- [HYCOM overview](https://www.hycom.org/hycom/overview)
- [HYCOM at ECMWF — Megann 2004](https://www.ecmwf.int/sites/default/files/elibrary/2004/11096-hybrid-coordinate-ocean-model-hycom.pdf)

---

### 1.6 ROMS (Regional Ocean Modeling System)

**Equations:** Free-surface hydrostatic primitive equations in flux form. Split-explicit time stepping.

**Grid:** Arakawa C-grid. Stretched terrain-following (sigma) vertical. Primarily regional (1-10 km). Supports AGRIF nesting.

**Key parameterizations:** KPP, GLS, MY2.5 for vertical mixing. 4D-Var and ensemble Kalman filter. Third-order upstream advection.

**Sources:**
- [ROMS GitHub](https://github.com/myroms/roms)
- [ROMS wiki](https://www.myroms.org/)
- [Shchepetkin & McWilliams 2005](https://www.sciencedirect.com/science/article/abs/pii/S1463500304000484)

---

## 2. Simplified and Idealized Models

### 2.1 Stommel Box Model (1961)

Two well-mixed boxes (tropical/polar) exchange heat and salt:
```
dT/dt = c(T* - T) - |q|T
dS/dt = -d*S_0 + |q|S
q = k(alpha*DeltaT - beta*DeltaS)
```

**Key result:** Multiple equilibria — thermally-dominated mode (modern AMOC) and salinity-dominated reverse mode. Hysteretic transition implies possible abrupt AMOC collapse. Over 40% of CMIP models may be in a bistable state, but most coupled GCMs appear monostable.

**Sources:**
- [Stommel model — Harvard lecture notes](https://courses.seas.harvard.edu/climate/eli/Courses/EPS281r/Sources/Thermohaline-circulation/2-THC-Stommel-model-notes.pdf)

### 2.2 Stommel-Munk Wind-Driven Circulation

**Stommel (1948):** Linear friction → western boundary layer width delta_S = r/beta. With r = 4e-4 m/s, beta = 2e-11, gives delta_S ~ 20 km.

**Munk (1950):** Lateral viscosity → boundary layer width delta_M = (A_H/beta)^(1/3). With A_H = 5e4 m^2/s, delta_M ~ 130 km.

At 1° resolution (~100 km), you marginally resolve the Munk layer. Viscosity must be large enough: A_H > beta * dx^3 ~ 2e4 m^2/s.

**Sources:**
- [Western boundary current theory — U. Hawaii](https://uhslc.soest.hawaii.edu/ocn620/notes/wbc_book.html)
- [Barotropic dynamics — GFDL](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/stg/ch_6.pdf)

### 2.3 Quasi-Geostrophic (QG) Models

Filters gravity waves, retains balanced motions. Prognostic variable: potential vorticity.
```
q = nabla^2(psi) + beta*y + f_0^2/N^2 * d^2(psi)/dz^2
```
Time step limited by flow speed (~0.1 m/s) not gravity wave speed (~200 m/s) → ~1000x larger steps than SWE.

**Sources:**
- [Thiry et al. 2024 — Unified QG/SWE](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024MS004510)

### 2.4 Model Hierarchy Summary

| Model | Dimensions | Waves | Eddies | Thermohaline | Time step |
|-------|-----------|-------|--------|-------------|-----------|
| Stommel box | 0D | No | No | Yes | None (ODE) |
| Stommel/Munk gyre | 2D barotropic | No | No | No | CFL viscous |
| QG (n-layer) | 2D+layers | Rossby only | Yes | No | ~U/dx |
| SWE (n-layer) | 2D+layers | All | Yes | Partial | ~c/dx |
| Primitive eq. GCM | 3D | All | If resolved | Yes | ~c/dx |

---

## 3. Key Parameterizations at 1-Degree Resolution

### 3.1 Lateral Viscosity

**Laplacian:** A_H = 2e4 - 1e5 m^2/s. Must resolve Munk layer: A_H > beta * dx^3 ~ 2e4 m^2/s.

**Biharmonic:** A_4 = 1e11 - 5e12 m^4/s. More scale-selective.

**Smagorinsky:** A_H = (C_smag/pi)^2 * dx^2 * |D|, C_smag ~ 0.06-0.2. Automatically adjusts to flow.

**Sources:**
- [Griffies & Hallberg 2000 — Biharmonic Smagorinsky](https://www.gfdl.noaa.gov/bibliography/related_files/smg0002.pdf)
- [Megann 2021 — Exploring viscosity space](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002263)

### 3.2 KPP Vertical Mixing

Boundary layer depth h from bulk Richardson number Ri_cr = 0.3. Diffusivity: K(sigma) = h * w(sigma) * G(sigma). Non-local transport under convective forcing.

Interior mixing: shear instability (Ri < 0.7), internal wave breaking (K_w ~ 1e-5 m^2/s), double diffusion.

| Parameter | Value | Units |
|-----------|-------|-------|
| Ri_cr | 0.3 | — |
| Ri_0 | 0.7 | — |
| von Karman | 0.4 | — |
| Background K_v | 1e-5 | m^2/s |

**Sources:**
- [MITgcm KPP docs](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/kpp.html)
- [Van Roekel et al. 2018 — KPP revisited](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018MS001336)
- [Large et al. 1994](https://ftp.soest.hawaii.edu/kelvin/OCN665/OCN665_full/ocean%20mixed%20layers/KPP_show.pdf)

### 3.3 GM/Redi Isopycnal Mixing

**Redi:** Diffuses tracers along isopycnal surfaces. Prevents spurious diapycnal mixing.

**GM:** Parameterizes adiabatic eddy transport: u* = -kappa_GM * nabla(S). Flattens isopycnals, mimics baroclinic instability.

| Parameter | Range | Default | Units |
|-----------|-------|---------|-------|
| kappa_GM | 200-1400 | 600-1000 | m^2/s |
| kappa_Redi | 400-2400 | 600-1000 | m^2/s |
| S_max (slope limit) | 0.002-0.01 | 0.005 | — |

**Sources:**
- [MITgcm GM/Redi docs](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/gmredi.html)
- [Gent 2011 — GM 20/20 hindsight](https://staff.cgd.ucar.edu/gent/gm20.pdf)
- [Pradal & Gnanadesikan 2014](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2013MS000273)

### 3.4 Bottom Drag

**Linear:** tau_b = -r * u_b, r ~ 4e-4 m/s (115-day timescale).

**Quadratic:** tau_b = -C_D * |u_b| * u_b, C_D ~ 0.001-0.003. OM4 uses C_D = 0.003, NEMO default C_D = 1e-3.

---

## 4. Consolidated Parameter Table for 1-Degree Models

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Lateral viscosity (Laplacian) | A_H | 2e4 - 1e5 | m^2/s |
| Lateral viscosity (biharmonic) | A_4 | 1e11 - 5e12 | m^4/s |
| Smagorinsky coefficient | C_smag | 0.06-0.2 | — |
| Lateral tracer diffusivity | kappa_H | 1e3 - 1e4 | m^2/s |
| GM coefficient | kappa_GM | 200-1400 | m^2/s |
| Redi coefficient | kappa_Redi | 400-2400 | m^2/s |
| Background vertical diffusivity | K_v | 1e-5 - 1e-4 | m^2/s |
| Vertical viscosity | A_v | 1e-4 | m^2/s |
| Bottom drag (quadratic) | C_D | 0.001-0.003 | — |
| Bottom drag (linear) | r | 4e-4 | m/s |
| Reference density | rho_0 | 1025-1035 | kg/m^3 |
| Beta (df/dy) | beta | 2e-11 | m^-1 s^-1 |
| Coriolis at 30°N | f_0 | 7.3e-5 | s^-1 |
| Thermal expansion | alpha | ~2e-4 | K^-1 |
| Wind stress peak | tau_0 | 0.1-0.2 | Pa |

### OLR / Thermal Forcing Coefficients

Linearized OLR: `OLR = A + B * T_s`

| Source | A (W/m^2) | B (W/m^2/K) |
|--------|-----------|-------------|
| Budyko (1969) | 210 | 2.0 |
| Sellers (1969) | 204 | 2.17 |
| North (1975) | 202 | 1.9 |
| Planck (no feedbacks) | — | 3.3 |

Difference between B_Planck (3.3) and B_observed (~2.0) = sum of positive feedbacks (water vapor, ice-albedo, lapse rate).

---

## 5. AMOC Representation

### 5.1 Observed AMOC

RAPID-MOCHA array at 26.5°N, continuous since April 2004:
- **Mean strength:** 16.9 Sv (1 Sv = 10^6 m^3/s) over 2004-2020
- **Variability:** Annual means 13-20 Sv
- **Structure:** ~17 Sv northward in upper ~1000 m, ~17 Sv southward below (NADW)

**Sources:**
- [RAPID array](https://rapid.ac.uk/)
- [UK Met Office AMOC dashboard](https://climate.metoffice.cloud/amoc.html)

### 5.2 AMOC in Climate Models

CMIP6 spread: 8-25 Sv (factor of 3). Multi-model mean ~17 Sv matches RAPID.

**Key controls:** Overturning depth, North Atlantic surface buoyancy loss, freshwater transport biases, resolution effects on eddy freshwater transport, Nordic Sea overflow representation, Southern Ocean wind-driven upwelling.

**CMIP6 projections:** AMOC decline of 6-8 Sv (34-45%) by 2100 under high emissions.

**Sources:**
- [Nayak et al. 2024 — Controls on AMOC strength](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL109055)
- [Weijer et al. 2020 — CMIP6 AMOC decline](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GL086075)
- [Westen et al. 2024 — AMOC tipping](https://www.science.org/doi/10.1126/sciadv.adk1189)

### 5.3 Minimum Physics for AMOC in a 2-Layer Model

1. Density difference between layers driven by temperature
2. Deep water formation regions (diapycnal mass flux in subpolar North Atlantic)
3. Return flow (Southern Ocean wind-driven + diffusive upwelling)
4. Surface restoring toward target temperature

---

## 6. Data Assimilation and Reanalysis Products

| Product | Method | Resolution | Assimilates | Source |
|---------|--------|-----------|------------|--------|
| **ERA5** | 4D-Var atmospheric | ~31 km, 137 levels | Satellite, radiosonde, aircraft, surface | [CDS](https://cds.climate.copernicus.eu/) |
| **ORAS5** | 3D-Var FGAT, NEMO 3.4 | 0.25°, 75 levels | Argo, XBT, altimetry, SST, sea ice | [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-oras5) |
| **GODAS** | 3D-Var, MOM4/5 | ~1°, 40 levels | T profiles, SST relaxation | [NCEP](https://www.cpc.ncep.noaa.gov/products/GODAS/) |
| **SODA3** | Optimal interpolation | 1/4° | WOD T/S, satellite SST | [Carton et al. 2019](https://journals.ametsoc.org/view/journals/clim/32/8/jcli-d-18-0605.1.xml) |
| **ECCO v4** | 4D-Var adjoint, MITgcm | 1° (LLC90) | All ocean obs | [ecco-group.org](https://ecco-group.org/) |

ECCO is unique: no data insertion — adjusts initial conditions and forcing to produce a free-running trajectory that fits observations. Obeys conservation laws exactly.

**Sources:**
- [ECCO adjoint modeling](https://ecco-group.org/adjoint.htm)
- [Forget et al. 2015 — ECCO v4](https://gmd.copernicus.org/articles/8/3071/2015/gmd-8-3071-2015.pdf)

---

## 7. Known Model Biases

### Double ITCZ
Most coupled models produce a spurious second ITCZ south of the equator. Cause: Southern Ocean cloud biases → too much solar absorption → SH warms → ITCZ drawn south. Cloud biases explain most model spread.

**Source:** [Hwang & Frierson 2013 (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1213302110)

### Equatorial Cold Tongue Bias
Cold tongue extends too far west, 1-3 K too cold. Cause: overly strong equatorial upwelling, excessive thermocline diffusion, Bjerknes feedback amplification.

### Gulf Stream Separation
At 1°, Gulf Stream overshoots Cape Hatteras → large warm SST bias in NW Atlantic. Largely resolves at 1/10°. Cause: insufficient resolution for inertial jet + topographic steering.

### Southern Ocean Warming Bias
Models warm Southern Ocean too slowly. Related to double ITCZ cloud biases + wind stress biases + Antarctic sea ice extent errors.

**Sources:**
- [Zhang et al. 2023 — CMIP5 to CMIP6 SST biases](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GL100888)
- [Westen et al. 2024 — Atlantic freshwater transport biases](https://os.copernicus.org/articles/20/549/2024/)

---

## 8. Emerging Approaches

### 8.1 Neural Network Parameterizations

Physics-informed deep learning for vertical mixing: trained on turbulence measurements with KPP constraints, reduced SST biases 30-50% in coupled models.

**Sources:**
- [Zhu et al. 2022 — Physics-informed DL mixing (NSR)](https://academic.oup.com/nsr/article/9/8/nwac044/6544687)

### 8.2 Differentiable Ocean Models

**NeuralOGCM** (Dec 2025): First hybrid ocean GCM with differentiable physics + deep learning. Learnable parameters, neural net corrects subgrid processes.

**Oceananigans.jl** (CliMA, Caltech): Julia-based GPU ocean model designed for differentiability. Global ocean at 488 m on 768 A100s. 9.9 simulated years/day at 10 km on 68 A100s.

**NeuralGCM** (Google/DeepMind, 2024, Nature): Differentiable atmospheric GCM with learned physics. Atmosphere-only currently; ocean coupling next.

**Sources:**
- [NeuralOGCM — arXiv 2512.11525](https://arxiv.org/abs/2512.11525)
- [Oceananigans.jl GitHub](https://github.com/CliMA/Oceananigans.jl)
- [Oceananigans paper — arXiv 2309.06662](https://arxiv.org/abs/2309.06662)
- [NeuralGCM — Nature 2024](https://www.nature.com/articles/s41586-024-07744-y)

### 8.3 ML-Accelerated Emulators

**Samudra** (2025, GRL): ML emulator of GFDL OM4. Modified ConvNeXt UNet. Stable for multi-century runs. **150x speedup** on single A100 (~100 simulated years in 1.3 hours).

**SamudrACE** (2025): Couples Samudra ocean + ACE atmosphere emulators → fully ML coupled climate model. Captures ENSO.

**ACE2-SOM** (2025): ML atmosphere + slab ocean. Reproduces equilibrium climate sensitivity at 25x less cost.

**Sources:**
- [Samudra — Dheeshjith et al. 2025](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL114318)
- [SamudrACE — arXiv 2509.12490](https://arxiv.org/html/2509.12490v1)
- [ACE2-SOM — Clark et al. 2025](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JH000575)

### 8.4 Physics-Informed Neural Operators

PINOs learn PDE solution maps with physics constraints in the loss. Train on coarse data, enforce PDE at high resolution.

**Sources:**
- [PINO — arXiv 2111.03794](https://arxiv.org/abs/2111.03794)
- [Rewiring climate modeling with ML — Nature 2026](https://www.nature.com/articles/s43247-026-03238-z)

---

## Key References (Consolidated)

### Model GitHub Repos
- [MOM6](https://github.com/NOAA-GFDL/MOM6)
- [MITgcm](https://github.com/MITgcm/MITgcm)
- [ROMS](https://github.com/myroms/roms)
- [Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)

### Model Documentation
- [MOM6 ReadTheDocs](https://mom6.readthedocs.io/)
- [NEMO docs](https://www.nemo-ocean.eu/doc/node1.html)
- [MITgcm docs](https://mitgcm.readthedocs.io/)
- [HYCOM](https://www.hycom.org/hycom/overview)
- [POP Reference Manual](https://www2.cesm.ucar.edu/models/cesm1.0/pop2/doc/sci/POPRefManual.pdf)

### Foundational Papers
- Adcroft et al. 2019 — OM4.0: [doi:10.1029/2019MS001726](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019MS001726)
- Large, McWilliams & Doney 1994 — KPP: [PDF](https://ftp.soest.hawaii.edu/kelvin/OCN665/OCN665_full/ocean%20mixed%20layers/KPP_show.pdf)
- Gent 2011 — GM hindsight: [PDF](https://staff.cgd.ucar.edu/gent/gm20.pdf)
- Griffies & Hallberg 2000 — Biharmonic Smagorinsky: [PDF](https://www.gfdl.noaa.gov/bibliography/related_files/smg0002.pdf)

### Reanalysis & Observations
- [ERA5 on CDS](https://cds.climate.copernicus.eu/)
- [ORAS5](https://cds.climate.copernicus.eu/datasets/reanalysis-oras5)
- [ECCO](https://ecco-group.org/)
- [RAPID AMOC array](https://rapid.ac.uk/)
