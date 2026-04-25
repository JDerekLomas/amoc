# Roadmap

## v1a — barotropic gyres (DONE 2026-04-25)

Wind-driven Sverdrup/Munk circulation under ERA5 wind-curl forcing on a
global lat-lon grid with land mask. Ships subpolar + subtropical gyres,
western boundary intensification, and a Southern Ocean ACC-like band.

Open follow-ups before v1b:

- **Higher-resolution stability.** 512×256 currently blows up at dt = 0.01.
  Halving Δx halves the CFL-limited dt, so 0.005 should work — needs verifying.
  At 1024×512 we will likely want adaptive dt or implicit treatment of the
  most rigid term (viscosity).
- **Diagnostics.** Compute and print on every save:
  - $\psi_{\max}$ in each basin (proxy for gyre transport)
  - Western boundary current speed and width
  - Total energy $\tfrac{1}{2}\sum|\nabla\psi|^2$ and enstrophy $\tfrac{1}{2}\sum\zeta^2$
  - Conservation: should be flat once forcing balances dissipation.
- **Validation oracle.** Replicate the same setup in JAXSW or Veros
  (Antarctic Circumpolar Channel test case in Veros is a near-match) and
  compare ψ patterns and gyre transport.
- **Decide Poisson convention.** Keep grid Laplacian (current) or switch to
  spherical Laplacian. See `docs/physics.md` §3a. Defer to v1b.

## v1b — two-layer baroclinic

Add a deep layer with its own ψ_d, ζ_d. Couple via a thermal-wind buoyancy
term: a meridional buoyancy gradient g'∂_y(b̄) drives the upper-layer
streamfunction independently from the wind.

Equations (sketched):

$$\partial_t\zeta_1 + \tfrac{1}{\cos\varphi}J(\psi_1,\zeta_1) + \beta\partial_\lambda\psi_1
= \mathrm{curl}(\tau) - r\zeta_1 + A\nabla^2\zeta_1 + F_{\text{buoy}}\partial_\lambda b$$
$$\partial_t\zeta_2 + \tfrac{1}{\cos\varphi}J(\psi_2,\zeta_2) + \beta\partial_\lambda\psi_2
= -r_d\zeta_2 + A\nabla^2\zeta_2 - F_{\text{buoy}}\partial_\lambda b$$

with prescribed buoyancy $b(y)$ for now. Diagnostic: meridional overturning
streamfunction $\Psi(y) = \int v\,dz\,d\lambda$ over the Atlantic basin.

Goal: produce a clean overturning cell visible in the (y, z) projection.

## v1c — temperature + salinity

Replace prescribed buoyancy with prognostic T and S, each obeying:

$$\partial_t T + \tfrac{1}{\cos\varphi}J(\psi_1, T) + \kappa\nabla^2 T
= \tfrac{1}{\tau_T}(T^*(\varphi) - T)$$

(and analogous S, with a freshwater flux instead of restoring). Linear
equation of state $b = -\alpha_T (T - T_0) + \alpha_S (S - S_0)$. Convective
adjustment when surface buoyancy < deep buoyancy.

Validation:

- SST RMSE vs NOAA OI < 3°C globally.
- Atlantic deep water formation in Labrador / Nordic seas.
- AMOC strength roughly scales like Bryan: $\Psi \propto \kappa^{2/3}$.

## v1d — hosing experiment

Add a freshwater flux $F$ to the North Atlantic surface salinity equation.
Run a slow ramp up + ramp down. Plot $(F, \Psi_{\text{AMOC}})$ — must
exhibit hysteresis (Stommel saddle-node bifurcation).

This is the AMOC science showpiece. References: Rahmstorf 1996, Cessi 1994,
Castellana et al. 2024, van Westen et al. 2024.

Diagnostics to add:

- $F_{\text{ovS}}$ at 34°S (van Westen et al. 2025 indicator)
- Time series of $\Psi_{\text{AMOC}}$ at 26°N (RAPID-style)
- Deep convection on/off indicator (mixed-layer depth proxy)

## Past v1d

- **GPU runs.** Move to a rented A100/H100; verify same results;
  benchmark steps/sec at 1024×512.
- **Browser viewer integration.** Adapt the existing `simamoc/renderer.js`
  to read frames from saved zarr instead of running WGSL. The Atlantic-vs-
  Pacific viz, paleoclimate scenarios, and editable coastlines all stay.
- **Differentiable parameter fitting.** With JAX autodiff, optimize
  $(r, A, \kappa, \alpha_T, \alpha_S)$ to minimize SST RMSE. Output reproducible
  best-fit parameter set.
- **Atmosphere coupling (v2).** 1-layer atmosphere with moisture/latent heat,
  closing the cloud-radiation-evaporation loop.
