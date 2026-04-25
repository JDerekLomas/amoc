# amoc-jax

Compute-first ocean circulation simulator. JAX core (Mac CPU/Metal, cloud GPU
via CUDA). Full coupled ocean-atmosphere-ice physics with 25+ observational
datasets for initialization, forcing, and validation.

## Status

**Full coupled model running.** Two-layer ocean (vorticity + T/S) + 1-layer
atmosphere (temperature + moisture) + prognostic sea ice + seasonal forcing
cycle from monthly climatologies. 34-layer interactive viewer with sim fields,
observational maps, and diagnostics.

Physics stack:
- **Ocean dynamics**: Arakawa Jacobian, beta-plane, wind-driven gyres, interfacial coupling
- **Ocean tracers**: SST, salinity (surface + deep), Haney restoring, Ekman advection
- **Atmosphere**: diffusion, ocean/land exchange, Clausius-Clapeyron moisture, precipitation
- **Radiation**: seasonal solar, ice-albedo feedback, cloud albedo (blended model + MODIS), water vapor greenhouse
- **Sea ice**: thermodynamic growth/melt, albedo feedback, brine rejection
- **Seasonal cycle**: monthly SST target + albedo from observations
- **Deep circulation**: density-driven vertical exchange, meridional overturning tendency

## Quickstart

```bash
cd amoc-jax
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/pytest                                      # 28 tests should pass

# Interactive viewer with seasonal forcing:
.venv/bin/python app.py --nx 128 --ny 64

# Fast startup (skip monthly data loading):
.venv/bin/python app.py --nx 128 --ny 64 --no-seasonal

# Headless batch run:
.venv/bin/python run.py --nx 256 --ny 128 --steps 10000
```

The viewer opens at `http://127.0.0.1:8765` with:
- 12 live simulation fields (SST, speed, streamfunction, vorticity, air temp, moisture, deep T, salinity, sea ice, etc.)
- 20 observational maps (NOAA SST, WOA23 salinity, ERA5 winds, ETOPO1 bathymetry, MODIS clouds, GPM precipitation, HYCOM currents, ...)
- 2 diagnostic difference maps (sim - obs for SST and salinity)
- Coastline overlay, lat/lon grid, per-field colormaps with colorbar
- Time series panel tracking SST, circulation strength, N. Atlantic salinity
- Physics parameter sliders (solar, wind, freshwater forcing, mixing)

## Code layout

```
amoc-jax/
  src/amoc/
    grid.py        # spherical lat-lon mesh, cos(lat) metric, Coriolis
    state.py       # State, Params, Forcing, SeasonalForcing pytrees
    data.py        # load 25+ observational datasets (JSON + binary)
    physics.py     # Arakawa Jacobian, vorticity RHS, tracer RHS
    atmosphere.py  # radiation, clouds, atmosphere time stepping
    poisson.py     # FFT-x + DST-I-y solver for psi from zeta
    step.py        # forward Euler + seasonal variants, jax.lax.scan
    render.py      # matplotlib diagnostic plots
    diagnostics.py # AMOC streamfunction, meridional velocity
  tests/           # 28 tests (grid, poisson, physics, correctness, data, step)
  app.py           # interactive browser viewer (HTTP + polling)
  run.py           # headless batch runner
  lab.py           # daemon mode with HTTP RPC
  calibrate.py     # jax.grad autodiff parameter calibration
  assimilate.py    # data assimilation pipeline
  scripts/         # rendering, evaluation, hysteresis experiments
  docs/
    physics.md     # equations, derivations, discretization choices
    data.md        # data layer reference
    limitations.md # honest audit of simplifications
    roadmap.md     # development phases
```

## Data layer

25+ observational datasets at 1024x512 in `../data/bin/`:

| Dataset | Source | Role |
|---------|--------|------|
| SST | NOAA OI SST v2.1 | Init, restoring target, validation |
| SST monthly | NOAA OI SST (12 months) | Seasonal forcing cycle |
| Deep temperature | WOA23 | Init, restoring target |
| Salinity | WOA23 | Init, restoring, validation |
| Wind stress + curl | ERA5 reanalysis | Vorticity forcing, Ekman transport |
| Wind monthly | ERA5 (12 months) | Seasonal wind forcing |
| Bathymetry | ETOPO1 | Ocean depth, variable MLD |
| Ocean mask | Derived from ETOPO1 | Land/ocean geometry |
| Land surface temp | MODIS | Atmosphere-land exchange |
| Air temperature | ERA5 | Atmosphere initialization |
| Mixed layer depth | Observed/estimated | Vertical mixing structure |
| Cloud fraction | MODIS | Radiation (blended with model) |
| Albedo | MODIS | Radiation, ice-albedo feedback |
| Albedo monthly | MODIS (12 months) | Seasonal albedo cycle |
| Sea ice | NOAA OISST | Ice initialization |
| Precipitation | NASA GPM IMERG | Validation |
| Evaporation | ERA5 | Validation |
| Water vapor | MODIS | Moisture initialization |
| Ocean currents | HYCOM/GODAS | Validation |
| NDVI, snow, chlorophyll, pressure | Various | Visualization |

```bash
python scripts/data_status.py             # check everything
python scripts/data_status.py --by-role   # group by role
```

## Validation results

After 10,000 steps at 128x64:
- **SST RMSE**: 1.08 C (bias: -0.06 C)
- **Stability**: No NaN at 256x128 over 10k steps
- **Hosing response**: N. Atlantic salinity drops 0.29 psu at F=2.0, deep circulation weakens to 79%
- **Parameter sensitivity**: Stable across solar x0.7-1.5, wind x0-2

## What this model gets right and wrong

See [`docs/limitations.md`](docs/limitations.md) for a severity-rated audit.
Key simplifications: linear EOS, single-layer atmosphere (no jet stream/Hadley
cells), no land hydrology, thermodynamic ice only (no dynamics).

## Reading list

- **Stommel (1948)**, **Munk (1950)**, **Sverdrup (1947)** — wind-driven gyres
- **Arakawa (1966)** — energy/enstrophy-conserving Jacobian
- **Stommel (1961)** — 2-box thermohaline bistability
- **Rahmstorf (1996, 2002)** — hosing experiments and AMOC thresholds
- **Ditlevsen & Ditlevsen (2023)** — early-warning AMOC collapse
- **van Westen et al. (2024, 2025)** — AMOC tipping in eddying GCMs

## License

TBD.
