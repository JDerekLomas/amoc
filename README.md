# SimAMOC — Ocean Circulation Simulator

**Live demo: [amoc-sim.vercel.app/simamoc/](https://amoc-sim.vercel.app/simamoc/)**
**System docs: [amoc-sim.vercel.app/docs/](https://amoc-sim.vercel.app/docs/)**

A real-time ocean circulation simulator running in the browser. The barotropic vorticity equation computes wind-driven gyres, western boundary currents (Gulf Stream, Kuroshio), and the Antarctic Circumpolar Current from first principles — on the GPU via WebGPU compute shaders.

## What it does

- **512x160 global grid** (~0.7° resolution) with periodic longitude boundaries
- **WebGPU compute shaders** run the physics: vorticity timestep, FFT Poisson solver, temperature/salinity advection, boundary enforcement
- **Two-layer ocean** with temperature, salinity, and linear equation of state for density-driven circulation
- **AMOC** — thermohaline overturning that responds to freshwater forcing (Stommel bifurcation)
- **7-regime cloud model** with shortwave albedo and longwave greenhouse effects
- **1-layer atmosphere** with two-way ocean-air coupling and moisture/latent heat
- **Editable coastlines** — paint land or ocean to reshape continents in real time (SimEarth-style)
- **Paleoclimate scenarios**: Open Drake Passage, Close Panama Seaway, Melt Greenland, Ice Age
- **14 visualization modes**: SST, deep ocean, currents, speed, vorticity, salinity, density, bathymetry, clouds, air temp, moisture, precipitation, and more
- **AI self-improvement loop** — agents diagnose physics errors and tune parameters at $0.03/run

## The physics

The barotropic vorticity equation:

```
dq/dt + J(ψ,q) = curl(τ) − rζ + A∇²ζ − α∂ρ/∂x + F(ψ_deep − ψ)
```

One equation produces all major ocean currents. Wind stress curl drives the gyres. The β effect creates western intensification (Stommel, 1948). Lateral viscosity sets the boundary layer width (Munk, 1950). Density gradients (from temperature and salinity) drive the thermohaline overturning.

## Data sources

- **Coastlines**: Natural Earth 110m (public domain)
- **SST**: NOAA OI SST v2.1 (1991-2020 climatology)
- **Deep temperature**: WOA23 at 1000m
- **Bathymetry**: ETOPO1 (seafloor + land elevation)
- **Salinity**: WOA23 surface salinity
- **Wind stress**: NCEP Reanalysis (tau_x, tau_y)
- **Clouds**: MODIS MOD08_M3 (fraction + low/high types)
- **Albedo**: MODIS MCD43A3
- **Precipitation**: GPM IMERG

## Running locally

```bash
python3 -m http.server 8765
open http://localhost:8765/simamoc/
```

WebGPU requires Chrome 113+, Firefox 128+, Safari 18+, or Edge 113+. Falls back to CPU if unavailable.

## Documentation

- **[SYSTEM.md](SYSTEM.md)** — comprehensive system documentation (architecture, physics, data, AI loop)
- **[PHYSICS_REGISTRY.md](PHYSICS_REGISTRY.md)** — every physical process with equations, data, status, and gaps
- **[CLIMATE-MODELS.md](CLIMATE-MODELS.md)** — survey of MOM6, NEMO, MITgcm, HYCOM, ROMS, and ML models
- **[simamoc/ARCHITECTURE.md](simamoc/ARCHITECTURE.md)** — code architecture and module structure
- **[knowledge/](knowledge/)** — 16 research files covering equations, parameters, diagnostics, observations, and more
- **[docs/](https://amoc-sim.vercel.app/docs/)** — public documentation website

## ⚠ Don't open the simulator over `file://`

If you double-click `simamoc/index.html` (or `v4-physics/console.html`)
to open it directly, browsers block its `fetch()` calls for the mask /
coastlines / SST data — silently — and the engine renders a degenerate
**continent-less zonal-band ocean** with `AMOC = +0.0e+0`. That's not
the simulator working; that's the simulator without its inputs.

Always serve over HTTP:

```bash
python3 -m http.server 8000
open http://localhost:8000/simamoc/
```

Or `node helm-lab/cli.mjs serve` if you're using helm-lab. The Vercel
deployment is also fine because it serves over HTTPS.

See [helm-lab/KNOWN-LIMITATIONS.md](helm-lab/KNOWN-LIMITATIONS.md) for
the full list of known issues, surprises (the model is bistable —
results depend on which spinup branch you land on), and what's
explicitly broken.

## Programmatic experimentation — `helm-lab/`

A headless harness that drives the simulator without a user-facing
browser. Useful for parameter sweeps, hysteresis searches, time-lapse
movies, and anything else that wants to script the planet.

```
helm-lab/
├── README.md            usage + architecture
├── server.mjs           daemon (Playwright + WebGPU + HTTP RPC)
├── cli.mjs              full Node CLI with daemon auto-detect
├── q                    bash + curl client (~30-50ms per call)
├── lab.mjs              HelmLab class
├── composer.js          in-page Canvas2D image composers
├── bench.mjs            latency benchmark
└── experiments/
    ├── amoc-bifurcation-v2.mjs   Stommel hysteresis
    ├── amoc-collapse-window.mjs  dense F* search
    ├── earth-movie.mjs           time-lapse MP4
    └── RESULTS.md                findings log
```

The simulator's API contract is documented at
**[v4-physics/CONTRACT.md](v4-physics/CONTRACT.md)** — anything calling
`window.lab.*` is supported; anything else is private. `helm-lab` is
the first consumer; the contract is stable for any future driver
(MCP server, CI runner, other LLM agent).

**Quick start:**

```bash
node helm-lab/cli.mjs serve &                     # start daemon
helm-lab/q step 5000                              # ~30 ms per call
helm-lab/q render '"frame.png"' '{"view":"temp"}'
node helm-lab/experiments/amoc-bifurcation-v2.mjs # ~12 min run
helm-lab/q --shutdown                             # stop daemon
```

## Other views

Earlier explorations preserved in the repo:

- `/v2/` — 3D globe with Three.js
- `/v3-oscar/` — Real OSCAR satellite ocean current data
- `/v4-physics/` — Single-file simulator (`window.lab` API stable;
  primary target for `helm-lab` programmatic experiments)
- `/v5-story/` — AMOC collapse narrative

## Credits

Built by Luke Barrington and Derek Lomas with Claude Code. Coastline data from Natural Earth. SST data from NOAA. Bathymetry from ETOPO1.
