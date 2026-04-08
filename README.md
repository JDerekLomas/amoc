# Ocean Circulation Simulator

**Live demo: [amoc-sim.vercel.app/v4-physics/](https://amoc-sim.vercel.app/v4-physics/)**

A real-time ocean circulation simulator running in the browser. The barotropic vorticity equation computes wind-driven gyres, western boundary currents (Gulf Stream, Kuroshio), and the Antarctic Circumpolar Current from first principles — on the GPU via WebGPU compute shaders.

## What it does

- **360x180 global grid** (1-degree resolution) with periodic longitude boundaries
- **WebGPU compute shaders** run the physics: timestep, Poisson solver (Jacobi iteration), temperature advection, boundary enforcement
- **Editable coastlines** — paint land or ocean to reshape continents in real time (SimEarth-style)
- **Paleoclimate scenarios**: Open Drake Passage, Close Panama Seaway, Melt Greenland, Ice Age
- **Temperature field** with seasonal solar heating and buoyancy coupling
- **AMOC freshwater forcing** — push the slider to collapse the Atlantic overturning circulation

## The physics

The barotropic vorticity equation:

```
dζ/dt + J(ψ,ζ) + βv = curl(τ)/ρH - rζ + A∇⁴ψ
```

One equation produces all major ocean currents. Wind stress curl drives the gyres. The β effect creates western intensification (Stommel, 1948). Lateral viscosity sets the boundary layer width (Munk, 1950). Temperature gradients provide buoyancy coupling.

**[Interactive equation explainer →](https://amoc-sim.vercel.app/v4-physics/equation.html)**

## Data sources

- **Coastlines**: Natural Earth 110m (public domain)
- **SST**: NOAA OISST v2.1 (used in other views)
- **Ocean currents**: OSCAR satellite data (used in v3 currents view)

## Running locally

```bash
python3 -m http.server 8765
open http://localhost:8765/v4-physics/
```

WebGPU requires Chrome 113+, Firefox 128+, Safari 18+, or Edge 113+. Falls back to CPU if unavailable.

## Other views

The repo also contains earlier explorations:

- `/` — Flat map with Stommel 2-box tipping model
- `/v2/` — 3D globe with Three.js
- `/v3-oscar/` — Real OSCAR satellite ocean current data
- `/v5-story/` — AMOC collapse narrative

## Credits

Built with Claude Code. Coastline data from Natural Earth. SST data from NOAA. Blue Marble texture from NASA.
