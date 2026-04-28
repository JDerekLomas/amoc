# SimAMOC -- Design Notes & Project Plan

*A first-principles, browser-native, interactive ocean circulation simulator.*

---

## What it is, and where it sits

SimAMOC is a real-time ocean circulation simulator that runs in the browser. It computes the Atlantic Meridional Overturning Circulation -- the global conveyor belt that moves heat from the tropics to the poles -- from first principles: wind stress, solar heating, density-driven flow. The goal is to reproduce observed ocean patterns (SST, salinity, currents) well enough that you can break things realistically -- open the Panama seaway, melt Greenland, crank CO2 -- and see physically meaningful responses.

The niche is genuinely empty. Existing projects tile around it but don't hit the same target:

| Project | First-principles? | Browser-native? | Real-time? | Interactive perturbations? |
|---|---|---|---|---|
| **SimAMOC** | yes | yes | yes | yes |
| PlaSim / ExoPlaSim | yes | no | fast batch | parameter-driven |
| Build Your Own Earth | yes (FOAM) | yes | no (cached runs) | yes (preset scenarios) |
| CLIMLAB | yes | no | varies | yes |
| Oceananigans | yes | no | research-grade | scriptable |
| Samudra / SamudrACE | no (ML emulator) | no | yes | no (training-distribution only) |
| En-ROADS, C-ROADS | no (stock-flow) | yes | yes | yes (policy sliders) |

The honest framing: **SimAMOC is a browser-native PlaSim that you can poke.** Nothing else is that.

### Current status (as of this writing)

- Wind-driven gyres work.
- Temperature patterns are recognizable (RMSE ~1.3-7C depending on version).
- ERA5 observed winds drive circulation on GPU.
- Cloud feedbacks exist but are crude.
- AMOC itself is weak.

### Known physics gaps

- **Atmospheric moisture** -- the dominant tropical energy transport. Without it, you can't get tropical heat balance right.
- **Salinity forcing from P-E** -- precipitation minus evaporation is the freshwater flux that drives AMOC. Without it, AMOC has no buoyancy-forcing mechanism worth the name.
- **Cloud-radiation closure** -- current cloud feedbacks don't match observed OLR or shortwave reflection.
- **Tighter land boundaries** -- coastlines and bathymetry need to interact properly with circulation.

---

## Architecture

Build it as components that don't know about each other, talking through a flux coupler. Different physical subsystems have radically different timescales and natural grids. The only thing components share are fluxes at their interfaces.

```
                    Flux Coupler
         (exchanges + conservation checks)
      |             |             |             |
   [Atmos]       [Ocean]       [Ice]        [Land]
      ^             ^             ^             ^
   wind tau      SST, S       albedo,        albedo,
   heat Q     currents      melt/salt        ET, runoff
   freshwater P-E             reject
```

### Core principles

- **State is data, physics is code.** Flat typed arrays you can serialize and snapshot.
- **Geometry and forcing are configuration, not code.** "Open Panama" = swap a mask.
- **Timescales drive the timestepping hierarchy.**
- **Diagnostics are separate from simulation.**
- **Spin-up and runtime are different modes.**

---

## Resolution

**256x128 as default, 360x180 as fidelity mode.** Skip 512x1024. AMOC structure doesn't need 0.35 degree resolution. Western boundary current needs ~4-5 cells across -- 256x128 gives that.

---

## Roadmap

### Phase 1 -- Architectural refactor
Component + coupler structure. State as flat typed arrays. Geometry as loadable data. Drop to 256x128.

### Phase 2 -- Close the energy balance
Individual fluxes match observations (CERES, MODIS), not just total.

### Phase 3 -- Add missing physics
1. P-E freshwater forcing on salinity (biggest AMOC lever)
2. Atmospheric moisture transport
3. Cloud-radiation closure
4. Tighter coastlines and bathymetry

### Phase 4 -- Spin-up & snapshot infrastructure
Offline spin-up runner producing equilibrium snapshots. Library of snapshots for different configurations.

### Phase 5 -- Scenario library
Named perturbation experiments: open Panama, close Bering, melt Greenland, double CO2.

---

## Validation targets

- Global mean SST
- AMOC strength at 26.5N (RAPID array)
- Sea ice extent
- ITCZ position
- Western boundary current structure
