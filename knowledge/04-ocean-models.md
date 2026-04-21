# Ocean Model Survey

See `../CLIMATE-MODELS.md` for the full deep dive with all citations.

## Quick Reference

| Model | Grid | Vertical | Unique Feature | GitHub |
|-------|------|----------|---------------|--------|
| **MOM6** | C-grid, tripolar | ALE (z*/isopycnal/sigma) | Scale-aware params, ePBL | [NOAA-GFDL/MOM6](https://github.com/NOAA-GFDL/MOM6) |
| **NEMO** | C-grid, ORCA | z/s/hybrid | Huge EU community, SI3 ice | [nemo-ocean.eu](https://www.nemo-ocean.eu/) |
| **MITgcm** | C-grid, cubed-sphere | z/z*/p/p* | Non-hydrostatic + adjoint (ECCO) | [MITgcm/MITgcm](https://github.com/MITgcm/MITgcm) |
| **HYCOM** | C-grid | Hybrid isopycnal/z/sigma | US Navy operational | [hycom.org](https://www.hycom.org/) |
| **POP** | B-grid | z-only | CESM legacy (→ MOM6) | [CESM docs](https://ncar.github.io/POP/) |
| **ROMS** | C-grid | Sigma (terrain-following) | Coastal/regional, 4D-Var | [myroms/roms](https://github.com/myroms/roms) |

## What They Get Right That We Don't

1. **Split-explicit time stepping** — barotropic sub-stepped at ~5-15s, baroclinic at ~600-1800s
2. **GM/Redi** — isopycnal mixing prevents spurious diapycnal diffusion
3. **KPP** — physically-based boundary layer mixing (we use simple vertical exchange)
4. **Realistic wind forcing** — from ERA5 or JRA-55 reanalysis
5. **Partial cells** — better topography representation than our BFS-derived depth

## What Our Sim Does That's Unique

1. **Interactive browser** — WebGPU real-time, no compilation
2. **Editable coastlines** — paint tools for paleoclimate scenarios
3. **Educational speed** — parameters accelerated ~10-20x for visible dynamics
4. **Arakawa Jacobian** — many simple models skip this and get spectral pile-up
