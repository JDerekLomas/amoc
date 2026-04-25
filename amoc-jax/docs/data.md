# Data layer

This document is the human-readable companion to `data_manifest.yaml`.
Everything observational the simulator depends on lives in the manifest;
this doc explains the conventions, the roles, and how to add or refresh
data.

## Quick check

```bash
python scripts/data_status.py             # everything
python scripts/data_status.py --by-role   # grouped by role
python scripts/data_status.py --missing-only
python scripts/data_status.py --sha       # also compute hashes
```

The script reads `data_manifest.yaml`, walks the filesystem, and prints a
status row per field: present? right shape? values within expected bounds?

## Layout

```
<repo root>/
  data/                                    # 1024x512 hires JSONs (preferred)
    wind_stress.json, mask.json, sst.json, salinity.json, deep_temp.json,
    bathymetry.json, mixed_layer_depth.json, ocean_currents.json,
    sst_monthly.json, wind_stress_monthly.json, ...
  earth-data/timeseries/                   # 1D series
    rapid_amoc_monthly.csv, co2_annual_mlo.csv, hadcrut5_global_annual.csv
  *_1deg.json                              # 360x160 lores legacy fields
  fetch-data-hires.py                      # produces data/*.json (Earth Engine)

  amoc-jax/
    data_manifest.yaml                     # single source of truth
    docs/data.md                           # this file
    scripts/data_status.py                 # checker
    src/amoc/data.py                       # loaders
```

## Roles a field can play

| Role | What the simulator does with it |
|---|---|
| `forcing` | Drives the model. Wind curl, freshwater flux, surface heat flux. |
| `geometry` | Land/ocean mask, bathymetry, coastlines. Doesn't change in time. |
| `ic` | Initial condition for a prognostic variable at t=0. |
| `restoring` | Target the model relaxes toward (Haney-style boundary condition). |
| `validation` | Compared against model output. **Never** fed to the model. |
| `atmosphere` | Atmospheric forcing/coupling fields (v2 onwards). |
| `scenario` | For paleoclimate (open Drake Passage) or future (CMIP) experiments. |
| `timeseries` | 0D/1D series — RAPID AMOC, Mauna Loa CO₂, HadCRUT5. |

Fields can have more than one role (e.g. SST is `[ic, restoring, validation]`).

## Resolution

Two tiers, both global lat-lon:

- **Hires** — 1024×512, ~0.35° lon × 0.31° lat. From `fetch-data-hires.py`.
  This is what we actually run on.
- **Lores** — 360×160, 1°. Legacy. Kept as fallback.

If we ever need eddy-resolving (~0.1°, 3600×1800), `fetch-data-hires.py`
takes a `--resolution` flag. Earth Engine quota will tell us very quickly
if we're being unreasonable.

## File schema (for 2D fields)

Standard JSON shape for a hires field:

```json
{
  "nx": 1024,
  "ny": 512,
  "lat0": -79.5,         // *cell-center* of southernmost row (NOT the edge)
  "lat1":  79.5,         // *cell-center* of northernmost row
  "lon0": -179.82,       // cell-center, west edge
  "lon1":  179.82,       // cell-center, east edge
  "source": "ERA5 Monthly 10m wind, 1991-2020 mean, ...",
  "<field_key>": [...float array, length nx*ny, row-major lat-major...]
}
```

`amoc.data.load_to_grid(path, field_key, grid)` handles bilinear resampling
to whatever target Grid you give it; periodic in longitude, clamped at the
poles.

## Mask format (special)

`data/mask.json` and `simamoc/mask*.json` use a packed-bits encoding:

- 4 bits per hex character, MSB first.
- Total bits = nx · ny.
- 1 = ocean, 0 = land.

`amoc.data.load_mask(path, grid=grid)` decodes to a (ny, nx) float32 array.

## Bathymetry sign convention (gotcha)

`bathymetry_1deg.json` and `data/bathymetry.json` store **unsigned
magnitudes**: ocean depth and land elevation both come out as positive
numbers. To distinguish ocean from land, **use the `ocean_mask` field**, not
a sign test on `depth`. My first attempt at v1a used `depth < 0` and
silently produced a zero ocean mask.

Recorded in `data_manifest.yaml` under the `bathymetry` entry.

## Provenance

Each manifest entry records:

- `source` — dataset name + period + version (free text).
- `fetched_via` — script that produced the file.
- `fetched_field` — the `--field` argument to that script.

For full forensic provenance:

```bash
python scripts/data_status.py --sha
```

prints SHA-256 (truncated to 12 chars) of every file. Pin those in the
manifest if a downstream calibration depends on a specific snapshot.

## Adding or refreshing a field

1. Edit/add an entry in `data_manifest.yaml`. Capture role, source, fetch
   script, expected shape, sanity bounds.
2. Refetch:
   ```bash
   python fetch-data-hires.py --field <name>
   ```
   See `fetch-data-hires.py --help` for the list. Fields are independent;
   you can refetch one without touching others.
3. Verify:
   ```bash
   python scripts/data_status.py
   ```
4. If the field needs a different convention (sign, encoding, units), add
   a section in this document and a note on the manifest entry.

## What data is used for in this project (a checklist for v1b/c/d planning)

| # | Use | v1a | v1b | v1c | v1d |
|---|---|---|---|---|---|
| 1 | Forcing the dynamical equations | wind_curl | wind_curl | + heat flux, P-E | + freshwater hosing |
| 2 | Initial conditions | rest | rest | SST, S, deep_T | spun-up state |
| 3 | Restoring targets (Haney BC) | — | — | T*(y), S*(y) | T*(y), S*(y) |
| 4 | Geometry | mask | mask, bathymetry | mask, bathymetry, MLD | + reconfigurable masks |
| 5 | Validation | currents (GODAS, HYCOM) | + MOC ψ(y,z) | SST RMSE, currents | RAPID AMOC time series |
| 6 | Calibration targets | — | — | SST, S RMSE | F_ovS, hysteresis |
| 7 | Visualization overlays | coastlines | coastlines | coastlines | coastlines |
| 8 | Scenario setups | — | — | — | hosing flux pattern |
| 9 | Time series context | — | — | — | RAPID, CO₂, HadCRUT5 |
| 10 | Atmosphere coupling (v2) | — | — | — | — |

The manifest's `used_in` field tags each entry against the phases above.
`scripts/data_status.py --by-role` groups by role for a clean view.

## Why a manifest at all

Without one, "where did this number come from?" answers depend on memory
and grep. A simulator that's tuned to a forcing dataset is only as
reproducible as the dataset; a hosing experiment that disagrees with
Rahmstorf 1996 by 20% needs a forensic trail. The manifest is the trail.
