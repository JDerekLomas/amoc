#!/usr/bin/env python3
"""Render every key data field as a map PNG.

Reads `data_manifest.yaml`, walks the entries we care about visually, loads
each field through `amoc.data.load_to_grid` (or `load_mask` for the mask),
and writes a PNG into `output/atlas/`. Used both as a sanity check and as
the source of map images for blog posts and the README.

Usage:
    python scripts/render_atlas.py [--nx 1024 --ny 512] [--out output/atlas]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from amoc.data import load_mask, load_to_grid
from amoc.grid import Grid
from amoc.render import render_field

REPO_ROOT = Path(__file__).resolve().parents[2]


# What to render: (output filename, field-spec, plot kwargs, ocean_only).
# field-spec = either ("file", "json_key") or ("mask", "file") for hex mask.
# ocean_only = True means apply the ocean mask before rendering (NaN over land
# so matplotlib renders it transparent / off-colormap).
ATLAS = [
    ("01_wind_curl.png",
     ("data/wind_stress.json", "wind_curl"),
     dict(title="Wind stress curl  (ERA5 1991-2020 mean, Pa/m)", cmap="RdBu_r", diverging=True),
     False),
    ("02_tau_x.png",
     ("data/wind_stress.json", "tau_x"),
     dict(title="Zonal wind stress τ_x  (ERA5, Pa)", cmap="RdBu_r", diverging=True),
     False),
    ("03_tau_y.png",
     ("data/wind_stress.json", "tau_y"),
     dict(title="Meridional wind stress τ_y  (ERA5, Pa)", cmap="RdBu_r", diverging=True),
     False),
    ("04_ocean_mask.png",
     ("mask", "data/mask.json"),
     dict(title="Ocean mask  (1=ocean, 0=land; ETOPO1)", cmap="Blues", diverging=False),
     False),
    ("05_bathymetry.png",
     ("data/bathymetry.json", "depth"),
     dict(title="Bathymetry / elevation  (ETOPO1, unsigned magnitude, m)", cmap="terrain", diverging=False),
     False),
    ("06_sst.png",
     ("data/sst.json", "sst"),
     dict(title="Sea surface temperature  (NOAA OISST 2015-2023, °C)", cmap="RdYlBu_r", diverging=False),
     True),
    ("07_salinity.png",
     ("data/salinity.json", "salinity"),
     dict(title="Surface salinity  (WOA23, psu)", cmap="viridis", diverging=False),
     True),
    ("08_deep_temp.png",
     ("data/deep_temp.json", "temp"),
     dict(title="Temperature at 1000 m  (WOA23, °C)", cmap="RdYlBu_r", diverging=False),
     True),
    ("09_mixed_layer_depth.png",
     ("data/mixed_layer_depth.json", "mld"),
     dict(title="Mixed layer depth  (heuristic, m)", cmap="viridis", diverging=False),
     True),
    ("10_precipitation.png",
     ("data/precipitation.json", "precipitation"),
     dict(title="Precipitation  (GPM IMERG 2015-2023, mm/yr)", cmap="Blues", diverging=False),
     False),
    ("11_evaporation.png",
     ("data/evaporation.json", "evaporation"),
     dict(title="Evaporation  (ERA5 Land 2015-2023, mm/yr)", cmap="Oranges", diverging=False),
     False),
    ("12_air_temp.png",
     ("data/air_temp.json", "air_temp"),
     dict(title="2 m air temperature  (ERA5 2015-2023, °C)", cmap="RdYlBu_r", diverging=False),
     False),
    ("13_cloud_fraction.png",
     ("data/cloud_fraction.json", "cloud_fraction"),
     dict(title="Cloud fraction  (MODIS 2020-2023)", cmap="Greys_r", diverging=False),
     False),
    ("14_albedo.png",
     ("data/albedo.json", "albedo"),
     dict(title="Surface albedo  (MODIS MCD43A3)", cmap="cividis", diverging=False),
     False),
    ("15_sea_ice.png",
     ("data/sea_ice.json", "ice_fraction"),
     dict(title="Sea ice fraction  (NOAA OISST 2015-2023)", cmap="Blues_r", diverging=False),
     True),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=1024)
    p.add_argument("--ny", type=int, default=512)
    p.add_argument("--out", default="output/atlas")
    args = p.parse_args()

    grid = Grid.create(nx=args.nx, ny=args.ny, lat0=-79.5, lat1=79.5)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load the ocean mask once, for ocean-only field display.
    ocean_mask = load_mask(REPO_ROOT / "data" / "mask.json", grid=grid)
    ocean_mask_np = np.asarray(ocean_mask)

    print(f"Rendering atlas at {args.nx}x{args.ny} -> {out_dir}/")

    for fname, spec, kw, ocean_only in ATLAS:
        try:
            kind = spec[0]
            if kind == "mask":
                arr = load_mask(REPO_ROOT / spec[1], grid=grid)
            else:
                path, key = spec
                arr = load_to_grid(REPO_ROOT / path, key, grid)
            if arr is None:
                print(f"  skipped {fname}: load returned None")
                continue
            arr_np = np.asarray(arr)
            if ocean_only:
                arr_np = np.where(ocean_mask_np > 0.5, arr_np, np.nan)
            render_field(arr_np, grid, out_dir / fname, **kw)
            finite = arr_np[np.isfinite(arr_np)]
            mn = float(finite.min()) if finite.size else float("nan")
            mx = float(finite.max()) if finite.size else float("nan")
            print(f"  {fname}  range=[{mn:+.3g}, {mx:+.3g}]")
        except FileNotFoundError as e:
            print(f"  MISSING {fname}: {e.filename}")
        except Exception as e:
            print(f"  ERROR {fname}: {type(e).__name__}: {e}")

    print(f"Done. {sum(1 for f in out_dir.iterdir() if f.suffix == '.png')} PNGs in {out_dir}")


if __name__ == "__main__":
    main()
