"""Load observational JSON fields and resample to the model grid.

The existing data files (`*_1deg.json`) store flat row-major arrays of length
nx*ny where lat0/lat1/lon0/lon1 in the JSON are *cell-center* extremes — i.e.,
lat0 is the latitude of row 0's center, not the bottom edge of row 0. The
target Grid in this package uses cell-edge bounds, so we interpolate from
source cell centers to target cell centers.

Longitude is treated as periodic (360°). Latitude is clamped to source range.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from .grid import Grid


def load_field(path: str | Path, field: str) -> jnp.ndarray:
    """Load a (ny, nx) array from a 1deg JSON file."""
    with open(path) as f:
        d = json.load(f)
    nx, ny = int(d["nx"]), int(d["ny"])
    arr = jnp.asarray(d[field], dtype=jnp.float32).reshape(ny, nx)
    return arr


def field_extents(path: str | Path) -> dict:
    """Return source grid metadata: nx, ny, and cell-center lat/lon arrays."""
    with open(path) as f:
        d = json.load(f)
    nx, ny = int(d["nx"]), int(d["ny"])
    lat = np.linspace(float(d["lat0"]), float(d["lat1"]), ny)
    lon = np.linspace(float(d["lon0"]), float(d["lon1"]), nx)
    return {"nx": nx, "ny": ny, "lat": lat, "lon": lon}


def resample_to_grid(
    src: jnp.ndarray,
    *,
    src_lat: Sequence[float] | np.ndarray,
    src_lon: Sequence[float] | np.ndarray,
    grid: Grid,
) -> jnp.ndarray:
    """Bilinear-interpolate source field onto the model grid.

    Args:
        src: (ny_src, nx_src) array of source values.
        src_lat: (ny_src,) cell-center latitudes (degrees, ascending).
        src_lon: (nx_src,) cell-center longitudes (degrees).
        grid: target Grid.

    Returns: (grid.ny, grid.nx) array.

    Periodic in longitude. Latitude is clamped to the source range.
    """
    src = jnp.asarray(src)
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)
    ny_src, nx_src = src.shape

    # Latitude: linear coordinate on [src_lat[0], src_lat[-1]].
    tgt_lat = np.asarray(grid.lat, dtype=np.float64)
    fj = (tgt_lat - src_lat[0]) / (src_lat[-1] - src_lat[0]) * (ny_src - 1)
    fj = np.clip(fj, 0.0, ny_src - 1 - 1e-9)
    j0 = np.floor(fj).astype(np.int32)
    j1 = np.minimum(j0 + 1, ny_src - 1)
    tj = fj - j0  # (ny,)

    # Longitude: periodic. Convert target lon to fractional source-cell index.
    # Source assumed evenly spaced; first center at src_lon[0], step dlon_src.
    dlon_src = (src_lon[-1] - src_lon[0]) / (nx_src - 1) if nx_src > 1 else 360.0 / nx_src
    tgt_lon = np.asarray(grid.lon, dtype=np.float64)
    fi = (tgt_lon - src_lon[0]) / dlon_src
    fi = np.mod(fi, nx_src)  # periodic wrap
    i0 = np.floor(fi).astype(np.int32) % nx_src
    i1 = (i0 + 1) % nx_src
    ti = fi - np.floor(fi)  # (nx,)

    # Gather the four corners. Shapes: (ny, nx).
    j0v = jnp.asarray(j0)[:, None]
    j1v = jnp.asarray(j1)[:, None]
    i0v = jnp.asarray(i0)[None, :]
    i1v = jnp.asarray(i1)[None, :]
    s00 = src[j0v, i0v]
    s01 = src[j0v, i1v]
    s10 = src[j1v, i0v]
    s11 = src[j1v, i1v]

    tj_b = jnp.asarray(tj, dtype=src.dtype)[:, None]
    ti_b = jnp.asarray(ti, dtype=src.dtype)[None, :]
    out = (
        s00 * (1 - tj_b) * (1 - ti_b)
        + s01 * (1 - tj_b) * ti_b
        + s10 * tj_b * (1 - ti_b)
        + s11 * tj_b * ti_b
    )
    return out


def load_to_grid(path: str | Path, field: str, grid: Grid) -> jnp.ndarray:
    """Convenience: load a JSON field and resample to a target grid."""
    meta = field_extents(path)
    src = load_field(path, field)
    return resample_to_grid(src, src_lat=meta["lat"], src_lon=meta["lon"], grid=grid)
