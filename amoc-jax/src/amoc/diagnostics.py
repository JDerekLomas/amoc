"""Diagnostics computed from simulation state.

For v1b the headline diagnostic is the meridional overturning streamfunction
Ψ(y) over a chosen basin (Atlantic by default for AMOC). It's the
zonally-integrated, depth-integrated meridional velocity.

In our 2-layer non-dimensional units:
    v_k = (1/cos φ) ∂_λ ψ_k                  (per layer)
    Ψ_AMOC(y) = H_s · <v_s>_basin - H_d · <v_d>_basin    (depth-integrated)

Sign convention: positive Ψ means surface flow is northward (warm water
heading toward NADW formation), deep flow is southward — the canonical
AMOC sense.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .grid import Grid
from .state import State


def basin_mask(grid: Grid, lon_min: float, lon_max: float, ocean_mask=None) -> np.ndarray:
    """Return a (ny, nx) boolean mask over [lon_min, lon_max] degrees."""
    lon = np.asarray(grid.lon)
    mask_x = (lon >= lon_min) & (lon <= lon_max)
    out = np.broadcast_to(mask_x[None, :], grid.shape).astype(np.float32).copy()
    if ocean_mask is not None:
        out *= np.asarray(ocean_mask)
    return out


def meridional_velocity(psi: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """v = (1/cos φ) ∂_λψ. Centered, periodic. Grid units."""
    dpsi_dx = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) * 0.5
    cos_lat = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    return dpsi_dx / cos_lat


def amoc_streamfunction(
    state: State, grid: Grid,
    *, lon_min: float = -80.0, lon_max: float = 0.0,
    H_s: float = 100.0, H_d: float = 3900.0,
    ocean_mask=None,
) -> np.ndarray:
    """Atlantic-basin meridional overturning streamfunction Ψ(y).

    Returns a (ny,) array. Positive = northward surface flow / southward
    deep return. Units are arbitrary in v1b (grid-unit ψ × thickness).
    """
    bmask = basin_mask(grid, lon_min, lon_max, ocean_mask=ocean_mask)
    bmask_j = jnp.asarray(bmask)
    v_s = meridional_velocity(state.psi_s, grid)
    v_d = meridional_velocity(state.psi_d, grid)
    # Zonal *sum* weighted by the basin mask, divided by basin width per row
    # (so latitudes with narrow basin don't dominate). If a row has zero
    # basin cells, set Ψ to zero there.
    width = jnp.maximum(bmask_j.sum(axis=1), 1.0)
    v_s_basin = (v_s * bmask_j).sum(axis=1) / width
    v_d_basin = (v_d * bmask_j).sum(axis=1) / width
    return np.asarray(H_s * v_s_basin - H_d * v_d_basin)


def gyre_transport(state: State, grid: Grid, *, ocean_mask=None) -> dict:
    """Quick scalar diagnostics for sanity / regression tests."""
    psi_s = np.asarray(state.psi_s)
    psi_d = np.asarray(state.psi_d)
    if ocean_mask is not None:
        ocean_mask_np = np.asarray(ocean_mask) > 0.5
        psi_s_ocean = psi_s[ocean_mask_np]
        psi_d_ocean = psi_d[ocean_mask_np]
    else:
        psi_s_ocean = psi_s.ravel()
        psi_d_ocean = psi_d.ravel()
    return {
        "psi_s_min": float(psi_s_ocean.min()),
        "psi_s_max": float(psi_s_ocean.max()),
        "psi_s_rms": float(np.sqrt((psi_s_ocean ** 2).mean())),
        "psi_d_min": float(psi_d_ocean.min()),
        "psi_d_max": float(psi_d_ocean.max()),
        "psi_d_rms": float(np.sqrt((psi_d_ocean ** 2).mean())),
    }
