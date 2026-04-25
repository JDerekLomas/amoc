"""Matplotlib rendering of simulation state.

For v1b: 4-panel diagnostic — surface ψ, deep ψ, |U|_surface, MOC(y).
PNGs only for now; the browser viewer reads zarr later.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import amoc_streamfunction, meridional_velocity
from .grid import Grid
from .state import State


def _diverging_lim(arr: np.ndarray, percentile: float = 99.0) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    return float(np.percentile(np.abs(finite), percentile)) or 1.0


def velocities_from_psi(psi: jnp.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """u = -∂ψ/∂y, v = ∂ψ/∂x. Centered differences."""
    u = -(jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) * 0.5
    v = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) * 0.5
    u = u.at[0, :].set(0.0).at[-1, :].set(0.0)
    return np.asarray(u), np.asarray(v)


def render_panel(
    state: State,
    grid: Grid,
    out_path: str | Path,
    *,
    title: str | None = None,
    ocean_mask=None,
    moc_kwargs: dict | None = None,
):
    """Four-panel diagnostic: ψ_surface, ψ_deep, |U|_s with quivers, AMOC(y)."""
    psi_s = np.asarray(state.psi_s)
    psi_d = np.asarray(state.psi_d)
    u, v = velocities_from_psi(state.psi_s, grid)
    speed = np.hypot(u, v)

    if ocean_mask is not None:
        mask_np = np.asarray(ocean_mask) > 0.5
        # Mask off land for visual clarity.
        psi_s_p = np.where(mask_np, psi_s, np.nan)
        psi_d_p = np.where(mask_np, psi_d, np.nan)
        speed_p = np.where(mask_np, speed, np.nan)
    else:
        psi_s_p, psi_d_p, speed_p = psi_s, psi_d, speed

    extent = [grid.lon0, grid.lon1, grid.lat0, grid.lat1]
    lim_psi_s = _diverging_lim(psi_s_p)
    lim_psi_d = _diverging_lim(psi_d_p)

    fig = plt.figure(figsize=(11, 11), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.7])
    if title:
        fig.suptitle(title, fontsize=11)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(psi_s_p, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-lim_psi_s, vmax=lim_psi_s, aspect="auto")
    ax.set_title(r"surface streamfunction $\psi_s$  (red=anticyclonic, blue=cyclonic)")
    ax.set_ylabel("latitude")
    plt.colorbar(im, ax=ax, shrink=0.85)

    ax = fig.add_subplot(gs[1])
    im = ax.imshow(psi_d_p, origin="lower", extent=extent, cmap="PuOr_r",
                   vmin=-lim_psi_d, vmax=lim_psi_d, aspect="auto")
    ax.set_title(r"deep streamfunction $\psi_d$")
    ax.set_ylabel("latitude")
    plt.colorbar(im, ax=ax, shrink=0.85)

    ax = fig.add_subplot(gs[2])
    im = ax.imshow(speed_p, origin="lower", extent=extent, cmap="viridis", aspect="auto")
    ax.set_title(r"surface speed $|U_s|$  (with current vectors)")
    ax.set_ylabel("latitude")
    plt.colorbar(im, ax=ax, shrink=0.85)
    sx = max(1, grid.nx // 32)
    sy = max(1, grid.ny // 16)
    lon_q = np.asarray(grid.lon)[::sx]
    lat_q = np.asarray(grid.lat)[::sy]
    ax.quiver(lon_q, lat_q, u[::sy, ::sx], v[::sy, ::sx],
              color="white", scale=None, width=0.002, alpha=0.7)

    # AMOC streamfunction Ψ(y) over the Atlantic.
    moc_kwargs = moc_kwargs or {}
    psi_amoc = amoc_streamfunction(state, grid, ocean_mask=ocean_mask, **moc_kwargs)
    lat = np.asarray(grid.lat)
    ax = fig.add_subplot(gs[3])
    ax.plot(lat, psi_amoc, color="#2c7fb8", linewidth=2)
    ax.axhline(0, color="#444", linewidth=0.5)
    ax.fill_between(lat, 0, psi_amoc, alpha=0.25, color="#2c7fb8")
    ax.set_title(r"Atlantic meridional overturning $\Psi_{\rm AMOC}(y)$"
                 r"  (positive = surface northward)")
    ax.set_xlabel("latitude")
    ax.set_ylabel("Ψ (model units)")
    ax.set_xlim(grid.lat0, grid.lat1)
    ax.grid(True, alpha=0.2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_field(
    field: np.ndarray,
    grid: Grid,
    out_path: str | Path,
    *,
    title: str | None = None,
    cmap: str = "RdBu_r",
    diverging: bool = True,
):
    """Single-panel field plot — useful for forcing fields, etc."""
    arr = np.asarray(field)
    extent = [grid.lon0, grid.lon1, grid.lat0, grid.lat1]
    if diverging:
        lim = _diverging_lim(arr)
        kw = dict(cmap=cmap, vmin=-lim, vmax=lim)
    else:
        kw = dict(cmap=cmap)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", **kw)
    if title:
        ax.set_title(title)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    plt.colorbar(im, ax=ax, shrink=0.85)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
