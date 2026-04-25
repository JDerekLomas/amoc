"""Matplotlib rendering of simulation state. PNG-only for now; the browser
viewer will read the same fields from zarr later.

Conventions:
- ψ is plotted in centered diverging colormap (positive/negative gyres).
- ζ is plotted in centered diverging colormap.
- Currents (u = -∂ψ/∂y, v = ∂ψ/∂x) shown as |U| with optional quivers.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .grid import Grid
from .state import State


def _diverging_lim(arr: np.ndarray, percentile: float = 99.0) -> float:
    """Symmetric color limit at given percentile, robust to outliers."""
    return float(np.percentile(np.abs(arr), percentile))


def velocities_from_psi(psi: jnp.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """u = -∂ψ/∂y, v = ∂ψ/∂x. Centered differences in grid units."""
    u = -(jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) * 0.5
    v = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) * 0.5
    # Zero out velocities at the y-boundaries (Dirichlet ψ=0 outside).
    u = u.at[0, :].set(0.0).at[-1, :].set(0.0)
    return np.asarray(u), np.asarray(v)


def render_panel(
    state: State,
    grid: Grid,
    out_path: str | Path,
    *,
    title: str | None = None,
):
    """Three-panel diagnostic: ψ, ζ, |U| with quivers."""
    psi = np.asarray(state.psi)
    zeta = np.asarray(state.zeta)
    u, v = velocities_from_psi(state.psi, grid)
    speed = np.hypot(u, v)

    extent = [grid.lon0, grid.lon1, grid.lat0, grid.lat1]
    lim_psi = _diverging_lim(psi) or 1.0
    lim_zeta = _diverging_lim(zeta) or 1.0

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=11)

    ax = axes[0]
    im = ax.imshow(
        psi, origin="lower", extent=extent, cmap="RdBu_r",
        vmin=-lim_psi, vmax=lim_psi, aspect="auto",
    )
    ax.set_title(r"streamfunction $\psi$")
    ax.set_ylabel("latitude")
    plt.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1]
    im = ax.imshow(
        zeta, origin="lower", extent=extent, cmap="PuOr_r",
        vmin=-lim_zeta, vmax=lim_zeta, aspect="auto",
    )
    ax.set_title(r"vorticity $\zeta$")
    ax.set_ylabel("latitude")
    plt.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[2]
    im = ax.imshow(
        speed, origin="lower", extent=extent, cmap="viridis", aspect="auto",
    )
    ax.set_title(r"speed $|U|$")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    plt.colorbar(im, ax=ax, shrink=0.85)
    # Sparse quivers so the arrows are readable.
    sx = max(1, grid.nx // 32)
    sy = max(1, grid.ny // 16)
    lon_q = np.asarray(grid.lon)[::sx]
    lat_q = np.asarray(grid.lat)[::sy]
    ax.quiver(
        lon_q, lat_q, u[::sy, ::sx], v[::sy, ::sx],
        color="white", scale=None, width=0.002, alpha=0.7,
    )

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
        lim = _diverging_lim(arr) or 1.0
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
