#!/usr/bin/env python3
"""Quantitative scorer for a simulation final state.

Evaluates whether the output matches what an "impressive" ocean simulation
should look like:

  - magnitude_balance: |ψ_min| / |ψ_max| >= 0.3   (both gyre signs present)
  - wbi_ratio:        max|U| in west / max|U| in interior >= 2.5
  - n_extrema_NH/SH:  count of distinct local ψ extrema in Atlantic strip
  - amoc_sign:        Atlantic Ψ(y) at 30-45N positive

Returns a dict of scores and a boolean "impressive" verdict that drives
the iteration loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp


def _velocities(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) * 0.5
    v =  (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) * 0.5
    u[0,:] = 0; u[-1,:] = 0
    return u, v


def evaluate(state, grid, ocean_mask) -> dict:
    psi = np.asarray(state.psi_s)
    om = np.asarray(ocean_mask) > 0.5
    psi_ocean = psi[om]
    if psi_ocean.size == 0 or not np.all(np.isfinite(psi_ocean)):
        return {"impressive": False, "reason": "non-finite or empty"}

    psi_min = float(psi_ocean.min())
    psi_max = float(psi_ocean.max())
    mag_balance = min(abs(psi_min), abs(psi_max)) / max(abs(psi_min), abs(psi_max), 1e-9)

    # Western boundary intensification: compare max speed in 0-15% of each
    # latitude row (western edge of any basin) vs 25-75% (interior).
    u, v = _velocities(psi)
    speed = np.hypot(u, v)
    speed_ocean = np.where(om, speed, np.nan)
    nx = psi.shape[1]
    west_band = speed_ocean[:, : int(0.15 * nx)]
    interior   = speed_ocean[:, int(0.25 * nx): int(0.75 * nx)]
    max_west = float(np.nanmax(west_band)) if np.isfinite(np.nanmax(west_band)) else 0.0
    max_int  = float(np.nanmax(interior)) if np.isfinite(np.nanmax(interior)) else 1e-9
    wbi_ratio = max_west / max(max_int, 1e-9)

    # Local extrema: find points where ψ is min/max in a 5x5 neighborhood,
    # restricted to ocean cells in the Atlantic strip (lon -80..0).
    lat = np.asarray(grid.lat)
    lon = np.asarray(grid.lon)
    atl_x = (lon >= -80) & (lon <= 0)
    nh_y = (lat >= 5) & (lat <= 60)
    sh_y = (lat <= -5) & (lat >= -60)

    def count_extrema(strip_psi: np.ndarray, strip_mask: np.ndarray) -> int:
        from scipy.ndimage import maximum_filter, minimum_filter
        psi_m = np.where(strip_mask, strip_psi, np.nan)
        if not np.any(strip_mask):
            return 0
        # local maxima
        mx = maximum_filter(np.nan_to_num(psi_m, nan=-1e30), size=7) == psi_m
        mn = minimum_filter(np.nan_to_num(psi_m, nan=1e30), size=7) == psi_m
        # Filter to require values significantly different from zero
        thresh = 0.1 * max(abs(psi_min), abs(psi_max))
        is_signif = np.abs(psi_m) > thresh
        n_max = int(np.sum(mx & is_signif & strip_mask))
        n_min = int(np.sum(mn & is_signif & strip_mask))
        return n_max + n_min

    atl_psi = psi
    nh_atl_mask = om & np.outer(nh_y, atl_x)
    sh_atl_mask = om & np.outer(sh_y, atl_x)
    n_nh = count_extrema(atl_psi, nh_atl_mask)
    n_sh = count_extrema(atl_psi, sh_atl_mask)

    # AMOC: average meridional v in Atlantic between 30-45N. Positive = good.
    nh_band = (lat >= 30) & (lat <= 45)
    v_atl_nh = v[np.outer(nh_band, atl_x) & om]
    amoc_proxy = float(v_atl_nh.mean()) if v_atl_nh.size else 0.0

    scores = {
        "psi_min": psi_min, "psi_max": psi_max,
        "magnitude_balance": mag_balance,
        "wbi_ratio": wbi_ratio,
        "max_speed_west": max_west, "max_speed_interior": max_int,
        "n_extrema_nh_atlantic": n_nh,
        "n_extrema_sh_atlantic": n_sh,
        "amoc_proxy_30_45N": amoc_proxy,
    }
    passes = {
        "balance":   mag_balance >= 0.3,
        "wbi":       wbi_ratio >= 2.5,
        "nh_gyres":  n_nh >= 2,
        "sh_gyres":  n_sh >= 1,
    }
    scores["passes"] = passes
    scores["impressive"] = all(passes.values())
    return scores


def print_report(scores: dict, *, label: str = ""):
    if label:
        print(f"=== {label} ===")
    print(f"  ψ range:          [{scores['psi_min']:+.2f}, {scores['psi_max']:+.2f}]")
    print(f"  magnitude balance: {scores['magnitude_balance']:.3f}  (want ≥ 0.30)  "
          + ("✓" if scores['passes']['balance'] else "✗"))
    print(f"  WBI ratio:         {scores['wbi_ratio']:.2f}  "
          f"(west {scores['max_speed_west']:.2f} / int {scores['max_speed_interior']:.2f})  "
          f"(want ≥ 2.5)  " + ("✓" if scores['passes']['wbi'] else "✗"))
    print(f"  NH Atl extrema:    {scores['n_extrema_nh_atlantic']}  (want ≥ 2)  "
          + ("✓" if scores['passes']['nh_gyres'] else "✗"))
    print(f"  SH Atl extrema:    {scores['n_extrema_sh_atlantic']}  (want ≥ 1)  "
          + ("✓" if scores['passes']['sh_gyres'] else "✗"))
    print(f"  AMOC proxy 30-45N: {scores['amoc_proxy_30_45N']:+.4f}")
    print(f"  IMPRESSIVE: {'YES ✓' if scores['impressive'] else 'NO ✗'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="path to final_state.npz")
    p.add_argument("--mask", default="../data/mask.json")
    args = p.parse_args()

    from amoc.grid import Grid
    from amoc.data import load_mask

    d = np.load(args.npz)
    nx = d["lon"].size; ny = d["lat"].size
    grid = Grid.create(nx=nx, ny=ny)
    mask = load_mask(args.mask)
    if mask.shape != (ny, nx):
        from amoc.data import resample_to_grid
        src_lat = np.linspace(-79.5, 79.5, mask.shape[0])
        src_lon = np.linspace(-180, 180, mask.shape[1], endpoint=False)
        mask = np.asarray(resample_to_grid(mask, src_lat=src_lat, src_lon=src_lon, grid=grid))
        mask = (mask >= 0.5).astype(np.float32)

    class _S:
        psi_s = d["psi_s"]
    scores = evaluate(_S, grid, mask)
    print_report(scores, label=Path(args.npz).parent.name)
