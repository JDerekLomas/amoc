#!/usr/bin/env python3
"""v1a driver: spin up barotropic gyres from rest under NCEP wind curl.

Loads the existing 1deg wind stress, normalizes it, runs the JIT'd integrator
for N steps, and saves PNG diagnostics + zarr history.

Usage:
    python scripts/run.py [--nx 256 --ny 128 --steps 4000 --dt 0.05]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from amoc.data import load_mask, load_to_grid
from amoc.grid import Grid
from amoc.physics import vorticity_rhs  # noqa: F401  (keep imported for jit cache)
from amoc.render import render_field, render_panel
from amoc.state import Forcing, Params, State
from amoc.step import run_with_history


REPO_ROOT = Path(__file__).resolve().parents[2]
# Prefer 1024x512 hires fields (data/) and fall back to 1deg if not present.
WIND_HIRES = REPO_ROOT / "data" / "wind_stress.json"          # 1024x512 ERA5
WIND_FALLBACK = REPO_ROOT / "wind_stress_1deg.json"           # 360x160 NCEP
MASK_HIRES = REPO_ROOT / "data" / "mask.json"                 # 1024x512 hex-packed
MASK_FALLBACK = REPO_ROOT / "simamoc" / "mask.json"           # 360x160


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--r", type=float, default=0.04, help="linear friction")
    p.add_argument("--A", type=float, default=2.0e-4, help="Laplacian viscosity")
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--wind", type=float, default=1.0, help="wind strength multiplier")
    p.add_argument("--out", type=str, default="output")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax devices: {jax.devices()}")
    print(f"grid: {args.nx} x {args.ny}")

    grid = Grid.create(nx=args.nx, ny=args.ny, lat0=-79.5, lat1=79.5)

    wind_path = WIND_HIRES if WIND_HIRES.exists() else WIND_FALLBACK
    print(f"loading wind curl from {wind_path.relative_to(REPO_ROOT)}")
    curl = load_to_grid(wind_path, "wind_curl", grid)

    mask_path = MASK_HIRES if MASK_HIRES.exists() else MASK_FALLBACK
    print(f"loading land mask from {mask_path.relative_to(REPO_ROOT)}")
    ocean_mask = load_mask(mask_path, grid=grid)
    n_ocean = float(ocean_mask.sum())
    n_total = grid.nx * grid.ny
    print(f"ocean mask: {n_ocean:.0f}/{n_total} cells ({100*n_ocean/n_total:.1f}% ocean)")

    rms = float(jnp.sqrt(jnp.sum((curl * ocean_mask) ** 2) / jnp.maximum(n_ocean, 1.0)))
    print(f"wind curl ocean-RMS = {rms:.3e} Pa/m  (normalizing to 1.0)")
    curl_norm = (curl / rms) * ocean_mask
    forcing = Forcing(wind_curl=curl_norm, ocean_mask=ocean_mask)

    # Save the forcing + mask for sanity-checking.
    render_field(
        np.asarray(curl_norm), grid, out_dir / "00_wind_curl.png",
        title="NCEP wind curl (ocean-RMS normalized, land=0)", cmap="RdBu_r",
    )
    render_field(
        np.asarray(ocean_mask), grid, out_dir / "00_ocean_mask.png",
        title="Ocean mask (1=ocean, 0=land)", cmap="Blues", diverging=False,
    )
    print(f"  saved forcing + mask diagnostics to {out_dir}/")

    state0 = State(psi=jnp.zeros(grid.shape), zeta=jnp.zeros(grid.shape))
    params = Params(
        r_friction=args.r,
        A_visc=args.A,
        beta=args.beta,
        wind_strength=args.wind,
        dt=args.dt,
    )

    print(f"running {args.steps} steps @ dt={args.dt} (sim time {args.steps*args.dt:.0f})")
    t0 = time.time()
    final, history = run_with_history(
        state0, forcing, params, grid,
        n_steps=args.steps, save_every=args.save_every,
    )
    # Wait for async result.
    final.zeta.block_until_ready()
    elapsed = time.time() - t0
    sps = args.steps / elapsed
    print(f"  done in {elapsed:.2f}s  ({sps:.0f} steps/sec)")

    # Diagnostics on final state.
    psi = np.asarray(final.psi)
    zeta = np.asarray(final.zeta)
    print(f"  ψ  range: [{psi.min():+.3f}, {psi.max():+.3f}]")
    print(f"  ζ  range: [{zeta.min():+.3f}, {zeta.max():+.3f}]")
    print(f"  ψ  RMS:   {np.sqrt(np.mean(psi**2)):.3f}")
    if not np.all(np.isfinite(psi)):
        print("WARNING: ψ contains non-finite values (NaN/Inf). Likely numerical blow-up.")

    render_panel(final, grid, out_dir / "99_final.png", title="v1a final state")
    print(f"  saved {out_dir / '99_final.png'}")

    # Render a few history snapshots so the spin-up is visible.
    psi_hist = np.asarray(history.psi)  # (n_saves, ny, nx)
    n_saves = psi_hist.shape[0]
    chosen = [0, n_saves // 4, n_saves // 2, 3 * n_saves // 4, n_saves - 1]
    chosen = sorted(set(max(0, min(c, n_saves - 1)) for c in chosen))
    for k in chosen:
        snapshot = State(psi=jnp.asarray(psi_hist[k]), zeta=jnp.asarray(history.zeta[k]))
        step = (k + 1) * args.save_every
        render_panel(
            snapshot, grid,
            out_dir / f"snap_{step:06d}.png",
            title=f"step {step}  (sim time {step*args.dt:.0f})",
        )
    print(f"  saved {len(chosen)} snapshots to {out_dir}")

    # Save the final state for the renderer/browser viewer to pick up later.
    np.savez(
        out_dir / "final_state.npz",
        psi=psi, zeta=zeta,
        lat=np.asarray(grid.lat), lon=np.asarray(grid.lon),
        params=np.array([args.r, args.A, args.beta, args.wind, args.dt]),
    )
    print(f"  saved {out_dir / 'final_state.npz'}")
    print("done.")


if __name__ == "__main__":
    main()
