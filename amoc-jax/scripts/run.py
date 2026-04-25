#!/usr/bin/env python3
"""v1b driver: two-layer ocean with prescribed-buoyancy MOC.

Loads ERA5 wind curl + the hex-packed ocean mask + NOAA SST (used as the
prescribed buoyancy field b ~ -alpha_T * SST), runs the JIT'd two-layer
integrator, and saves a 4-panel diagnostic PNG plus an .npz of the final
state.

Usage:
    python scripts/run.py [--nx 256 --ny 128 --steps 10000 --dt 0.01]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from amoc.data import load_mask, load_to_grid
from amoc.diagnostics import gyre_transport
from amoc.grid import Grid
from amoc.render import render_field, render_panel
from amoc.state import Forcing, Params, State
from amoc.step import run_with_history


REPO_ROOT = Path(__file__).resolve().parents[2]
WIND_HIRES    = REPO_ROOT / "data" / "wind_stress.json"
WIND_FALLBACK = REPO_ROOT / "wind_stress_1deg.json"
MASK_HIRES    = REPO_ROOT / "data" / "mask.json"
MASK_FALLBACK = REPO_ROOT / "simamoc"   / "mask.json"
SST_HIRES     = REPO_ROOT / "data" / "sst.json"
SST_FALLBACK  = REPO_ROOT / "sst_global_1deg.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--rs", type=float, default=0.04, help="surface friction")
    p.add_argument("--rd", type=float, default=0.10, help="deep friction")
    p.add_argument("--A", type=float, default=2.0e-4, help="viscosity (both layers)")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--wind", type=float, default=0.04, help="wind strength multiplier")
    p.add_argument("--alpha", type=float, default=0.05, help="buoyancy coupling strength")
    p.add_argument("--Fcs", type=float, default=0.5, help="surface->deep coupling")
    p.add_argument("--Fcd", type=float, default=0.0125, help="deep->surface coupling")
    p.add_argument("--out", type=str, default="output")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax devices: {jax.devices()}")
    print(f"grid: {args.nx} x {args.ny}")

    grid = Grid.create(nx=args.nx, ny=args.ny, lat0=-79.5, lat1=79.5)

    wind_path = WIND_HIRES if WIND_HIRES.exists() else WIND_FALLBACK
    mask_path = MASK_HIRES if MASK_HIRES.exists() else MASK_FALLBACK
    sst_path  = SST_HIRES  if SST_HIRES.exists()  else SST_FALLBACK
    print(f"loading wind curl   from {wind_path.relative_to(REPO_ROOT)}")
    print(f"loading land mask   from {mask_path.relative_to(REPO_ROOT)}")
    print(f"loading SST (->buoy) from {sst_path.relative_to(REPO_ROOT)}")

    curl = load_to_grid(wind_path, "wind_curl", grid)
    ocean_mask = load_mask(mask_path, grid=grid)
    # SST: use whichever key is present (1deg uses "sst", hires uses "sst")
    sst = load_to_grid(sst_path, "sst", grid)

    n_ocean = float(ocean_mask.sum())
    n_total = grid.nx * grid.ny
    print(f"ocean mask: {n_ocean:.0f}/{n_total} cells ({100*n_ocean/n_total:.1f}% ocean)")

    # Wind curl: ocean-RMS-normalize.
    rms = float(jnp.sqrt(jnp.sum((curl * ocean_mask) ** 2) / jnp.maximum(n_ocean, 1.0)))
    curl_norm = (curl / rms) * ocean_mask
    print(f"wind curl ocean-RMS = {rms:.3e} -> normalized to 1.0")

    # Buoyancy: derive from SST climatology, remove ocean mean so the
    # zonal gradient (the only thing the physics sees) is unaffected by
    # the offset, and ocean-RMS-normalize to unit magnitude. Sign: warm
    # water = positively buoyant (rises) so b ∝ +SST anomaly.
    sst_mean_ocean = float(jnp.sum(sst * ocean_mask) / jnp.maximum(n_ocean, 1.0))
    sst_anom = (sst - sst_mean_ocean) * ocean_mask
    sst_rms = float(jnp.sqrt(jnp.sum(sst_anom ** 2) / jnp.maximum(n_ocean, 1.0)))
    buoyancy = (sst_anom / sst_rms).astype(jnp.float32) * ocean_mask
    print(f"SST mean (ocean): {sst_mean_ocean:.2f} °C; SST RMS: {sst_rms:.2f} °C")
    print(f"  buoyancy (normalized) range "
          f"[{float(buoyancy.min()):+.2f}, {float(buoyancy.max()):+.2f}]")

    forcing = Forcing(wind_curl=curl_norm, ocean_mask=ocean_mask, buoyancy=buoyancy)

    # Save inputs as PNGs.
    render_field(np.asarray(curl_norm), grid, out_dir / "00_wind_curl.png",
                 title="Wind curl (ocean-RMS-normalized, land=0)")
    render_field(np.asarray(ocean_mask), grid, out_dir / "00_ocean_mask.png",
                 title="Ocean mask", cmap="Blues", diverging=False)
    render_field(np.asarray(buoyancy), grid, out_dir / "00_buoyancy.png",
                 title="Prescribed buoyancy b = SST anomaly  (warm = positive)",
                 cmap="RdYlBu_r", diverging=True)
    print(f"  saved input diagnostics to {out_dir}/")

    state0 = State(
        psi_s=jnp.zeros(grid.shape), zeta_s=jnp.zeros(grid.shape),
        psi_d=jnp.zeros(grid.shape), zeta_d=jnp.zeros(grid.shape),
    )
    params = Params(
        r_friction_s=args.rs, A_visc_s=args.A,
        r_friction_d=args.rd, A_visc_d=args.A,
        beta=args.beta, wind_strength=args.wind, dt=args.dt,
        F_couple_s=args.Fcs, F_couple_d=args.Fcd, alpha_buoy=args.alpha,
    )

    print(f"running {args.steps} steps @ dt={args.dt} (sim time {args.steps*args.dt:.1f})")
    t0 = time.time()
    final, history = run_with_history(
        state0, forcing, params, grid,
        n_steps=args.steps, save_every=args.save_every,
    )
    final.zeta_s.block_until_ready()
    elapsed = time.time() - t0
    sps = args.steps / elapsed
    print(f"  done in {elapsed:.2f}s  ({sps:.0f} steps/sec)")

    diags = gyre_transport(final, grid, ocean_mask=ocean_mask)
    print("Final state diagnostics (ocean cells only):")
    for k, v in diags.items():
        print(f"  {k:<12} {v:+.4f}")
    if not np.all(np.isfinite(np.asarray(final.psi_s))):
        print("WARNING: ψ_s contains non-finite values — numerical blow-up.")

    render_panel(final, grid, out_dir / "99_final.png",
                 title="v1b two-layer steady state",
                 ocean_mask=ocean_mask)
    print(f"  saved {out_dir / '99_final.png'}")

    # Snapshot sequence for spin-up visualization.
    psi_s_h = np.asarray(history.psi_s)
    n_saves = psi_s_h.shape[0]
    chosen = sorted(set(max(0, min(c, n_saves - 1))
                        for c in [0, n_saves // 4, n_saves // 2, 3 * n_saves // 4, n_saves - 1]))
    for k in chosen:
        snapshot = State(
            psi_s=jnp.asarray(history.psi_s[k]),
            zeta_s=jnp.asarray(history.zeta_s[k]),
            psi_d=jnp.asarray(history.psi_d[k]),
            zeta_d=jnp.asarray(history.zeta_d[k]),
        )
        step = (k + 1) * args.save_every
        render_panel(snapshot, grid,
                     out_dir / f"snap_{step:06d}.png",
                     title=f"step {step}  (sim time {step*args.dt:.1f})",
                     ocean_mask=ocean_mask)
    print(f"  saved {len(chosen)} snapshots to {out_dir}")

    np.savez(
        out_dir / "final_state.npz",
        psi_s=np.asarray(final.psi_s),  zeta_s=np.asarray(final.zeta_s),
        psi_d=np.asarray(final.psi_d),  zeta_d=np.asarray(final.zeta_d),
        lat=np.asarray(grid.lat), lon=np.asarray(grid.lon),
    )
    print(f"  saved {out_dir / 'final_state.npz'}")
    print("done.")


if __name__ == "__main__":
    main()
