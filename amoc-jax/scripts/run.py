#!/usr/bin/env python3
"""v1c driver: two-layer ocean with prognostic T, S and derived buoyancy.

Loads observed SST + SSS climatology as both initial conditions and
restoring targets; ERA5 wind curl forces the surface vorticity; the
buoyancy that drives baroclinic shear is now derived from the T,S fields
themselves via the linear EOS — no prescribed buoyancy field anymore.

Usage:
    python scripts/run.py [--nx 256 --ny 128 --steps 10000 --dt 0.005]
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
from amoc.state import Forcing, Params, State, zero_state
from amoc.step import run_with_history


REPO_ROOT     = Path(__file__).resolve().parents[2]
WIND_HIRES    = REPO_ROOT / "data" / "wind_stress.json"
WIND_FALLBACK = REPO_ROOT / "wind_stress_1deg.json"
MASK_HIRES    = REPO_ROOT / "data" / "mask.json"
MASK_FALLBACK = REPO_ROOT / "simamoc" / "mask.json"
SST_HIRES     = REPO_ROOT / "data" / "sst.json"
SST_FALLBACK  = REPO_ROOT / "sst_global_1deg.json"
SSS_HIRES     = REPO_ROOT / "data" / "salinity.json"
SSS_FALLBACK  = REPO_ROOT / "salinity_1deg.json"
DEEPT_HIRES   = REPO_ROOT / "data" / "deep_temp.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--rs", type=float, default=0.04)
    p.add_argument("--rd", type=float, default=0.10)
    p.add_argument("--A", type=float, default=2e-4)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--wind", type=float, default=0.04)
    p.add_argument("--alpha-BC", type=float, default=1.0,
                   help="thermal-wind shear forcing strength")
    p.add_argument("--Fcs", type=float, default=0.05)
    p.add_argument("--Fcd", type=float, default=0.001)
    p.add_argument("--tauT", type=float, default=60.0,
                   help="surface T restoring time-scale (model units)")
    p.add_argument("--tauS", type=float, default=180.0)
    p.add_argument("--kappa-T", type=float, default=5e-4)
    p.add_argument("--kappa-S", type=float, default=5e-4)
    p.add_argument("--out", type=str, default="output")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"jax devices: {jax.devices()}")
    print(f"grid: {args.nx} x {args.ny}")

    grid = Grid.create(nx=args.nx, ny=args.ny, lat0=-79.5, lat1=79.5)

    wind_path  = WIND_HIRES  if WIND_HIRES.exists()  else WIND_FALLBACK
    mask_path  = MASK_HIRES  if MASK_HIRES.exists()  else MASK_FALLBACK
    sst_path   = SST_HIRES   if SST_HIRES.exists()   else SST_FALLBACK
    sss_path   = SSS_HIRES   if SSS_HIRES.exists()   else SSS_FALLBACK
    print(f"  wind   {wind_path.relative_to(REPO_ROOT)}")
    print(f"  mask   {mask_path.relative_to(REPO_ROOT)}")
    print(f"  SST    {sst_path.relative_to(REPO_ROOT)}")
    print(f"  SSS    {sss_path.relative_to(REPO_ROOT)}")
    if DEEPT_HIRES.exists():
        print(f"  deep T {DEEPT_HIRES.relative_to(REPO_ROOT)}")

    curl       = load_to_grid(wind_path, "wind_curl", grid)
    ocean_mask = load_mask(mask_path, grid=grid)
    sst        = load_to_grid(sst_path, "sst", grid)
    sss        = load_to_grid(sss_path, "salinity", grid)
    deep_temp  = (load_to_grid(DEEPT_HIRES, "temp", grid)
                  if DEEPT_HIRES.exists()
                  else jnp.full(grid.shape, 4.0, dtype=jnp.float32))

    n_ocean = float(ocean_mask.sum())
    print(f"ocean: {n_ocean:.0f}/{grid.nx*grid.ny} cells "
          f"({100*n_ocean/(grid.nx*grid.ny):.1f}%)")

    # Wind: ocean-RMS-normalize.
    rms = float(jnp.sqrt(jnp.sum((curl * ocean_mask) ** 2) / jnp.maximum(n_ocean, 1.0)))
    curl_norm = (curl / rms) * ocean_mask

    # Replace land cells in T, S with the global ocean mean so the
    # Laplacian/restoring at coastlines isn't pulled toward 0.
    sst_mean  = float(jnp.sum(sst * ocean_mask) / jnp.maximum(n_ocean, 1.0))
    sss_mean  = float(jnp.sum(sss * ocean_mask) / jnp.maximum(n_ocean, 1.0))
    deepT_mean = float(jnp.sum(deep_temp * ocean_mask) / jnp.maximum(n_ocean, 1.0))
    sst_filled = jnp.where(ocean_mask > 0.5, sst, sst_mean).astype(jnp.float32)
    sss_filled = jnp.where(ocean_mask > 0.5, sss, sss_mean).astype(jnp.float32)
    deepT_filled = jnp.where(ocean_mask > 0.5, deep_temp, deepT_mean).astype(jnp.float32)

    print(f"  SST    mean={sst_mean:5.2f} °C   range=[{float(sst.min()):+.1f}, {float(sst.max()):+.1f}]")
    print(f"  SSS    mean={sss_mean:5.2f} psu  range=[{float(sss.min()):+.2f}, {float(sss.max()):+.2f}]")
    print(f"  deepT  mean={deepT_mean:5.2f} °C")

    # Save inputs as PNGs.
    render_field(np.asarray(curl_norm), grid, out_dir / "00_wind_curl.png",
                 title="Wind curl (RMS-normalized)")
    render_field(np.where(ocean_mask>0.5, np.asarray(sst), np.nan), grid,
                 out_dir / "00_sst_target.png",
                 title="SST climatology  (NOAA OISST 2015-2023, °C)",
                 cmap="RdYlBu_r", diverging=False)
    render_field(np.where(ocean_mask>0.5, np.asarray(sss), np.nan), grid,
                 out_dir / "00_sss_target.png",
                 title="SSS climatology  (WOA23, psu)",
                 cmap="viridis", diverging=False)

    forcing = Forcing(
        wind_curl=curl_norm,
        ocean_mask=ocean_mask,
        buoyancy=jnp.zeros(grid.shape),  # legacy v1b knob, unused here
        T_target=sst_filled,
        S_target=sss_filled,
        S_d_const=jnp.full(grid.shape, sss_mean, dtype=jnp.float32),
        F_fresh=jnp.zeros(grid.shape),
    )

    state0 = zero_state(grid.shape)._replace(
        T_s=sst_filled, S_s=sss_filled, T_d=deepT_filled,
    )
    params = Params(
        r_friction_s=args.rs, A_visc_s=args.A,
        r_friction_d=args.rd, A_visc_d=args.A,
        beta=args.beta, wind_strength=args.wind, dt=args.dt,
        F_couple_s=args.Fcs, F_couple_d=args.Fcd,
        alpha_buoy=0.0, alpha_BC=args.alpha_BC,
        kappa_T=args.kappa_T, kappa_S=args.kappa_S,
        tau_T=args.tauT, tau_S=args.tauS,
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

    # Diagnostics.
    diags = gyre_transport(final, grid, ocean_mask=ocean_mask)
    print("Final state diagnostics (ocean cells only):")
    for k, v in diags.items():
        print(f"  {k:<12} {v:+.4f}")
    T_s = np.asarray(final.T_s); T_target = np.asarray(sst_filled)
    om = np.asarray(ocean_mask) > 0.5
    rmse = float(np.sqrt(np.mean((T_s[om] - T_target[om]) ** 2)))
    bias = float(np.mean(T_s[om] - T_target[om]))
    print(f"  SST    rmse={rmse:.3f} °C   bias={bias:+.3f} °C  (vs restoring target)")

    if not np.all(np.isfinite(np.asarray(final.psi_s))):
        print("WARNING: ψ_s contains non-finite values — numerical blow-up.")

    render_panel(final, grid, out_dir / "99_final.png",
                 title=f"v1c steady state  (SST RMSE = {rmse:.2f} °C)",
                 ocean_mask=ocean_mask)
    print(f"  saved {out_dir / '99_final.png'}")

    # Save SST/SSS panels too.
    render_field(np.where(om, T_s, np.nan), grid, out_dir / "98_T_s.png",
                 title=f"T_s (model)  RMSE vs target = {rmse:.2f} °C",
                 cmap="RdYlBu_r", diverging=False)
    render_field(np.where(om, np.asarray(final.S_s), np.nan), grid,
                 out_dir / "98_S_s.png",
                 title="S_s (model)", cmap="viridis", diverging=False)

    np.savez(
        out_dir / "final_state.npz",
        psi_s=np.asarray(final.psi_s), zeta_s=np.asarray(final.zeta_s),
        psi_d=np.asarray(final.psi_d), zeta_d=np.asarray(final.zeta_d),
        T_s=T_s, S_s=np.asarray(final.S_s), T_d=np.asarray(final.T_d),
        lat=np.asarray(grid.lat), lon=np.asarray(grid.lon),
    )
    print(f"  saved {out_dir / 'final_state.npz'}")
    print("done.")


if __name__ == "__main__":
    main()
