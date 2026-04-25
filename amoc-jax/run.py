#!/usr/bin/env python3
"""Run the full coupled ocean-atmosphere simulation.

Usage: python run.py [--steps N] [--nx NX] [--ny NY]
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from amoc.grid import make_grid
from amoc.data import build_forcing, build_initial_state
from amoc.step import step, run
from amoc.state import Params

# Resolve data directory (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=5e-5)
    parser.add_argument("--batch", type=int, default=50,
                        help="Steps per JIT batch (for scan)")
    parser.add_argument("--render", action="store_true", default=True)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Grid: {args.nx}x{args.ny}, dt={args.dt}, steps={args.steps}")

    # Build grid
    grid = make_grid(args.nx, args.ny)
    print(f"Grid lat range: [{float(grid.lat[0]):.1f}, {float(grid.lat[-1]):.1f}]")

    # Build forcing from observational data
    print(f"Loading data from {DATA_DIR}...")
    t0 = time.time()
    forcing = build_forcing(DATA_DIR, grid)
    print(f"  Forcing built in {time.time()-t0:.1f}s")
    ocean_frac = float(jnp.mean(forcing.ocean_mask))
    print(f"  Ocean fraction: {ocean_frac:.1%}")

    # Build initial state
    state = build_initial_state(DATA_DIR, grid, forcing)
    print(f"  SST range: [{float(jnp.min(state.T_s[forcing.ocean_mask > 0.5])):.1f}, "
          f"{float(jnp.max(state.T_s))}]")
    print(f"  Air temp range: [{float(jnp.min(state.air_temp)):.1f}, "
          f"{float(jnp.max(state.air_temp)):.1f}]")

    params = Params(dt=args.dt)

    # Warm up JIT
    print("\nJIT compiling (first step)...")
    t0 = time.time()
    state = step(state, forcing, params, grid)
    jax.block_until_ready(state.T_s)
    print(f"  JIT compile: {time.time()-t0:.1f}s")

    # Check for NaN
    has_nan = bool(jnp.any(jnp.isnan(state.T_s)))
    print(f"  NaN after first step: {has_nan}")
    if has_nan:
        print("  ABORTING — NaN on first step. Check physics parameters.")
        return

    # Run simulation in batches
    print(f"\nRunning {args.steps} steps in batches of {args.batch}...")
    n_batches = args.steps // args.batch
    remaining = args.steps % args.batch

    t_start = time.time()
    for b in range(n_batches):
        state = run(state, forcing, params, grid, args.batch)
        jax.block_until_ready(state.T_s)

        # Progress
        done = (b + 1) * args.batch
        elapsed = time.time() - t_start
        rate = done / elapsed
        sst_min = float(jnp.min(state.T_s[forcing.ocean_mask > 0.5]))
        sst_max = float(jnp.max(state.T_s))
        sst_mean = float(jnp.mean(state.T_s[forcing.ocean_mask > 0.5]))
        has_nan = bool(jnp.any(jnp.isnan(state.T_s)))
        print(f"  step {done:5d}/{args.steps} | "
              f"{rate:.0f} steps/s | "
              f"SST [{sst_min:.1f}, {sst_mean:.1f}, {sst_max:.1f}] | "
              f"NaN: {has_nan}")
        if has_nan:
            print("  ABORTING — NaN detected")
            break

    if remaining > 0 and not has_nan:
        state = run(state, forcing, params, grid, remaining)
        jax.block_until_ready(state.T_s)

    total_time = time.time() - t_start
    print(f"\nDone: {args.steps} steps in {total_time:.1f}s "
          f"({args.steps/total_time:.0f} steps/s)")

    # Final diagnostics
    mask = forcing.ocean_mask > 0.5
    print(f"\nFinal state:")
    print(f"  SST:      [{float(jnp.min(state.T_s[mask])):.2f}, "
          f"{float(jnp.mean(state.T_s[mask])):.2f}, "
          f"{float(jnp.max(state.T_s))}]")
    print(f"  Deep T:   [{float(jnp.min(state.T_d[mask])):.2f}, "
          f"{float(jnp.mean(state.T_d[mask])):.2f}, "
          f"{float(jnp.max(state.T_d))}]")
    print(f"  Salinity: [{float(jnp.min(state.S_s[mask])):.2f}, "
          f"{float(jnp.mean(state.S_s[mask])):.2f}, "
          f"{float(jnp.max(state.S_s))}]")
    print(f"  Psi_s:    [{float(jnp.min(state.psi_s)):.4f}, "
          f"{float(jnp.max(state.psi_s)):.4f}]")
    print(f"  Air temp: [{float(jnp.min(state.air_temp)):.2f}, "
          f"{float(jnp.max(state.air_temp)):.2f}]")
    print(f"  Moisture: [{float(jnp.min(state.moisture)):.5f}, "
          f"{float(jnp.max(state.moisture)):.5f}]")
    print(f"  Sim time: {float(state.sim_time):.4f}")

    # Render if requested
    if args.render:
        try:
            from amoc.render import render_panel, render_field
            out_dir = Path(__file__).parent / "output"
            out_dir.mkdir(exist_ok=True)

            print(f"\nRendering to {out_dir}/...")
            render_panel(state, grid, out_dir / "panel.png",
                         title=f"Step {args.steps}, sim_time={float(state.sim_time):.3f}",
                         ocean_mask=forcing.ocean_mask)

            render_field(np.asarray(state.T_s), grid,
                         out_dir / "sst.png",
                         title="Surface Temperature (C)",
                         cmap="RdYlBu_r", diverging=False)

            render_field(np.asarray(state.air_temp), grid,
                         out_dir / "air_temp.png",
                         title="Air Temperature (C)",
                         cmap="RdYlBu_r", diverging=False)

            render_field(np.asarray(state.moisture), grid,
                         out_dir / "moisture.png",
                         title="Specific Humidity (kg/kg)",
                         cmap="YlGnBu", diverging=False)

            print("  Done!")
        except Exception as e:
            print(f"  Render error: {e}")


if __name__ == "__main__":
    main()
