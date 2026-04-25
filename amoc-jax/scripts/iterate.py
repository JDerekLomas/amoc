#!/usr/bin/env python3
"""Run sim with a Params, score, return result. Single-iteration of the loop.

Usage from another script:
    from iterate import run_one
    state, scores = run_one(params, grid_nx=256, grid_ny=128, n_steps=12000)

Usage from CLI (single run):
    python scripts/iterate.py [--nx 256 --ny 128 --steps 12000 --dt 0.005]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from amoc.data import build_forcing, build_initial_state
from amoc.grid import Grid
from amoc.state import Params
from amoc.step import run

# Local import (script-relative)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate, print_report


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "data"


def run_one(params: Params, *, grid_nx=256, grid_ny=128, n_steps=12000,
            label: str = "run", verbose: bool = True):
    grid = Grid.create(nx=grid_nx, ny=grid_ny, lat0=-79.5, lat1=79.5)
    forcing = build_forcing(DATA_DIR, grid)
    state0 = build_initial_state(DATA_DIR, grid, forcing)

    t0 = time.time()
    final = run(state0, forcing, params, grid, n_steps=n_steps)
    final.zeta_s.block_until_ready()
    elapsed = time.time() - t0

    scores = evaluate(final, grid, forcing.ocean_mask)
    if verbose:
        print(f"--- {label} :: {elapsed:.1f}s ({n_steps/elapsed:.0f} steps/s) ---")
        print_report(scores)
    return final, forcing, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--wind", type=float, default=1.0)
    args = p.parse_args()

    params = Params(dt=args.dt, beta=args.beta, wind_strength=args.wind)
    run_one(params, grid_nx=args.nx, grid_ny=args.ny, n_steps=args.steps,
            label=f"dt={args.dt} beta={args.beta} wind={args.wind}")


if __name__ == "__main__":
    main()
