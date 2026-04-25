#!/usr/bin/env python3
"""Run JAX sim and dump per-frame state to a directory the simamoc browser
viewer can replay.

Output format (deliberately simple — no zarr dependency on the JS side):

    OUTDIR/
      manifest.json     # {nx, ny, n_frames, dt, sim_time_per_frame, fields}
      temp.f32          # float32 LE, shape (n_frames, ny, nx) — surface T, °C
      sal.f32           # ditto, surface salinity, psu
      psi.f32           # ditto, surface streamfunction
      air_temp.f32      # ditto, atmosphere
      ice_frac.f32      # ditto, sea ice 0–1

Each .f32 is one big contiguous Float32Array. JS reads via fetch +
ArrayBuffer. Same orientation convention as simamoc (j=0 at south pole).

Usage:
    python amoc-jax/scripts/write-replay.py \\
        --nx 256 --ny 128 --steps 1000 --frames 50 --out runs/replay-baseline
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "amoc-jax" / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from amoc.grid import make_grid
from amoc.data import build_forcing, build_initial_state
from amoc.state import Params
from amoc.step import step

DATA_DIR = REPO_ROOT / "data"

FIELDS = ["T_s", "S_s", "psi_s", "air_temp", "ice_frac"]
JS_NAMES = {  # map JAX field → simamoc array name (for UI consistency)
    "T_s":      "temp",
    "S_s":      "sal",
    "psi_s":    "psi",
    "air_temp": "air_temp",
    "ice_frac": "ice_frac",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nx",     type=int, default=256)
    p.add_argument("--ny",     type=int, default=128)
    p.add_argument("--steps",  type=int, default=1000)
    p.add_argument("--frames", type=int, default=50,
                   help="Number of snapshots to write (evenly spaced over --steps).")
    p.add_argument("--dt",     type=float, default=5e-5)
    p.add_argument("--out",    type=Path,  required=True,
                   help="Output directory (will be created).")
    args = p.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"# JAX devices: {jax.devices()}")
    print(f"# Grid {args.nx}x{args.ny}, {args.steps} steps, {args.frames} frames")

    grid    = make_grid(args.nx, args.ny)
    forcing = build_forcing(DATA_DIR, grid)
    state   = build_initial_state(DATA_DIR, grid, forcing)
    params  = Params(dt=args.dt)

    steps_per_frame = max(1, args.steps // args.frames)
    n_frames        = args.steps // steps_per_frame

    # Pre-allocate output buffers (shape: n_frames × ny × nx, float32).
    bufs = {f: np.zeros((n_frames, args.ny, args.nx), dtype=np.float32) for f in FIELDS}

    # JIT a single-step function for speed.
    step_jit = jax.jit(lambda s: step(s, forcing, params, grid))

    print(f"# Running ({steps_per_frame} steps per frame)…")
    t0 = time.time()
    for fi in range(n_frames):
        for _ in range(steps_per_frame):
            state = step_jit(state)
        for f in FIELDS:
            bufs[f][fi] = np.asarray(getattr(state, f), dtype=np.float32)
        if (fi + 1) % 10 == 0 or fi == n_frames - 1:
            elapsed = time.time() - t0
            print(f"  frame {fi+1}/{n_frames}  sim_time={float(state.sim_time):.4f}  ({elapsed:.1f}s)")

    # Write each field as one contiguous f32 file.
    for f in FIELDS:
        path = out / f"{JS_NAMES[f]}.f32"
        bufs[f].tofile(path)
        print(f"  wrote {path.name}  ({path.stat().st_size/1e6:.1f} MB)")

    manifest = {
        "version":           1,
        "nx":                args.nx,
        "ny":                args.ny,
        "n_frames":          n_frames,
        "steps_per_frame":   steps_per_frame,
        "total_steps":       n_frames * steps_per_frame,
        "dt":                args.dt,
        "sim_time_per_frame": float(state.sim_time) / n_frames,
        "lat0":              -79.5,
        "lat1":               79.5,
        "lon0":              -180.0,
        "lon1":               180.0,
        "fields":            list(JS_NAMES.values()),
        "dtype":             "float32",
        "byteorder":         "little",
        "source":            "amoc-jax",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote manifest.json")

    elapsed = time.time() - t0
    print(f"\n# Done in {elapsed:.1f}s.")
    print(f"# Output: {out}")
    try:
        rel = out.resolve().relative_to(REPO_ROOT)
        print(f"# View:   open simamoc/replay.html?run={rel}")
    except ValueError:
        print(f"# View:   serve {out} from a static server, then open replay.html")


if __name__ == "__main__":
    main()
