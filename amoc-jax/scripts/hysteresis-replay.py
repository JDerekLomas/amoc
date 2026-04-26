#!/usr/bin/env python3
"""Hysteresis experiment + per-segment replay snapshots.

Same physics setup as scripts/hysteresis.py (quasi-static freshwater
ramp up + down), but at every ramp segment also saves the full surface
state to a replay directory. The simamoc viewer can then scrub through
the loop and watch the AMOC collapse + recovery in space, not just on
a curve.

Output:
    OUTDIR/
      manifest.json     # extends write-replay format with .trajectory[]
      curve.json        # legacy hysteresis.py format (config + trajectory)
      hysteresis.png    # static curve plot
      temp.f32          # per-segment surface T snapshots (n_frames × ny × nx)
      sal.f32, psi.f32, air_temp.f32, ice_frac.f32

n_frames = 1 (init) + 2 * N_SEGMENTS (up + down).

Usage (default ~3-5 min on Mac CPU at 128×64):
    python amoc-jax/scripts/hysteresis-replay.py \\
        --out amoc-jax/runs/hysteresis-default
Override anything via CLI:
    --nx 256 --ny 128 --segments 16 --steps-per-seg 4000 --spinup 20000 --F-max 5.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "amoc-jax" / "src"))

from amoc.data import build_forcing, build_initial_state
from amoc.grid import Grid
from amoc.state import Params
from amoc.step import run

DATA_DIR = REPO_ROOT / "data"

FIELDS = ["T_s", "S_s", "psi_s", "air_temp", "ice_frac"]
JS_NAMES = {"T_s": "temp", "S_s": "sal", "psi_s": "psi",
            "air_temp": "air_temp", "ice_frac": "ice_frac"}


def amoc_strength(state, om_np) -> float:
    psi_s = np.asarray(state.psi_s); psi_d = np.asarray(state.psi_d)
    v_s = (np.roll(psi_s, -1, axis=1) - np.roll(psi_s, 1, axis=1)) * 0.5
    v_d = (np.roll(psi_d, -1, axis=1) - np.roll(psi_d, 1, axis=1)) * 0.5
    # ny-dependent indices for 25-35°N Atlantic strip
    ny, nx = psi_s.shape
    j0 = int((25 - (-79.5)) / 159.0 * (ny - 1))
    j1 = int((35 - (-79.5)) / 159.0 * (ny - 1))
    i0 = int((-80 - (-180)) / 360.0 * (nx - 1))
    i1 = int((0   - (-180)) / 360.0 * (nx - 1))
    region = np.zeros_like(om_np, dtype=bool)
    region[j0:j1+1, i0:i1+1] = om_np[j0:j1+1, i0:i1+1] > 0.5
    if not region.any():
        return 0.0
    H_s, H_d = 100.0, 3900.0
    return float(H_s * v_s[region].mean() - H_d * v_d[region].mean())


def natl_sst(state, om_np) -> float:
    T = np.asarray(state.T_s)
    ny, nx = T.shape
    j0 = int((40 - (-79.5)) / 159.0 * (ny - 1))
    j1 = int((60 - (-79.5)) / 159.0 * (ny - 1))
    i0 = int((-60 - (-180)) / 360.0 * (nx - 1))
    i1 = int((-10 - (-180)) / 360.0 * (nx - 1))
    region = np.zeros_like(om_np, dtype=bool)
    region[j0:j1+1, i0:i1+1] = om_np[j0:j1+1, i0:i1+1] > 0.5
    return float(T[region].mean()) if region.any() else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx",            type=int,   default=128)
    p.add_argument("--ny",            type=int,   default=64)
    p.add_argument("--dt",            type=float, default=5e-5)
    p.add_argument("--spinup",        type=int,   default=5000,
                   help="Spinup steps before the ramp begins.")
    p.add_argument("--segments",      type=int,   default=10,
                   help="Number of ramp segments per leg (up and down get equal segments).")
    p.add_argument("--steps-per-seg", type=int,   default=1500,
                   help="Sim steps per ramp segment (quasi-static parameter).")
    p.add_argument("--F-max",         type=float, default=5.0,
                   help="Peak freshwater forcing.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"# JAX devices: {jax.devices()}")
    print(f"# Grid {args.nx}x{args.ny}, spinup={args.spinup}, segments={args.segments}x{args.steps_per_seg} per leg, F_max={args.F_max}")

    grid    = Grid.create(nx=args.nx, ny=args.ny, lat0=-79.5, lat1=79.5)
    forcing = build_forcing(DATA_DIR, grid)
    state   = build_initial_state(DATA_DIR, grid, forcing)
    om_np   = np.asarray(forcing.ocean_mask)

    n_frames = 1 + 2 * args.segments
    bufs = {f: np.zeros((n_frames, args.ny, args.nx), dtype=np.float32) for f in FIELDS}
    trajectory = []
    step_count = 0

    def snapshot(idx: int, phase: str, F: float):
        amoc = amoc_strength(state, om_np)
        sst  = natl_sst(state, om_np)
        for f in FIELDS:
            bufs[f][idx] = np.asarray(getattr(state, f), dtype=np.float32)
        trajectory.append({
            "frame": idx, "phase": phase, "F": F,
            "amoc": amoc, "sst_natl": sst, "step": step_count,
        })
        marker = "↑" if phase == "up" else ("↓" if phase == "down" else "○")
        print(f"  [{idx:>3}/{n_frames}] {marker} F={F:.2f}  AMOC={amoc:+.4f}  N.Atl SST={sst:+.2f}")

    t0 = time.time()
    print(f"\n# Spinup ({args.spinup} steps)")
    state = run(state, forcing, Params(dt=args.dt, freshwater_forcing=0.0),
                grid, n_steps=args.spinup)
    state.zeta_s.block_until_ready()
    step_count += args.spinup
    snapshot(0, "init", 0.0)

    idx = 1
    print(f"\n# Ramp UP F: 0 → {args.F_max}")
    for k in range(1, args.segments + 1):
        F = args.F_max * k / args.segments
        state = run(state, forcing, Params(dt=args.dt, freshwater_forcing=F),
                    grid, n_steps=args.steps_per_seg)
        state.zeta_s.block_until_ready()
        step_count += args.steps_per_seg
        snapshot(idx, "up", F); idx += 1

    print(f"\n# Ramp DOWN F: {args.F_max} → 0")
    for k in range(1, args.segments + 1):
        F = args.F_max * (args.segments - k) / args.segments
        state = run(state, forcing, Params(dt=args.dt, freshwater_forcing=F),
                    grid, n_steps=args.steps_per_seg)
        state.zeta_s.block_until_ready()
        step_count += args.steps_per_seg
        snapshot(idx, "down", F); idx += 1

    elapsed = time.time() - t0
    print(f"\n# Sim done in {elapsed/60:.1f} min ({step_count} total steps)")

    # ── Write replay-format binaries + manifest ──
    for f in FIELDS:
        path = out / f"{JS_NAMES[f]}.f32"
        bufs[f].tofile(path)
        print(f"  wrote {path.name}  ({path.stat().st_size/1e6:.2f} MB)")

    manifest = {
        "version":   1,
        "kind":      "hysteresis",
        "nx":        args.nx,
        "ny":        args.ny,
        "n_frames":  n_frames,
        "dt":        args.dt,
        "lat0":      -79.5,
        "lat1":       79.5,
        "lon0":      -180.0,
        "lon1":       180.0,
        "fields":    list(JS_NAMES.values()),
        "dtype":     "float32",
        "byteorder": "little",
        "source":    "amoc-jax/hysteresis-replay",
        "spinup_steps":     args.spinup,
        "segments":         args.segments,
        "steps_per_seg":    args.steps_per_seg,
        "F_max":            args.F_max,
        "trajectory": trajectory,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote manifest.json")

    # Also dump curve.json in the legacy hysteresis.py format so existing tools work.
    (out / "curve.json").write_text(json.dumps({
        "config": {"nx": args.nx, "ny": args.ny, "dt": args.dt,
                   "spinup_steps": args.spinup,
                   "n_segments": args.segments,
                   "steps_per_segment": args.steps_per_seg,
                   "F_max": args.F_max},
        "trajectory": trajectory,
    }, indent=2))
    print(f"  wrote curve.json")

    # ── Static PNG of the curve ──
    F_arr    = np.array([p["F"]    for p in trajectory])
    amoc_arr = np.array([p["amoc"] for p in trajectory])
    phases   = [p["phase"] for p in trajectory]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.patch.set_facecolor("#0a1320"); ax.set_facecolor("#08111c")
    ups   = [i for i,p in enumerate(phases) if p in ("init", "up")]
    downs = [i for i,p in enumerate(phases) if p == "down"]
    ax.plot(F_arr[ups], amoc_arr[ups], "-o", color="#de9c7a",
            label="ramp up", markersize=4)
    if downs:
        ax.plot(np.concatenate([F_arr[ups[-1:]], F_arr[downs]]),
                np.concatenate([amoc_arr[ups[-1:]], amoc_arr[downs]]),
                "-o", color="#5a9ec8", label="ramp down", markersize=4)
    ax.axhline(0, color="#3a5468", lw=0.5)
    ax.set_xlabel("freshwater forcing F", color="#a0b8c8")
    ax.set_ylabel("AMOC strength (model units)", color="#a0b8c8")
    ax.set_title("AMOC hysteresis under freshwater forcing", color="#a0d4f0", fontsize=11)
    ax.tick_params(colors="#5a7e98")
    for spine in ax.spines.values(): spine.set_color("#1a2838")
    ax.legend(facecolor="#0c1420", edgecolor="#1a2838", labelcolor="#a0b8c8", framealpha=0.9)
    ax.grid(True, color="#1a2838", linewidth=0.5)
    fig.savefig(out / "hysteresis.png", dpi=140, facecolor="#0a1320")
    plt.close(fig)
    print(f"  wrote hysteresis.png")

    print(f"\n# View:")
    try:
        rel = out.resolve().relative_to(REPO_ROOT)
        print(f"  open simamoc/hysteresis.html?run={rel}")
    except ValueError:
        print(f"  serve {out} from a static server")


if __name__ == "__main__":
    main()
