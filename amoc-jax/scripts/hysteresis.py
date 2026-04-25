#!/usr/bin/env python3
"""Run a quasi-static freshwater-forcing sweep (up then down) and trace the
AMOC vs F trajectory — the spatial-model analog of the Stommel two-box
hysteresis curve.

Procedure:
  1. Spin up to baseline steady state.
  2. Ramp freshwater_forcing from 0 → F_max linearly across N segments,
     running M sub-steps per segment so the model approximately tracks
     quasi-equilibrium.
  3. Ramp back from F_max → 0 the same way.
  4. At every segment, record (F, AMOC strength, mean SST in N. Atlantic).

Output: interactives/05-hysteresis/curve.json with the full trajectory,
plus a static matplotlib PNG of the curve.

Tunable knobs: N_SEGMENTS (resolution along the loop), M_STEPS_PER_SEGMENT
(quasi-static parameter), F_MAX (peak freshwater forcing).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from amoc.data import build_forcing, build_initial_state
from amoc.grid import Grid
from amoc.state import Params
from amoc.step import run

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = REPO_ROOT / "interactives" / "05-hysteresis"

NX, NY = 256, 128
DT = 5e-5
SPINUP_STEPS = 20000
N_SEGMENTS = 16            # 16 up + 16 down = 32 points along the loop
M_STEPS_PER_SEGMENT = 4000  # ~2 sec per segment of "physical" sim time
F_MAX = 5.0


def _amoc_strength(state, mask_np, grid) -> float:
    psi_s = np.asarray(state.psi_s); psi_d = np.asarray(state.psi_d)
    om = mask_np > 0.5
    v_s = (np.roll(psi_s, -1, axis=1) - np.roll(psi_s, 1, axis=1)) * 0.5
    v_d = (np.roll(psi_d, -1, axis=1) - np.roll(psi_d, 1, axis=1)) * 0.5
    lat = np.asarray(grid.lat); lon = np.asarray(grid.lon)
    band = (lat >= 25) & (lat <= 35)
    atl  = (lon >= -80) & (lon <= 0)
    region = np.outer(band, atl) & om
    if not region.any():
        return 0.0
    H_s, H_d = 100.0, 3900.0
    return float(H_s * v_s[region].mean() - H_d * v_d[region].mean())


def _natl_sst(state, mask_np, grid) -> float:
    T = np.asarray(state.T_s); om = mask_np > 0.5
    lat = np.asarray(grid.lat); lon = np.asarray(grid.lon)
    band = (lat >= 40) & (lat <= 60)
    atl  = (lon >= -60) & (lon <= -10)
    region = np.outer(band, atl) & om
    return float(T[region].mean()) if region.any() else float("nan")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"jax devices: {jax.devices()}")
    print(f"grid: {NX}x{NY}, segments: {N_SEGMENTS} up + {N_SEGMENTS} down,"
          f" {M_STEPS_PER_SEGMENT} steps/segment")

    grid = Grid.create(nx=NX, ny=NY, lat0=-79.5, lat1=79.5)
    forcing = build_forcing(DATA_DIR, grid)
    state = build_initial_state(DATA_DIR, grid, forcing)
    om_np = np.asarray(forcing.ocean_mask)

    # Spin up to baseline (F=0).
    t0 = time.time()
    print(f"\nSpinup ({SPINUP_STEPS} steps)...")
    state = run(state, forcing, Params(dt=DT, freshwater_forcing=0.0),
                grid, n_steps=SPINUP_STEPS)
    state.zeta_s.block_until_ready()
    print(f"  done in {time.time()-t0:.1f}s")

    trajectory = []   # list of {phase, F, amoc, sst_natl, step}
    step_count = SPINUP_STEPS

    def record(phase, F):
        amoc = _amoc_strength(state, om_np, grid)
        sst_natl = _natl_sst(state, om_np, grid)
        trajectory.append({
            "phase": phase, "F": F, "amoc": amoc,
            "sst_natl": sst_natl, "step": step_count,
        })
        marker = "↑" if phase == "up" else ("↓" if phase == "down" else "○")
        print(f"  {marker} F={F:.2f}  AMOC={amoc:+.4f}  N.Atl SST={sst_natl:.2f}")

    record("init", 0.0)

    print(f"\nRamp UP F: 0 → {F_MAX}")
    for k in range(1, N_SEGMENTS + 1):
        F = F_MAX * k / N_SEGMENTS
        state = run(state, forcing, Params(dt=DT, freshwater_forcing=F),
                    grid, n_steps=M_STEPS_PER_SEGMENT)
        state.zeta_s.block_until_ready()
        step_count += M_STEPS_PER_SEGMENT
        record("up", F)

    print(f"\nRamp DOWN F: {F_MAX} → 0")
    for k in range(1, N_SEGMENTS + 1):
        F = F_MAX * (N_SEGMENTS - k) / N_SEGMENTS
        state = run(state, forcing, Params(dt=DT, freshwater_forcing=F),
                    grid, n_steps=M_STEPS_PER_SEGMENT)
        state.zeta_s.block_until_ready()
        step_count += M_STEPS_PER_SEGMENT
        record("down", F)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")

    # Save JSON.
    out_json = OUT_DIR / "curve.json"
    with open(out_json, "w") as f:
        json.dump({
            "config": {"nx": NX, "ny": NY, "dt": DT,
                       "spinup_steps": SPINUP_STEPS,
                       "n_segments": N_SEGMENTS,
                       "steps_per_segment": M_STEPS_PER_SEGMENT,
                       "F_max": F_MAX},
            "trajectory": trajectory,
        }, f, indent=2)
    print(f"  wrote {out_json}")

    # Quick PNG of the trajectory.
    F_arr    = np.array([p["F"]    for p in trajectory])
    amoc_arr = np.array([p["amoc"] for p in trajectory])
    phases   = [p["phase"] for p in trajectory]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.patch.set_facecolor("#0a1320"); ax.set_facecolor("#08111c")
    # Up-leg in orange, down-leg in blue
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
    ax.set_title("Hysteresis loop in the spatial JAX model",
                 color="#a0d4f0", fontsize=11)
    ax.tick_params(colors="#5a7e98")
    for spine in ax.spines.values(): spine.set_color("#1a2838")
    ax.legend(facecolor="#0c1420", edgecolor="#1a2838",
              labelcolor="#a0b8c8", framealpha=0.9)
    ax.grid(True, color="#1a2838", linewidth=0.5)
    fig.savefig(OUT_DIR / "hysteresis.png", dpi=140, facecolor="#0a1320")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'hysteresis.png'}")


if __name__ == "__main__":
    main()
