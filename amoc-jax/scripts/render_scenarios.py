#!/usr/bin/env python3
"""Render coupled-physics scenarios as PNG sequences for the browser viewer.

Uses the new build_forcing / build_initial_state API. For each scenario:
spin up from observed SST/SSS climatology, save N snapshots of T_s + ψ_s
during the integration. The viewer at interactives/04-amoc-jax-viewer/
loads these.
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

NX, NY = 256, 128
N_FRAMES = 8
N_STEPS_PER_FRAME = 500       # 8 * 500 = 4k steps; 0.5°C RMSE at 2k per memo
DT = 5e-5


SCENARIOS = [
    {"id": "baseline",   "label": "Baseline (full coupled)",
     "summary": "Default coupled ocean + atmosphere. ERA5 wind, NOAA SST/SSS, 1-layer atmosphere with moisture cycle."},
    {"id": "strong_wind","label": "Strong wind",
     "summary": "Wind doubled. Stronger gyres, faster western boundary currents.",
     "kw": {"wind_strength": 2.0}},
    {"id": "weak_beta",  "label": "Weak β",
     "summary": "Halved planetary vorticity gradient. Gyres get larger.",
     "kw": {"beta": 0.5}},
    {"id": "high_visc",  "label": "High viscosity",
     "summary": "10× viscosity. Boundary currents thicken and slow.",
     "kw": {"A_visc_s": 2e-3, "A_visc_d": 2e-3}},
    {"id": "warm",       "label": "Warmer climate (+5°C)",
     "summary": "Global temperature offset +5°C. Watch SST and atmosphere respond.",
     "kw": {"global_temp_offset": 5.0}},
    {"id": "weak_amoc",  "label": "Reduced overturning",
     "summary": "Halved meridional overturning tendency. Less vertical heat transport.",
     "kw": {"mot_strength": 0.025}},
]


def _velocities(psi):
    u = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) * 0.5
    v =  (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) * 0.5
    u[0, :] = 0; u[-1, :] = 0
    return u, v


def _render_frame(T, psi, mask_np, out, *, lat0, lat1, lon0, lon1,
                  title, T_min, T_max):
    om = mask_np > 0.5
    T_disp = np.where(om, T, np.nan)
    u, v = _velocities(psi)
    speed = np.hypot(u, v)
    speed_disp = np.where(om, speed, np.nan)

    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("#0a1320")
    ax.set_facecolor("#08111c")

    ax.imshow(
        T_disp, origin="lower", extent=[lon0, lon1, lat0, lat1],
        cmap="RdYlBu_r", vmin=T_min, vmax=T_max, aspect="auto",
        interpolation="bilinear",
    )

    sp_max = np.nanpercentile(speed_disp, 99) if np.isfinite(np.nanpercentile(speed_disp, 99)) else 1.0
    if sp_max > 1e-9:
        ny, nx = psi.shape
        x_coords = np.linspace(lon0, lon1, nx)
        y_coords = np.linspace(lat0, lat1, ny)
        u_m = np.where(om, u, 0.0)
        v_m = np.where(om, v, 0.0)
        try:
            ax.streamplot(
                x_coords, y_coords, u_m, v_m,
                density=[2.4, 1.2],
                color=speed_disp, cmap="cividis",
                norm=plt.Normalize(0, sp_max or 1.0),
                linewidth=0.7, arrowsize=0.7,
            )
        except Exception:
            pass

    # Continent fill
    ax.imshow(
        np.where(om, np.nan, 1.0), origin="lower",
        extent=[lon0, lon1, lat0, lat1], cmap="gray_r", vmin=0, vmax=1,
        aspect="auto", interpolation="nearest",
    )

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, color="#a0d4f0", fontsize=11, pad=6)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor="#0a1320")
    plt.close(fig)


def main():
    out_root = REPO_ROOT / "interactives" / "04-amoc-jax-viewer" / "scenarios"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"jax devices: {jax.devices()}")
    print(f"grid: {NX}x{NY}, {N_FRAMES} frames per scenario, {N_STEPS_PER_FRAME} steps/frame")

    grid = Grid.create(nx=NX, ny=NY, lat0=-79.5, lat1=79.5)
    print("loading forcing + initial state...")
    forcing = build_forcing(DATA_DIR, grid)
    state0  = build_initial_state(DATA_DIR, grid, forcing)
    om_np = np.asarray(forcing.ocean_mask)

    manifest = {
        "grid": {"nx": NX, "ny": NY, "lat0": -79.5, "lat1": 79.5,
                 "lon0": -180, "lon1": 180},
        "frames_per_scenario": N_FRAMES,
        "scenarios": [],
    }

    for sc in SCENARIOS:
        print(f"\n=== {sc['id']}: {sc['label']} ===")
        sc_dir = out_root / sc["id"]
        sc_dir.mkdir(parents=True, exist_ok=True)

        params_kw = {"dt": DT}
        params_kw.update(sc.get("kw", {}))
        params = Params(**params_kw)

        # Render frames by chunking the run.
        state = state0
        frames_T = []
        frames_psi = []
        t0 = time.time()
        for k in range(N_FRAMES):
            state = run(state, forcing, params, grid, n_steps=N_STEPS_PER_FRAME)
            jax.block_until_ready(state.T_s)
            frames_T.append(np.asarray(state.T_s))
            frames_psi.append(np.asarray(state.psi_s))
        elapsed = time.time() - t0
        print(f"  {N_FRAMES * N_STEPS_PER_FRAME} steps in {elapsed:.1f}s "
              f"({(N_FRAMES * N_STEPS_PER_FRAME)/elapsed:.0f} steps/s)")

        # Common temp range across frames so colormap is stable.
        T_arr = np.array(frames_T)
        T_finite = T_arr[np.broadcast_to(om_np > 0.5, T_arr.shape)]
        T_finite = T_finite[np.isfinite(T_finite)]
        T_min = float(np.percentile(T_finite, 1)) if T_finite.size else -2.0
        T_max = float(np.percentile(T_finite, 99)) if T_finite.size else 30.0
        if T_max - T_min < 5.0:
            mid = (T_min + T_max) / 2; T_min, T_max = mid - 4, mid + 4

        for k in range(N_FRAMES):
            sim_time = (k + 1) * N_STEPS_PER_FRAME * DT
            _render_frame(
                frames_T[k], frames_psi[k], om_np,
                sc_dir / f"frame_{k:03d}.png",
                lat0=-79.5, lat1=79.5, lon0=-180, lon1=180,
                title=f"{sc['label']}  ·  t = {sim_time:.2f}",
                T_min=T_min, T_max=T_max,
            )

        psi_finite = np.asarray(frames_psi[-1])
        psi_finite = psi_finite[np.isfinite(psi_finite)]
        manifest["scenarios"].append({
            "id": sc["id"],
            "label": sc["label"],
            "summary": sc["summary"],
            "params": {k: v for k, v in params_kw.items()},
            "psi_min": float(psi_finite.min()) if psi_finite.size else 0,
            "psi_max": float(psi_finite.max()) if psi_finite.size else 0,
            "T_range": [T_min, T_max],
        })

    with open(out_root.parent / "scenarios.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {out_root.parent / 'scenarios.json'}")


if __name__ == "__main__":
    main()
