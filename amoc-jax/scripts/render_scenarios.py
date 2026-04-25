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
N_STEPS = 20000               # 20k steps × dt=5e-5 = sim time 1.0; well past spinup
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
    # ---- Hosing experiments (the AMOC science) ----
    {"id": "hosing_weak",     "label": "Hosing (weak)",
     "summary": "Light Greenland-style freshwater flux into N. Atlantic surface. AMOC weakens but stays in thermal mode.",
     "kw": {"freshwater_forcing": 1.0}},
    {"id": "hosing_strong",   "label": "Hosing (strong)",
     "summary": "Heavy freshwater flux. Salt-advection feedback breaks down; AMOC begins to collapse.",
     "kw": {"freshwater_forcing": 2.5}},
    {"id": "hosing_collapse", "label": "Collapsed AMOC",
     "summary": "Sustained heavy hosing. Stommel salt mode — overturning shut down. North Atlantic cools.",
     "kw": {"freshwater_forcing": 5.0}},
]


def _velocities(psi):
    u = -(np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) * 0.5
    v =  (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) * 0.5
    u[0, :] = 0; u[-1, :] = 0
    return u, v


def _amoc_strength(state, mask_np, grid) -> float:
    """Atlantic-basin overturning proxy at 30°N: average surface − deep
    meridional velocity weighted by layer thickness.

    Positive = thermal mode (surface northward, deep return southward),
    near zero = collapsed."""
    psi_s = np.asarray(state.psi_s); psi_d = np.asarray(state.psi_d)
    om = mask_np > 0.5
    _, v_s = _velocities(psi_s)
    _, v_d = _velocities(psi_d)
    lat = np.asarray(grid.lat); lon = np.asarray(grid.lon)
    band = (lat >= 25) & (lat <= 35)
    atl  = (lon >= -80) & (lon <= 0)
    region = np.outer(band, atl) & om
    if not region.any():
        return 0.0
    H_s, H_d = 100.0, 3900.0
    return float(H_s * v_s[region].mean() - H_d * v_d[region].mean())


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
    print(f"grid: {NX}x{NY}, {N_STEPS} steps per scenario @ dt={DT}")

    grid = Grid.create(nx=NX, ny=NY, lat0=-79.5, lat1=79.5)
    print("loading forcing + initial state...")
    forcing = build_forcing(DATA_DIR, grid)
    state0  = build_initial_state(DATA_DIR, grid, forcing)
    om_np = np.asarray(forcing.ocean_mask)

    manifest = {
        "grid": {"nx": NX, "ny": NY, "lat0": -79.5, "lat1": 79.5,
                 "lon0": -180, "lon1": 180},
        "scenarios": [],
    }

    for sc in SCENARIOS:
        print(f"\n=== {sc['id']}: {sc['label']} ===")
        sc_dir = out_root / sc["id"]
        sc_dir.mkdir(parents=True, exist_ok=True)

        params_kw = {"dt": DT}
        params_kw.update(sc.get("kw", {}))
        params = Params(**params_kw)

        t0 = time.time()
        state = run(state0, forcing, params, grid, n_steps=N_STEPS)
        jax.block_until_ready(state.T_s)
        elapsed = time.time() - t0
        print(f"  {N_STEPS} steps in {elapsed:.1f}s ({N_STEPS/elapsed:.0f} steps/s)")

        T_final = np.asarray(state.T_s)
        psi_final = np.asarray(state.psi_s)

        # Compute u, v from psi for particle advection in browser.
        u_full, v_full = _velocities(psi_final)
        u_full = np.where(om_np > 0.5, u_full, 0.0).astype(np.float32)
        v_full = np.where(om_np > 0.5, v_full, 0.0).astype(np.float32)
        # Pack as [u, v] interleaved Float32 LE for a single binary fetch.
        # Layout: ny rows, nx cols, [u_ji, v_ji] per cell.
        packed = np.stack([u_full, v_full], axis=-1).astype(np.float32)  # (ny, nx, 2)
        with open(sc_dir / "velocity.bin", "wb") as f:
            f.write(packed.tobytes())
        # Also pack the ocean mask so the JS can avoid spawning land particles.
        with open(sc_dir / "mask.bin", "wb") as f:
            f.write((om_np > 0.5).astype(np.uint8).tobytes())

        # SST color range with a reasonable spread.
        T_finite = T_final[(om_np > 0.5) & np.isfinite(T_final)]
        T_min = float(np.percentile(T_finite, 1)) if T_finite.size else -2.0
        T_max = float(np.percentile(T_finite, 99)) if T_finite.size else 30.0
        if T_max - T_min < 8.0:
            mid = (T_min + T_max) / 2; T_min, T_max = mid - 5, mid + 5

        # One static composite image (SST + streamlines).
        _render_frame(
            T_final, psi_final, om_np,
            sc_dir / "background.png",
            lat0=-79.5, lat1=79.5, lon0=-180, lon1=180,
            title=f"{sc['label']}  ·  steady state",
            T_min=T_min, T_max=T_max,
        )

        # Stats for the manifest
        sp_max = float(np.nanmax(np.hypot(u_full, v_full)))
        amoc = _amoc_strength(state, om_np, grid)
        print(f"  AMOC proxy (30°N Atlantic): {amoc:+.4f}")
        manifest["scenarios"].append({
            "id": sc["id"],
            "label": sc["label"],
            "summary": sc["summary"],
            "params": {k: v for k, v in params_kw.items()},
            "psi_min": float(np.nanmin(psi_final[om_np > 0.5])),
            "psi_max": float(np.nanmax(psi_final[om_np > 0.5])),
            "speed_max": sp_max,
            "amoc": amoc,
            "T_range": [T_min, T_max],
        })

    with open(out_root.parent / "scenarios.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {out_root.parent / 'scenarios.json'}")


if __name__ == "__main__":
    main()
