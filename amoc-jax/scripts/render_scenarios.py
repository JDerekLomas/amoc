#!/usr/bin/env python3
"""Render a small set of v1a/v1c scenarios as PNG sequences for the browser
viewer at interactives/04-amoc-jax-viewer/.

Each scenario runs the simulator from rest with a chosen parameter set,
saving N snapshots of the surface streamfunction ψ_s during spin-up.
Output goes to `interactives/04-amoc-jax-viewer/scenarios/<name>/`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from amoc.data import load_mask, load_to_grid
from amoc.grid import Grid
from amoc.state import Forcing, Params, State, zero_state
from amoc.step import run_with_history


REPO_ROOT = Path(__file__).resolve().parents[2]
WIND      = REPO_ROOT / "data" / "wind_stress.json"
MASK      = REPO_ROOT / "data" / "mask.json"
SST       = REPO_ROOT / "data" / "sst.json"
SSS       = REPO_ROOT / "data" / "salinity.json"
DEEPT     = REPO_ROOT / "data" / "deep_temp.json"

# All scenarios share grid, all variations are in Params.
NX, NY = 256, 128
N_FRAMES = 8
N_STEPS_PER_FRAME = 1500   # 8 frames × 1500 = 12000 steps total
DT = 0.005


SCENARIOS = [
    {
        "id": "baseline",
        "label": "Baseline v1a",
        "summary": "ERA5 wind curl, no thermodynamics. The default subtropical/subpolar gyre response.",
        "kw": dict(wind_strength=0.04, beta=2.0, A_visc_s=0.005, A_visc_d=0.005,
                   r_friction_s=0.04, r_friction_d=0.10,
                   alpha_BC=0.0, F_couple_s=0.0, F_couple_d=0.0,
                   kappa_T=0.0, kappa_S=0.0, tau_T=1e9, tau_S=1e9,
                   gamma_TS=0.0, gamma_conv=0.0, dt=DT),
    },
    {
        "id": "strong_wind",
        "label": "Strong wind",
        "summary": "Wind doubled. Gyres intensify; western boundary currents speed up.",
        "kw": dict(wind_strength=0.08, beta=2.0, A_visc_s=0.005, A_visc_d=0.005,
                   r_friction_s=0.04, r_friction_d=0.10,
                   alpha_BC=0.0, F_couple_s=0.0, F_couple_d=0.0,
                   kappa_T=0.0, kappa_S=0.0, tau_T=1e9, tau_S=1e9,
                   gamma_TS=0.0, gamma_conv=0.0, dt=DT),
    },
    {
        "id": "weak_beta",
        "label": "Weak β",
        "summary": "Halved planetary vorticity gradient. Gyres get larger; Munk layer widens.",
        "kw": dict(wind_strength=0.04, beta=1.0, A_visc_s=0.005, A_visc_d=0.005,
                   r_friction_s=0.04, r_friction_d=0.10,
                   alpha_BC=0.0, F_couple_s=0.0, F_couple_d=0.0,
                   kappa_T=0.0, kappa_S=0.0, tau_T=1e9, tau_S=1e9,
                   gamma_TS=0.0, gamma_conv=0.0, dt=DT),
    },
    {
        "id": "high_viscosity",
        "label": "High viscosity",
        "summary": "Tripled viscosity. Boundary currents thicken and slow.",
        "kw": dict(wind_strength=0.04, beta=2.0, A_visc_s=0.020, A_visc_d=0.020,
                   r_friction_s=0.04, r_friction_d=0.10,
                   alpha_BC=0.0, F_couple_s=0.0, F_couple_d=0.0,
                   kappa_T=0.0, kappa_S=0.0, tau_T=1e9, tau_S=1e9,
                   gamma_TS=0.0, gamma_conv=0.0, dt=DT),
    },
    {
        "id": "with_thermo",
        "label": "v1c with thermodynamics",
        "summary": "Wind + observed SST,SSS as initial T,S + thermal-wind shear forcing. The deep layer responds.",
        "kw": dict(wind_strength=0.04, beta=2.0, A_visc_s=2e-4, A_visc_d=2e-4,
                   r_friction_s=0.04, r_friction_d=0.10,
                   alpha_BC=1.0, F_couple_s=0.05, F_couple_d=0.001,
                   kappa_T=5e-4, kappa_S=5e-4, tau_T=60.0, tau_S=180.0,
                   gamma_TS=0.001, gamma_conv=0.05, dt=DT),
        "use_thermo": True,
    },
    {
        "id": "thermo_strong_wind",
        "label": "v1c, strong wind",
        "summary": "Same as v1c thermodynamics but doubled wind. Compare to baseline + strong-wind to see thermal-wind shear effect.",
        "kw": dict(wind_strength=0.08, beta=2.0, A_visc_s=2e-4, A_visc_d=2e-4,
                   r_friction_s=0.04, r_friction_d=0.10,
                   alpha_BC=1.0, F_couple_s=0.05, F_couple_d=0.001,
                   kappa_T=5e-4, kappa_S=5e-4, tau_T=60.0, tau_S=180.0,
                   gamma_TS=0.001, gamma_conv=0.05, dt=DT),
        "use_thermo": True,
    },
]


def _load_inputs(grid: Grid):
    curl = load_to_grid(WIND, "wind_curl", grid)
    ocean = load_mask(MASK, grid=grid)
    sst = load_to_grid(SST, "sst", grid)
    sss = load_to_grid(SSS, "salinity", grid)
    deepT = load_to_grid(DEEPT, "temp", grid)
    n_ocean = float(ocean.sum())
    rms = float(jnp.sqrt(jnp.sum((curl * ocean) ** 2) / jnp.maximum(n_ocean, 1.0)))
    curl_n = (curl / rms) * ocean
    sst_mean = float(jnp.sum(sst * ocean) / jnp.maximum(n_ocean, 1.0))
    sss_mean = float(jnp.sum(sss * ocean) / jnp.maximum(n_ocean, 1.0))
    deepT_mean = float(jnp.sum(deepT * ocean) / jnp.maximum(n_ocean, 1.0))
    sst_f = jnp.where(ocean > 0.5, sst, sst_mean).astype(jnp.float32)
    sss_f = jnp.where(ocean > 0.5, sss, sss_mean).astype(jnp.float32)
    deepT_f = jnp.where(ocean > 0.5, deepT, deepT_mean).astype(jnp.float32)
    return curl_n, ocean, sst_f, sss_f, deepT_f, sss_mean


def _render_psi_png(psi: np.ndarray, ocean_mask: np.ndarray, out: Path,
                    *, vmax: float, lat0: float, lat1: float, lon0: float, lon1: float):
    """Single-panel ψ render with land mask, no axes, square colormap."""
    arr = np.where(ocean_mask > 0.5, psi, np.nan)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.imshow(
        arr, origin="lower", extent=[lon0, lon1, lat0, lat1],
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#0c1420")
    fig.patch.set_facecolor("#080e18")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, facecolor="#080e18")
    plt.close(fig)


def main():
    out_root = REPO_ROOT / "interactives" / "04-amoc-jax-viewer" / "scenarios"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"jax devices: {jax.devices()}")
    print(f"grid: {NX}x{NY}, {N_FRAMES} frames per scenario, {N_STEPS_PER_FRAME} steps/frame")
    grid = Grid.create(nx=NX, ny=NY, lat0=-79.5, lat1=79.5)

    print("loading inputs...")
    curl_n, ocean, sst_f, sss_f, deepT_f, sss_mean = _load_inputs(grid)

    forcing = Forcing(
        wind_curl=curl_n,
        ocean_mask=ocean,
        buoyancy=jnp.zeros(grid.shape),
        T_target=sst_f, S_target=sss_f,
        S_d_const=jnp.full(grid.shape, sss_mean, dtype=jnp.float32),
        F_fresh=jnp.zeros(grid.shape),
    )
    ocean_np = np.asarray(ocean)

    manifest = {"grid": {"nx": NX, "ny": NY, "lat0": -79.5, "lat1": 79.5,
                         "lon0": -180, "lon1": 180},
                "frames_per_scenario": N_FRAMES,
                "scenarios": []}

    for sc in SCENARIOS:
        print(f"\n=== {sc['id']}: {sc['label']} ===")
        sc_dir = out_root / sc["id"]
        sc_dir.mkdir(parents=True, exist_ok=True)

        if sc.get("use_thermo"):
            state = zero_state(grid.shape)._replace(
                T_s=sst_f, S_s=sss_f, T_d=deepT_f,
            )
        else:
            state = zero_state(grid.shape)

        params = Params(**sc["kw"])

        t0 = time.time()
        final, hist = run_with_history(
            state, forcing, params, grid,
            n_steps=N_FRAMES * N_STEPS_PER_FRAME,
            save_every=N_STEPS_PER_FRAME,
        )
        final.zeta_s.block_until_ready()
        elapsed = time.time() - t0

        # Determine common color range across frames so animation doesn't flicker.
        psi_hist = np.asarray(hist.psi_s)  # (N_FRAMES, ny, nx)
        if not np.all(np.isfinite(psi_hist)):
            print(f"  WARNING: blow-up; using last finite frame")
        finite_max = np.percentile(np.abs(psi_hist[np.isfinite(psi_hist)]), 99)
        vmax = max(float(finite_max), 1.0)

        for k in range(N_FRAMES):
            _render_psi_png(
                psi_hist[k], ocean_np, sc_dir / f"frame_{k:03d}.png",
                vmax=vmax, lat0=-79.5, lat1=79.5, lon0=-180, lon1=180,
            )
        print(f"  {N_FRAMES} frames saved in {elapsed:.1f}s ({(N_FRAMES * N_STEPS_PER_FRAME)/elapsed:.0f} steps/sec)")

        psi_finite = psi_hist[np.isfinite(psi_hist)]
        manifest["scenarios"].append({
            "id": sc["id"],
            "label": sc["label"],
            "summary": sc["summary"],
            "params": {k: float(v) if isinstance(v, (int, float)) else v for k, v in sc["kw"].items()},
            "psi_min": float(psi_finite.min()) if psi_finite.size else 0,
            "psi_max": float(psi_finite.max()) if psi_finite.size else 0,
            "vmax": vmax,
        })

    with open(out_root.parent / "scenarios.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {out_root.parent / 'scenarios.json'}")


if __name__ == "__main__":
    main()
