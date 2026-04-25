#!/usr/bin/env python3
"""Re-render only the 3 hosing scenarios + patch their entries in the
existing scenarios.json (leaving baseline/strong_wind/etc untouched).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Reuse the heavy lifting from render_scenarios.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "render_scenarios",
    Path(__file__).resolve().parent / "render_scenarios.py",
)
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "data"


HOSING = [s for s in rs.SCENARIOS if s["id"].startswith("hosing")]


def main():
    out_root = REPO_ROOT / "interactives" / "04-amoc-jax-viewer" / "scenarios"
    manifest_path = out_root.parent / "scenarios.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    grid = rs.Grid.create(nx=rs.NX, ny=rs.NY, lat0=-79.5, lat1=79.5)
    forcing = rs.build_forcing(DATA_DIR, grid)
    state0  = rs.build_initial_state(DATA_DIR, grid, forcing)
    om_np = np.asarray(forcing.ocean_mask)

    by_id = {s["id"]: i for i, s in enumerate(manifest["scenarios"])}

    for sc in HOSING:
        print(f"\n=== {sc['id']}: {sc['label']} ===")
        sc_dir = out_root / sc["id"]
        sc_dir.mkdir(parents=True, exist_ok=True)
        params_kw = {"dt": rs.DT}
        params_kw.update(sc.get("kw", {}))
        params = rs.Params(**params_kw)

        t0 = time.time()
        state = rs.run(state0, forcing, params, grid, n_steps=rs.N_STEPS)
        jax.block_until_ready(state.T_s)
        print(f"  {rs.N_STEPS} steps in {time.time()-t0:.1f}s")

        T_final = np.asarray(state.T_s)
        psi_final = np.asarray(state.psi_s)

        u_full, v_full = rs._velocities(psi_final)
        u_full = np.where(om_np > 0.5, u_full, 0.0).astype(np.float32)
        v_full = np.where(om_np > 0.5, v_full, 0.0).astype(np.float32)
        packed = np.stack([u_full, v_full], axis=-1).astype(np.float32)
        with open(sc_dir / "velocity.bin", "wb") as f:
            f.write(packed.tobytes())
        with open(sc_dir / "mask.bin", "wb") as f:
            f.write((om_np > 0.5).astype(np.uint8).tobytes())

        T_finite = T_final[(om_np > 0.5) & np.isfinite(T_final)]
        T_min = float(np.percentile(T_finite, 1)) if T_finite.size else -2.0
        T_max = float(np.percentile(T_finite, 99)) if T_finite.size else 30.0
        if T_max - T_min < 8.0:
            mid = (T_min + T_max) / 2; T_min, T_max = mid - 5, mid + 5

        rs._render_frame(
            T_final, psi_final, om_np,
            sc_dir / "background.png",
            lat0=-79.5, lat1=79.5, lon0=-180, lon1=180,
            title=f"{sc['label']}  ·  steady state",
            T_min=T_min, T_max=T_max,
        )
        sp_max = float(np.nanmax(np.hypot(u_full, v_full)))
        amoc = rs._amoc_strength(state, om_np, grid)
        print(f"  AMOC proxy (30°N Atlantic): {amoc:+.4f}")

        entry = {
            "id": sc["id"],
            "label": sc["label"],
            "summary": sc["summary"],
            "params": {k: v for k, v in params_kw.items()},
            "psi_min": float(np.nanmin(psi_final[om_np > 0.5])),
            "psi_max": float(np.nanmax(psi_final[om_np > 0.5])),
            "speed_max": sp_max,
            "amoc": amoc,
            "T_range": [T_min, T_max],
        }
        if sc["id"] in by_id:
            manifest["scenarios"][by_id[sc["id"]]] = entry
        else:
            manifest["scenarios"].append(entry)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nUpdated {manifest_path}")


if __name__ == "__main__":
    main()
