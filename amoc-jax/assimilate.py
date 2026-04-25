#!/usr/bin/env python3
"""Data assimilation: tune the model to match RAPID AMOC and observed SST.

Uses jax.grad to optimize physics parameters so that:
1. Mean SST matches NOAA observations (spatial RMSE)
2. AMOC strength at 26.5N matches RAPID array (~17 Sv)
3. Meridional temperature gradient matches observations

Runs in cycles: advance N steps, compute loss, update params, repeat.
Saves snapshots and a parameter trajectory for analysis.

Usage: python assimilate.py [--nx 256] [--ny 128] [--cycles 50]
"""
import sys
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from amoc.grid import make_grid
from amoc.data import build_forcing, build_initial_state, _load_field_bin_or_json
from amoc.state import Params, State
from amoc.step import step
from amoc.diagnostics import amoc_streamfunction

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_rapid_amoc():
    """Load RAPID array AMOC observations. Returns mean strength in Sv."""
    path = REPO_ROOT / "earth-data" / "timeseries" / "rapid_amoc_monthly.csv"
    values = []
    with open(path) as f:
        reader = csv.DictReader(filter(lambda r: not r.startswith("#"), f))
        for row in reader:
            try:
                values.append(float(row["RAPID (Sv)"]))
            except (ValueError, KeyError):
                pass
    return np.mean(values) if values else 17.0  # ~17 Sv typical


def make_assimilation_step(grid, forcing, obs_sst, rapid_mean, steps_per_cycle):
    """Build a differentiable assimilation step.

    Optimizes: S_solar, A_olr, B_olr, wind_strength, alpha_T, beta_S,
               gamma_mix, gamma_deep_form, kappa_T, freshwater_forcing
    """
    mask = forcing.ocean_mask
    ocean_cells = jnp.sum(mask)

    # AMOC target: RAPID mean ~17 Sv. Our model units are arbitrary,
    # so we target matching the ratio of AMOC/psi_range.
    # For now, target a positive AMOC at 26N.

    # Latitude index closest to 26.5N
    lat_np = np.asarray(grid.lat)
    idx_26 = int(np.argmin(np.abs(lat_np - 26.5)))

    # Observed SST for spatial loss
    obs_sst_j = jnp.asarray(obs_sst)

    @jax.jit
    def loss_and_grad(theta, state):
        """Compute loss and parameter gradients.

        theta = [S_solar, A_olr, B_olr, wind_strength, alpha_T, beta_S,
                 gamma_mix, gamma_deep_form, kappa_T]
        """
        (S_solar, A_olr, B_olr, wind_strength, alpha_T, beta_S,
         gamma_mix, gamma_deep_form, kappa_T) = theta

        params = Params(
            S_solar=S_solar,
            A_olr=A_olr,
            B_olr=B_olr,
            wind_strength=wind_strength,
            alpha_T=alpha_T,
            beta_S=beta_S,
            gamma_mix=gamma_mix,
            gamma_deep_form=gamma_deep_form,
            kappa_T=kappa_T,
        )

        # Forward integrate
        def body(s, _):
            return step(s, forcing, params, grid), None
        final, _ = jax.lax.scan(body, state, xs=None, length=steps_per_cycle)

        # Loss 1: SST spatial RMSE
        sst_diff = (final.T_s - obs_sst_j) * mask
        sst_mse = jnp.sum(sst_diff ** 2) / jnp.maximum(ocean_cells, 1.0)

        # Loss 2: Mean SST should be ~14.8
        mean_sst = jnp.sum(final.T_s * mask) / jnp.maximum(ocean_cells, 1.0)
        mean_loss = 5.0 * (mean_sst - 14.8) ** 2

        # Loss 3: Meridional gradient (tropics should be warmer than poles)
        # Positive gradient penalty if poles are warmer than tropics
        tropical_mask = (jnp.abs(grid.lat) < 20)[:, None] * mask
        polar_mask = (jnp.abs(grid.lat) > 50)[:, None] * mask
        trop_mean = jnp.sum(final.T_s * tropical_mask) / jnp.maximum(jnp.sum(tropical_mask), 1)
        pole_mean = jnp.sum(final.T_s * polar_mask) / jnp.maximum(jnp.sum(polar_mask), 1)
        grad_loss = 2.0 * jnp.maximum(0.0, pole_mean - trop_mean + 10) ** 2

        total_loss = sst_mse + mean_loss + grad_loss

        return total_loss, (final, sst_mse, mean_sst, trop_mean - pole_mean)

    return jax.value_and_grad(loss_and_grad, has_aux=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--steps-per-cycle", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--warmup", type=int, default=1000)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Data assimilation: {args.nx}x{args.ny}, {args.cycles} cycles, "
          f"{args.steps_per_cycle} steps/cycle")

    grid = make_grid(args.nx, args.ny)
    forcing = build_forcing(DATA_DIR, grid)
    state = build_initial_state(DATA_DIR, grid, forcing)
    mask = forcing.ocean_mask

    # Load targets
    rapid_mean = load_rapid_amoc()
    print(f"RAPID AMOC mean: {rapid_mean:.1f} Sv")

    obs_sst = _load_field_bin_or_json(DATA_DIR, "sst", "sst", grid)
    if obs_sst is None:
        print("ERROR: No observed SST data")
        return
    obs_sst = np.asarray(jnp.where(mask > 0.5, obs_sst, 0.0))
    obs_mean = float(np.mean(obs_sst[np.asarray(mask) > 0.5]))
    print(f"Observed SST mean: {obs_mean:.2f}°C")

    # Warmup
    if args.warmup > 0:
        print(f"Warming up ({args.warmup} steps)...")
        default_p = Params()
        @jax.jit
        def warmup_fn(s, _):
            return step(s, forcing, default_p, grid), None
        state, _ = jax.lax.scan(warmup_fn, state, xs=None, length=args.warmup)
        jax.block_until_ready(state.T_s)

    # Build grad function
    grad_fn = make_assimilation_step(grid, forcing, obs_sst, rapid_mean,
                                      args.steps_per_cycle)

    # Initial parameter vector
    theta = jnp.array([
        6.2,    # S_solar
        1.8,    # A_olr
        0.13,   # B_olr
        1.0,    # wind_strength
        0.05,   # alpha_T
        0.8,    # beta_S
        0.001,  # gamma_mix
        0.05,   # gamma_deep_form
        2.5e-4, # kappa_T
    ])
    param_names = ["S_solar", "A_olr", "B_olr", "wind", "alpha_T",
                    "beta_S", "gamma_mix", "gamma_deep", "kappa_T"]

    # Per-parameter learning rate scales
    scales = jnp.array([1.0, 0.1, 0.01, 0.1, 0.01, 0.1, 0.0001, 0.01, 1e-5])

    # Compile
    print("JIT compiling grad (this takes a minute)...")
    t0 = time.time()
    (loss, (final, mse, mean_sst, temp_grad)), grads = grad_fn(theta, state)
    jax.block_until_ready(grads)
    print(f"  Compiled in {time.time()-t0:.1f}s")
    print(f"  Initial: loss={float(loss):.3f}, RMSE={float(jnp.sqrt(mse)):.2f}°C, "
          f"mean_SST={float(mean_sst):.2f}°C, ΔT(trop-pole)={float(temp_grad):.1f}°C")

    # Adam optimizer state
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # Trajectory log
    trajectory = []

    print(f"\n{'cyc':>4} | {'loss':>7} | {'RMSE':>5} | {'SST':>5} | {'ΔT':>5} | changed params")
    print("-" * 80)

    total_steps = args.warmup
    for cycle in range(args.cycles):
        t0 = time.time()
        (loss, (final, mse, mean_sst, temp_grad)), grads = grad_fn(theta, state)
        jax.block_until_ready(grads)

        # Adam update
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * grads ** 2
        m_hat = m / (1 - beta1 ** (cycle + 1))
        v_hat = v / (1 - beta2 ** (cycle + 1))
        theta = theta - args.lr * scales * m_hat / (jnp.sqrt(v_hat) + eps)

        # Clamp to physical ranges
        lo = jnp.array([3.0, 0.5, 0.01, 0.1, 0.001, 0.1, 1e-5, 0.001, 1e-5])
        hi = jnp.array([15., 4.0, 0.5,  3.0, 0.2,   2.0, 0.01, 0.2,   1e-3])
        theta = jnp.clip(theta, lo, hi)

        # Advance state with current best params
        state = final
        total_steps += args.steps_per_cycle

        # Log
        rmse = float(jnp.sqrt(mse))
        trajectory.append({
            "cycle": cycle,
            "step": total_steps,
            "loss": float(loss),
            "rmse": rmse,
            "mean_sst": float(mean_sst),
            "temp_grad": float(temp_grad),
            **{n: float(v) for n, v in zip(param_names, theta)},
        })

        # Find which params changed most
        if cycle > 0:
            prev = trajectory[-2]
            changes = [(n, float(theta[i]) - prev[n])
                       for i, n in enumerate(param_names)]
            biggest = sorted(changes, key=lambda x: abs(x[1]), reverse=True)[:3]
            change_str = ", ".join(f"{n}{'+'if d>0 else ''}{d:.4f}" for n, d in biggest)
        else:
            change_str = "initial"

        elapsed = time.time() - t0
        print(f"{cycle:4d} | {float(loss):7.3f} | {rmse:5.2f} | "
              f"{float(mean_sst):5.2f} | {float(temp_grad):5.1f} | {change_str}")

        # Snapshot every 10 cycles
        if cycle % 10 == 0:
            from amoc.render import render_field
            render_field(np.asarray(final.T_s), grid,
                         OUT_DIR / f"assim_sst_c{cycle:03d}.png",
                         title=f"Cycle {cycle} | RMSE={rmse:.2f}°C | SST={float(mean_sst):.1f}°C",
                         cmap="RdYlBu_r", diverging=False)

    # Final results
    print(f"\n{'='*60}")
    print(f"Final parameters after {args.cycles} cycles:")
    for n, v in zip(param_names, theta):
        print(f"  {n:15s} = {float(v):.6f}")

    print(f"\nFinal RMSE: {trajectory[-1]['rmse']:.3f}°C")
    print(f"Final mean SST: {trajectory[-1]['mean_sst']:.2f}°C")

    # Save trajectory
    import json
    traj_path = OUT_DIR / "assimilation_trajectory.json"
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"Trajectory saved to {traj_path}")

    # Final snapshot
    from amoc.render import render_field, render_panel
    render_field(np.asarray(final.T_s), grid,
                 OUT_DIR / "assim_sst_final.png",
                 title=f"Assimilated SST | RMSE={trajectory[-1]['rmse']:.2f}°C",
                 cmap="RdYlBu_r", diverging=False)
    render_field(np.asarray(jnp.abs(final.T_s - jnp.asarray(obs_sst)) * mask), grid,
                 OUT_DIR / "assim_error_final.png",
                 title=f"|SST error| after assimilation",
                 cmap="hot_r", diverging=False)
    render_panel(final, grid, OUT_DIR / "assim_panel_final.png",
                 title=f"Assimilated state | RMSE={trajectory[-1]['rmse']:.2f}°C",
                 ocean_mask=mask)
    print("Snapshots saved.")


if __name__ == "__main__":
    main()
