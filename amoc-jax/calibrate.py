#!/usr/bin/env python3
"""Autodiff-based parameter calibration for the coupled ocean-atmosphere model.

Uses jax.grad to find parameters that minimize SST error against observations.
Differentiates through short simulation windows (100-500 steps).

Usage: python calibrate.py [--steps 200] [--epochs 30] [--lr 0.1]
"""
import sys
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

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def make_loss_fn(grid, forcing, obs_sst, n_steps: int):
    """Build a differentiable loss function over tunable parameters.

    We parameterize 4 key radiation/heat-balance knobs as a vector:
        theta = [S_solar, A_olr, B_olr, kappa_T]
    and return a function loss(theta, state0) -> scalar.
    """
    mask = forcing.ocean_mask
    ocean_cells = jnp.sum(mask)

    @jax.jit
    def loss_fn(theta, state0):
        """Run n_steps, return MSE(mean_SST - observed) + penalty for drift."""
        S_solar, A_olr, B_olr, kappa_T = theta

        params = Params(
            S_solar=S_solar,
            A_olr=A_olr,
            B_olr=B_olr,
            kappa_T=kappa_T,
        )

        # Forward integration using scan
        def body(s, _):
            return step(s, forcing, params, grid), None

        final, _ = jax.lax.scan(body, state0, xs=None, length=n_steps)

        # Loss: RMSE of SST vs observed (ocean cells only)
        diff = (final.T_s - obs_sst) * mask
        mse = jnp.sum(diff ** 2) / jnp.maximum(ocean_cells, 1.0)

        # Penalty: mean SST should be ~14.5C (global ocean average)
        mean_sst = jnp.sum(final.T_s * mask) / jnp.maximum(ocean_cells, 1.0)
        mean_penalty = 10.0 * (mean_sst - 14.5) ** 2

        return mse + mean_penalty, (final, mse, mean_sst)

    return loss_fn


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200,
                        help="Steps per gradient evaluation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=500,
                        help="Warmup steps before calibration")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Calibration: {args.nx}x{args.ny}, {args.steps} steps/eval, "
          f"{args.epochs} epochs, lr={args.lr}")

    grid = make_grid(args.nx, args.ny)
    forcing = build_forcing(DATA_DIR, grid)
    state0 = build_initial_state(DATA_DIR, grid, forcing)
    mask = forcing.ocean_mask

    # Load observed SST as target
    obs_sst = _load_field_bin_or_json(DATA_DIR, "sst", "sst", grid)
    if obs_sst is None:
        print("ERROR: Cannot load observed SST for calibration")
        return
    obs_sst = jnp.where(mask > 0.5, obs_sst, 0.0)
    obs_mean = float(jnp.sum(obs_sst * mask) / jnp.sum(mask))
    print(f"Observed SST mean: {obs_mean:.2f}C")

    # Warmup: run forward with default params to let transients settle
    if args.warmup > 0:
        print(f"\nWarming up ({args.warmup} steps with default params)...")
        default_params = Params()
        @jax.jit
        def warmup_body(s, _):
            return step(s, forcing, default_params, grid), None
        state0, _ = jax.lax.scan(warmup_body, state0, xs=None, length=args.warmup)
        jax.block_until_ready(state0.T_s)
        warm_mean = float(jnp.sum(state0.T_s * mask) / jnp.sum(mask))
        print(f"  Post-warmup SST mean: {warm_mean:.2f}C")

    # Build loss and grad
    loss_fn = make_loss_fn(grid, forcing, obs_sst, args.steps)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    # Initial parameters
    theta = jnp.array([6.2, 1.8, 0.13, 2.5e-4])  # S_solar, A_olr, B_olr, kappa_T
    param_names = ["S_solar", "A_olr", "B_olr", "kappa_T"]
    param_scales = jnp.array([1.0, 0.1, 0.01, 1e-5])  # Scale for Adam-like stepping

    print(f"\nInitial: {dict(zip(param_names, theta.tolist()))}")

    # JIT compile
    print("JIT compiling grad (this takes a minute)...")
    t0 = time.time()
    (loss, (final, mse, mean_sst)), grads = grad_fn(theta, state0)
    jax.block_until_ready(grads)
    print(f"  Compiled in {time.time()-t0:.1f}s")
    print(f"  Initial loss={float(loss):.2f}, RMSE={float(jnp.sqrt(mse)):.2f}, "
          f"mean_SST={float(mean_sst):.2f}")
    print(f"  Gradients: {dict(zip(param_names, grads.tolist()))}")

    # Gradient descent with momentum
    velocity = jnp.zeros_like(theta)
    momentum = 0.9
    best_loss = float(loss)
    best_theta = theta.copy()

    print(f"\n{'epoch':>5} | {'loss':>8} | {'RMSE':>6} | {'mean_SST':>8} | params")
    print("-" * 80)

    state_running = state0
    for epoch in range(args.epochs):
        t0 = time.time()
        (loss, (final, mse, mean_sst)), grads = grad_fn(theta, state_running)
        jax.block_until_ready(grads)

        # Clip gradients
        grad_norm = jnp.sqrt(jnp.sum(grads ** 2))
        grads = jnp.where(grad_norm > 10.0, grads * 10.0 / grad_norm, grads)

        # Momentum update with per-parameter learning rates
        velocity = momentum * velocity - args.lr * grads * param_scales
        theta = theta + velocity

        # Clamp to physical ranges
        theta = theta.at[0].set(jnp.clip(theta[0], 3.0, 15.0))   # S_solar
        theta = theta.at[1].set(jnp.clip(theta[1], 0.5, 4.0))    # A_olr
        theta = theta.at[2].set(jnp.clip(theta[2], 0.01, 0.5))   # B_olr
        theta = theta.at[3].set(jnp.clip(theta[3], 1e-5, 1e-3))  # kappa_T

        # Track best
        if float(loss) < best_loss:
            best_loss = float(loss)
            best_theta = theta.copy()

        # Advance the running state (so we calibrate on evolving state)
        state_running = final

        elapsed = time.time() - t0
        param_str = " ".join(f"{n}={float(v):.4f}" for n, v in zip(param_names, theta))
        print(f"{epoch:5d} | {float(loss):8.2f} | {float(jnp.sqrt(mse)):6.2f} | "
              f"{float(mean_sst):8.2f} | {param_str} ({elapsed:.1f}s)")

        # Check for NaN
        if jnp.any(jnp.isnan(grads)):
            print("  NaN in gradients — stopping")
            break

    print(f"\nBest parameters (loss={best_loss:.2f}):")
    for n, v in zip(param_names, best_theta.tolist()):
        print(f"  {n} = {v}")

    # Run a validation with best params
    print(f"\nValidation run with best params (2000 steps)...")
    S_solar, A_olr, B_olr, kappa_T = best_theta
    best_params = Params(S_solar=float(S_solar), A_olr=float(A_olr),
                         B_olr=float(B_olr), kappa_T=float(kappa_T))
    state_val = build_initial_state(DATA_DIR, grid, forcing)

    @jax.jit
    def val_body(s, _):
        return step(s, forcing, best_params, grid), None
    state_val, _ = jax.lax.scan(val_body, state_val, xs=None, length=2000)
    jax.block_until_ready(state_val.T_s)

    val_mean = float(jnp.sum(state_val.T_s * mask) / jnp.sum(mask))
    val_diff = (state_val.T_s - obs_sst) * mask
    val_rmse = float(jnp.sqrt(jnp.sum(val_diff ** 2) / jnp.sum(mask)))
    print(f"  Mean SST: {val_mean:.2f}C (obs: {obs_mean:.2f}C)")
    print(f"  RMSE: {val_rmse:.2f}C")

    # Render
    try:
        from amoc.render import render_field
        out_dir = Path(__file__).parent / "output"
        out_dir.mkdir(exist_ok=True)
        render_field(np.asarray(state_val.T_s), grid,
                     out_dir / "sst_calibrated.png",
                     title=f"Calibrated SST (RMSE={val_rmse:.1f}C, mean={val_mean:.1f}C)",
                     cmap="RdYlBu_r", diverging=False)
        render_field(np.asarray(jnp.abs(state_val.T_s - obs_sst) * mask), grid,
                     out_dir / "sst_error.png",
                     title=f"|SST - observed| (RMSE={val_rmse:.1f}C)",
                     cmap="hot_r", diverging=False)
        print(f"  Rendered to {out_dir}/")
    except Exception as e:
        print(f"  Render error: {e}")


if __name__ == "__main__":
    main()
