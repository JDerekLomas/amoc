#!/usr/bin/env python3
"""Animate the coupled ocean-atmosphere simulation.

Renders SST, streamfunction, and diagnostics as a live matplotlib animation,
saving frames to an mp4 video.

Usage: python play.py [--nx 256] [--ny 128] [--frames 200] [--steps-per-frame 50]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for rendering
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import TwoSlopeNorm

from amoc.grid import make_grid
from amoc.data import build_forcing, build_initial_state
from amoc.state import Params
from amoc.step import run
from amoc.diagnostics import amoc_streamfunction
from amoc.render import velocities_from_psi

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--steps-per-frame", type=int, default=100)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", type=str, default="output/simulation.mp4")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Animation: {args.nx}x{args.ny}, {args.frames} frames, "
          f"{args.steps_per_frame} steps/frame")

    grid = make_grid(args.nx, args.ny)
    forcing = build_forcing(DATA_DIR, grid)
    state = build_initial_state(DATA_DIR, grid, forcing)
    params = Params()
    mask = forcing.ocean_mask
    mask_np = np.asarray(mask) > 0.5
    extent = [grid.lon0, grid.lon1, grid.lat0, grid.lat1]
    lat_np = np.asarray(grid.lat)

    # Warm up JIT
    print("JIT compiling...")
    state = run(state, forcing, params, grid, 10)
    jax.block_until_ready(state.T_s)
    print("  Done")

    # Set up figure: 3-panel layout
    fig = plt.figure(figsize=(14, 10), facecolor="black")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.93, bottom=0.05,
                        hspace=0.35)

    # Panel 1: SST
    ax1 = fig.add_subplot(3, 1, 1)
    sst_data = np.where(mask_np, np.asarray(state.T_s), np.nan)
    im_sst = ax1.imshow(sst_data, origin="lower", extent=extent,
                         cmap="RdYlBu_r", vmin=-2, vmax=30, aspect="auto")
    ax1.set_title("Sea Surface Temperature (°C)", color="white", fontsize=13)
    ax1.set_ylabel("latitude", color="white")
    ax1.tick_params(colors="white")
    cb1 = plt.colorbar(im_sst, ax=ax1, shrink=0.8, pad=0.02)
    cb1.ax.yaxis.set_tick_params(color="white")
    cb1.ax.yaxis.set_ticklabels(cb1.ax.get_yticks(), color="white")

    # Panel 2: Surface streamfunction with current vectors
    ax2 = fig.add_subplot(3, 1, 2)
    psi_data = np.where(mask_np, np.asarray(state.psi_s), np.nan)
    im_psi = ax2.imshow(psi_data, origin="lower", extent=extent,
                         cmap="RdBu_r", vmin=-0.3, vmax=0.3, aspect="auto")
    ax2.set_title("Surface Streamfunction + Currents", color="white", fontsize=13)
    ax2.set_ylabel("latitude", color="white")
    ax2.tick_params(colors="white")
    cb2 = plt.colorbar(im_psi, ax=ax2, shrink=0.8, pad=0.02)
    cb2.ax.yaxis.set_tick_params(color="white")
    cb2.ax.yaxis.set_ticklabels(cb2.ax.get_yticks(), color="white")

    # Quiver (subsampled)
    sx = max(1, args.nx // 24)
    sy = max(1, args.ny // 12)
    lon_q = np.asarray(grid.lon)[::sx]
    lat_q = np.asarray(grid.lat)[::sy]
    u, v = velocities_from_psi(state.psi_s, grid)
    quiv = ax2.quiver(lon_q, lat_q, u[::sy, ::sx], v[::sy, ::sx],
                       color="cyan", scale=2.0, width=0.002, alpha=0.6)

    # Panel 3: AMOC profile + diagnostics text
    ax3 = fig.add_subplot(3, 1, 3)
    amoc = amoc_streamfunction(state, grid, ocean_mask=mask)
    line_amoc, = ax3.plot(lat_np, amoc, color="#4fc3f7", linewidth=2)
    ax3.axhline(0, color="#666", linewidth=0.5)
    fill_amoc = ax3.fill_between(lat_np, 0, amoc, alpha=0.2, color="#4fc3f7")
    ax3.set_title("Atlantic Meridional Overturning (AMOC)", color="white", fontsize=13)
    ax3.set_xlabel("latitude", color="white")
    ax3.set_ylabel("Ψ (model units)", color="white")
    ax3.set_xlim(grid.lat0, grid.lat1)
    ax3.set_ylim(-0.5, 0.5)
    ax3.tick_params(colors="white")
    ax3.grid(True, alpha=0.15, color="white")

    # Time annotation
    time_text = fig.text(0.5, 0.96,
                          f"Step 0 | SST mean: {float(jnp.mean(state.T_s[mask > 0.5])):.1f}°C",
                          ha="center", va="center", color="white", fontsize=14,
                          fontweight="bold")

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_color("#444")

    # Precompute: collect frames
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRendering {args.frames} frames to {out_path}...")
    t_start = time.time()

    # Use FFMpeg writer
    try:
        writer = FFMpegWriter(fps=args.fps, metadata={"title": "SimAMOC JAX"},
                               bitrate=3000)
    except Exception:
        print("  FFMpeg not available, falling back to pillow gif")
        writer = None

    total_steps = 0

    def update(frame_num):
        nonlocal state, total_steps, fill_amoc

        # Advance simulation
        state = run(state, forcing, params, grid, args.steps_per_frame)
        jax.block_until_ready(state.T_s)
        total_steps += args.steps_per_frame

        # Update SST
        sst = np.where(mask_np, np.asarray(state.T_s), np.nan)
        im_sst.set_data(sst)

        # Update streamfunction
        psi = np.asarray(state.psi_s)
        psi_lim = max(float(np.nanpercentile(np.abs(psi[mask_np]), 99)), 0.01)
        psi_show = np.where(mask_np, psi, np.nan)
        im_psi.set_data(psi_show)
        im_psi.set_clim(-psi_lim, psi_lim)

        # Update quiver
        u, v = velocities_from_psi(state.psi_s, grid)
        quiv.set_UVC(u[::sy, ::sx], v[::sy, ::sx])

        # Update AMOC
        amoc = amoc_streamfunction(state, grid, ocean_mask=mask)
        line_amoc.set_ydata(amoc)
        # Remove old fill, add new
        fill_amoc.remove()
        fill_amoc = ax3.fill_between(lat_np, 0, amoc, alpha=0.2, color="#4fc3f7")
        amoc_max = float(np.max(amoc))
        amoc_lim = max(abs(float(np.min(amoc))), abs(amoc_max), 0.1) * 1.3
        ax3.set_ylim(-amoc_lim, amoc_lim)

        # Update text
        mean_sst = float(jnp.mean(state.T_s[mask > 0.5]))
        sim_t = float(state.sim_time)
        time_text.set_text(
            f"Step {total_steps:,} | Year {sim_t / 10:.2f} | "
            f"SST mean: {mean_sst:.1f}°C | AMOC max: {amoc_max:.3f}"
        )

        # Progress
        elapsed = time.time() - t_start
        rate = total_steps / elapsed
        if frame_num % 10 == 0:
            print(f"  frame {frame_num:4d}/{args.frames} | "
                  f"step {total_steps:6,} | {rate:.0f} steps/s | "
                  f"SST {mean_sst:.1f}°C")

        return [im_sst, im_psi, quiv, line_amoc, fill_amoc, time_text]

    if writer is not None:
        anim = FuncAnimation(fig, update, frames=args.frames, blit=False)
        anim.save(str(out_path), writer=writer, dpi=100)
    else:
        # Fallback: save individual PNGs
        png_dir = out_path.parent / "frames"
        png_dir.mkdir(exist_ok=True)
        for i in range(args.frames):
            update(i)
            fig.savefig(png_dir / f"frame_{i:04d}.png", dpi=100,
                        facecolor="black")
        print(f"  Saved {args.frames} PNGs to {png_dir}/")

    total_time = time.time() - t_start
    print(f"\nDone: {args.frames} frames, {total_steps:,} steps in {total_time:.0f}s")
    print(f"  Output: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
