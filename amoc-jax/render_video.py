#!/usr/bin/env python3
"""Render simulation frames and stitch into video with ffmpeg.

Much faster than matplotlib animation — renders PNGs in parallel then stitches.

Usage: python render_video.py [--frames 120] [--steps-per-frame 500]
"""
import sys
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amoc.grid import make_grid
from amoc.data import build_forcing, build_initial_state
from amoc.state import Params
from amoc.step import run
from amoc.diagnostics import amoc_streamfunction
from amoc.render import velocities_from_psi

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def render_frame(state, grid, forcing, step_num, out_path):
    """Render a single 3-panel frame."""
    mask_np = np.asarray(forcing.ocean_mask) > 0.5
    extent = [grid.lon0, grid.lon1, grid.lat0, grid.lat1]
    lat_np = np.asarray(grid.lat)

    fig = plt.figure(figsize=(12, 9), facecolor="#0a0a1a")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.06, hspace=0.38)

    # SST
    ax1 = fig.add_subplot(3, 1, 1)
    sst = np.where(mask_np, np.asarray(state.T_s), np.nan)
    ax1.imshow(sst, origin="lower", extent=extent, cmap="RdYlBu_r",
               vmin=-2, vmax=30, aspect="auto")
    ax1.set_title("Sea Surface Temperature (°C)", color="white", fontsize=12)
    ax1.set_ylabel("lat", color="white", fontsize=9)
    ax1.tick_params(colors="white", labelsize=8)
    ax1.set_facecolor("#1a1a2e")
    for s in ax1.spines.values(): s.set_color("#333")

    # Streamfunction + currents
    ax2 = fig.add_subplot(3, 1, 2)
    psi = np.asarray(state.psi_s)
    psi_lim = max(float(np.nanpercentile(np.abs(psi[mask_np]), 99)), 0.01)
    psi_show = np.where(mask_np, psi, np.nan)
    ax2.imshow(psi_show, origin="lower", extent=extent, cmap="RdBu_r",
               vmin=-psi_lim, vmax=psi_lim, aspect="auto")
    # Current vectors
    sx = max(1, grid.nx // 24)
    sy = max(1, grid.ny // 12)
    u, v = velocities_from_psi(state.psi_s, grid)
    ax2.quiver(np.asarray(grid.lon)[::sx], np.asarray(grid.lat)[::sy],
               u[::sy, ::sx], v[::sy, ::sx],
               color="cyan", scale=2.0, width=0.002, alpha=0.5)
    ax2.set_title("Streamfunction + Currents", color="white", fontsize=12)
    ax2.set_ylabel("lat", color="white", fontsize=9)
    ax2.tick_params(colors="white", labelsize=8)
    ax2.set_facecolor("#1a1a2e")
    for s in ax2.spines.values(): s.set_color("#333")

    # AMOC
    ax3 = fig.add_subplot(3, 1, 3)
    amoc = amoc_streamfunction(state, grid, ocean_mask=forcing.ocean_mask)
    ax3.fill_between(lat_np, 0, amoc, alpha=0.3, color="#4fc3f7")
    ax3.plot(lat_np, amoc, color="#4fc3f7", linewidth=1.5)
    ax3.axhline(0, color="#444", linewidth=0.5)
    amoc_lim = max(abs(float(np.min(amoc))), abs(float(np.max(amoc))), 0.05) * 1.5
    ax3.set_ylim(-amoc_lim, amoc_lim)
    ax3.set_xlim(grid.lat0, grid.lat1)
    ax3.set_title("AMOC Streamfunction", color="white", fontsize=12)
    ax3.set_xlabel("latitude", color="white", fontsize=9)
    ax3.set_ylabel("Ψ", color="white", fontsize=9)
    ax3.tick_params(colors="white", labelsize=8)
    ax3.set_facecolor("#1a1a2e")
    ax3.grid(True, alpha=0.1, color="white")
    for s in ax3.spines.values(): s.set_color("#333")

    # Header
    mean_sst = float(jnp.mean(state.T_s[forcing.ocean_mask > 0.5]))
    sim_time = float(state.sim_time)
    year = sim_time / 10.0
    amoc_max = float(np.max(amoc))
    fig.suptitle(
        f"Step {step_num:,}  |  Year {year:.2f}  |  "
        f"SST {mean_sst:.1f}°C  |  AMOC max {amoc_max:.3f}",
        color="white", fontsize=13, fontweight="bold", y=0.97
    )

    fig.savefig(out_path, dpi=100, facecolor="#0a0a1a")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--frames", type=int, default=120,
                        help="Total frames (at 15fps, 120 = 8 seconds)")
    parser.add_argument("--steps-per-frame", type=int, default=500,
                        help="Simulation steps between frames")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", type=str, default="output/first-year.mp4")
    args = parser.parse_args()

    total_steps = args.frames * args.steps_per_frame
    sim_years = total_steps * 5e-5 * 1.0 / 10.0  # dt * year_speed / T_YEAR
    print(f"Rendering {args.frames} frames at {args.nx}x{args.ny}")
    print(f"  {args.steps_per_frame} steps/frame = {total_steps:,} total steps")
    print(f"  ~{sim_years:.2f} simulated years at 15fps = {args.frames/args.fps:.0f}s video")

    grid = make_grid(args.nx, args.ny)
    forcing = build_forcing(DATA_DIR, grid)
    state = build_initial_state(DATA_DIR, grid, forcing)
    params = Params()

    # JIT warmup
    print("JIT compiling...")
    state = run(state, forcing, params, grid, 10)
    jax.block_until_ready(state.T_s)

    # Frame directory
    frame_dir = Path("output/frames")
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering frames to {frame_dir}/...")
    t_start = time.time()
    step_count = 0

    for i in range(args.frames):
        # Advance simulation
        state = run(state, forcing, params, grid, args.steps_per_frame)
        jax.block_until_ready(state.T_s)
        step_count += args.steps_per_frame

        # Render frame
        frame_path = frame_dir / f"frame_{i:04d}.png"
        render_frame(state, grid, forcing, step_count, frame_path)

        if i % 10 == 0 or i == args.frames - 1:
            elapsed = time.time() - t_start
            rate = step_count / elapsed
            mean_sst = float(jnp.mean(state.T_s[forcing.ocean_mask > 0.5]))
            print(f"  frame {i+1:4d}/{args.frames} | step {step_count:6,} | "
                  f"{rate:.0f} steps/s | SST {mean_sst:.1f}°C")

    # Stitch with ffmpeg
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nStitching video with ffmpeg -> {out_path}")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "23", "-preset", "medium",
        str(out_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    total_time = time.time() - t_start
    file_size = out_path.stat().st_size / (1024 * 1024)
    print(f"\nDone: {out_path} ({file_size:.1f} MB)")
    print(f"  {args.frames} frames, {step_count:,} steps in {total_time:.0f}s")


if __name__ == "__main__":
    main()
