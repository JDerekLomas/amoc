#!/usr/bin/env python3
"""Headless simulation server with REST API for programmatic control.

Designed to be operated by Claude Code (or any HTTP client). No browser needed.
The simulation runs in the background; API calls adjust parameters, advance
the sim, and retrieve snapshots and diagnostics.

API:
  GET  /diagnostics              → JSON with step, SST, AMOC, psi, etc.
  GET  /snapshot?field=sst       → renders PNG to output/snapshot.png
  GET  /snapshot?field=psi       → streamfunction
  GET  /snapshot?field=panel     → 3-panel diagnostic (SST + psi + AMOC)
  POST /params                   → set physics parameters (JSON body)
  POST /run    {"steps": 1000}   → advance N steps, return diagnostics
  POST /reset                    → reset to initial conditions
  POST /pause                    → toggle pause

Start: python lab.py [--nx 128] [--ny 64] [--port 8766]
"""
import sys
import io
import json
import time
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

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
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)

# --- Global simulation state ---
sim = None  # set in main()


class SimState:
    def __init__(self, nx, ny):
        self.grid = make_grid(nx, ny)
        self.forcing = build_forcing(DATA_DIR, self.grid)
        self.state = build_initial_state(DATA_DIR, self.grid, self.forcing)
        self.params = Params()
        self.step_count = 0
        self.lock = threading.Lock()
        self.running = True
        self.paused = True  # start paused — advance via API
        self.bg_steps_per_tick = 50

    def advance(self, n_steps):
        """Run n_steps synchronously. Returns diagnostics."""
        with self.lock:
            self.state = run(self.state, self.forcing, self.params, self.grid, n_steps)
            jax.block_until_ready(self.state.T_s)
            self.step_count += n_steps
        return self.diagnostics()

    def diagnostics(self):
        """Compute key metrics from current state."""
        s = self.state
        f = self.forcing
        g = self.grid
        m = f.ocean_mask > 0.5
        m_np = np.asarray(m)

        sst = np.asarray(s.T_s)
        psi_s = np.asarray(s.psi_s)
        psi_d = np.asarray(s.psi_d)

        amoc = amoc_streamfunction(s, g, ocean_mask=f.ocean_mask)
        amoc_max = float(np.max(amoc))
        amoc_26n = 0.0
        # Find index closest to 26.5N
        lat_np = np.asarray(g.lat)
        idx_26 = int(np.argmin(np.abs(lat_np - 26.5)))
        amoc_26n = float(amoc[idx_26])

        return {
            "step": self.step_count,
            "sim_time": float(s.sim_time),
            "year": round(float(s.sim_time) / 10.0, 3),
            "sst_mean": round(float(np.mean(sst[m_np])), 2),
            "sst_min": round(float(np.min(sst[m_np])), 2),
            "sst_max": round(float(np.max(sst[m_np])), 2),
            "psi_s_range": [round(float(psi_s.min()), 4), round(float(psi_s.max()), 4)],
            "psi_d_range": [round(float(psi_d.min()), 4), round(float(psi_d.max()), 4)],
            "amoc_max": round(amoc_max, 4),
            "amoc_26n": round(amoc_26n, 4),
            "air_temp_mean": round(float(np.mean(np.asarray(s.air_temp))), 2),
            "deep_T_mean": round(float(np.mean(np.asarray(s.T_d)[m_np])), 2),
            "sal_mean": round(float(np.mean(np.asarray(s.S_s)[m_np])), 2),
        }

    def snapshot(self, field="sst"):
        """Render a field to PNG, return the file path."""
        s = self.state
        g = self.grid
        mask_np = np.asarray(self.forcing.ocean_mask) > 0.5
        extent = [g.lon0, g.lon1, g.lat0, g.lat1]
        lat_np = np.asarray(g.lat)

        if field == "panel":
            return self._render_panel()

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0a0a1a")
        ax.set_facecolor("#1a1a2e")

        if field == "sst":
            data = np.where(mask_np, np.asarray(s.T_s), np.nan)
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap="RdYlBu_r", vmin=-2, vmax=30, aspect="auto")
            ax.set_title("Sea Surface Temperature (°C)", color="white", fontsize=13)
        elif field == "psi":
            psi = np.asarray(s.psi_s)
            lim = max(float(np.nanpercentile(np.abs(psi[mask_np]), 99)), 0.01)
            data = np.where(mask_np, psi, np.nan)
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            # Add current vectors
            sx = max(1, g.nx // 24)
            sy = max(1, g.ny // 12)
            u, v = velocities_from_psi(s.psi_s, g)
            ax.quiver(np.asarray(g.lon)[::sx], lat_np[::sy],
                      u[::sy, ::sx], v[::sy, ::sx],
                      color="cyan", scale=2.0, width=0.002, alpha=0.5)
            ax.set_title("Streamfunction + Currents", color="white", fontsize=13)
        elif field == "air":
            data = np.where(mask_np, np.asarray(s.air_temp), np.nan)
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap="RdYlBu_r", vmin=-40, vmax=35, aspect="auto")
            ax.set_title("Air Temperature (°C)", color="white", fontsize=13)
        elif field == "deep":
            data = np.where(mask_np, np.asarray(s.T_d), np.nan)
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap="RdYlBu_r", vmin=-2, vmax=20, aspect="auto")
            ax.set_title("Deep Temperature (°C)", color="white", fontsize=13)
        elif field == "salinity":
            data = np.where(mask_np, np.asarray(s.S_s), np.nan)
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap="YlOrBr", vmin=30, vmax=37, aspect="auto")
            ax.set_title("Surface Salinity (PSU)", color="white", fontsize=13)
        else:
            data = np.where(mask_np, np.asarray(s.T_s), np.nan)
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap="RdYlBu_r", vmin=-2, vmax=30, aspect="auto")

        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        ax.set_xlabel("longitude", color="white", fontsize=9)
        ax.set_ylabel("latitude", color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333")

        d = self.diagnostics()
        fig.suptitle(
            f"Step {d['step']:,} | Year {d['year']} | "
            f"SST {d['sst_mean']}°C | AMOC {d['amoc_26n']:.3f}",
            color="white", fontsize=11, y=0.98
        )

        path = OUT_DIR / f"snapshot_{field}.png"
        fig.savefig(path, dpi=120, facecolor="#0a0a1a", bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def _render_panel(self):
        """3-panel diagnostic: SST + streamfunction + AMOC."""
        s = self.state
        g = self.grid
        mask_np = np.asarray(self.forcing.ocean_mask) > 0.5
        extent = [g.lon0, g.lon1, g.lat0, g.lat1]
        lat_np = np.asarray(g.lat)

        fig = plt.figure(figsize=(12, 9), facecolor="#0a0a1a")
        fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.06, hspace=0.38)

        # SST
        ax1 = fig.add_subplot(3, 1, 1)
        sst = np.where(mask_np, np.asarray(s.T_s), np.nan)
        ax1.imshow(sst, origin="lower", extent=extent, cmap="RdYlBu_r",
                   vmin=-2, vmax=30, aspect="auto")
        ax1.set_title("Sea Surface Temperature (°C)", color="white", fontsize=11)
        ax1.set_ylabel("lat", color="white", fontsize=9)
        ax1.tick_params(colors="white", labelsize=7)
        ax1.set_facecolor("#1a1a2e")
        for sp in ax1.spines.values(): sp.set_color("#333")

        # Streamfunction
        ax2 = fig.add_subplot(3, 1, 2)
        psi = np.asarray(s.psi_s)
        lim = max(float(np.nanpercentile(np.abs(psi[mask_np]), 99)), 0.01)
        ax2.imshow(np.where(mask_np, psi, np.nan), origin="lower", extent=extent,
                   cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        sx, sy = max(1, g.nx // 24), max(1, g.ny // 12)
        u, v = velocities_from_psi(s.psi_s, g)
        ax2.quiver(np.asarray(g.lon)[::sx], lat_np[::sy],
                   u[::sy, ::sx], v[::sy, ::sx],
                   color="cyan", scale=2.0, width=0.002, alpha=0.5)
        ax2.set_title("Streamfunction + Currents", color="white", fontsize=11)
        ax2.set_ylabel("lat", color="white", fontsize=9)
        ax2.tick_params(colors="white", labelsize=7)
        ax2.set_facecolor("#1a1a2e")
        for sp in ax2.spines.values(): sp.set_color("#333")

        # AMOC
        ax3 = fig.add_subplot(3, 1, 3)
        amoc = amoc_streamfunction(s, g, ocean_mask=self.forcing.ocean_mask)
        ax3.fill_between(lat_np, 0, amoc, alpha=0.3, color="#4fc3f7")
        ax3.plot(lat_np, amoc, color="#4fc3f7", linewidth=1.5)
        ax3.axhline(0, color="#444", linewidth=0.5)
        alim = max(abs(float(np.min(amoc))), abs(float(np.max(amoc))), 0.05) * 1.5
        ax3.set_ylim(-alim, alim)
        ax3.set_xlim(g.lat0, g.lat1)
        ax3.set_title("AMOC", color="white", fontsize=11)
        ax3.set_xlabel("latitude", color="white", fontsize=9)
        ax3.tick_params(colors="white", labelsize=7)
        ax3.set_facecolor("#1a1a2e")
        ax3.grid(True, alpha=0.1, color="white")
        for sp in ax3.spines.values(): sp.set_color("#333")

        d = self.diagnostics()
        fig.suptitle(
            f"Step {d['step']:,} | Year {d['year']} | "
            f"SST {d['sst_mean']}°C | AMOC@26N {d['amoc_26n']:.3f}",
            color="white", fontsize=12, fontweight="bold", y=0.97
        )

        path = OUT_DIR / "snapshot_panel.png"
        fig.savefig(path, dpi=100, facecolor="#0a0a1a")
        plt.close(fig)
        return str(path)

    def set_params(self, updates: dict):
        """Update physics parameters. Returns the full params dict."""
        with self.lock:
            current = self.params._asdict()
            for k, v in updates.items():
                if k in current:
                    current[k] = type(current[k])(v)
            self.params = Params(**current)
        return {k: float(v) if isinstance(v, (int, float)) else v
                for k, v in self.params._asdict().items()}

    def get_params(self):
        return {k: float(v) if isinstance(v, (int, float)) else v
                for k, v in self.params._asdict().items()}

    def reset(self):
        with self.lock:
            self.state = build_initial_state(DATA_DIR, self.grid, self.forcing)
            self.step_count = 0
        return self.diagnostics()

    def bg_loop(self):
        """Background simulation loop (only runs when not paused)."""
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            with self.lock:
                self.state = run(self.state, self.forcing, self.params,
                                 self.grid, self.bg_steps_per_tick)
                jax.block_until_ready(self.state.T_s)
                self.step_count += self.bg_steps_per_tick
            time.sleep(0.005)


class Handler(SimpleHTTPRequestHandler):
    def _json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/diagnostics":
            self._json(sim.diagnostics())

        elif parsed.path == "/snapshot":
            field = qs.get("field", ["sst"])[0]
            path = sim.snapshot(field)
            self._json({"path": path, "field": field})

        elif parsed.path == "/params":
            self._json(sim.get_params())

        elif parsed.path == "/help":
            self._json({
                "endpoints": {
                    "GET /diagnostics": "Current simulation metrics",
                    "GET /snapshot?field=sst|psi|air|deep|salinity|panel": "Render PNG snapshot",
                    "GET /params": "Current parameter values",
                    "POST /run {steps: N}": "Advance N steps, return diagnostics",
                    "POST /params {key: value, ...}": "Update parameters",
                    "POST /reset": "Reset to initial conditions",
                    "POST /pause": "Toggle background simulation",
                }
            })
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        body = self._read_body()

        if parsed.path == "/run":
            n = body.get("steps", 1000)
            d = sim.advance(n)
            self._json(d)

        elif parsed.path == "/params":
            result = sim.set_params(body)
            self._json(result)

        elif parsed.path == "/reset":
            d = sim.reset()
            self._json(d)

        elif parsed.path == "/pause":
            sim.paused = not sim.paused
            self._json({"paused": sim.paused})

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    global sim

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Initializing {args.nx}x{args.ny} simulation...")

    sim = SimState(args.nx, args.ny)

    # JIT warmup
    print("JIT compiling...")
    sim.advance(10)
    sim.step_count = 0
    print("  Ready")

    # Start background loop (paused by default)
    bg = threading.Thread(target=sim.bg_loop, daemon=True)
    bg.start()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  Lab API: http://127.0.0.1:{args.port}")
    print(f"  GET  /diagnostics")
    print(f"  GET  /snapshot?field=sst|psi|panel")
    print(f"  POST /run {{\"steps\": 1000}}")
    print(f"  POST /params {{\"freshwater_forcing\": 1.0}}")
    print(f"  POST /reset\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        sim.running = False


if __name__ == "__main__":
    main()
