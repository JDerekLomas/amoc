#!/usr/bin/env python3
"""Interactive ocean simulation viewer.

Runs the JAX simulation and displays live SST + streamfunction in the browser.
Uses a simple HTTP server with server-sent events for frame streaming.

Usage: python app.py [--nx 128] [--ny 64] [--port 8765]
"""
import sys
import io
import json
import time
import base64
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).parent / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from amoc.grid import make_grid
from amoc.data import build_forcing, build_initial_state
from amoc.state import Params
from amoc.step import run

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Global state
sim_state = None
sim_grid = None
sim_forcing = None
sim_params = None
sim_step_count = 0
sim_lock = threading.Lock()
sim_running = True
STEPS_PER_TICK = 50


def state_to_json():
    """Extract key fields as base64-encoded float32 arrays for the browser."""
    with sim_lock:
        s = sim_state
        g = sim_grid
        f = sim_forcing
        steps = sim_step_count

    mask = np.asarray(f.ocean_mask)
    sst = np.asarray(s.T_s).astype(np.float32)
    psi = np.asarray(s.psi_s).astype(np.float32)
    air = np.asarray(s.air_temp).astype(np.float32)

    ny, nx = sst.shape
    ocean_mask = mask > 0.5
    mean_sst = float(np.mean(sst[ocean_mask])) if np.any(ocean_mask) else 0.0
    psi_range = [float(np.min(psi)), float(np.max(psi))]

    return json.dumps({
        "nx": nx, "ny": ny,
        "step": steps,
        "sim_time": float(s.sim_time),
        "mean_sst": round(mean_sst, 2),
        "psi_range": [round(psi_range[0], 4), round(psi_range[1], 4)],
        "sst": base64.b64encode(sst.tobytes()).decode(),
        "psi": base64.b64encode(psi.tobytes()).decode(),
        "mask": base64.b64encode(mask.astype(np.float32).tobytes()).decode(),
    })


def sim_loop():
    """Background thread that advances the simulation."""
    global sim_state, sim_step_count
    while sim_running:
        with sim_lock:
            sim_state = run(sim_state, sim_forcing, sim_params, sim_grid, STEPS_PER_TICK)
            jax.block_until_ready(sim_state.T_s)
            sim_step_count += STEPS_PER_TICK
        time.sleep(0.01)  # yield to HTTP thread


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SimAMOC — JAX Ocean Simulation</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a1a; color: #ddd; font-family: system-ui, sans-serif; }
#header { padding: 12px 20px; background: #111; border-bottom: 1px solid #333;
           display: flex; justify-content: space-between; align-items: center; }
#header h1 { font-size: 18px; color: #4fc3f7; }
#stats { font-size: 13px; color: #aaa; }
#canvas-wrap { display: flex; flex-direction: column; align-items: center;
               padding: 20px; gap: 8px; }
canvas { image-rendering: pixelated; border: 1px solid #333; border-radius: 4px; }
#controls { padding: 10px 20px; display: flex; gap: 20px; align-items: center; }
#controls label { font-size: 13px; color: #888; }
select, button { background: #222; color: #ddd; border: 1px solid #444;
                 padding: 4px 10px; border-radius: 3px; cursor: pointer; }
button:hover { background: #333; }
</style>
</head>
<body>
<div id="header">
  <h1>SimAMOC — JAX Ocean Circulation</h1>
  <div id="stats">Loading...</div>
</div>
<div id="controls">
  <label>Field: <select id="field-select">
    <option value="sst" selected>Sea Surface Temperature</option>
    <option value="psi">Streamfunction</option>
  </select></label>
  <button onclick="togglePause()">Pause/Resume</button>
</div>
<div id="canvas-wrap">
  <canvas id="c"></canvas>
</div>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let nx = 0, ny = 0, paused = false;
let field = 'sst';

document.getElementById('field-select').onchange = e => { field = e.target.value; };
function togglePause() { paused = !paused; }

// Color maps
function sstColor(t) {
  // RdYlBu_r: blue(-2) -> yellow(15) -> red(30)
  let f = (t + 2) / 32; // normalize to 0-1
  f = Math.max(0, Math.min(1, f));
  let r, g, b;
  if (f < 0.25) { // deep blue to light blue
    const s = f / 0.25;
    r = Math.round(50 + 30 * s); g = Math.round(60 + 120 * s); b = Math.round(180 + 60 * s);
  } else if (f < 0.5) { // light blue to yellow
    const s = (f - 0.25) / 0.25;
    r = Math.round(80 + 175 * s); g = Math.round(180 + 60 * s); b = Math.round(240 - 200 * s);
  } else if (f < 0.75) { // yellow to orange
    const s = (f - 0.5) / 0.25;
    r = Math.round(255); g = Math.round(240 - 120 * s); b = Math.round(40 - 30 * s);
  } else { // orange to dark red
    const s = (f - 0.75) / 0.25;
    r = Math.round(255 - 60 * s); g = Math.round(120 - 80 * s); b = Math.round(10);
  }
  return [r, g, b];
}

function psiColor(v, lim) {
  let f = (v / lim + 1) / 2; // normalize to 0-1
  f = Math.max(0, Math.min(1, f));
  let r, g, b;
  if (f < 0.5) { // blue (cyclonic)
    const s = f / 0.5;
    r = Math.round(20 + 60 * s); g = Math.round(40 + 80 * s); b = Math.round(180 + 40 * s);
  } else { // red (anticyclonic)
    const s = (f - 0.5) / 0.5;
    r = Math.round(80 + 175 * s); g = Math.round(120 - 80 * s); b = Math.round(220 - 200 * s);
  }
  return [r, g, b];
}

let imageData = null;

function render(data) {
  if (!data.sst) return;
  const w = data.nx, h = data.ny;
  if (w !== nx || h !== ny) {
    nx = w; ny = h;
    // Scale canvas: 3x or 4x
    const scale = Math.max(1, Math.min(4, Math.floor(1200 / nx)));
    canvas.width = nx * scale; canvas.height = ny * scale;
    canvas.style.width = (nx * scale) + 'px';
    canvas.style.height = (ny * scale) + 'px';
    ctx.imageSmoothingEnabled = false;
  }

  // Decode base64 Float32Arrays
  const sstBuf = new Float32Array(Uint8Array.from(atob(data.sst), c => c.charCodeAt(0)).buffer);
  const psiBuf = new Float32Array(Uint8Array.from(atob(data.psi), c => c.charCodeAt(0)).buffer);
  const maskBuf = new Float32Array(Uint8Array.from(atob(data.mask), c => c.charCodeAt(0)).buffer);

  // Draw to offscreen then scale
  const off = new OffscreenCanvas(nx, ny);
  const offCtx = off.getContext('2d');
  const img = offCtx.createImageData(nx, ny);
  const d = img.data;

  const psiLim = Math.max(Math.abs(data.psi_range[0]), Math.abs(data.psi_range[1]), 0.01);

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      // Flip y: row 0 is south, but canvas row 0 is top
      const srcIdx = (ny - 1 - j) * nx + i;
      const dstIdx = (j * nx + i) * 4;
      const m = maskBuf[srcIdx];

      if (m < 0.5) {
        d[dstIdx] = 30; d[dstIdx+1] = 30; d[dstIdx+2] = 40; d[dstIdx+3] = 255;
        continue;
      }

      let rgb;
      if (field === 'sst') {
        rgb = sstColor(sstBuf[srcIdx]);
      } else {
        rgb = psiColor(psiBuf[srcIdx], psiLim);
      }
      d[dstIdx] = rgb[0]; d[dstIdx+1] = rgb[1]; d[dstIdx+2] = rgb[2]; d[dstIdx+3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

  // Stats
  document.getElementById('stats').textContent =
    `Step ${data.step.toLocaleString()} | Year ${(data.sim_time/10).toFixed(2)} | ` +
    `SST mean: ${data.mean_sst}°C | ψ: [${data.psi_range[0]}, ${data.psi_range[1]}]`;
}

// Poll for frames
async function poll() {
  while (true) {
    if (!paused) {
      try {
        const resp = await fetch('/state');
        const data = await resp.json();
        render(data);
      } catch(e) {}
    }
    await new Promise(r => setTimeout(r, 80)); // ~12 fps
  }
}
poll();
</script>
</body>
</html>"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())
        elif self.path == "/state":
            data = state_to_json()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data.encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    global sim_state, sim_grid, sim_forcing, sim_params

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"Initializing {args.nx}x{args.ny} simulation...")

    sim_grid = make_grid(args.nx, args.ny)
    sim_forcing = build_forcing(DATA_DIR, sim_grid)
    sim_state = build_initial_state(DATA_DIR, sim_grid, sim_forcing)
    sim_params = Params()

    # JIT warmup
    print("JIT compiling...")
    sim_state = run(sim_state, sim_forcing, sim_params, sim_grid, 10)
    jax.block_until_ready(sim_state.T_s)
    print("  Ready")

    # Start simulation thread
    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    sim_thread.start()

    # Start HTTP server
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  Open {url} in your browser\n")

    # Try to open browser automatically
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")
        sim_running = False


if __name__ == "__main__":
    main()
