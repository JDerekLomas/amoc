#!/usr/bin/env python3
"""Interactive ocean simulation viewer.

Runs the JAX simulation and displays live SST + streamfunction in the browser.
Controls: speed, field view, and tunable physics parameters.

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
from urllib.parse import urlparse, parse_qs

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
sim_paused = False
STEPS_PER_TICK = 50


def state_to_json():
    with sim_lock:
        s = sim_state
        f = sim_forcing
        steps = sim_step_count
        p = sim_params

    mask = np.asarray(f.ocean_mask)
    sst = np.asarray(s.T_s).astype(np.float32)
    psi = np.asarray(s.psi_s).astype(np.float32)
    air = np.asarray(s.air_temp).astype(np.float32)
    moisture = np.asarray(s.moisture).astype(np.float32)

    ocean_mask = mask > 0.5
    ny, nx = sst.shape
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
        "air": base64.b64encode(air.tobytes()).decode(),
        "moisture": base64.b64encode(moisture.tobytes()).decode(),
        "mask": base64.b64encode(mask.astype(np.float32).tobytes()).decode(),
        "paused": sim_paused,
        "steps_per_tick": STEPS_PER_TICK,
        "params": {
            "S_solar": float(p.S_solar),
            "A_olr": float(p.A_olr),
            "B_olr": float(p.B_olr),
            "wind_strength": float(p.wind_strength),
            "freshwater_forcing": float(p.freshwater_forcing),
            "gamma_mix": float(p.gamma_mix),
            "gamma_deep_form": float(p.gamma_deep_form),
            "kappa_T": float(p.kappa_T),
        },
    })


def sim_loop():
    global sim_state, sim_step_count
    while sim_running:
        if sim_paused:
            time.sleep(0.05)
            continue
        with sim_lock:
            sim_state = run(sim_state, sim_forcing, sim_params, sim_grid, STEPS_PER_TICK)
            jax.block_until_ready(sim_state.T_s)
            sim_step_count += STEPS_PER_TICK
        time.sleep(0.005)


HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SimAMOC — JAX Ocean Simulation</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a1a; color: #ddd; font-family: system-ui, -apple-system, sans-serif;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
#header { padding: 10px 20px; background: #111; border-bottom: 1px solid #333;
           display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
#header h1 { font-size: 16px; color: #4fc3f7; font-weight: 600; }
#stats { font-size: 12px; color: #999; font-variant-numeric: tabular-nums; }
#main { display: flex; flex: 1; overflow: hidden; }
#canvas-area { flex: 1; display: flex; justify-content: center; align-items: center; padding: 10px; }
canvas { image-rendering: pixelated; border-radius: 4px; max-width: 100%; max-height: 100%; }
#sidebar { width: 260px; background: #111; border-left: 1px solid #333; padding: 14px;
           overflow-y: auto; flex-shrink: 0; }
.section { margin-bottom: 18px; }
.section h3 { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em;
              color: #666; margin-bottom: 8px; font-weight: 600; }
.control { margin-bottom: 10px; }
.control label { display: block; font-size: 12px; color: #aaa; margin-bottom: 3px; }
.control input[type=range] { width: 100%; accent-color: #4fc3f7; }
.control .val { float: right; font-size: 11px; color: #4fc3f7; font-variant-numeric: tabular-nums; }
select, button { background: #1a1a2e; color: #ddd; border: 1px solid #333;
                 padding: 5px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; width: 100%; }
button:hover { background: #252540; border-color: #4fc3f7; }
button.active { background: #1a2a3e; border-color: #4fc3f7; color: #4fc3f7; }
.btn-row { display: flex; gap: 6px; }
.btn-row button { flex: 1; }
</style>
</head>
<body>
<div id="header">
  <h1>SimAMOC</h1>
  <div id="stats">Connecting...</div>
</div>
<div id="main">
  <div id="canvas-area"><canvas id="c"></canvas></div>
  <div id="sidebar">
    <div class="section">
      <h3>Display</h3>
      <div class="control">
        <select id="field-select">
          <option value="sst" selected>Sea Surface Temperature</option>
          <option value="psi">Streamfunction</option>
          <option value="air">Air Temperature</option>
          <option value="moisture">Moisture</option>
        </select>
      </div>
    </div>

    <div class="section">
      <h3>Speed</h3>
      <div class="control">
        <label>Steps/tick <span class="val" id="speed-val">50</span></label>
        <input type="range" id="speed" min="10" max="500" step="10" value="50">
      </div>
      <div class="btn-row">
        <button id="pause-btn" onclick="togglePause()">Pause</button>
        <button onclick="resetSim()">Reset</button>
      </div>
    </div>

    <div class="section">
      <h3>Radiation</h3>
      <div class="control">
        <label>Solar heating <span class="val" id="ssolar-val">6.20</span></label>
        <input type="range" id="ssolar" min="2" max="12" step="0.1" value="6.2">
      </div>
      <div class="control">
        <label>OLR constant <span class="val" id="aolr-val">1.80</span></label>
        <input type="range" id="aolr" min="0.5" max="4" step="0.05" value="1.8">
      </div>
      <div class="control">
        <label>OLR slope <span class="val" id="bolr-val">0.13</span></label>
        <input type="range" id="bolr" min="0.01" max="0.5" step="0.01" value="0.13">
      </div>
    </div>

    <div class="section">
      <h3>Circulation</h3>
      <div class="control">
        <label>Wind strength <span class="val" id="wind-val">1.00</span></label>
        <input type="range" id="wind" min="0" max="3" step="0.05" value="1.0">
      </div>
      <div class="control">
        <label>Freshwater forcing <span class="val" id="fw-val">0.00</span></label>
        <input type="range" id="fw" min="0" max="2" step="0.05" value="0">
      </div>
    </div>

    <div class="section">
      <h3>Mixing</h3>
      <div class="control">
        <label>Vertical mixing <span class="val" id="gmix-val">0.001</span></label>
        <input type="range" id="gmix" min="0.0001" max="0.01" step="0.0001" value="0.001">
      </div>
      <div class="control">
        <label>Deep water formation <span class="val" id="gdeep-val">0.050</span></label>
        <input type="range" id="gdeep" min="0.001" max="0.2" step="0.001" value="0.05">
      </div>
      <div class="control">
        <label>Thermal diffusion <span class="val" id="kappa-val">2.5e-4</span></label>
        <input type="range" id="kappa" min="0.00001" max="0.001" step="0.00001" value="0.00025">
      </div>
    </div>
  </div>
</div>

<script>
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let nx = 0, ny = 0;
let field = 'sst';
let paused = false;

document.getElementById('field-select').onchange = e => { field = e.target.value; };

// Slider bindings
const sliders = {
  speed:  { el: 'speed',  val: 'speed-val',  fmt: v => Math.round(v) },
  ssolar: { el: 'ssolar', val: 'ssolar-val',  fmt: v => v.toFixed(2) },
  aolr:   { el: 'aolr',   val: 'aolr-val',   fmt: v => v.toFixed(2) },
  bolr:   { el: 'bolr',   val: 'bolr-val',   fmt: v => v.toFixed(2) },
  wind:   { el: 'wind',   val: 'wind-val',   fmt: v => v.toFixed(2) },
  fw:     { el: 'fw',     val: 'fw-val',     fmt: v => v.toFixed(2) },
  gmix:   { el: 'gmix',   val: 'gmix-val',   fmt: v => v.toFixed(4) },
  gdeep:  { el: 'gdeep',  val: 'gdeep-val',  fmt: v => v.toFixed(3) },
  kappa:  { el: 'kappa',  val: 'kappa-val',  fmt: v => v.toExponential(1) },
};

for (const [k, s] of Object.entries(sliders)) {
  const el = document.getElementById(s.el);
  el.oninput = () => {
    document.getElementById(s.val).textContent = s.fmt(parseFloat(el.value));
    sendParams();
  };
}

function togglePause() {
  paused = !paused;
  document.getElementById('pause-btn').textContent = paused ? 'Resume' : 'Pause';
  document.getElementById('pause-btn').classList.toggle('active', paused);
  fetch('/cmd', { method: 'POST', body: JSON.stringify({ action: paused ? 'pause' : 'resume' }) });
}

function resetSim() {
  fetch('/cmd', { method: 'POST', body: JSON.stringify({ action: 'reset' }) });
}

function sendParams() {
  const p = {
    action: 'params',
    steps_per_tick: parseInt(document.getElementById('speed').value),
    S_solar: parseFloat(document.getElementById('ssolar').value),
    A_olr: parseFloat(document.getElementById('aolr').value),
    B_olr: parseFloat(document.getElementById('bolr').value),
    wind_strength: parseFloat(document.getElementById('wind').value),
    freshwater_forcing: parseFloat(document.getElementById('fw').value),
    gamma_mix: parseFloat(document.getElementById('gmix').value),
    gamma_deep_form: parseFloat(document.getElementById('gdeep').value),
    kappa_T: parseFloat(document.getElementById('kappa').value),
  };
  fetch('/cmd', { method: 'POST', body: JSON.stringify(p) });
}

// Color maps
function sstColor(t) {
  let f = (t + 2) / 32;
  f = Math.max(0, Math.min(1, f));
  let r, g, b;
  if (f < 0.25) {
    const s = f / 0.25;
    r = 50 + 30*s | 0; g = 60 + 120*s | 0; b = 180 + 60*s | 0;
  } else if (f < 0.5) {
    const s = (f-0.25) / 0.25;
    r = 80 + 175*s | 0; g = 180 + 60*s | 0; b = 240 - 200*s | 0;
  } else if (f < 0.75) {
    const s = (f-0.5) / 0.25;
    r = 255; g = 240 - 120*s | 0; b = 40 - 30*s | 0;
  } else {
    const s = (f-0.75) / 0.25;
    r = 255 - 60*s | 0; g = 120 - 80*s | 0; b = 10;
  }
  return [r, g, b];
}

function divColor(v, lim) {
  let f = (v / lim + 1) / 2;
  f = Math.max(0, Math.min(1, f));
  if (f < 0.5) {
    const s = f / 0.5;
    return [20+60*s|0, 40+100*s|0, 200+40*s|0];
  } else {
    const s = (f-0.5) / 0.5;
    return [140+115*s|0, 140-100*s|0, 240-220*s|0];
  }
}

function seqColor(v, lo, hi) {
  let f = (v - lo) / (hi - lo || 1);
  f = Math.max(0, Math.min(1, f));
  return [20+60*f|0, 80+140*f|0, 120+120*f|0];
}

function render(data) {
  if (!data.sst) return;
  const w = data.nx, h = data.ny;
  if (w !== nx || h !== ny) {
    nx = w; ny = h;
    const scale = Math.max(2, Math.min(5, Math.floor(1100 / nx)));
    canvas.width = nx * scale; canvas.height = ny * scale;
    ctx.imageSmoothingEnabled = false;
  }

  const sstBuf = new Float32Array(Uint8Array.from(atob(data.sst), c => c.charCodeAt(0)).buffer);
  const psiBuf = new Float32Array(Uint8Array.from(atob(data.psi), c => c.charCodeAt(0)).buffer);
  const airBuf = new Float32Array(Uint8Array.from(atob(data.air), c => c.charCodeAt(0)).buffer);
  const moistBuf = new Float32Array(Uint8Array.from(atob(data.moisture), c => c.charCodeAt(0)).buffer);
  const maskBuf = new Float32Array(Uint8Array.from(atob(data.mask), c => c.charCodeAt(0)).buffer);

  const off = new OffscreenCanvas(nx, ny);
  const offCtx = off.getContext('2d');
  const img = offCtx.createImageData(nx, ny);
  const d = img.data;

  const psiLim = Math.max(Math.abs(data.psi_range[0]), Math.abs(data.psi_range[1]), 0.01);

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const srcIdx = (ny-1-j) * nx + i;
      const dstIdx = (j * nx + i) * 4;
      if (maskBuf[srcIdx] < 0.5) {
        d[dstIdx] = 25; d[dstIdx+1] = 25; d[dstIdx+2] = 35; d[dstIdx+3] = 255;
        continue;
      }
      let rgb;
      switch (field) {
        case 'sst': rgb = sstColor(sstBuf[srcIdx]); break;
        case 'psi': rgb = divColor(psiBuf[srcIdx], psiLim); break;
        case 'air': rgb = sstColor(airBuf[srcIdx]); break;
        case 'moisture': rgb = seqColor(moistBuf[srcIdx], 0, 0.025); break;
        default: rgb = sstColor(sstBuf[srcIdx]);
      }
      d[dstIdx] = rgb[0]; d[dstIdx+1] = rgb[1]; d[dstIdx+2] = rgb[2]; d[dstIdx+3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

  const yr = (data.sim_time / 10).toFixed(2);
  document.getElementById('stats').textContent =
    `Step ${data.step.toLocaleString()} | Year ${yr} | ` +
    `SST ${data.mean_sst}°C | ψ [${data.psi_range[0].toFixed(3)}, ${data.psi_range[1].toFixed(3)}]` +
    (data.paused ? ' | PAUSED' : '');
}

async function poll() {
  while (true) {
    try {
      const resp = await fetch('/state');
      const data = await resp.json();
      render(data);
    } catch(e) {}
    await new Promise(r => setTimeout(r, 80));
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

    def do_POST(self):
        global sim_paused, sim_params, sim_state, sim_step_count, STEPS_PER_TICK
        if self.path == "/cmd":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            action = body.get("action", "")

            if action == "pause":
                sim_paused = True
            elif action == "resume":
                sim_paused = False
            elif action == "reset":
                with sim_lock:
                    sim_state = build_initial_state(DATA_DIR, sim_grid, sim_forcing)
                    sim_step_count = 0
            elif action == "params":
                STEPS_PER_TICK = max(10, min(500, body.get("steps_per_tick", STEPS_PER_TICK)))
                with sim_lock:
                    sim_params = Params(
                        S_solar=body.get("S_solar", sim_params.S_solar),
                        A_olr=body.get("A_olr", sim_params.A_olr),
                        B_olr=body.get("B_olr", sim_params.B_olr),
                        wind_strength=body.get("wind_strength", sim_params.wind_strength),
                        freshwater_forcing=body.get("freshwater_forcing", sim_params.freshwater_forcing),
                        gamma_mix=body.get("gamma_mix", sim_params.gamma_mix),
                        gamma_deep_form=body.get("gamma_deep_form", sim_params.gamma_deep_form),
                        kappa_T=body.get("kappa_T", sim_params.kappa_T),
                    )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


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

    print("JIT compiling...")
    sim_state = run(sim_state, sim_forcing, sim_params, sim_grid, 10)
    jax.block_until_ready(sim_state.T_s)
    print("  Ready")

    sim_thread = threading.Thread(target=sim_loop, daemon=True)
    sim_thread.start()

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n  Open {url}\n")

    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
