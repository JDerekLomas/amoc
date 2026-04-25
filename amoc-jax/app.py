#!/usr/bin/env python3
"""Interactive ocean simulation viewer with observational data maps.

Runs the JAX simulation and displays live fields + all observational data in the browser.
Controls: speed, field view (sim + obs), physics parameters, coastline overlay.

Usage: python app.py [--nx 128] [--ny 64] [--port 8765]
"""
import sys
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
sim_paused = False
STEPS_PER_TICK = 50


def _b64(arr):
    """Base64-encode a float32 array."""
    return base64.b64encode(np.asarray(arr).astype(np.float32).tobytes()).decode()


def state_to_json():
    with sim_lock:
        s = sim_state
        f = sim_forcing
        steps = sim_step_count
        p = sim_params

    mask = np.asarray(f.ocean_mask)
    sst = np.asarray(s.T_s).astype(np.float32)
    ocean_mask = mask > 0.5
    ny, nx = sst.shape
    mean_sst = float(np.mean(sst[ocean_mask])) if np.any(ocean_mask) else 0.0
    psi_s = np.asarray(s.psi_s).astype(np.float32)
    psi_range = [float(np.min(psi_s)), float(np.max(psi_s))]

    return json.dumps({
        "nx": nx, "ny": ny,
        "step": steps,
        "sim_time": float(s.sim_time),
        "mean_sst": round(mean_sst, 2),
        "psi_range": [round(psi_range[0], 4), round(psi_range[1], 4)],
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
        # --- Simulation fields ---
        "mask": _b64(mask),
        "sst": _b64(s.T_s),
        "psi_s": _b64(s.psi_s),
        "psi_d": _b64(s.psi_d),
        "zeta_s": _b64(s.zeta_s),
        "air_temp": _b64(s.air_temp),
        "moisture": _b64(s.moisture),
        "T_d": _b64(s.T_d),
        "S_s": _b64(s.S_s),
        "S_d": _b64(s.S_d),
        # --- Observed fields ---
        "obs_sst": _b64(f.T_target),
        "obs_deep_temp": _b64(f.T_deep_target),
        "obs_salinity": _b64(f.sal_climatology),
        "obs_wind_curl": _b64(f.wind_curl),
        "obs_depth": _b64(f.depth_field),
        "obs_land_temp": _b64(f.land_temp),
        "obs_mld": _b64(f.obs_mld),
        "obs_cloud": _b64(f.obs_cloud),
        "obs_albedo": _b64(f.obs_albedo),
        "obs_sea_ice": _b64(f.obs_sea_ice),
        "obs_precip": _b64(f.obs_precip),
        "obs_evap": _b64(f.obs_evap),
        "obs_water_vapor": _b64(f.obs_water_vapor),
        "obs_u": _b64(f.obs_u),
        "obs_v": _b64(f.obs_v),
        "obs_ndvi": _b64(f.obs_ndvi),
        "obs_snow": _b64(f.obs_snow),
        "obs_chlorophyll": _b64(f.obs_chlorophyll),
        "obs_pressure": _b64(f.obs_pressure),
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
#header { padding: 8px 20px; background: #111; border-bottom: 1px solid #333;
           display: flex; justify-content: space-between; align-items: center; flex-shrink: 0; }
#header h1 { font-size: 16px; color: #4fc3f7; font-weight: 600; }
#stats { font-size: 12px; color: #999; font-variant-numeric: tabular-nums; }
#main { display: flex; flex: 1; overflow: hidden; }
#canvas-wrap { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
#canvas-area { flex: 1; display: flex; justify-content: center; align-items: center; padding: 10px; position: relative; }
canvas { image-rendering: pixelated; border-radius: 4px; max-width: 100%; max-height: 100%; }
#colorbar { height: 24px; margin: 0 10px 6px; display: flex; align-items: center; gap: 8px; }
#colorbar canvas { height: 14px; width: 200px; border-radius: 2px; image-rendering: auto; }
#colorbar .cb-label { font-size: 10px; color: #888; font-variant-numeric: tabular-nums; white-space: nowrap; }
#sidebar { width: 270px; background: #111; border-left: 1px solid #333; padding: 14px;
           overflow-y: auto; flex-shrink: 0; }
.section { margin-bottom: 16px; }
.section h3 { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em;
              color: #555; margin-bottom: 6px; font-weight: 600; }
.control { margin-bottom: 8px; }
.control label { display: block; font-size: 11px; color: #aaa; margin-bottom: 2px; }
.control input[type=range] { width: 100%; accent-color: #4fc3f7; }
.control .val { float: right; font-size: 10px; color: #4fc3f7; font-variant-numeric: tabular-nums; }
select, button { background: #1a1a2e; color: #ddd; border: 1px solid #333;
                 padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 11px; width: 100%; }
button:hover { background: #252540; border-color: #4fc3f7; }
button.active { background: #1a2a3e; border-color: #4fc3f7; color: #4fc3f7; }
.btn-row { display: flex; gap: 6px; }
.btn-row button { flex: 1; }
optgroup { font-weight: 600; color: #4fc3f7; }
option { font-weight: 400; color: #ddd; }
.check-row { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }
.check-row input { accent-color: #4fc3f7; }
.check-row label { font-size: 11px; color: #aaa; }
#field-info { font-size: 10px; color: #666; margin-top: 4px; line-height: 1.4; }
</style>
</head>
<body>
<div id="header">
  <h1>SimAMOC</h1>
  <div id="stats">Connecting...</div>
</div>
<div id="main">
  <div id="canvas-wrap">
    <div id="canvas-area"><canvas id="c"></canvas></div>
    <div id="colorbar">
      <span class="cb-label" id="cb-lo"></span>
      <canvas id="cb" width="200" height="14"></canvas>
      <span class="cb-label" id="cb-hi"></span>
      <span class="cb-label" id="cb-unit"></span>
    </div>
  </div>
  <div id="sidebar">
    <div class="section">
      <h3>Display</h3>
      <div class="control">
        <select id="field-select">
          <optgroup label="Simulation (live)">
            <option value="sst" selected>SST (surface temperature)</option>
            <option value="psi_s">Streamfunction (surface)</option>
            <option value="psi_d">Streamfunction (deep)</option>
            <option value="zeta_s">Vorticity (surface)</option>
            <option value="air_temp">Air Temperature</option>
            <option value="moisture">Moisture (humidity)</option>
            <option value="T_d">Deep Temperature</option>
            <option value="S_s">Surface Salinity</option>
            <option value="S_d">Deep Salinity</option>
          </optgroup>
          <optgroup label="Observations (static)">
            <option value="obs_sst">SST (NOAA OI)</option>
            <option value="obs_deep_temp">Deep Temperature</option>
            <option value="obs_salinity">Salinity (WOA23)</option>
            <option value="obs_wind_curl">Wind Stress Curl (ERA5)</option>
            <option value="obs_depth">Bathymetry (ETOPO1)</option>
            <option value="obs_land_temp">Land Surface Temp</option>
            <option value="obs_mld">Mixed Layer Depth</option>
            <option value="obs_cloud">Cloud Fraction (MODIS)</option>
            <option value="obs_albedo">Surface Albedo</option>
            <option value="obs_sea_ice">Sea Ice Fraction</option>
            <option value="obs_precip">Precipitation (GPM)</option>
            <option value="obs_evap">Evaporation</option>
            <option value="obs_water_vapor">Column Water Vapor</option>
            <option value="obs_u">Ocean Current U (HYCOM)</option>
            <option value="obs_v">Ocean Current V (HYCOM)</option>
            <option value="obs_ndvi">NDVI (vegetation)</option>
            <option value="obs_snow">Snow Cover</option>
            <option value="obs_chlorophyll">Chlorophyll (MODIS)</option>
            <option value="obs_pressure">Surface Pressure</option>
          </optgroup>
        </select>
      </div>
      <div class="check-row">
        <input type="checkbox" id="show-coast" checked>
        <label for="show-coast">Coastline overlay</label>
      </div>
      <div class="check-row">
        <input type="checkbox" id="show-grid">
        <label for="show-grid">Lat/lon grid</label>
      </div>
      <div id="field-info"></div>
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
const cbCanvas = document.getElementById('cb');
const cbCtx = cbCanvas.getContext('2d');
let nx = 0, ny = 0;
let field = 'sst';
let paused = false;
let showCoast = true;
let showGrid = false;
let coastPath = null; // cached coastline path
let maskBuf = null;   // cached mask for coastline extraction

document.getElementById('field-select').onchange = e => { field = e.target.value; updateFieldInfo(); };
document.getElementById('show-coast').onchange = e => { showCoast = e.target.checked; };
document.getElementById('show-grid').onchange = e => { showGrid = e.target.checked; };

// Field metadata for colorbar and info
const FIELDS = {
  // Simulation
  sst:       { cmap: 'thermal', lo: -2, hi: 30, unit: 'C', info: 'Live sea surface temperature' },
  psi_s:     { cmap: 'diverge', lo: 'auto', hi: 'auto', unit: '', info: 'Surface streamfunction (red=anticyclonic)' },
  psi_d:     { cmap: 'diverge', lo: 'auto', hi: 'auto', unit: '', info: 'Deep streamfunction' },
  zeta_s:    { cmap: 'diverge', lo: 'auto', hi: 'auto', unit: '', info: 'Surface vorticity' },
  air_temp:  { cmap: 'thermal', lo: -30, hi: 40, unit: 'C', info: 'Atmospheric temperature (1-layer model)' },
  moisture:  { cmap: 'humid',   lo: 0, hi: 0.025, unit: 'kg/kg', info: 'Specific humidity' },
  T_d:       { cmap: 'thermal', lo: -2, hi: 15, unit: 'C', info: 'Deep ocean temperature' },
  S_s:       { cmap: 'salinity',lo: 32, hi: 37, unit: 'psu', info: 'Surface salinity' },
  S_d:       { cmap: 'salinity',lo: 34, hi: 36, unit: 'psu', info: 'Deep salinity' },
  // Observations
  obs_sst:       { cmap: 'thermal', lo: -2, hi: 30, unit: 'C', info: 'NOAA OI SST v2.1, 2015-2023 annual mean' },
  obs_deep_temp: { cmap: 'thermal', lo: -2, hi: 15, unit: 'C', info: 'Deep ocean temperature (WOA23)' },
  obs_salinity:  { cmap: 'salinity',lo: 32, hi: 37, unit: 'psu', info: 'WOA23 surface salinity' },
  obs_wind_curl: { cmap: 'diverge', lo: 'auto', hi: 'auto', unit: '', info: 'ERA5 wind stress curl (scaled to model units)' },
  obs_depth:     { cmap: 'bathy',   lo: 0, hi: 6000, unit: 'm', info: 'ETOPO1 ocean depth' },
  obs_land_temp: { cmap: 'thermal', lo: -30, hi: 40, unit: 'C', info: 'MODIS land surface temperature' },
  obs_mld:       { cmap: 'depth',   lo: 10, hi: 400, unit: 'm', info: 'Mixed layer depth (observed/estimated)' },
  obs_cloud:     { cmap: 'gray',    lo: 0, hi: 1, unit: '', info: 'MODIS cloud fraction, annual mean' },
  obs_albedo:    { cmap: 'gray',    lo: 0, hi: 0.8, unit: '', info: 'Surface albedo' },
  obs_sea_ice:   { cmap: 'ice',     lo: 0, hi: 1, unit: '', info: 'Sea ice fraction (annual mean)' },
  obs_precip:    { cmap: 'precip',  lo: 0, hi: 3000, unit: 'mm/yr', info: 'NASA GPM IMERG precipitation' },
  obs_evap:      { cmap: 'precip',  lo: 0, hi: 3000, unit: 'mm/yr', info: 'Evaporation' },
  obs_water_vapor:{ cmap: 'humid',  lo: 0, hi: 1, unit: '', info: 'MODIS column water vapor (normalized)' },
  obs_u:         { cmap: 'diverge', lo: 'auto', hi: 'auto', unit: 'm/s', info: 'Observed ocean current (zonal)' },
  obs_v:         { cmap: 'diverge', lo: 'auto', hi: 'auto', unit: 'm/s', info: 'Observed ocean current (meridional)' },
  obs_ndvi:      { cmap: 'veg',     lo: 0, hi: 1, unit: '', info: 'NDVI vegetation index (MODIS)' },
  obs_snow:      { cmap: 'ice',     lo: 0, hi: 1, unit: '', info: 'Snow cover fraction' },
  obs_chlorophyll:{ cmap: 'chlor',  lo: 0, hi: 5, unit: 'mg/m3', info: 'MODIS ocean chlorophyll' },
  obs_pressure:  { cmap: 'pressure',lo: 960, hi: 1040, unit: 'hPa', info: 'Surface pressure' },
};

function updateFieldInfo() {
  const f = FIELDS[field] || {};
  document.getElementById('field-info').textContent = f.info || '';
}
updateFieldInfo();

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

// --- Color maps ---
function lerp3(a, b, t) { return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]; }
function clamp01(x) { return Math.max(0, Math.min(1, x)); }

// Thermal: deep blue -> cyan -> yellow -> red
function thermalColor(f) {
  f = clamp01(f);
  const stops = [[10,20,120],[40,100,220],[80,200,240],[200,240,80],[255,180,30],[200,50,20]];
  const t = f * (stops.length-1);
  const i = Math.min(Math.floor(t), stops.length-2);
  return lerp3(stops[i], stops[i+1], t-i);
}

// Diverging: blue -> white -> red
function divergeColor(f) {
  f = clamp01(f);
  if (f < 0.5) {
    const t = f / 0.5;
    return lerp3([20,40,180], [220,220,230], t);
  } else {
    const t = (f-0.5) / 0.5;
    return lerp3([220,220,230], [200,30,30], t);
  }
}

// Humidity: white -> teal -> dark blue
function humidColor(f) {
  f = clamp01(f);
  const stops = [[240,245,250],[150,220,200],[40,160,160],[20,80,140],[10,30,80]];
  const t = f * (stops.length-1);
  const i = Math.min(Math.floor(t), stops.length-2);
  return lerp3(stops[i], stops[i+1], t-i);
}

// Salinity: yellow-green -> blue
function salinityColor(f) {
  f = clamp01(f);
  const stops = [[255,255,200],[140,220,100],[40,160,80],[30,100,180],[20,40,140]];
  const t = f * (stops.length-1);
  const i = Math.min(Math.floor(t), stops.length-2);
  return lerp3(stops[i], stops[i+1], t-i);
}

// Bathymetry: light blue -> dark navy
function bathyColor(f) {
  f = clamp01(f);
  return lerp3([180,220,255], [10,20,60], f);
}

// Depth (MLD): yellow -> red -> dark
function depthColor(f) {
  f = clamp01(f);
  const stops = [[255,255,180],[255,180,50],[200,60,30],[80,20,60]];
  const t = f * (stops.length-1);
  const i = Math.min(Math.floor(t), stops.length-2);
  return lerp3(stops[i], stops[i+1], t-i);
}

// Gray
function grayColor(f) {
  f = clamp01(f);
  const v = 30 + 200*f;
  return [v, v, v];
}

// Ice: dark -> white-blue
function iceColor(f) {
  f = clamp01(f);
  return lerp3([20,30,50], [200,230,255], f);
}

// Precipitation: tan -> green -> blue -> purple
function precipColor(f) {
  f = clamp01(f);
  const stops = [[240,230,200],[100,200,80],[30,120,200],[60,20,160]];
  const t = f * (stops.length-1);
  const i = Math.min(Math.floor(t), stops.length-2);
  return lerp3(stops[i], stops[i+1], t-i);
}

// Vegetation (NDVI): brown -> green
function vegColor(f) {
  f = clamp01(f);
  const stops = [[180,160,120],[200,200,100],[80,180,40],[20,100,20]];
  const t = f * (stops.length-1);
  const i = Math.min(Math.floor(t), stops.length-2);
  return lerp3(stops[i], stops[i+1], t-i);
}

// Chlorophyll: dark blue -> green -> yellow
function chlorColor(f) {
  f = clamp01(f);
  const stops = [[10,20,80],[20,80,120],[40,160,80],[180,220,40]];
  const t = f * (stops.length-1);
  const i = Math.min(Math.floor(t), stops.length-2);
  return lerp3(stops[i], stops[i+1], t-i);
}

// Pressure: blue -> white -> red
function pressureColor(f) { return divergeColor(f); }

const CMAPS = {
  thermal: thermalColor, diverge: divergeColor, humid: humidColor,
  salinity: salinityColor, bathy: bathyColor, depth: depthColor,
  gray: grayColor, ice: iceColor, precip: precipColor,
  veg: vegColor, chlor: chlorColor, pressure: pressureColor,
};

function decodeF32(b64) {
  return new Float32Array(Uint8Array.from(atob(b64), c => c.charCodeAt(0)).buffer);
}

// Build coastline path from mask (run once when mask arrives)
function buildCoastPath(mask, w, h) {
  // Find cells where mask transitions from ocean to land
  const edges = [];
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const idx = j * w + i;
      if (mask[idx] < 0.5) continue;
      // Check 4 neighbors - if any is land, this is a coast cell
      const left = i > 0 ? mask[idx-1] : 0;
      const right = i < w-1 ? mask[idx+1] : 0;
      const up = j > 0 ? mask[idx-w] : 0;
      const down = j < h-1 ? mask[idx+w] : 0;
      if (left < 0.5 || right < 0.5 || up < 0.5 || down < 0.5) {
        edges.push([i, j]);
      }
    }
  }
  return edges;
}

function drawCoastline(ctx, edges, scaleX, scaleY) {
  if (!edges || edges.length === 0) return;
  ctx.fillStyle = 'rgba(180, 180, 180, 0.5)';
  for (const [i, j] of edges) {
    ctx.fillRect(i * scaleX, j * scaleY, Math.max(1, scaleX), Math.max(1, scaleY));
  }
}

function drawGridLines(ctx, w, h, cw, ch) {
  ctx.strokeStyle = 'rgba(100, 100, 100, 0.3)';
  ctx.lineWidth = 0.5;
  // Latitude lines every 30 deg: lat range -79.5 to 79.5, so y = (lat+79.5)/159 * h
  for (let lat = -60; lat <= 60; lat += 30) {
    const y = (1 - (lat + 79.5) / 159) * ch;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
  }
  // Longitude lines every 60 deg: lon range -180 to 180
  for (let lon = -120; lon <= 120; lon += 60) {
    const x = (lon + 180) / 360 * cw;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
  }
  // Equator
  ctx.strokeStyle = 'rgba(150, 150, 150, 0.4)';
  const eqY = (1 - (0 + 79.5) / 159) * ch;
  ctx.beginPath(); ctx.moveTo(0, eqY); ctx.lineTo(cw, eqY); ctx.stroke();
}

function drawColorbar(cmapName, lo, hi, unit) {
  const cmap = CMAPS[cmapName] || thermalColor;
  const img = cbCtx.createImageData(200, 14);
  for (let x = 0; x < 200; x++) {
    const f = x / 199;
    const [r, g, b] = cmap(f);
    for (let y = 0; y < 14; y++) {
      const i = (y * 200 + x) * 4;
      img.data[i] = r|0; img.data[i+1] = g|0; img.data[i+2] = b|0; img.data[i+3] = 255;
    }
  }
  cbCtx.putImageData(img, 0, 0);
  const fmt = v => Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toPrecision(2);
  document.getElementById('cb-lo').textContent = fmt(lo);
  document.getElementById('cb-hi').textContent = fmt(hi);
  document.getElementById('cb-unit').textContent = unit;
}

function render(data) {
  if (!data.mask) return;
  const w = data.nx, h = data.ny;
  if (w !== nx || h !== ny) {
    nx = w; ny = h;
    const scale = Math.max(2, Math.min(6, Math.floor(1200 / nx)));
    canvas.width = nx * scale; canvas.height = ny * scale;
    ctx.imageSmoothingEnabled = false;
    coastPath = null; // rebuild
  }

  const mask = decodeF32(data.mask);
  if (!maskBuf || maskBuf.length !== mask.length) {
    maskBuf = mask;
    // Build coastline from south-first data (flip for display)
    const flipped = new Float32Array(w * h);
    for (let j = 0; j < h; j++)
      for (let i = 0; i < w; i++)
        flipped[j * w + i] = mask[(h-1-j) * w + i];
    coastPath = buildCoastPath(flipped, w, h);
  }

  // Get the field data
  const fieldKey = field;
  if (!data[fieldKey]) return;
  const buf = decodeF32(data[fieldKey]);
  const meta = FIELDS[fieldKey] || { cmap: 'thermal', lo: 0, hi: 1, unit: '' };
  const cmapFn = CMAPS[meta.cmap] || thermalColor;

  // Compute auto range for diverging fields
  let lo = meta.lo, hi = meta.hi;
  if (lo === 'auto' || hi === 'auto') {
    let maxAbs = 0.01;
    for (let k = 0; k < buf.length; k++) {
      if (mask[k] > 0.5) maxAbs = Math.max(maxAbs, Math.abs(buf[k]));
    }
    // Use 99th percentile to avoid outliers
    const vals = [];
    for (let k = 0; k < buf.length; k++) {
      if (mask[k] > 0.5) vals.push(Math.abs(buf[k]));
    }
    vals.sort((a,b) => a-b);
    const p99 = vals[Math.floor(vals.length * 0.99)] || maxAbs;
    lo = -p99; hi = p99;
  }

  // Determine if this field uses ocean mask for land rendering
  const isOceanField = !fieldKey.startsWith('obs_land') && !fieldKey.startsWith('obs_ndvi') &&
                        !fieldKey.startsWith('obs_snow') && !fieldKey.startsWith('obs_pressure') &&
                        fieldKey !== 'air_temp' && fieldKey !== 'obs_albedo';

  // Render to offscreen canvas
  const off = new OffscreenCanvas(nx, ny);
  const offCtx = off.getContext('2d');
  const img = offCtx.createImageData(nx, ny);
  const d = img.data;

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const srcIdx = (ny-1-j) * nx + i;
      const dstIdx = (j * nx + i) * 4;
      const isLand = mask[srcIdx] < 0.5;

      if (isOceanField && isLand) {
        // Dark land
        d[dstIdx] = 30; d[dstIdx+1] = 32; d[dstIdx+2] = 28; d[dstIdx+3] = 255;
        continue;
      }

      const v = buf[srcIdx];
      const f = clamp01((v - lo) / (hi - lo || 1));
      const [r, g, b] = cmapFn(f);
      d[dstIdx] = r|0; d[dstIdx+1] = g|0; d[dstIdx+2] = b|0; d[dstIdx+3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);

  // Overlays
  const sx = canvas.width / nx;
  const sy = canvas.height / ny;
  if (showCoast && coastPath) drawCoastline(ctx, coastPath, sx, sy);
  if (showGrid) drawGridLines(ctx, nx, ny, canvas.width, canvas.height);

  // Colorbar
  drawColorbar(meta.cmap, lo, hi, meta.unit);

  // Stats
  const yr = (data.sim_time / 10).toFixed(2);
  document.getElementById('stats').textContent =
    `Step ${data.step.toLocaleString()} | Year ${yr} | ` +
    `SST ${data.mean_sst} C | psi [${data.psi_range[0].toFixed(3)}, ${data.psi_range[1].toFixed(3)}]` +
    (data.paused ? ' | PAUSED' : '');
}

async function poll() {
  while (true) {
    try {
      const resp = await fetch('/state');
      const data = await resp.json();
      render(data);
    } catch(e) {}
    await new Promise(r => setTimeout(r, 100));
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

    # Report loaded obs fields
    obs_fields = ['obs_mld', 'obs_cloud', 'obs_albedo', 'obs_sea_ice', 'obs_precip',
                  'obs_evap', 'obs_water_vapor', 'obs_u', 'obs_v', 'obs_ndvi',
                  'obs_snow', 'obs_chlorophyll', 'obs_pressure']
    for name in obs_fields:
        arr = getattr(sim_forcing, name)
        nz = float(jnp.sum(jnp.abs(arr) > 1e-10))
        total = arr.size
        pct = nz / total * 100
        print(f"  {name:20s}: {pct:5.1f}% non-zero, range [{float(jnp.min(arr)):.3g}, {float(jnp.max(arr)):.3g}]")

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
