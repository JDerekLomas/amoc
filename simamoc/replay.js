// simamoc/replay.js
// Loads a JAX-produced replay directory (manifest.json + *.f32 binary fields)
// and animates frames in the canvas using the same colormaps as the live sim.
//
// URL params:
//   ?run=runs/replay-baseline   (path relative to repo root)
//
// Self-contained: no dependency on model.js / renderer.js, just the colormaps
// inlined below (kept minimal — port more if you want all 14 viz modes).

const $ = (s) => document.querySelector(s);
const cv = $('#cv'), ctx = cv.getContext('2d');
const status_ = $('#status');
const scrubber = $('#scrubber');
const frameLabel = $('#frameLabel');
const playBtn = $('#playBtn');
const legendBar = $('#legendBar'), legendCtx = legendBar.getContext('2d');
const legendLo = $('#legendLo'), legendHi = $('#legendHi'), legendUnit = $('#legendUnit');

const params = new URLSearchParams(location.search);
const runPath = params.get('run') || 'amoc-jax/runs/replay-baseline';

// ───────── colormaps (inlined from renderer.js) ─────────

function tempToRGB(t) {
  // -2 → deep blue, 0 → cyan, 15 → green, 25 → yellow, 30+ → red
  const stops = [
    [-2, [10, 30, 100]], [0, [40, 120, 200]], [10, [80, 200, 200]],
    [18, [200, 220, 100]], [25, [240, 160, 60]], [30, [220, 60, 40]], [40, [120, 0, 30]],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ca] = stops[i], [b, cb] = stops[i + 1];
    if (t <= b) {
      const u = Math.max(0, Math.min(1, (t - a) / (b - a)));
      return [ca[0] + (cb[0]-ca[0])*u, ca[1] + (cb[1]-ca[1])*u, ca[2] + (cb[2]-ca[2])*u];
    }
  }
  return stops[stops.length - 1][1];
}

function salToRGB(s) {
  // Salinity 30–38 psu, low → green, high → magenta
  const u = Math.max(0, Math.min(1, (s - 30) / 8));
  return [40 + 200*u, 200 - 100*u, 80 + 120*u];
}

function psiToRGB(p, absMax) {
  // Diverging blue-white-red around 0
  const u = absMax > 0 ? p / absMax : 0;
  if (u >= 0) return [255 - 100*(1-Math.abs(u)), 255 - 200*Math.abs(u), 255 - 200*Math.abs(u)];
  return [255 - 200*Math.abs(u), 255 - 200*Math.abs(u), 255 - 100*(1-Math.abs(u))];
}

function iceToRGB(f) {
  const u = Math.max(0, Math.min(1, f));
  return [40 + 215*u, 60 + 195*u, 90 + 165*u];
}

const COLORMAPS = {
  temp:     { fn: tempToRGB,    range: [-2, 32],  unit: '°C',         dynamic: false },
  air_temp: { fn: tempToRGB,    range: [-30, 35], unit: '°C (air)',   dynamic: false },
  sal:      { fn: salToRGB,     range: [30, 38],  unit: 'psu',        dynamic: false },
  psi:      { fn: psiToRGB,     range: null,      unit: 'streamfn',   dynamic: true  },
  ice_frac: { fn: iceToRGB,     range: [0, 1],    unit: 'fraction',   dynamic: false },
};

// ───────── state ─────────

let manifest = null;
let arrays = {};       // { fieldName: Float32Array of total_frames * ny * nx }
let currentField = 'temp';
let currentFrame = 0;
let playing = false;

// ───────── load ─────────

async function load() {
  try {
    status_.textContent = `loading manifest from /${runPath}/`;
    const r = await fetch(`/${runPath}/manifest.json`);
    if (!r.ok) throw new Error(`manifest not found at /${runPath}/manifest.json`);
    manifest = await r.json();
    cv.width = manifest.nx;
    cv.height = manifest.ny;
    cv.style.width = '100%';
    cv.style.maxWidth = '1200px';
    scrubber.max = manifest.n_frames - 1;
    status_.textContent = `loading ${manifest.fields.length} fields…`;

    for (const field of manifest.fields) {
      const fr = await fetch(`/${runPath}/${field}.f32`);
      if (!fr.ok) throw new Error(`field ${field} not found`);
      const buf = await fr.arrayBuffer();
      arrays[field] = new Float32Array(buf);
    }
    status_.textContent = `${manifest.nx}×${manifest.ny}, ${manifest.n_frames} frames, ${manifest.steps_per_frame} steps/frame`;
    drawFrame(0);
    drawLegend();
  } catch (e) {
    status_.textContent = `ERROR: ${e.message}`;
    console.error(e);
  }
}

// ───────── render ─────────

function drawFrame(f) {
  if (!manifest) return;
  currentFrame = f;
  const { nx, ny } = manifest;
  const cellsPerFrame = nx * ny;
  const arr = arrays[currentField];
  if (!arr) return;
  const off = f * cellsPerFrame;

  // Compute dynamic range if needed (psi)
  const cmap = COLORMAPS[currentField];
  let absMax = 0;
  if (cmap.dynamic) {
    for (let k = 0; k < cellsPerFrame; k++) {
      const v = arr[off + k]; if (Math.abs(v) > absMax) absMax = Math.abs(v);
    }
    if (absMax === 0) absMax = 1;
  }

  // Render with vertical flip: data j=0 is south pole, screen y=0 is top → invert.
  const img = ctx.createImageData(nx, ny);
  const data = img.data;
  for (let j = 0; j < ny; j++) {
    const dstRow = ny - 1 - j;
    for (let i = 0; i < nx; i++) {
      const v = arr[off + j*nx + i];
      const rgb = cmap.dynamic ? cmap.fn(v, absMax) : cmap.fn(v);
      const di = (dstRow*nx + i) * 4;
      data[di] = rgb[0]|0; data[di+1] = rgb[1]|0; data[di+2] = rgb[2]|0; data[di+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  scrubber.value = f;
  const t = (f + 1) * manifest.sim_time_per_frame;
  frameLabel.textContent = `frame ${f+1} / ${manifest.n_frames}  t=${t.toFixed(4)}`;
}

function drawLegend() {
  const cmap = COLORMAPS[currentField];
  const w = legendBar.width, h = legendBar.height;
  const range = cmap.dynamic ? [-1, 1] : cmap.range;
  const [lo, hi] = range;
  for (let x = 0; x < w; x++) {
    const v = lo + (hi - lo) * (x / (w - 1));
    const rgb = cmap.dynamic ? cmap.fn(v, 1) : cmap.fn(v);
    legendCtx.fillStyle = `rgb(${rgb[0]|0},${rgb[1]|0},${rgb[2]|0})`;
    legendCtx.fillRect(x, 0, 1, h);
  }
  legendLo.textContent = lo.toFixed(1);
  legendHi.textContent = hi.toFixed(1);
  legendUnit.textContent = cmap.unit;
}

// ───────── controls ─────────

document.querySelectorAll('header .field button').forEach((b) => {
  b.addEventListener('click', () => {
    document.querySelectorAll('header .field button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    currentField = b.dataset.field;
    drawFrame(currentFrame);
    drawLegend();
  });
});

scrubber.addEventListener('input', (e) => {
  drawFrame(parseInt(e.target.value, 10));
});

playBtn.addEventListener('click', () => {
  playing = !playing;
  playBtn.textContent = playing ? '⏸ pause' : '▶ play';
  if (playing) loop();
});

function loop() {
  if (!playing || !manifest) return;
  let next = currentFrame + 1;
  if (next >= manifest.n_frames) next = 0;
  drawFrame(next);
  setTimeout(loop, 50);  // 20 fps
}

load();
