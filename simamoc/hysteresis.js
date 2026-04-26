// simamoc/hysteresis.js  (v2 — Windy-style UI matching simamoc/index.html)
//
// Loads a hysteresis-replay run produced by amoc-jax/scripts/hysteresis-replay.py
// and animates it inside the same look as the live SimAMOC: full-bleed map,
// floating layer pills, HUD with friendly labels, slide-up drawer for the curve.
//
// URL params:
//   ?run=amoc-jax/runs/hysteresis-default

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ── DOM refs ──
const cv = $('#sim'), ctx = cv.getContext('2d');
const hCv = $('#hyst-canvas'), hCtx = hCv.getContext('2d');
const cap = $('#cap-text');
const mhF = $('#mh-F'), mhA = $('#mh-amoc'), mhN = $('#mh-natl');
const hudA = $('#hud-amoc');
const dsF = $('#ds-F'), dsA = $('#ds-A'), dsS = $('#ds-S');
const scrubber = $('#scrubber'), scrubLabel = $('#scrub-label');
const fab = $('#m-fab-pause');

const params = new URLSearchParams(location.search);
const runPath = params.get('run') || 'amoc-jax/runs/hysteresis-default';

// ── colormaps (match simamoc/renderer.js conventions) ──
function tempToRGB(t) {
  const s = [
    [-2,[10,30,100]],[0,[40,120,200]],[10,[80,200,200]],
    [18,[200,220,100]],[25,[240,160,60]],[30,[220,60,40]],[40,[120,0,30]],
  ];
  for (let i = 0; i < s.length-1; i++) {
    const [a,ca]=s[i],[b,cb]=s[i+1];
    if (t <= b) { const u=Math.max(0,Math.min(1,(t-a)/(b-a)));
      return [ca[0]+(cb[0]-ca[0])*u, ca[1]+(cb[1]-ca[1])*u, ca[2]+(cb[2]-ca[2])*u]; }
  }
  return s[s.length-1][1];
}
function salToRGB(v) { const u=Math.max(0,Math.min(1,(v-30)/8)); return [40+200*u,200-100*u,80+120*u]; }
function psiToRGB(p,M){ const u=M>0?p/M:0;
  if (u>=0) return [255-100*(1-Math.abs(u)),255-200*Math.abs(u),255-200*Math.abs(u)];
  return [255-200*Math.abs(u),255-200*Math.abs(u),255-100*(1-Math.abs(u))]; }
function iceToRGB(f){ const u=Math.max(0,Math.min(1,f)); return [40+215*u,60+195*u,90+165*u]; }
const CMAPS = {
  temp:    {fn:tempToRGB,range:[-2,32],unit:'°C',dynamic:false,nice:'sea-surface temperature'},
  air_temp:{fn:tempToRGB,range:[-30,35],unit:'°C',dynamic:false,nice:'air temperature'},
  sal:     {fn:salToRGB,range:[30,38],unit:'psu',dynamic:false,nice:'salinity'},
  psi:     {fn:psiToRGB,range:null,unit:'',dynamic:true,nice:'streamfunction'},
  ice_frac:{fn:iceToRGB,range:[0,1],unit:'',dynamic:false,nice:'sea ice fraction'},
};

// ── state ──
let manifest=null, arrays={}, traj=[];
let field='temp', playing=false;
// Continuous time index (float). Integer part = segment, fractional = interp.
let curT = 0;
// Speed: how many segments per second of real time. 0.4 → 21 segments in ~52s.
let segmentsPerSec = 0.4;
let lastTick = 0, rafId = 0;

// Particles drift on the current frame's streamfunction for life.
const NP = 1500, MAX_AGE = 240;
const px = new Float32Array(NP), py = new Float32Array(NP), page = new Float32Array(NP);
let particlesInit = false;

// Land mask derived from the loaded SST: cells with T near 0 (the JAX
// ice-mask placeholder) are treated as land for particle reset purposes.
let oceanMask = null;

// ── load ──
async function load() {
  try {
    cap.textContent = 'Loading replay…';
    const r = await fetch(`/${runPath}/manifest.json`);
    if (!r.ok) throw new Error(`No replay at /${runPath}/`);
    manifest = await r.json();
    if (!manifest.trajectory) throw new Error('Manifest missing trajectory');
    traj = manifest.trajectory;
    cv.width = manifest.nx; cv.height = manifest.ny;
    scrubber.max = traj.length - 1;
    for (const f of manifest.fields) {
      const fr = await fetch(`/${runPath}/${f}.f32`);
      if (!fr.ok) throw new Error(`Missing field: ${f}`);
      arrays[f] = new Float32Array(await fr.arrayBuffer());
    }
    buildOceanMask();
    initParticles();
    curT = 0;
    drawAll();
    // Start the rAF loop for particle motion + time advancement.
    requestAnimationFrame(tick);
  } catch (e) {
    cap.innerHTML = `<strong>Couldn't load replay.</strong> ${e.message}`;
    console.error(e);
  }
}

// ── world map render with inter-frame interpolation ──
// f can be a float; we blend frame[floor(f)] and frame[ceil(f)] linearly.
function drawWorld(f) {
  const {nx, ny} = manifest;
  const cells = nx * ny;
  const arr = arrays[field];
  if (!arr) return;
  const f0 = Math.floor(f), f1 = Math.min(traj.length - 1, f0 + 1);
  const a  = f - f0;  // interpolation weight 0..1
  const off0 = f0 * cells, off1 = f1 * cells;
  const cmap = CMAPS[field];

  let absMax = 0;
  if (cmap.dynamic) {
    for (let k = 0; k < cells; k++) {
      const v = arr[off0+k]*(1-a) + arr[off1+k]*a;
      if (Math.abs(v) > absMax) absMax = Math.abs(v);
    }
    if (absMax === 0) absMax = 1;
  }
  const img = ctx.createImageData(nx, ny);
  const d = img.data;
  for (let j = 0; j < ny; j++) {
    const dstRow = ny - 1 - j;  // south-up data → north-up screen
    for (let i = 0; i < nx; i++) {
      const k = j*nx + i;
      const v = arr[off0+k]*(1-a) + arr[off1+k]*a;
      const rgb = cmap.dynamic ? cmap.fn(v, absMax) : cmap.fn(v);
      const di = (dstRow*nx + i) * 4;
      d[di]=rgb[0]|0; d[di+1]=rgb[1]|0; d[di+2]=rgb[2]|0; d[di+3]=255;
    }
  }
  ctx.putImageData(img, 0, 0);

  // Particles overlay
  drawParticles(f);
}

// ── ocean mask derived from temp data (T==0 → likely land in JAX output) ──
function buildOceanMask() {
  const {nx, ny} = manifest;
  oceanMask = new Uint8Array(nx * ny);
  const T = arrays.temp;
  if (!T) return;
  // Sample across all frames: a cell is "ocean" if temp ever exceeds a tiny
  // threshold (land cells stay at 0).
  for (let f = 0; f < traj.length; f++) {
    const off = f * nx * ny;
    for (let k = 0; k < nx * ny; k++) {
      if (Math.abs(T[off + k]) > 0.01) oceanMask[k] = 1;
    }
  }
}

function spawnInOcean() {
  const {nx, ny} = manifest;
  for (let t = 0; t < 50; t++) {
    const x = Math.random() * nx;
    const y = 2 + Math.random() * (ny - 4);
    const k = (Math.floor(y) * nx) + Math.floor(x);
    if (oceanMask && oceanMask[k]) return [x, y];
  }
  return [Math.random() * nx, ny / 2];
}

function initParticles() {
  for (let p = 0; p < NP; p++) {
    const [x, y] = spawnInOcean();
    px[p] = x; py[p] = y;
    page[p] = Math.random() * MAX_AGE;
  }
  particlesInit = true;
}

function getVelInterp(psiArr, off, fi, fj, nx, ny) {
  // Centered differences on psi at integer cell of (fi, fj). Returns [u, v]
  // in cells/unit_time. Periodic in i, clamped in j.
  let i = Math.floor(fi);
  let j = Math.min(Math.max(Math.floor(fj), 1), ny - 2);
  i = ((i % nx) + nx) % nx;
  const ip = (i + 1) % nx, im = (i - 1 + nx) % nx;
  // u = -d psi/dy, v = +d psi/dx (no cos(lat) here — we're working in grid units,
  // and the particle motion just needs to look like advection on the streamfn).
  const u = -(psiArr[off + (j+1)*nx + i] - psiArr[off + (j-1)*nx + i]) * 0.5;
  const v =  (psiArr[off + j*nx + ip]    - psiArr[off + j*nx + im])    * 0.5;
  return [u, v];
}

function advectParticles(f, dtSec) {
  if (!particlesInit || !arrays.psi) return;
  const {nx, ny} = manifest;
  const psiArr = arrays.psi;
  const f0 = Math.floor(f), f1 = Math.min(traj.length - 1, f0 + 1);
  const a = f - f0;
  const off0 = f0 * nx * ny, off1 = f1 * nx * ny;
  // Particle velocity scale: tune so motion is visible but not chaotic.
  // psi values in JAX output are O(1), so vel ~1 cells/unit_time. We multiply
  // by an empirical advection rate to get cells per second of real time.
  const ADVEC = 8.0;
  const cellsX = ADVEC * dtSec;
  for (let p = 0; p < NP; p++) {
    const [u0, v0] = getVelInterp(psiArr, off0, px[p], py[p], nx, ny);
    const [u1, v1] = getVelInterp(psiArr, off1, px[p], py[p], nx, ny);
    const u = u0*(1-a) + u1*a, v = v0*(1-a) + v1*a;
    px[p] += u * cellsX;
    py[p] += v * cellsX;
    if (px[p] >= nx) px[p] -= nx;
    if (px[p] < 0) px[p] += nx;
    page[p] += dtSec * 60;  // ~60 ticks per second of real time
    const gi = ((Math.floor(px[p]) % nx) + nx) % nx;
    const gj = Math.floor(py[p]);
    if (gj < 1 || gj >= ny - 1 || (oceanMask && !oceanMask[gj * nx + gi]) || page[p] > MAX_AGE) {
      const [x, y] = spawnInOcean();
      px[p] = x; py[p] = y; page[p] = 0;
    }
  }
}

function drawParticles(f) {
  if (!particlesInit) return;
  const {nx, ny} = manifest;
  // Find max speed for brightness scaling
  const psiArr = arrays.psi;
  const f0 = Math.floor(f), f1 = Math.min(traj.length - 1, f0 + 1);
  const a = f - f0;
  const off0 = f0 * nx * ny, off1 = f1 * nx * ny;
  // Sample max speed at a subset of cells to set brightness scale
  let maxSpd = 0;
  for (let s = 0; s < 200; s++) {
    const k = (Math.random() * nx * ny) | 0;
    const j = (k / nx) | 0, i = k - j*nx;
    if (j < 1 || j >= ny-1) continue;
    const [u0, v0] = getVelInterp(psiArr, off0, i, j, nx, ny);
    const [u1, v1] = getVelInterp(psiArr, off1, i, j, nx, ny);
    const u = u0*(1-a) + u1*a, v = v0*(1-a) + v1*a;
    const spd = Math.sqrt(u*u + v*v);
    if (spd > maxSpd) maxSpd = spd;
  }
  if (maxSpd < 1e-6) maxSpd = 1;

  ctx.save();
  for (let p = 0; p < NP; p++) {
    const dstRow = ny - 1 - py[p];   // south-up → north-up flip
    const x = px[p], y = dstRow;
    const [u0, v0] = getVelInterp(psiArr, off0, px[p], py[p], nx, ny);
    const [u1, v1] = getVelInterp(psiArr, off1, px[p], py[p], nx, ny);
    const u = u0*(1-a) + u1*a, v = v0*(1-a) + v1*a;
    const spd = Math.sqrt(u*u + v*v);
    const alpha = Math.min(1, page[p]/20) * Math.min(1, (MAX_AGE - page[p])/20);
    const bright = Math.min(1, spd/maxSpd * 2.5);
    ctx.fillStyle = `rgba(${200+55*bright|0},${220+35*bright|0},${240+15*bright|0},${(alpha*0.65).toFixed(2)})`;
    ctx.fillRect(x - 0.4, y - 0.4, 1.0, 1.0);
  }
  ctx.restore();
}

// ── hysteresis curve ──
function drawCurve(f) {
  const W = hCv.width, H = hCv.height;
  hCtx.fillStyle = '#08111c'; hCtx.fillRect(0, 0, W, H);

  const Fs = traj.map(p => p.F), As = traj.map(p => p.amoc);
  const Fmin = Math.min(...Fs), Fmax = Math.max(...Fs);
  const Amin = Math.min(...As, 0), Amax = Math.max(...As, 0);
  const padL = 56, padR = 16, padT = 18, padB = 32;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const xOf = (F) => padL + plotW * (F - Fmin) / Math.max(1e-9, Fmax - Fmin);
  const yOf = (A) => padT + plotH * (1 - (A - Amin) / Math.max(1e-9, Amax - Amin));

  // Background grid
  hCtx.strokeStyle = '#1a2838'; hCtx.lineWidth = 1;
  hCtx.strokeRect(padL, padT, plotW, plotH);
  if (Amin < 0 && Amax > 0) {
    hCtx.strokeStyle = '#3a5468';
    hCtx.beginPath(); hCtx.moveTo(padL, yOf(0)); hCtx.lineTo(W-padR, yOf(0)); hCtx.stroke();
  }

  // Branches
  const branch = (color, indices) => {
    if (indices.length < 2) return;
    hCtx.strokeStyle = color; hCtx.lineWidth = 2;
    hCtx.beginPath();
    indices.forEach((i, k) => {
      const x = xOf(traj[i].F), y = yOf(traj[i].amoc);
      if (k === 0) hCtx.moveTo(x, y); else hCtx.lineTo(x, y);
    });
    hCtx.stroke();
    hCtx.fillStyle = color;
    indices.forEach(i => {
      hCtx.beginPath(); hCtx.arc(xOf(traj[i].F), yOf(traj[i].amoc), 2.8, 0, Math.PI*2); hCtx.fill();
    });
  };
  const ups   = traj.map((p,i) => ({i,p})).filter(x => x.p.phase==='init' || x.p.phase==='up').map(x => x.i);
  const downs = traj.map((p,i) => ({i,p})).filter(x => x.p.phase==='down').map(x => x.i);
  branch('#de9c7a', ups);
  if (downs.length) branch('#5a9ec8', [ups[ups.length-1], ...downs]);

  // Current point (white halo)
  const cur = traj[f];
  const cx = xOf(cur.F), cy = yOf(cur.amoc);
  hCtx.strokeStyle = 'rgba(255,255,255,0.95)'; hCtx.lineWidth = 2;
  hCtx.beginPath(); hCtx.arc(cx, cy, 6, 0, Math.PI*2); hCtx.stroke();
  hCtx.fillStyle = cur.phase === 'down' ? '#5a9ec8' : '#de9c7a';
  hCtx.beginPath(); hCtx.arc(cx, cy, 4, 0, Math.PI*2); hCtx.fill();

  // Axis labels
  hCtx.font = '10px ui-monospace, monospace';
  hCtx.fillStyle = '#7ab0d0';
  hCtx.textAlign = 'left'; hCtx.fillText('AMOC strength', padL, padT - 5);
  hCtx.textAlign = 'right'; hCtx.fillText('Meltwater forcing', W - padR, H - 6);
  hCtx.textAlign = 'center';
  for (let k = 0; k <= 4; k++) {
    const F = Fmin + (Fmax-Fmin)*k/4;
    hCtx.fillText(F.toFixed(1), xOf(F), H - padB + 14);
  }
  hCtx.textAlign = 'right';
  for (let k = 0; k <= 4; k++) {
    const A = Amin + (Amax-Amin)*k/4;
    hCtx.fillText(A.toExponential(1), padL - 4, yOf(A) + 3);
  }
  // Mini-legend
  hCtx.font = '10px system-ui, sans-serif'; hCtx.textAlign = 'left';
  hCtx.fillStyle = '#de9c7a'; hCtx.fillRect(padL+8, padT+4, 12, 2);
  hCtx.fillStyle = '#a0b8c8'; hCtx.fillText('ramp up', padL+24, padT+9);
  hCtx.fillStyle = '#5a9ec8'; hCtx.fillRect(padL+88, padT+4, 12, 2);
  hCtx.fillStyle = '#a0b8c8'; hCtx.fillText('ramp down', padL+104, padT+9);
}

// ── caption (friendly story below the map) ──
function captionFor(cur) {
  const Fp = cur.F.toFixed(1);
  const Ap = cur.amoc.toExponential(2);
  if (cur.phase === 'init') {
    return `<strong>Spinup.</strong> No extra meltwater. AMOC settling from observed ocean state.`;
  }
  if (cur.phase === 'up') {
    return `<strong>Ramping meltwater up.</strong> F = ${Fp}. Cooler, fresher North Atlantic; AMOC weakening (${Ap}).`;
  }
  return `<strong>Ramping meltwater back down.</strong> F = ${Fp}. AMOC stays in the weak branch (${Ap}) — this is the bistability.`;
}

// Linear interp of a numeric trajectory field between two segments.
function interpField(f, key) {
  const f0 = Math.floor(f), f1 = Math.min(traj.length - 1, f0 + 1);
  const a = f - f0;
  return traj[f0][key] * (1 - a) + traj[f1][key] * a;
}

// ── master draw (called both from rAF tick and from scrubber input) ──
function drawAll() {
  if (!manifest) return;
  drawWorld(curT);
  drawCurve(curT);
  scrubber.value = Math.round(curT);
  const f0 = Math.floor(curT);
  const cur = traj[f0];
  if (cur) {
    const F = interpField(curT, 'F');
    const A = interpField(curT, 'amoc');
    const S = interpField(curT, 'sst_natl');
    mhF.textContent = F.toFixed(2);
    mhA.textContent = A.toExponential(2);
    mhN.textContent = S.toFixed(1) + '°';
    dsF.textContent = F.toFixed(2);
    dsA.textContent = A.toExponential(2);
    dsS.textContent = S.toFixed(1) + '°C';
    hudA.classList.remove('warn', 'crit');
    if (A < -0.05) hudA.classList.add('crit');
    else if (A < 0) hudA.classList.add('warn');
    scrubLabel.textContent = `${f0+1} / ${traj.length}`;
    cap.innerHTML = captionFor(cur);
  }
}

// ── rAF loop: advance curT (when playing) and tick particles every frame ──
function tick(now) {
  if (!lastTick) lastTick = now;
  const dtSec = Math.min(0.1, (now - lastTick) / 1000);  // cap big dts
  lastTick = now;

  if (playing) {
    curT += segmentsPerSec * dtSec;
    if (curT >= traj.length - 1) curT = 0;  // loop back to start
  }
  // Particles always animate, even when paused — feels alive.
  advectParticles(curT, dtSec);
  drawAll();
  rafId = requestAnimationFrame(tick);
}

// ── controls ──
$$('.m-viewbar #grp-views button').forEach((b) => {
  b.addEventListener('click', () => {
    $$('.m-viewbar #grp-views button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    field = b.dataset.field;
  });
});

scrubber.addEventListener('input', e => { curT = parseInt(e.target.value, 10); });

function setPlaying(v) {
  playing = v;
  fab.classList.toggle('playing', v);
  fab.classList.toggle('paused', !v);
}
fab.addEventListener('click', () => setPlaying(!playing));

// Speed buttons re-map their fps attr to segments-per-second of the loop.
// 1x = walk the 21-segment loop in ~52 sec (so each segment "lives" ~2.5s)
$$('.m-speed button').forEach(b => {
  b.addEventListener('click', () => {
    $$('.m-speed button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    const f = parseInt(b.dataset.fps, 10);
    // Map fps attr {2,4,8,16} → {0.2, 0.4, 0.8, 1.6} segments/sec
    segmentsPerSec = f * 0.1;
  });
});

// Bottom toolbar drawers (mutually exclusive)
const scrim = $('.m-scrim');
function closeDrawers() {
  $$('.m-drawer').forEach(d => d.classList.remove('open'));
  $$('.m-tb-btn').forEach(b => b.classList.remove('active'));
  scrim.classList.remove('open');
}
$$('.m-tb-btn').forEach(b => {
  b.addEventListener('click', () => {
    const id = b.dataset.drawer;
    const d = document.getElementById(id);
    const wasOpen = d.classList.contains('open');
    closeDrawers();
    if (!wasOpen) {
      d.classList.add('open');
      b.classList.add('active');
      scrim.classList.add('open');
    }
  });
});
scrim.addEventListener('click', closeDrawers);

// Onboarding
$('#onb-go').addEventListener('click', () => {
  $('#onb').classList.add('hidden');
  setPlaying(true);
});

// Auto-skip onboarding if loaded via reload
if (sessionStorage.getItem('amoc-onb-seen')) {
  $('#onb').classList.add('hidden');
} else {
  sessionStorage.setItem('amoc-onb-seen', '1');
}

load();
