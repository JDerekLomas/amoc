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
let field='temp', frame=0, playing=false, fps=4;
let playTimer=null;

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
    drawAll(0);
  } catch (e) {
    cap.innerHTML = `<strong>Couldn't load replay.</strong> ${e.message}`;
    console.error(e);
  }
}

// ── world map render (south-up data → north-up screen) ──
function drawWorld(f) {
  const {nx, ny} = manifest;
  const cells = nx * ny;
  const arr = arrays[field];
  if (!arr) return;
  const off = f * cells;
  const cmap = CMAPS[field];
  let absMax = 0;
  if (cmap.dynamic) {
    for (let k = 0; k < cells; k++) { const v=arr[off+k]; if(Math.abs(v)>absMax)absMax=Math.abs(v); }
    if (absMax === 0) absMax = 1;
  }
  const img = ctx.createImageData(nx, ny);
  const d = img.data;
  for (let j = 0; j < ny; j++) {
    const dstRow = ny - 1 - j;
    for (let i = 0; i < nx; i++) {
      const v = arr[off + j*nx + i];
      const rgb = cmap.dynamic ? cmap.fn(v, absMax) : cmap.fn(v);
      const di = (dstRow*nx + i) * 4;
      d[di]=rgb[0]|0; d[di+1]=rgb[1]|0; d[di+2]=rgb[2]|0; d[di+3]=255;
    }
  }
  ctx.putImageData(img, 0, 0);
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

// ── master draw ──
function drawAll(f) {
  if (!manifest) return;
  frame = f;
  drawWorld(f);
  drawCurve(f);
  scrubber.value = f;
  const cur = traj[f];
  if (cur) {
    mhF.textContent = cur.F.toFixed(2);
    mhA.textContent = cur.amoc.toExponential(2);
    mhN.textContent = cur.sst_natl.toFixed(1) + '°';
    dsF.textContent = cur.F.toFixed(2);
    dsA.textContent = cur.amoc.toExponential(2);
    dsS.textContent = cur.sst_natl.toFixed(1) + '°C';
    // Color the AMOC HUD by sign/strength
    hudA.classList.remove('warn', 'crit');
    if (cur.amoc < -0.05) hudA.classList.add('crit');
    else if (cur.amoc < 0) hudA.classList.add('warn');
    scrubLabel.textContent = `${f+1} / ${traj.length}`;
    cap.innerHTML = captionFor(cur);
  }
}

// ── controls ──
$$('.m-viewbar #grp-views button').forEach((b) => {
  b.addEventListener('click', () => {
    $$('.m-viewbar #grp-views button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    field = b.dataset.field;
    drawAll(frame);
  });
});

scrubber.addEventListener('input', e => drawAll(parseInt(e.target.value, 10)));

function setPlaying(v) {
  playing = v;
  fab.classList.toggle('playing', v);
  fab.classList.toggle('paused', !v);
  if (playTimer) { clearTimeout(playTimer); playTimer = null; }
  if (playing) loop();
}
fab.addEventListener('click', () => setPlaying(!playing));

function loop() {
  if (!playing || !manifest) return;
  let next = frame + 1;
  if (next >= traj.length) next = 0;
  drawAll(next);
  playTimer = setTimeout(loop, Math.max(40, 1000 / fps));
}

$$('.m-speed button').forEach(b => {
  b.addEventListener('click', () => {
    $$('.m-speed button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    fps = parseInt(b.dataset.fps, 10);
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
  // Auto-open the curve drawer once on first start so people see the loop.
  setTimeout(() => $$('.m-tb-btn[data-drawer=m-drawer-curve]')[0]?.click(), 200);
  setPlaying(true);
});

// Auto-skip onboarding if loaded via reload
if (sessionStorage.getItem('amoc-onb-seen')) {
  $('#onb').classList.add('hidden');
} else {
  sessionStorage.setItem('amoc-onb-seen', '1');
}

load();
