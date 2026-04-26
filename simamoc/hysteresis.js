// simamoc/hysteresis.js
// Side-by-side hysteresis viewer:
//   Left  — world map at the currently-selected ramp segment.
//   Right — full F-vs-AMOC loop with the current segment highlighted.
//
// Loads a directory produced by amoc-jax/scripts/hysteresis-replay.py:
//   manifest.json   { ..., trajectory: [{frame, phase, F, amoc, sst_natl, ...}] }
//   {temp,sal,psi,air_temp,ice_frac}.f32
//
// URL params:
//   ?run=amoc-jax/runs/hysteresis-default

const $ = (s) => document.querySelector(s);
const cv = $('#cv'), ctx = cv.getContext('2d');
const curve = $('#curve'), curveCtx = curve.getContext('2d');
const status_ = $('#status'), scrubber = $('#scrubber');
const frameInfo = $('#frameInfo'), playBtn = $('#playBtn');
const legendBar = $('#legendBar'), legendCtx = legendBar.getContext('2d');
const legendLo = $('#legendLo'), legendHi = $('#legendHi'), legendUnit = $('#legendUnit');
const sF = $('#sF'), sA = $('#sA'), sS = $('#sS');

const params = new URLSearchParams(location.search);
const runPath = params.get('run') || 'amoc-jax/runs/hysteresis-default';

// ─── colormaps (inlined; matches replay.js) ───
function tempToRGB(t) {
  const s = [
    [-2,[10,30,100]],[0,[40,120,200]],[10,[80,200,200]],
    [18,[200,220,100]],[25,[240,160,60]],[30,[220,60,40]],[40,[120,0,30]],
  ];
  for (let i = 0; i < s.length-1; i++) {
    const [a,ca]=s[i],[b,cb]=s[i+1];
    if (t <= b) { const u=Math.max(0,Math.min(1,(t-a)/(b-a)));
      return [ca[0]+(cb[0]-ca[0])*u,ca[1]+(cb[1]-ca[1])*u,ca[2]+(cb[2]-ca[2])*u]; }
  }
  return s[s.length-1][1];
}
function salToRGB(s) { const u=Math.max(0,Math.min(1,(s-30)/8)); return [40+200*u,200-100*u,80+120*u]; }
function psiToRGB(p, M) { const u = M>0?p/M:0;
  if (u>=0) return [255-100*(1-Math.abs(u)),255-200*Math.abs(u),255-200*Math.abs(u)];
  return [255-200*Math.abs(u),255-200*Math.abs(u),255-100*(1-Math.abs(u))]; }
function iceToRGB(f) { const u=Math.max(0,Math.min(1,f)); return [40+215*u,60+195*u,90+165*u]; }
const CMAPS = {
  temp:    {fn:tempToRGB,range:[-2,32],unit:'°C',dynamic:false},
  air_temp:{fn:tempToRGB,range:[-30,35],unit:'°C (air)',dynamic:false},
  sal:     {fn:salToRGB,range:[30,38],unit:'psu',dynamic:false},
  psi:     {fn:psiToRGB,range:null,unit:'streamfn',dynamic:true},
  ice_frac:{fn:iceToRGB,range:[0,1],unit:'fraction',dynamic:false},
};

// ─── state ───
let manifest=null, arrays={}, traj=[];
let field='temp', frame=0, playing=false;

// ─── load ───
async function load() {
  try {
    status_.textContent = `loading /${runPath}/manifest.json`;
    const r = await fetch(`/${runPath}/manifest.json`);
    if (!r.ok) throw new Error(`manifest not found at /${runPath}/`);
    manifest = await r.json();
    if (!manifest.trajectory) throw new Error('manifest missing .trajectory');
    traj = manifest.trajectory;
    cv.width = manifest.nx; cv.height = manifest.ny;
    cv.style.maxWidth = '700px';
    scrubber.max = traj.length - 1;
    status_.textContent = `loading ${manifest.fields.length} fields…`;
    for (const f of manifest.fields) {
      const fr = await fetch(`/${runPath}/${f}.f32`);
      if (!fr.ok) throw new Error(`field ${f} missing`);
      arrays[f] = new Float32Array(await fr.arrayBuffer());
    }
    status_.textContent = `${manifest.nx}×${manifest.ny}, ${traj.length} segments along the loop`;
    drawAll(0);
  } catch (e) {
    status_.textContent = `ERROR: ${e.message}`;
    console.error(e);
  }
}

// ─── world map render ───
function drawWorld(f) {
  if (!manifest) return;
  const {nx, ny} = manifest;
  const cells = nx * ny;
  const arr = arrays[field];
  if (!arr) return;
  const off = f * cells;
  const cmap = CMAPS[field];
  let absMax = 0;
  if (cmap.dynamic) { for (let k=0;k<cells;k++){ const v=arr[off+k]; if(Math.abs(v)>absMax)absMax=Math.abs(v); } if(absMax===0)absMax=1; }
  const img = ctx.createImageData(nx, ny);
  const d = img.data;
  for (let j = 0; j < ny; j++) {
    const dstRow = ny - 1 - j;  // south-up data → north-up screen
    for (let i = 0; i < nx; i++) {
      const v = arr[off + j*nx + i];
      const rgb = cmap.dynamic ? cmap.fn(v, absMax) : cmap.fn(v);
      const di = (dstRow*nx + i) * 4;
      d[di]=rgb[0]|0; d[di+1]=rgb[1]|0; d[di+2]=rgb[2]|0; d[di+3]=255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ─── legend ───
function drawLegend() {
  const cmap = CMAPS[field];
  const w = legendBar.width, h = legendBar.height;
  const [lo, hi] = cmap.dynamic ? [-1, 1] : cmap.range;
  for (let x = 0; x < w; x++) {
    const v = lo + (hi-lo)*(x/(w-1));
    const rgb = cmap.dynamic ? cmap.fn(v, 1) : cmap.fn(v);
    legendCtx.fillStyle = `rgb(${rgb[0]|0},${rgb[1]|0},${rgb[2]|0})`;
    legendCtx.fillRect(x, 0, 1, h);
  }
  legendLo.textContent = lo.toFixed(1);
  legendHi.textContent = hi.toFixed(1);
  legendUnit.textContent = cmap.unit;
}

// ─── hysteresis curve render ───
function drawCurve(f) {
  if (!traj.length) return;
  const W = curve.width, H = curve.height;
  curveCtx.fillStyle = '#08111c'; curveCtx.fillRect(0, 0, W, H);

  // Auto-scale axes
  const Fs = traj.map(p => p.F), As = traj.map(p => p.amoc);
  const Fmin = Math.min(...Fs), Fmax = Math.max(...Fs);
  const Amin = Math.min(...As, 0), Amax = Math.max(...As, 0);
  const padL = 50, padR = 12, padT = 14, padB = 28;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const xOf = (F) => padL + plotW * (F - Fmin) / Math.max(1e-9, Fmax - Fmin);
  const yOf = (A) => padT + plotH * (1 - (A - Amin) / Math.max(1e-9, Amax - Amin));

  // Grid + zero line
  curveCtx.strokeStyle = '#1a2838'; curveCtx.lineWidth = 1;
  curveCtx.strokeRect(padL, padT, plotW, plotH);
  if (Amin < 0 && Amax > 0) {
    curveCtx.strokeStyle = '#3a5468';
    curveCtx.beginPath(); curveCtx.moveTo(padL, yOf(0)); curveCtx.lineTo(W-padR, yOf(0)); curveCtx.stroke();
  }

  // Up-leg (orange) and down-leg (blue)
  const draw = (color, indices) => {
    if (indices.length < 2) return;
    curveCtx.strokeStyle = color; curveCtx.lineWidth = 1.5;
    curveCtx.beginPath();
    indices.forEach((i, k) => {
      const x = xOf(traj[i].F), y = yOf(traj[i].amoc);
      if (k === 0) curveCtx.moveTo(x, y); else curveCtx.lineTo(x, y);
    });
    curveCtx.stroke();
    curveCtx.fillStyle = color;
    indices.forEach(i => {
      const x = xOf(traj[i].F), y = yOf(traj[i].amoc);
      curveCtx.beginPath(); curveCtx.arc(x, y, 2.5, 0, Math.PI*2); curveCtx.fill();
    });
  };
  const ups = traj.map((p, i) => ({i, p})).filter(x => x.p.phase === 'init' || x.p.phase === 'up').map(x => x.i);
  const downs = traj.map((p, i) => ({i, p})).filter(x => x.p.phase === 'down').map(x => x.i);
  draw('#de9c7a', ups);
  if (downs.length) draw('#5a9ec8', [ups[ups.length-1], ...downs]);

  // Current point — bigger dot, contrasting
  const cur = traj[f];
  curveCtx.strokeStyle = '#fff'; curveCtx.lineWidth = 1.5; curveCtx.fillStyle = '#fff';
  const cx = xOf(cur.F), cy = yOf(cur.amoc);
  curveCtx.beginPath(); curveCtx.arc(cx, cy, 5, 0, Math.PI*2); curveCtx.stroke();
  curveCtx.fillStyle = cur.phase === 'down' ? '#5a9ec8' : '#de9c7a';
  curveCtx.beginPath(); curveCtx.arc(cx, cy, 3.5, 0, Math.PI*2); curveCtx.fill();

  // Axis labels
  curveCtx.fillStyle = '#a0b8c8'; curveCtx.font = '10px ui-monospace, monospace';
  curveCtx.textAlign = 'left';
  curveCtx.fillText('AMOC ↑', 4, padT - 2);
  curveCtx.textAlign = 'right';
  curveCtx.fillText('F →', W - 4, H - 8);
  // Tick labels
  curveCtx.textAlign = 'center';
  for (let k = 0; k <= 4; k++) {
    const F = Fmin + (Fmax-Fmin)*k/4;
    curveCtx.fillText(F.toFixed(1), xOf(F), H - padB + 13);
  }
  curveCtx.textAlign = 'right';
  for (let k = 0; k <= 4; k++) {
    const A = Amin + (Amax-Amin)*k/4;
    curveCtx.fillText(A.toExponential(1), padL - 3, yOf(A) + 3);
  }

  // Legend
  curveCtx.font = '10px system-ui, sans-serif'; curveCtx.textAlign = 'left';
  curveCtx.fillStyle = '#de9c7a'; curveCtx.fillRect(padL + 8, padT + 6, 12, 2);
  curveCtx.fillStyle = '#a0b8c8'; curveCtx.fillText('ramp up', padL + 24, padT + 11);
  curveCtx.fillStyle = '#5a9ec8'; curveCtx.fillRect(padL + 80, padT + 6, 12, 2);
  curveCtx.fillStyle = '#a0b8c8'; curveCtx.fillText('ramp down', padL + 96, padT + 11);
}

function drawAll(f) {
  frame = f;
  drawWorld(f);
  drawCurve(f);
  drawLegend();
  scrubber.value = f;
  const cur = traj[f];
  if (cur) {
    sF.textContent = cur.F.toFixed(2);
    sA.textContent = cur.amoc.toExponential(2);
    sS.textContent = cur.sst_natl.toFixed(2);
    const phaseLabel = cur.phase === 'init' ? 'spinup' : cur.phase === 'up' ? '↑ ramp up' : '↓ ramp down';
    frameInfo.textContent = `frame ${f+1}/${traj.length}  ${phaseLabel}  step=${cur.step}`;
  }
}

// ─── controls ───
document.querySelectorAll('header .field button').forEach((b) => {
  b.addEventListener('click', () => {
    document.querySelectorAll('header .field button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    field = b.dataset.field;
    drawAll(frame);
  });
});
scrubber.addEventListener('input', e => drawAll(parseInt(e.target.value, 10)));
playBtn.addEventListener('click', () => {
  playing = !playing;
  playBtn.textContent = playing ? '⏸ pause' : '▶ play';
  if (playing) loop();
});
function loop() {
  if (!playing || !manifest) return;
  let next = frame + 1;
  if (next >= traj.length) next = 0;
  drawAll(next);
  setTimeout(loop, 250);  // 4 fps — slow enough to see each segment
}

load();
