// experiments/amoc-bifurcation-v2.mjs
// ----------------------------------------------------------------------
// V2 of the Stommel hysteresis search — informed by the long-monitor
// finding that AMOC undergoes a natural sign-flip transient around
// model year 1.2 and only settles to a quasi-steady state after year ~2.
// V1 ran with a 60k-step (0.3 yr) spinup, which caught the system mid-
// transient and produced small-amplitude hysteresis.
//
// V2:
//   spinup = 400000 steps  (~2 model years; AMOC has flipped + settled)
//   dwell  = 60000 steps   (~0.3 yr per F; long enough to see response)
//   F grid = 0.0, 0.4, 0.8, 1.2, 1.6, 2.0 forward + reverse
//   total  = 400k + 11×60k = 1.06M steps  ≈ 17 min wall on Apple Metal
//
// Same RPC pattern as v1 — small calls, no long-held connections.

import { writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const PORT = Number(process.env.HELM_LAB_PORT || 8830);
const URL  = `http://127.0.0.1:${PORT}`;

async function rpc(method, args = []) {
  const r = await fetch(`${URL}/rpc`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ method, args }),
  });
  const j = await r.json();
  if (!j.ok) throw new Error(`${method}: ${j.error}`);
  return j.result;
}

const OUT = resolve('helm-lab/runs/amoc-bifurcation-v2');
await mkdir(join(OUT, 'frames'), { recursive: true });

const FORWARD  = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0];
const BACKWARD = [1.6, 1.2, 0.8, 0.4, 0.0];
const SPINUP   = 400000;
const DWELL    = 60000;
const VIEWS    = ['temp'];

console.log(`=== AMOC bifurcation v2 (informed spinup) ===
out:    ${OUT}
spinup: ${SPINUP.toLocaleString()} steps (~${(SPINUP * 5e-5 / 10).toFixed(1)} model years)
dwell:  ${DWELL.toLocaleString()} steps (~${(DWELL * 5e-5 / 10).toFixed(2)} years per F)
F grid: ${FORWARD.join(', ')} forward, ${BACKWARD.join(', ')} backward
total:  ${(SPINUP + (FORWARD.length + BACKWARD.length) * DWELL).toLocaleString()} steps
`);

const tStart = Date.now();

console.log('[1/4] reset + long spin-up at F=0...');
await rpc('reset');
await rpc('setParams', [{
  windStrength: 1.0, r: 0.04, A: 2e-4,
  freshwaterForcing: 0.0, yearSpeed: 1.0, stepsPerFrame: 50,
}]);

// Stream spin-up in chunks so we can see progress.
const SPIN_CHUNK = 50000;
let spinDone = 0;
while (spinDone < SPINUP) {
  const n = Math.min(SPIN_CHUNK, SPINUP - spinDone);
  await rpc('step', [n]);
  spinDone += n;
  const d = await rpc('diag', [{}]);
  console.log(`     ${spinDone.toString().padStart(6)} / ${SPINUP}  ` +
    `simYr=${d.simYears.toFixed(2)} KE=${d.KE.toExponential(2)} AMOC=${d.amoc.toFixed(4)} ` +
    `T_glob=${d.globalSST.toFixed(2)}° ice=${d.iceArea}`);
}
const baseDiag = await rpc('diag', [{}]);
const tSpin = ((Date.now() - tStart) / 1000).toFixed(1);
console.log(`     spinup done in ${tSpin}s · baseline AMOC=${baseDiag.amoc.toFixed(4)}`);

async function ramp(values, branch) {
  const points = [];
  for (const F of values) {
    const t0 = Date.now();
    await rpc('setParams', [{ freshwaterForcing: F }]);
    await rpc('step', [DWELL]);
    const d = await rpc('diag', [{}]);
    const tag = `${branch}_F${F.toFixed(1).replace('.', 'p')}`;
    const frames = {};
    for (const v of VIEWS) {
      const fp = join(OUT, 'frames', `${tag}_${v}.png`);
      await rpc('render', [fp, { view: v }]);
      frames[v] = fp;
    }
    points.push({ branch, F, frames, diag: d });
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    const status = (Math.abs(d.amoc) < 0.005) ? '[COLLAPSED]' :
                   (Math.abs(d.amoc) < 0.02)  ? '[weak]'      : '[STRONG]';
    console.log(
      `  ${branch.padEnd(8)} F=${F.toFixed(1)}  ` +
      `AMOC=${d.amoc.toFixed(5)} ${status.padEnd(11)} ` +
      `T_glob=${d.globalSST.toFixed(2)}°  T_pol=${d.polarSST.toFixed(2)}°  ` +
      `ice=${d.iceArea.toString().padStart(5)}  (${dt}s)`
    );
  }
  return points;
}

console.log(`\n[2/4] forward branch (${FORWARD.length} × ${DWELL} steps)...`);
const fwd = await ramp(FORWARD, 'forward');

console.log(`\n[3/4] backward branch (${BACKWARD.length} × ${DWELL} steps)...`);
const bwd = await ramp(BACKWARD, 'backward');

const all = [...fwd, ...bwd];
const stripArr = d => { const { zonalMeanT, zonalMeanPsi, zonalMeanU, latitudes, ...r } = d; return r; };

console.log('\n[4/4] writing artifacts...');
await writeFile(join(OUT, 'samples.json'), JSON.stringify(
  all.map(p => ({ branch: p.branch, F: p.F, frames: p.frames, diag: stripArr(p.diag) })), null, 2));
await writeFile(join(OUT, 'amoc-vs-F.jsonl'), all.map(p => JSON.stringify({
  branch: p.branch, F: p.F, amoc: p.diag.amoc, KE: p.diag.KE,
  globalSST: p.diag.globalSST, polarSST: p.diag.polarSST,
  tropicalSST: p.diag.tropicalSST, iceArea: p.diag.iceArea,
  simYears: p.diag.simYears,
})).join('\n') + '\n');

// Try the in-page bifurcation composer if the daemon happens to have it
// loaded. Otherwise the data is the result.
try {
  const points = all.map(p => ({
    branch: p.branch, F: p.F, amoc: p.diag.amoc, KE: p.diag.KE,
    globalSST: p.diag.globalSST, polarSST: p.diag.polarSST,
    frames: p.branch === 'forward' ? { temp: p.frames.temp } : {},
  }));
  await rpc('composeBifurcation', [points, join(OUT, 'bifurcation.png'), {
    title: 'AMOC strength vs. freshwater forcing (v2 · long spinup)',
    subtitle: `forward ramp ●  ·  backward ramp ○  ·  spinup ${SPINUP.toLocaleString()} + dwell ${DWELL.toLocaleString()} per F`,
  }]);
  console.log(`     wrote ${OUT}/bifurcation.png`);
} catch (e) {
  console.warn(`     (skipped bifurcation.png: ${e.message})`);
}

function findThreshold(arr, sign) {
  let best = 0, F = null;
  for (let i = 1; i < arr.length; i++) {
    const d = sign * (Math.abs(arr[i - 1].diag.amoc) - Math.abs(arr[i].diag.amoc));
    if (d > best) { best = d; F = (arr[i - 1].F + arr[i].F) / 2; }
  }
  return { F, magnitude: best };
}
const collapse = findThreshold(fwd, +1);
const recovery = findThreshold(bwd, -1);
const summary = {
  baseline_amoc: baseDiag.amoc,
  forward_endpoint_amoc: fwd[fwd.length - 1].diag.amoc,
  backward_endpoint_amoc: bwd[bwd.length - 1].diag.amoc,
  collapse, recovery,
  hysteresis_window_F: (collapse.F != null && recovery.F != null) ? collapse.F - recovery.F : null,
  recovery_complete_ratio: bwd[bwd.length - 1].diag.amoc / baseDiag.amoc,
  steps_total: SPINUP + (FORWARD.length + BACKWARD.length) * DWELL,
  wall_clock_minutes: ((Date.now() - tStart) / 60000).toFixed(2),
};
await writeFile(join(OUT, 'summary.json'), JSON.stringify(summary, null, 2));

console.log('\n=== SUMMARY ===');
console.log(JSON.stringify(summary, null, 2));
console.log(`\nartifacts in: ${OUT}/`);
