// experiments/amoc-bifurcation.mjs
// ----------------------------------------------------------------------
// The Stommel bifurcation: ramp freshwater forcing slowly up, watch the
// AMOC weaken and ultimately collapse. Then ramp back down: a healthy
// thermohaline circulation does not return at the same threshold (this
// is the classic two-branch hysteresis loop, Stommel 1961).
//
// Method:
//   1. Spin up to a strong-AMOC equilibrium at F=0.
//   2. Forward branch: F = 0 → 2.0, holding each value for DWELL_STEPS
//      so AMOC can equilibrate.
//   3. Backward branch: F = 2.0 → 0 with the same dwell time.
//   4. Sample AMOC + key diagnostics + a temp/psi frame at each F.
//   5. Compose a bifurcation diagram (AMOC vs F, two branches) with
//      collapse / recovery thresholds annotated.
//
// Usage:
//   node helm-lab/cli.mjs serve &        # in another shell, or background
//   node helm-lab/experiments/amoc-bifurcation.mjs
//
// ~5 minutes wall-clock on Apple Metal GPU.

import { writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const PORT = Number(process.env.HELM_LAB_PORT || 8830);
const URL = `http://127.0.0.1:${PORT}`;

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

const OUT = resolve('helm-lab/runs/amoc-bifurcation');
await mkdir(join(OUT, 'frames'), { recursive: true });

const FORWARD  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0];
const BACKWARD = [1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0];
const SPINUP_STEPS = 60000;
const DWELL_STEPS  = 12000;
const VIEWS        = ['temp', 'psi'];

console.log(`=== AMOC bifurcation experiment ===
out: ${OUT}
forward branch:  F = ${FORWARD.join(', ')}
backward branch: F = ${BACKWARD.join(', ')}
spinup ${SPINUP_STEPS} steps · dwell ${DWELL_STEPS} steps per F
`);

const tStart = Date.now();

// 1. Reset & baseline spin-up.
console.log('[1/4] reset + initial spin-up at F=0...');
await rpc('reset');
await rpc('setParams', [{
  windStrength: 1.0, r: 0.04, A: 2e-4,
  freshwaterForcing: 0.0, yearSpeed: 1.0, stepsPerFrame: 50,
}]);
const tSpin = Date.now();
await rpc('step', [SPINUP_STEPS]);
const baseDiag = await rpc('diag', [{}]);
console.log(`     spin-up done in ${((Date.now() - tSpin) / 1000).toFixed(1)}s · ` +
  `AMOC=${baseDiag.amoc.toFixed(4)} KE=${baseDiag.KE.toExponential(2)} simYears=${baseDiag.simYears.toFixed(1)}`);

async function ramp(values, branch) {
  const points = [];
  for (const F of values) {
    const t0 = Date.now();
    await rpc('setParams', [{ freshwaterForcing: F }]);
    await rpc('step', [DWELL_STEPS]);
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
    const tag2 = (Math.abs(d.amoc) < 0.005) ? '[COLLAPSED]' :
                 (Math.abs(d.amoc) < 0.02)  ? '[weak]'      : '[strong]';
    console.log(
      `  ${branch.padEnd(8)} F=${F.toFixed(2)}  ` +
      `AMOC=${d.amoc.toFixed(5)} ${tag2.padEnd(11)} ` +
      `T_glob=${d.globalSST.toFixed(2)}°  ice=${d.iceArea.toString().padStart(5)}  (${dt}s)`
    );
  }
  return points;
}

console.log(`\n[2/4] forward branch (${FORWARD.length} F values × ${DWELL_STEPS} steps)...`);
const fwd = await ramp(FORWARD, 'forward');

console.log(`\n[3/4] backward branch (${BACKWARD.length} F values × ${DWELL_STEPS} steps)...`);
const bwd = await ramp(BACKWARD, 'backward');

const all = [...fwd, ...bwd];

// 4. Save artifacts.
console.log('\n[4/4] composing diagrams...');
const stripArr = d => { const { zonalMeanT, zonalMeanPsi, zonalMeanU, latitudes, ...r } = d; return r; };
await writeFile(join(OUT, 'samples.json'), JSON.stringify(
  all.map(p => ({ branch: p.branch, F: p.F, frames: p.frames, diag: stripArr(p.diag) })),
  null, 2));
await writeFile(join(OUT, 'amoc-vs-F.jsonl'), all.map(p => JSON.stringify({
  branch: p.branch, F: p.F, amoc: p.diag.amoc, KE: p.diag.KE,
  globalSST: p.diag.globalSST, polarSST: p.diag.polarSST,
  tropicalSST: p.diag.tropicalSST, iceArea: p.diag.iceArea,
  simYears: p.diag.simYears,
})).join('\n') + '\n');

// Bifurcation diagram (PNG via in-page composer, if the daemon has the
// method registered — added in a separate commit). Falls back gracefully.
try {
  const points = all.map(p => ({
    branch: p.branch, F: p.F,
    amoc: p.diag.amoc, KE: p.diag.KE,
    globalSST: p.diag.globalSST, polarSST: p.diag.polarSST,
    frames: { temp: p.frames.temp },
  }));
  await rpc('composeBifurcation', [points, join(OUT, 'bifurcation.png'), {
    title: 'AMOC strength vs. freshwater forcing',
    subtitle: `forward ramp ●  ·  backward ramp ○  ·  ${all.length} samples · ${(SPINUP_STEPS + (FORWARD.length + BACKWARD.length) * DWELL_STEPS).toLocaleString()} total steps`,
  }]);
} catch (e) {
  console.warn(`     (skipped bifurcation.png: ${e.message})`);
  console.warn('     numerical data still in samples.json + amoc-vs-F.jsonl');
}

// Findings summary.
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
  collapse: collapse,
  recovery: recovery,
  hysteresis_window_F: (collapse.F != null && recovery.F != null)
    ? collapse.F - recovery.F : null,
  recovery_complete: bwd[bwd.length - 1].diag.amoc / baseDiag.amoc,
  steps_total: SPINUP_STEPS + (FORWARD.length + BACKWARD.length) * DWELL_STEPS,
  wall_clock_minutes: ((Date.now() - tStart) / 60000).toFixed(2),
};
await writeFile(join(OUT, 'summary.json'), JSON.stringify(summary, null, 2));

console.log('\n=== SUMMARY ===');
console.log(JSON.stringify(summary, null, 2));
console.log(`\nartifacts:
  ${OUT}/bifurcation.png       <- main result
  ${OUT}/samples.json
  ${OUT}/amoc-vs-F.jsonl
  ${OUT}/summary.json
  ${OUT}/frames/*.png          (${all.length * VIEWS.length} files)
`);
