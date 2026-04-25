// experiments/amoc-collapse-window.mjs
// ----------------------------------------------------------------------
// V3 — pin down the AMOC collapse threshold F*.
//
// V2 found a big drop between F=0 and F=0.4, suggesting F* lies in
// that window. This script samples that window densely (every 0.05),
// then sparsely outside it, to map the bifurcation more precisely.
//
// Note: the model is bistable. The 400k spinup may land on either the
// strong-positive branch (~+0.0092) or the negative branch (~-0.002).
// We run the spinup, check baseline AMOC, and bail if we landed on the
// wrong branch (so we don't waste 12 minutes on an inconclusive run).
//
// Spinup 400k + 14 F values × 60k = 1.24M steps, ~13 min on Apple Metal.

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

const OUT = resolve('helm-lab/runs/amoc-collapse-window');
await mkdir(join(OUT, 'frames'), { recursive: true });

// Dense in the suspected collapse window, sparse outside.
const FORWARD  = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0];
const BACKWARD = [1.5, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0];
const SPINUP   = 400000;
const DWELL    = 60000;
const VIEWS    = ['temp'];
const POSITIVE_BRANCH_THRESHOLD = 0.003;  // baseline AMOC must exceed this

console.log(`=== AMOC collapse-window scan ===
out:    ${OUT}
spinup: ${SPINUP.toLocaleString()} steps (~${(SPINUP * 5e-5 / 10).toFixed(1)} model years)
dwell:  ${DWELL.toLocaleString()} steps (~${(DWELL * 5e-5 / 10).toFixed(2)} years per F)
F grid (forward):  ${FORWARD.join(', ')}
F grid (backward): ${BACKWARD.join(', ')}
total:  ${(SPINUP + (FORWARD.length + BACKWARD.length) * DWELL).toLocaleString()} steps
`);

const tStart = Date.now();

console.log('[1/4] reset + spin-up...');
await rpc('reset');
await rpc('setParams', [{
  windStrength: 1.0, r: 0.04, A: 2e-4,
  freshwaterForcing: 0.0, yearSpeed: 1.0, stepsPerFrame: 50,
}]);

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
console.log(`     spinup done · baseline AMOC = ${baseDiag.amoc.toFixed(4)}`);

if (baseDiag.amoc < POSITIVE_BRANCH_THRESHOLD) {
  console.log(`\n!!  spinup landed on the weak/negative branch (AMOC=${baseDiag.amoc.toFixed(4)} < ${POSITIVE_BRANCH_THRESHOLD})`);
  console.log(`    aborting — re-run to roll the dice on initial conditions, or lengthen SPINUP`);
  await writeFile(join(OUT, 'aborted.json'), JSON.stringify({
    reason: 'spinup landed on weak branch', baseline: baseDiag, threshold: POSITIVE_BRANCH_THRESHOLD,
  }, null, 2));
  process.exit(2);
}

async function ramp(values, branch) {
  const points = [];
  for (const F of values) {
    const t0 = Date.now();
    await rpc('setParams', [{ freshwaterForcing: F }]);
    await rpc('step', [DWELL]);
    const d = await rpc('diag', [{}]);
    const tag = `${branch}_F${F.toFixed(2).replace('.', 'p')}`;
    const frames = {};
    for (const v of VIEWS) {
      const fp = join(OUT, 'frames', `${tag}_${v}.png`);
      await rpc('render', [fp, { view: v }]);
      frames[v] = fp;
    }
    points.push({ branch, F, frames, diag: d });
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(
      `  ${branch.padEnd(8)} F=${F.toFixed(2)}  AMOC=${d.amoc.toFixed(5)}  ` +
      `T_glob=${d.globalSST.toFixed(2)}°  T_pol=${d.polarSST.toFixed(2)}°  ` +
      `ice=${d.iceArea.toString().padStart(5)}  (${dt}s)`
    );
  }
  return points;
}

console.log(`\n[2/4] forward ramp (${FORWARD.length} values × ${DWELL} steps)...`);
const fwd = await ramp(FORWARD, 'forward');

console.log(`\n[3/4] backward ramp (${BACKWARD.length} values × ${DWELL} steps)...`);
const bwd = await ramp(BACKWARD, 'backward');

console.log('\n[4/4] writing artifacts...');
const all = [...fwd, ...bwd];
const stripArr = d => { const { zonalMeanT, zonalMeanPsi, zonalMeanU, latitudes, ...r } = d; return r; };

await writeFile(join(OUT, 'samples.json'), JSON.stringify(
  all.map(p => ({ branch: p.branch, F: p.F, frames: p.frames, diag: stripArr(p.diag) })), null, 2));
await writeFile(join(OUT, 'amoc-vs-F.jsonl'), all.map(p => JSON.stringify({
  branch: p.branch, F: p.F, amoc: p.diag.amoc, KE: p.diag.KE,
  globalSST: p.diag.globalSST, polarSST: p.diag.polarSST,
  tropicalSST: p.diag.tropicalSST, iceArea: p.diag.iceArea,
  simYears: p.diag.simYears,
})).join('\n') + '\n');

// Try the bifurcation composer.
try {
  const points = all.map(p => ({
    branch: p.branch, F: p.F, amoc: p.diag.amoc, KE: p.diag.KE,
    globalSST: p.diag.globalSST, polarSST: p.diag.polarSST,
    frames: p.branch === 'forward' ? { temp: p.frames.temp } : {},
  }));
  await rpc('composeBifurcation', [points, join(OUT, 'bifurcation.png'), {
    title: 'AMOC vs F · collapse-window scan',
    subtitle: `dense in F=0…0.4 (Δ=0.05) · spinup ${SPINUP.toLocaleString()} · dwell ${DWELL.toLocaleString()} per F`,
  }]);
  console.log(`     wrote ${OUT}/bifurcation.png`);
} catch (e) {
  console.warn(`     (skipped bifurcation.png: ${e.message})`);
}

// Find collapse threshold by largest single-step drop in forward branch.
let bestDrop = 0, F_star = null, fromAMOC = null, toAMOC = null;
for (let i = 1; i < fwd.length; i++) {
  const drop = fwd[i - 1].diag.amoc - fwd[i].diag.amoc;
  if (drop > bestDrop) {
    bestDrop = drop; F_star = (fwd[i - 1].F + fwd[i].F) / 2;
    fromAMOC = fwd[i - 1].diag.amoc; toAMOC = fwd[i].diag.amoc;
  }
}
const summary = {
  baseline_amoc: baseDiag.amoc,
  forward_endpoint_amoc: fwd[fwd.length - 1].diag.amoc,
  backward_endpoint_amoc: bwd[bwd.length - 1].diag.amoc,
  collapse_F_star: F_star,
  collapse_drop: bestDrop,
  collapse_from: fromAMOC,
  collapse_to: toAMOC,
  collapse_drop_fraction: fromAMOC ? bestDrop / fromAMOC : null,
  recovery_complete_ratio: bwd[bwd.length - 1].diag.amoc / baseDiag.amoc,
  steps_total: SPINUP + (FORWARD.length + BACKWARD.length) * DWELL,
  wall_clock_minutes: ((Date.now() - tStart) / 60000).toFixed(2),
};
await writeFile(join(OUT, 'summary.json'), JSON.stringify(summary, null, 2));
console.log('\n=== SUMMARY ===');
console.log(JSON.stringify(summary, null, 2));
console.log(`\nartifacts in: ${OUT}/`);
