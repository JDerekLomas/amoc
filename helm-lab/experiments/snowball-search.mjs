// experiments/snowball-search.mjs
// ----------------------------------------------------------------------
// Sweep the solar constant downward to find the ice-albedo runaway
// threshold (snowball Earth bifurcation). At some critical S* the ice
// line crosses a latitude where reflection from new ice cools the planet
// faster than the sun warms it — and the ice line runs away to the
// equator. This is the canonical Budyko/Sellers result.
//
// Method:
//   1. Reset, spin up at S=100 (modern Earth) for 400k steps.
//   2. Step S downward: 100, 90, 80, 70, 60, 50, 40, 30, 20.
//   3. At each S: dwell 80k steps; snapshot diag + temp frame.
//   4. Look for: T_glob crashing, iceArea exploding, polar SST plunging.
//
// What we expect:
//   - At high S, world is warm, partial polar ice.
//   - Below S*, ice-albedo runaway: iceArea jumps to many tens of
//     thousands of cells, T_glob crashes by 10-20°C.
//
// Wall clock: 400k + 9 * 80k = 1.12M steps, ~12 min on Apple Metal.

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

const OUT = resolve('helm-lab/runs/snowball-search');
await mkdir(join(OUT, 'frames'), { recursive: true });

// 1.0 = modern; range slider min is 20% so we go down to 0.20.
// In the engine, S_solar is the solar amplitude param. The UI's "Solar
// Constant" slider goes 20-200; the engine internal default is 5.0
// scaled by some factor. We'll set S_solar relative to its default
// (5.0). Ratios from 0.2 to 1.0 of default.
const S_VALUES = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5];
const SPINUP   = 400000;
const DWELL    = 80000;

console.log(`=== snowball-search ===
out:    ${OUT}
spinup: ${SPINUP.toLocaleString()} steps at S_solar=5.0 (modern)
dwell:  ${DWELL.toLocaleString()} steps per S value
S values: ${S_VALUES.join(', ')}  (5.0 = engine default)
total:  ${(SPINUP + S_VALUES.length * DWELL).toLocaleString()} steps
`);

const tStart = Date.now();

console.log('[1/3] spin up at S_solar=5.0 (modern)...');
await rpc('reset');
await rpc('setParams', [{
  windStrength: 1.0, r: 0.04, A: 2e-4,
  freshwaterForcing: 0.0, globalTempOffset: 0.0,
  S_solar: 5.0, yearSpeed: 1.0, stepsPerFrame: 50,
}]);
const SPIN_CHUNK = 50000;
for (let done = 0; done < SPINUP; done += SPIN_CHUNK) {
  const n = Math.min(SPIN_CHUNK, SPINUP - done);
  await rpc('step', [n]);
  const d = await rpc('diag', [{}]);
  console.log(`     ${(done + n).toString().padStart(6)} / ${SPINUP}  ` +
    `simYr=${d.simYears.toFixed(2)} T_glob=${d.globalSST.toFixed(2)}° T_pol=${d.polarSST.toFixed(2)}° ice=${d.iceArea}`);
}
const baseDiag = await rpc('diag', [{}]);
console.log(`     baseline (S=5.0): T_glob=${baseDiag.globalSST.toFixed(2)}° T_pol=${baseDiag.polarSST.toFixed(2)}° ice=${baseDiag.iceArea}`);

console.log(`\n[2/3] cooling sweep (S 5.0 → 0.5)...`);
const points = [];
for (const S of S_VALUES) {
  const t0 = Date.now();
  await rpc('setParams', [{ S_solar: S }]);
  await rpc('step', [DWELL]);
  const d = await rpc('diag', [{}]);
  const tag = `S${S.toFixed(1).replace('.', 'p')}`;
  const fp = join(OUT, 'frames', `${tag}_temp.png`);
  await rpc('render', [fp, { view: 'temp' }]);
  points.push({ S, frames: { temp: fp }, diag: d });
  const dt = ((Date.now() - t0) / 1000).toFixed(1);
  // Detect runaway: ice > 60% of ocean cells (ocean has ~44k cells)
  const oceanCells = d.oceanCells || 43950;
  const iceFrac = d.iceArea / oceanCells;
  const tag2 = iceFrac > 0.6 ? '[SNOWBALL]' : iceFrac > 0.3 ? '[icy]' : '[ice-free]';
  console.log(
    `  S=${S.toFixed(1)}  ` +
    `T_glob=${d.globalSST.toFixed(2)}°  T_trop=${d.tropicalSST.toFixed(2)}°  ` +
    `T_pol=${d.polarSST.toFixed(2)}°  ice=${d.iceArea.toString().padStart(5)} (${(iceFrac * 100).toFixed(0)}%)  ` +
    `${tag2.padEnd(11)} (${dt}s)`
  );
}

console.log('\n[3/3] writing artifacts...');
const stripArr = d => { const { zonalMeanT, zonalMeanPsi, zonalMeanU, latitudes, ...r } = d; return r; };
await writeFile(join(OUT, 'samples.json'), JSON.stringify(
  points.map(p => ({ S: p.S, frames: p.frames, diag: stripArr(p.diag) })), null, 2));
await writeFile(join(OUT, 'response.jsonl'), points.map(p => JSON.stringify({
  S: p.S, T_glob: p.diag.globalSST, T_trop: p.diag.tropicalSST,
  T_pol: p.diag.polarSST, iceArea: p.diag.iceArea,
  iceFraction: p.diag.iceArea / (p.diag.oceanCells || 43950),
  amoc: p.diag.amoc, KE: p.diag.KE, simYears: p.diag.simYears,
})).join('\n') + '\n');

// Find threshold by largest single-step ice jump.
let bestJump = 0, S_star = null, fromIce = null, toIce = null;
for (let i = 1; i < points.length; i++) {
  const jump = points[i].diag.iceArea - points[i - 1].diag.iceArea;
  if (jump > bestJump) {
    bestJump = jump; S_star = (points[i - 1].S + points[i].S) / 2;
    fromIce = points[i - 1].diag.iceArea; toIce = points[i].diag.iceArea;
  }
}

const summary = {
  baseline: { S: 5.0, T_glob: baseDiag.globalSST, T_pol: baseDiag.polarSST, iceArea: baseDiag.iceArea },
  endpoint: { S: S_VALUES[S_VALUES.length - 1], ...points[points.length - 1].diag },
  snowball_threshold_S: S_star,
  ice_jump_at_threshold: bestJump,
  ice_before_threshold: fromIce,
  ice_after_threshold: toIce,
  T_drop_total: baseDiag.globalSST - points[points.length - 1].diag.globalSST,
  steps_total: SPINUP + S_VALUES.length * DWELL,
  wall_clock_minutes: ((Date.now() - tStart) / 60000).toFixed(2),
};
await writeFile(join(OUT, 'summary.json'), JSON.stringify(summary, null, 2));
console.log('\n=== SUMMARY ===');
console.log(JSON.stringify(summary, null, 2));
console.log(`\nartifacts in: ${OUT}/`);
