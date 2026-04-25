// experiments/snowball-escape.mjs
// ----------------------------------------------------------------------
// The companion to R7 (snowball-search). R7 found the freezing
// threshold S*_freeze ≈ 1.25 going COLD. Once frozen, what S does it
// take to thaw the planet?
//
// In the canonical Budyko/Sellers picture (and in the Cryogenian
// geological record), the de-glaciation threshold is dramatically
// higher than the freezing threshold — because the bright ice surface
// reflects most of the incoming sunlight, you need to overdrive the
// system to break out. Hoffman & Schrag (1998) argued the real Earth
// needed centuries of CO₂ buildup to escape. The hysteresis loop
// between freeze and thaw is the climatological version of R5's
// freshwater hysteresis.
//
// Method:
//   1. Reset, set S_solar = 0.3 (deeply frozen), spin up 500k steps.
//   2. Confirm snowball state (ice fraction > 95%).
//   3. Ramp S UPWARD: 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 14, 20.
//      (engine slider max is 10× default = 10.0; we use S_solar
//      directly which has no slider cap.)
//   4. At each S: dwell 80k, snapshot.
//   5. Find escape threshold where ice plummets.

import { writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const PORT = Number(process.env.HELM_LAB_PORT || 8830);
const URL  = `http://127.0.0.1:${PORT}`;
async function rpc(m, a = []) {
  const r = await fetch(`${URL}/rpc`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ method: m, args: a }) });
  const j = await r.json();
  if (!j.ok) throw new Error(`${m}: ${j.error}`);
  return j.result;
}

const OUT = resolve('helm-lab/runs/snowball-escape');
await mkdir(join(OUT, 'frames'), { recursive: true });

const S_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0, 20.0];
const FREEZE_SPINUP_STEPS = 500000;
const FREEZE_S = 0.3;
const DWELL = 80000;

console.log(`=== snowball-escape ===
freeze first: S=${FREEZE_S} for ${FREEZE_SPINUP_STEPS.toLocaleString()} steps
warming sweep: S = ${S_VALUES.join(', ')} (each ${DWELL.toLocaleString()} steps)
total: ${(FREEZE_SPINUP_STEPS + S_VALUES.length * DWELL).toLocaleString()} steps
`);

const tStart = Date.now();

console.log('[1/3] freeze the planet — spin up at low S...');
await rpc('reset');
await rpc('setParams', [{
  windStrength: 1.0, r: 0.04, A: 2e-4,
  freshwaterForcing: 0.0, globalTempOffset: 0.0,
  S_solar: FREEZE_S, yearSpeed: 1.0, stepsPerFrame: 50,
}]);
const SPIN_CHUNK = 50000;
for (let done = 0; done < FREEZE_SPINUP_STEPS; done += SPIN_CHUNK) {
  const n = Math.min(SPIN_CHUNK, FREEZE_SPINUP_STEPS - done);
  await rpc('step', [n]);
  const d = await rpc('diag', [{}]);
  const frac = d.iceArea / (d.oceanCells || 43950);
  console.log(`     ${(done + n).toString().padStart(6)} / ${FREEZE_SPINUP_STEPS}  ` +
    `simYr=${d.simYears.toFixed(2)} T_glob=${d.globalSST.toFixed(2)}° ice=${d.iceArea} (${(frac * 100).toFixed(0)}%)`);
}
const frozenDiag = await rpc('diag', [{}]);
const frozenFrac = frozenDiag.iceArea / (frozenDiag.oceanCells || 43950);
console.log(`     frozen state: T_glob=${frozenDiag.globalSST.toFixed(2)}° ice=${frozenDiag.iceArea} (${(frozenFrac * 100).toFixed(0)}%)`);

if (frozenFrac < 0.85) {
  console.warn(`!!  WARNING: ice fraction only ${(frozenFrac * 100).toFixed(0)}% — not a true snowball.`);
  console.warn(`    proceeding anyway, but the "escape" interpretation may not apply.`);
}

console.log(`\n[2/3] warming sweep (S = ${FREEZE_S} → up)...`);
const points = [];
let escapeSeen = false;
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
  const frac = d.iceArea / (d.oceanCells || 43950);
  let regime;
  if (frac > 0.7) regime = '[FROZEN]';
  else if (frac > 0.3) regime = '[melting]';
  else { regime = '[ESCAPED]'; if (!escapeSeen) escapeSeen = true; }
  console.log(
    `  S=${S.toFixed(1).padStart(5)}  T_glob=${d.globalSST.toFixed(2).padStart(7)}°  ` +
    `T_pol=${d.polarSST.toFixed(2).padStart(7)}°  ` +
    `ice=${d.iceArea.toString().padStart(5)} (${(frac * 100).toFixed(0).padStart(3)}%)  ` +
    `${regime.padEnd(11)} (${dt}s)`
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

// Find escape threshold = largest single-step ice DROP.
let bestDrop = 0, S_escape = null, fromIce = null, toIce = null;
for (let i = 1; i < points.length; i++) {
  const drop = points[i - 1].diag.iceArea - points[i].diag.iceArea;
  if (drop > bestDrop) {
    bestDrop = drop; S_escape = (points[i - 1].S + points[i].S) / 2;
    fromIce = points[i - 1].diag.iceArea; toIce = points[i].diag.iceArea;
  }
}

const summary = {
  freeze_state: { S: FREEZE_S, T_glob: frozenDiag.globalSST,
    T_pol: frozenDiag.polarSST, iceArea: frozenDiag.iceArea, iceFraction: frozenFrac },
  endpoint: { S: S_VALUES[S_VALUES.length - 1], ...stripArr(points[points.length - 1].diag) },
  escape_threshold_S: S_escape,
  ice_drop_at_escape: bestDrop,
  ice_before_escape: fromIce,
  ice_after_escape: toIce,
  T_rise_total: points[points.length - 1].diag.globalSST - frozenDiag.globalSST,
  // Compare with R7: freezing threshold was ~1.25
  R7_freeze_S_star: 1.25,
  hysteresis_window: S_escape != null ? S_escape - 1.25 : null,
  steps_total: FREEZE_SPINUP_STEPS + S_VALUES.length * DWELL,
  wall_clock_minutes: ((Date.now() - tStart) / 60000).toFixed(2),
};
await writeFile(join(OUT, 'summary.json'), JSON.stringify(summary, null, 2));
console.log('\n=== SUMMARY ===');
console.log(JSON.stringify(summary, null, 2));
console.log(`\nartifacts in: ${OUT}/`);
