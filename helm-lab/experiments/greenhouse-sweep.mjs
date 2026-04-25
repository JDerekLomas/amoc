// experiments/greenhouse-sweep.mjs
// ----------------------------------------------------------------------
// Sweep radiative forcing (globalTempOffset) — analogous to changing CO₂.
// Measures the equilibrium climate response: SST, ice cover, AMOC, the
// equator-pole gradient.
//
// Method:
//   1. Reset, spin up at neutral forcing for 400k steps (~2 model years).
//   2. For each ΔF in [-8, -4, -2, 0, +2, +4, +8] °C of equivalent
//      radiative forcing: set, dwell 80k steps, snapshot diag + a temp
//      frame.
//   3. Save jsonl + composed sweep sheet.
//
// Sign convention: positive globalTempOffset = greenhouse (warming).

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

const OUT = resolve('helm-lab/runs/greenhouse-sweep');
await mkdir(join(OUT, 'frames'), { recursive: true });

const FORCINGS = [-8, -4, -2, 0, 2, 4, 8];
const SPINUP   = 400000;
const DWELL    = 80000;
const VIEWS    = ['temp'];

console.log(`=== greenhouse-sweep ===
out:    ${OUT}
spinup: ${SPINUP.toLocaleString()} steps
dwell:  ${DWELL.toLocaleString()} steps per forcing
forcings: ${FORCINGS.map(f => (f > 0 ? '+' : '') + f + '°C').join(', ')}
total:  ${(SPINUP + FORCINGS.length * DWELL).toLocaleString()} steps
`);

const tStart = Date.now();

console.log('[1/3] reset + spin-up at neutral forcing...');
await rpc('reset');
await rpc('setParams', [{
  windStrength: 1.0, r: 0.04, A: 2e-4,
  freshwaterForcing: 0.0, globalTempOffset: 0.0,
  yearSpeed: 1.0, stepsPerFrame: 50,
}]);
const SPIN_CHUNK = 50000;
for (let done = 0; done < SPINUP; done += SPIN_CHUNK) {
  const n = Math.min(SPIN_CHUNK, SPINUP - done);
  await rpc('step', [n]);
  const d = await rpc('diag', [{}]);
  console.log(`     ${(done + n).toString().padStart(6)} / ${SPINUP}  simYr=${d.simYears.toFixed(2)} T_glob=${d.globalSST.toFixed(2)}° AMOC=${d.amoc.toFixed(4)} ice=${d.iceArea}`);
}
const baseDiag = await rpc('diag', [{}]);

console.log(`\n[2/3] forcing sweep (${FORCINGS.length} values × ${DWELL} steps)...`);
const points = [];
for (const F of FORCINGS) {
  const t0 = Date.now();
  await rpc('setParams', [{ globalTempOffset: F }]);
  await rpc('step', [DWELL]);
  const d = await rpc('diag', [{}]);
  const tag = `dT${F >= 0 ? '+' : ''}${F.toString().replace('.', 'p')}`;
  const fp = join(OUT, 'frames', `${tag}_temp.png`);
  await rpc('render', [fp, { view: 'temp' }]);
  points.push({ forcing: F, frames: { temp: fp }, diag: d });
  const dt = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(
    `  ΔT=${F >= 0 ? '+' : ''}${F.toString().padStart(2)}°C  ` +
    `T_glob=${d.globalSST.toFixed(2)}°  T_trop=${d.tropicalSST.toFixed(2)}°  ` +
    `T_pol=${d.polarSST.toFixed(2)}°  AMOC=${d.amoc.toFixed(4)}  ` +
    `ice=${d.iceArea.toString().padStart(5)}  (${dt}s)`
  );
}

console.log('\n[3/3] writing artifacts...');
const stripArr = d => { const { zonalMeanT, zonalMeanPsi, zonalMeanU, latitudes, ...r } = d; return r; };
await writeFile(join(OUT, 'samples.json'), JSON.stringify(
  points.map(p => ({ forcing: p.forcing, frames: p.frames, diag: stripArr(p.diag) })), null, 2));
await writeFile(join(OUT, 'response.jsonl'), points.map(p => JSON.stringify({
  forcing: p.forcing, globalSST: p.diag.globalSST, tropicalSST: p.diag.tropicalSST,
  polarSST: p.diag.polarSST, amoc: p.diag.amoc, KE: p.diag.KE,
  iceArea: p.diag.iceArea, simYears: p.diag.simYears,
})).join('\n') + '\n');

// Climate sensitivity: ΔT_global per °C forcing (a slope estimate)
// Derived from the linear fit of globalSST vs forcing.
const xs = points.map(p => p.forcing);
const ys = points.map(p => p.diag.globalSST);
const n = xs.length;
const xbar = xs.reduce((a, b) => a + b) / n;
const ybar = ys.reduce((a, b) => a + b) / n;
const num = xs.reduce((a, x, i) => a + (x - xbar) * (ys[i] - ybar), 0);
const den = xs.reduce((a, x) => a + (x - xbar) ** 2, 0);
const slope = den > 0 ? num / den : null;

const summary = {
  baseline_global_SST: baseDiag.globalSST,
  baseline_AMOC: baseDiag.amoc,
  endpoints: {
    coldest: points.find(p => p.forcing === Math.min(...FORCINGS)).diag,
    warmest: points.find(p => p.forcing === Math.max(...FORCINGS)).diag,
  },
  climate_sensitivity_slope: slope,   // °C global SST per unit globalTempOffset
  AMOC_at_max_warming: points[points.length - 1].diag.amoc,
  AMOC_at_max_cooling: points[0].diag.amoc,
  ice_at_max_cooling: points[0].diag.iceArea,
  ice_at_max_warming: points[points.length - 1].diag.iceArea,
  steps_total: SPINUP + FORCINGS.length * DWELL,
  wall_clock_minutes: ((Date.now() - tStart) / 60000).toFixed(2),
};
await writeFile(join(OUT, 'summary.json'), JSON.stringify(summary, null, 2));
console.log('\n=== SUMMARY ===');
console.log(JSON.stringify(summary, null, 2));
console.log(`\nartifacts in: ${OUT}/`);
