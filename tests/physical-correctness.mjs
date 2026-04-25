#!/usr/bin/env node
/**
 * tests/physical-correctness.mjs
 *
 * Tests that should ALWAYS pass for the simamoc model. They drive the sim
 * via sim-control's HTTP API and check invariants of the physics, not just
 * "does it run." If any of these fail, the model has regressed in a way
 * that visual inspection wouldn't have caught.
 *
 * Run:
 *   1. Start sim-control: node sim-control.mjs &
 *   2. Run tests:        node tests/physical-correctness.mjs
 *
 * Each test is a function that returns {pass: bool, message: string}.
 */

const API = 'http://localhost:8775';

async function call(path, method = 'GET', body = null, timeoutMs = 600000) {
  const opts = { method };
  if (body) opts.body = typeof body === 'string' ? body : JSON.stringify(body);
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  opts.signal = ac.signal;
  try {
    const r = await fetch(API + path, opts);
    return r.json();
  } finally {
    clearTimeout(timer);
  }
}
const status = () => call('/status');
const reset = () => call('/reset', 'POST');
const setParams = (p) => call('/params', 'POST', p);
const step = (n, chunk = 30) => call(`/step?n=${n}&chunk=${chunk}`);
const eval_ = (code) => call('/eval', 'POST', code).then(r => r.ok ? JSON.parse(r.value) : { error: r.error });

// ─── Tests ─────────────────────────────────────────────────────────────────

async function test_orientation() {
  // After loading SST data, equator must be > 5°C warmer than both poles.
  const r = await eval_(`
    (() => {
      const d = obsSSTData;
      if (!d || !d.sst || !d.nx || !d.ny) return { error: 'no SST data' };
      const nx = d.nx, ny = d.ny, sst = d.sst;
      const avg = (j) => { let s=0,c=0; for(let i=0;i<nx;i++){const v=sst[j*nx+i];if(isFinite(v)&&v>-90){s+=v;c++;}} return c?s/c:NaN; };
      return { south: avg(5), eq: avg(Math.floor(ny/2)), north: avg(ny-6) };
    })()
  `);
  if (r.error) return { pass: false, message: r.error };
  const { south, eq, north } = r;
  const ok = (eq > south + 5) && (eq > north + 5);
  return {
    pass: ok,
    message: `SST south=${south.toFixed(1)} eq=${eq.toFixed(1)} north=${north.toFixed(1)} — ` +
      (ok ? 'orientation correct (equator >5°C warmer than poles)'
          : 'ORIENTATION WRONG! Check flipVerticalFloat32 in loadBinData'),
  };
}

async function test_coriolis_sign() {
  // f = 2Ω·sin(lat). At lat=+45°, f should be positive; at lat=-45°, negative.
  // The model's beta = df/dy = 2Ω·cos(lat)/R. β should be positive everywhere
  // (cos(lat) ≥ 0 for both hemispheres). Sign flip in either is a hemisphere bug.
  const r = await eval_(`
    (() => {
      // Probe internal lat formula by computing what model uses at each j
      const checkJ = [Math.floor(NY*0.05), Math.floor(NY*0.5), Math.floor(NY*0.95)];
      const lats = checkJ.map(j => LAT0 + (j/(NY-1))*(LAT1-LAT0));
      return { lats, NY, LAT0, LAT1 };
    })()
  `);
  if (r.error) return { pass: false, message: r.error };
  const { lats } = r;
  // Expected: small j → south (negative lat), middle → equator, large j → north (positive lat)
  const ok = (lats[0] < -50) && (Math.abs(lats[1]) < 10) && (lats[2] > 50);
  return {
    pass: ok,
    message: `lat formula at j=[5%, 50%, 95%] = [${lats[0].toFixed(1)}°, ${lats[1].toFixed(1)}°, ${lats[2].toFixed(1)}°] — ` +
      (ok ? 'south-up convention correct' : 'LAT FORMULA WRONG (south-up expected)'),
  };
}

async function test_no_nan_after_short_spinup() {
  // After 100 steps from a fresh state, no cell should be NaN.
  await reset();
  await step(100, 30);
  const s = await status();
  return {
    pass: s.nanCells === 0,
    message: `${s.nanCells} NaN cells after 100 steps (range=[${s.rangeSST[0]},${s.rangeSST[1]}])`,
  };
}

async function test_sst_range_physical() {
  // After 200 steps, no SST should escape [-3, +40]°C. The clamps should hold.
  await reset();
  await step(200, 30);
  const s = await status();
  const [lo, hi] = s.rangeSST;
  const ok = lo >= -3 && hi <= 40;
  return {
    pass: ok,
    message: `range=[${lo.toFixed(1)}, ${hi.toFixed(1)}]°C after 200 steps — ` +
      (ok ? 'within physical bounds' : 'RANGE ESCAPED clamps; clamping logic broken'),
  };
}

// Shared spinup state: cooling_drift + amoc_sign both need the same 500-step run.
let _spinupCache = null;
async function _runSpinupOnce() {
  if (_spinupCache) return _spinupCache;
  await reset();
  const before = await status();
  await step(500, 50);
  const after = await status();
  _spinupCache = { before, after };
  return _spinupCache;
}

async function test_cooling_drift_acceptable() {
  // Demo-quality requirement: drift < 0.1°C per 100 steps. Anything more means
  // a regression in heat budget (e.g., the evapCool dt bug returning).
  const { before, after } = await _runSpinupOnce();
  const driftPer100 = (after.globalMeanSST - before.globalMeanSST) / 5;
  const ok = Math.abs(driftPer100) < 0.1;
  return {
    pass: ok,
    message: `drift = ${driftPer100.toFixed(3)}°C per 100 steps — ` +
      (ok ? 'within 0.1°C tolerance' : 'DRIFT TOO HIGH; check CPU/GPU divergence (evapCool dt bug?)'),
  };
}

async function test_amoc_sign_positive() {
  // After spinup, AMOC should be positive (northward upper-layer flow).
  // A reversed AMOC is a hemisphere/orientation bug.
  const { after } = await _runSpinupOnce();
  return {
    pass: after.amoc > 0,
    message: `AMOC = ${after.amoc.toExponential(3)} after 500 steps — ` +
      (after.amoc > 0 ? 'positive (correct sign)' : 'NEGATIVE (likely Coriolis/orientation bug)'),
  };
}

async function test_data_not_uniformly_zero() {
  // Sanity: loaded fields must not all be zero (data file fetch failure).
  // Sample widely (every 1000 cells) since the first 100 cells are pole/land
  // where temp[k] = 0 is normal.
  const r = await eval_(`
    (() => {
      const sample = (arr, oceanOnly) => {
        if (!arr) return false;
        for (let k = 0; k < arr.length; k += 1000) {
          if (oceanOnly && !mask[k]) continue;
          if (Math.abs(arr[k]) > 0.01) return true;
        }
        return false;
      };
      return {
        temp: sample(temp, true),
        sal: sample(sal, true),
        depth: sample(depth, true),
        mask: mask ? Array.from(mask.slice(1000, 100000)).some(v => v) : false,
      };
    })()
  `);
  if (r.error) return { pass: false, message: r.error };
  const allOk = Object.values(r).every(v => v);
  return {
    pass: allOk,
    message: `field sanity: ${JSON.stringify(r)} — ${allOk ? 'all loaded' : 'AT LEAST ONE FIELD MISSING/ZERO'}`,
  };
}

// ─── Runner ────────────────────────────────────────────────────────────────

const tests = [
  { name: 'data_not_uniformly_zero', fn: test_data_not_uniformly_zero, slow: false },
  { name: 'coriolis_sign',           fn: test_coriolis_sign,           slow: false },
  { name: 'orientation',             fn: test_orientation,             slow: false },
  { name: 'no_nan_short_spinup',     fn: test_no_nan_after_short_spinup, slow: true },
  { name: 'sst_range_physical',      fn: test_sst_range_physical,      slow: true },
  { name: 'amoc_sign_positive',      fn: test_amoc_sign_positive,      slow: true },
  { name: 'cooling_drift_acceptable',fn: test_cooling_drift_acceptable,slow: true },
];

async function main() {
  console.log(`Running ${tests.length} physical-correctness tests against ${API}\n`);
  let passed = 0, failed = 0;
  for (const t of tests) {
    const t0 = Date.now();
    let result;
    try { result = await t.fn(); }
    catch (e) { result = { pass: false, message: `THREW: ${e.message}` }; }
    const ms = Date.now() - t0;
    const tag = result.pass ? '\x1b[32m PASS \x1b[0m' : '\x1b[31m FAIL \x1b[0m';
    console.log(`[${tag}] ${t.name.padEnd(28)} (${(ms/1000).toFixed(1)}s)  ${result.message}`);
    if (result.pass) passed++; else failed++;
  }
  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
