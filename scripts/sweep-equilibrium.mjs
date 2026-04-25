#!/usr/bin/env node
/**
 * sweep-equilibrium.mjs — find a parameter set that holds equilibrium.
 *
 * Strategy: for each candidate {S_solar, A_olr}, reset the sim, run a short
 * spinup, and measure (a) the mean SST drift rate and (b) the AMOC sign.
 * Print a table; the user picks the winner.
 *
 * Requires sim-control running on :8775.
 */

const API = 'http://localhost:8775';

async function call(path, method = 'GET', body = null) {
  const opts = { method };
  if (body) opts.body = typeof body === 'string' ? body : JSON.stringify(body);
  const r = await fetch(API + path, opts);
  return r.json();
}
async function post(path, body) { return call(path, 'POST', body); }
async function status() { return call('/status'); }
async function reset() { return post('/reset'); }
async function setParams(p) { return post('/params', p); }
async function step(n, chunk = 30) { return call(`/step?n=${n}&chunk=${chunk}`); }

async function main() {
  console.log('# Sweep table: each row resets sim, applies params, runs N steps, reports drift\n');
  // Note: /reset reinitializes the sim back to the observation-matched state.

  const SPINUP = 500;
  // Candidate parameter combos. Start with S_solar (currently 6.5).
  // Net cooling ≈ 0.16°C / 1000 steps; need qNet to roughly double.
  const grid = [];
  for (const sSolar of [6.5, 9, 12, 15, 18]) {
    for (const aOlr of [1.8, 1.4, 1.0]) {
      grid.push({ S_solar: sSolar, A_olr: aOlr });
    }
  }

  console.log(`# ${grid.length} configs, ${SPINUP} steps each (~${grid.length * SPINUP * 0.4}s estimated)\n`);
  console.log('S_solar A_olr | startMean afterMean drift°C  startAMOC afterAMOC | range[min,max]');
  console.log('---------------+-------------------------+----------------------+---------------');

  for (const params of grid) {
    await reset();
    await setParams(params);
    const before = await status();
    await step(SPINUP, 30);
    const after = await status();
    const drift = after.globalMeanSST - before.globalMeanSST;
    const row = `${String(params.S_solar).padStart(7)} ${String(params.A_olr).padStart(5)} | ` +
      `${before.globalMeanSST.toFixed(2).padStart(9)} ${after.globalMeanSST.toFixed(2).padStart(9)} ${drift.toFixed(2).padStart(7)} | ` +
      `${before.amoc.toExponential(2).padStart(9)} ${after.amoc.toExponential(2).padStart(9)} | ` +
      `[${after.rangeSST[0].toFixed(1)}, ${after.rangeSST[1].toFixed(1)}]`;
    console.log(row);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
