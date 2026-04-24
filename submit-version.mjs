#!/usr/bin/env node
/**
 * Submit a new version of SimAMOC to the leaderboard.
 *
 * Snapshots the current simamoc/ code, runs evaluation via lab API,
 * records scores, and updates the leaderboard.
 *
 * Usage:
 *   node submit-version.mjs --author "Derek" --name "Regime clouds + tuned radiative" --description "..."
 *   node submit-version.mjs --author "Luke" --name "Spectral winds" --spinup 30
 *
 * Options:
 *   --author      Author name (required)
 *   --name        Short version name (required)
 *   --description Longer description
 *   --params      JSON object of param overrides (e.g. '{"S_solar":7}')
 *   --spinup      Spinup time in seconds (default: 30, OOM-safe)
 */

import { chromium } from 'playwright';
import { mkdirSync, writeFileSync, readFileSync, copyFileSync, existsSync } from 'fs';
import { createServer } from 'http';
import { resolve } from 'path';

const ROOT = '/Users/dereklomas/lukebarrington/amoc';
const SPINUP = parseInt(process.argv.find((_, i, a) => a[i - 1] === '--spinup') || '30');
const AUTHOR = process.argv.find((_, i, a) => a[i - 1] === '--author') || 'Unknown';
const NAME = process.argv.find((_, i, a) => a[i - 1] === '--name') || 'Unnamed';
const DESC = process.argv.find((_, i, a) => a[i - 1] === '--description') || '';
const PARAMS_JSON = process.argv.find((_, i, a) => a[i - 1] === '--params') || '{}';
const userParams = JSON.parse(PARAMS_JSON);

// Version ID
const now = new Date();
const ts = now.toISOString().replace(/[-:T]/g, '').slice(0, 15);
const slug = AUTHOR.toLowerCase().replace(/[^a-z0-9]/g, '');
const VERSION_ID = `${slug}-${ts}`;
const VERSION_DIR = `versions/${VERSION_ID}`;

// Reference SST (NOAA OI SST v2, 1991-2020 long-term mean)
// Loaded from actual 1-degree data for 15 latitude bands
function loadReferenceSST() {
  const sstPath = resolve(ROOT, 'sst_global_1deg.json');
  if (!existsSync(sstPath)) {
    // Fallback to approximate values
    return {
      '-70': -1.5, '-60': 1.3, '-50': 6.8, '-40': 15.1, '-30': 20.9,
      '-20': 24.7, '-10': 26.9, '0': 27.6, '10': 27.8, '20': 26.1,
      '30': 22.2, '40': 15.6, '50': 9.0, '60': 5.2, '70': 1.0,
    };
  }
  const raw = JSON.parse(readFileSync(sstPath, 'utf8'));
  const nx = 360, ny = 160;
  const latBins = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70];
  const ref = {};
  for (const lat of latBins) {
    const j = Math.round((lat + 79.5) / 1.0);
    if (j < 0 || j >= ny) continue;
    let sum = 0, count = 0;
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const v = raw.sst?.[k] ?? raw.data?.[k] ?? raw[k];
      if (v != null && !isNaN(v) && v > -90) { sum += v; count++; }
    }
    if (count > 0) ref[String(lat)] = +(sum / count).toFixed(1);
  }
  return ref;
}

// Map TUNABLE_PARAMS keys to lab API keys
function toLabKeys(params) {
  const p = { ...params };
  if ('r_friction' in p) { p.r = p.r_friction; delete p.r_friction; }
  if ('A_visc' in p) { p.A = p.A_visc; delete p.A_visc; }
  return p;
}

async function main() {
  console.log('SimAMOC Version Submission');
  console.log(`  Author:  ${AUTHOR}`);
  console.log(`  Name:    ${NAME}`);
  console.log(`  Version: ${VERSION_ID}`);
  console.log(`  Spinup:  ${SPINUP}s`);
  if (Object.keys(userParams).length) console.log(`  Params:  ${JSON.stringify(userParams)}`);
  console.log('');

  const REF_SST = loadReferenceSST();

  // 1. Snapshot current simulation code
  mkdirSync(VERSION_DIR, { recursive: true });
  const filesToCopy = [
    'index.html', 'model.js', 'gpu-solver.js', 'renderer.js',
    'main.js', 'ui.js', 'overlay.js', 'input-widget.js',
    'mask.json', 'coastlines.json',
  ];
  for (const f of filesToCopy) {
    const src = `simamoc/${f}`;
    if (existsSync(src)) copyFileSync(src, `${VERSION_DIR}/${f}`);
  }
  console.log(`  Snapshot saved to ${VERSION_DIR}/`);

  // 2. Launch browser and run evaluation
  const server = createServer((req, res) => {
    const urlPath = new URL(req.url, 'http://localhost').pathname;
    const filePath = resolve(ROOT, urlPath.replace(/^\//, ''));
    try {
      const data = readFileSync(filePath);
      const ext = filePath.split('.').pop();
      const mime = { html: 'text/html', json: 'application/json', js: 'text/javascript' }[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': mime }); res.end(data);
    } catch { res.writeHead(404); res.end('Not found'); }
  });
  await new Promise(r => server.listen(8775, r));

  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
  });
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 800 } });
  const page = await ctx.newPage();

  await page.goto('http://localhost:8775/simamoc/index.html', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(5000);
  try { await page.evaluate(() => { document.getElementById('btn-start-exploring')?.click(); }); } catch {}
  await page.waitForTimeout(1000);

  const backend = await page.evaluate(() => document.getElementById('backend-badge')?.textContent);
  console.log(`  Backend: ${backend}`);

  // Inject params via lab API
  if (Object.keys(userParams).length > 0) {
    const labParams = toLabKeys(userParams);
    await page.evaluate((p) => { lab.setParams(p); }, labParams);
  }

  // Set speed and run
  await page.evaluate(() => {
    lab.setParams({ stepsPerFrame: 100, yearSpeed: 3 });
    showField = 'temp';
  });

  console.log(`  Spinning up for ${SPINUP}s...`);
  await page.waitForTimeout(SPINUP * 1000);

  // 3. Extract diagnostics
  const data = await page.evaluate(() => {
    const nx = NX, ny = NY;
    const m = mask, t = temp;
    const dt = typeof deepTemp !== 'undefined' ? deepTemp : null;
    if (!t || !m) return null;

    const zonalTemp = {}, zonalDeep = {};
    const latBins = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70];
    for (const targetLat of latBins) {
      const j = Math.round((targetLat - (-80)) / 160 * (ny - 1));
      if (j < 0 || j >= ny) continue;
      let sum = 0, dsum = 0, count = 0;
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (m[k] && isFinite(t[k])) { sum += t[k]; if (dt && isFinite(dt[k])) dsum += dt[k]; count++; }
      }
      if (count > 0) {
        zonalTemp[targetLat] = sum / count;
        zonalDeep[targetLat] = dt ? dsum / count : null;
      }
    }

    let gSum = 0, gCount = 0, tSum = 0, tCount = 0, pSum = 0, pCount = 0, ice = 0;
    for (let k = 0; k < nx * ny; k++) {
      if (!m[k] || !isFinite(t[k])) continue;
      gSum += t[k]; gCount++;
      if (t[k] < -1.5) ice++;
      const j = Math.floor(k / nx);
      const lat = -80 + (j / (ny - 1)) * 160;
      if (Math.abs(lat) < 20) { tSum += t[k]; tCount++; }
      if (Math.abs(lat) > 60) { pSum += t[k]; pCount++; }
    }

    // Read actual current params from model globals
    const params = {};
    try {
      params.S_solar = S_solar; params.A_olr = A_olr; params.B_olr = B_olr;
      params.kappa_diff = kappa_diff; params.alpha_T = alpha_T;
      params.r_friction = r_friction; params.A_visc = A_visc;
      params.gamma_mix = gamma_mix; params.gamma_deep_form = gamma_deep_form;
      params.kappa_deep = kappa_deep; params.windStrength = windStrength;
      params.F_couple_s = F_couple_s; params.r_deep = r_deep;
    } catch {}

    return {
      zonalTemp, zonalDeep, params,
      globalSST: gSum / gCount,
      tropicalSST: tCount > 0 ? tSum / tCount : NaN,
      polarSST: pCount > 0 ? pSum / pCount : NaN,
      iceArea: ice,
      amoc: typeof amocStrength !== 'undefined' ? amocStrength : 0,
      step: totalSteps,
      simYears: simTime / T_YEAR,
    };
  });

  if (!data) {
    console.log('  ERROR: No simulation data');
    await browser.close(); server.close();
    return;
  }

  // Screenshots
  const views = ['temp', 'psi', 'speed', 'deeptemp', 'clouds', 'airtemp'];
  for (const view of views) {
    await page.evaluate((v) => { showField = v; }, view);
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${VERSION_DIR}/${view}.png` });
  }
  await page.evaluate(() => { showField = 'temp'; });
  await page.screenshot({ path: `${VERSION_DIR}/thumbnail.png` });

  await browser.close();
  server.close();

  // 4. Compute RMSE
  let sqSum = 0, nBands = 0;
  const errors = [];
  for (const [latStr, refT] of Object.entries(REF_SST)) {
    const lat = parseFloat(latStr);
    const simT = data.zonalTemp[lat];
    if (simT != null && isFinite(simT)) {
      const err = simT - refT;
      errors.push({ lat, simT: +simT.toFixed(1), refT, error: +err.toFixed(1) });
      sqSum += err * err;
      nBands++;
    }
  }
  const rmse = nBands > 0 ? Math.sqrt(sqSum / nBands) : 999;

  // 5. Build version metadata
  const version = {
    id: VERSION_ID,
    author: AUTHOR,
    name: NAME,
    description: DESC,
    date: now.toISOString(),
    params: data.params,
    rmse: +rmse.toFixed(2),
    globalSST: +data.globalSST.toFixed(1),
    tropicalSST: +data.tropicalSST.toFixed(1),
    polarSST: +data.polarSST.toFixed(1),
    amoc: +(data.amoc || 0).toFixed(4),
    iceArea: data.iceArea || 0,
    simYears: +data.simYears.toFixed(1),
    spinupSecs: SPINUP,
    errors,
    path: VERSION_DIR,
  };

  writeFileSync(`${VERSION_DIR}/metadata.json`, JSON.stringify(version, null, 2));

  // 6. Update leaderboard
  const scoresPath = 'versions/scores.json';
  let scores = [];
  if (existsSync(scoresPath)) {
    scores = JSON.parse(readFileSync(scoresPath, 'utf8'));
  }
  scores.push(version);
  scores.sort((a, b) => a.rmse - b.rmse);
  writeFileSync(scoresPath, JSON.stringify(scores, null, 2));

  // 7. Print results
  console.log(`\n  RMSE:     ${rmse.toFixed(2)}°C`);
  console.log(`  Global:   ${data.globalSST.toFixed(1)}°C`);
  console.log(`  Tropical: ${data.tropicalSST.toFixed(1)}°C`);
  console.log(`  Polar:    ${data.polarSST.toFixed(1)}°C`);
  console.log(`  AMOC:     ${data.amoc.toFixed(4)}`);
  console.log(`  SimYears: ${data.simYears.toFixed(1)}`);

  console.log(`\n  Zonal errors:`);
  console.log('    Lat    Sim     Obs    Error');
  for (const e of errors) {
    const sign = e.error >= 0 ? '+' : '';
    console.log(`    ${String(e.lat).padStart(4)}° ${e.simT.toFixed(1).padStart(7)} ${String(e.refT).padStart(7)} ${(sign + e.error.toFixed(1)).padStart(7)}`);
  }

  console.log(`\n  Leaderboard:`);
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    const me = s.id === VERSION_ID ? ' <-- NEW' : '';
    console.log(`    ${String(i+1).padStart(2)}. ${s.rmse.toFixed(2)}°C  ${s.author.padEnd(8)} ${s.name}${me}`);
  }
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
