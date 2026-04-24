#!/usr/bin/env node
/**
 * Simulation runner for manual parameter tuning.
 * Launches the sim, injects params, spins up, extracts diagnostics + screenshots.
 *
 * Usage:
 *   node tune.mjs [--spinup 120] [--params '{"S_solar":120}']
 *   node tune.mjs --params '{"S_solar":120,"A_olr":35}'
 */

import { chromium } from 'playwright';
import { mkdirSync, writeFileSync, readFileSync } from 'fs';
import { resolve } from 'path';
import { createServer } from 'http';

const ROOT = '/Users/dereklomas/lukebarrington/amoc';
const OUT = './screenshots/tune';
const SPINUP_SECS = parseInt(process.argv.find((_, i, a) => a[i - 1] === '--spinup') || '120');
const ITER_LABEL = process.argv.find((_, i, a) => a[i - 1] === '--label') || 'run';

let userParams = {};
const paramsArg = process.argv.find((_, i, a) => a[i - 1] === '--params');
if (paramsArg) userParams = JSON.parse(paramsArg);

mkdirSync(OUT, { recursive: true });

// Reference data
function loadReferenceData() {
  const sstRaw = JSON.parse(readFileSync(resolve(ROOT, 'sst_global_1deg.json'), 'utf8'));
  const nx = 360, ny = 160;
  const latBins = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70];
  const refSST = {};
  for (const targetLat of latBins) {
    const j = Math.round((targetLat + 79.5) / 1.0);
    if (j < 0 || j >= ny) continue;
    let sum = 0, count = 0;
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const sv = sstRaw.sst?.[k] ?? sstRaw.data?.[k] ?? sstRaw[k];
      if (sv != null && !isNaN(sv) && sv > -90) { sum += sv; count++; }
    }
    if (count > 0) refSST[targetLat] = sum / count;
  }
  let gSum = 0, gCount = 0;
  const data = sstRaw.sst || sstRaw.data || sstRaw;
  for (let k = 0; k < nx * ny; k++) {
    const v = data[k];
    if (v != null && !isNaN(v) && v > -90) { gSum += v; gCount++; }
  }
  return { refSST, globalMeanSST: gSum / gCount };
}

async function main() {
  const refData = loadReferenceData();
  console.log(`Reference global mean SST: ${refData.globalMeanSST.toFixed(1)}°C\n`);

  // Local server
  const server = createServer((req, res) => {
    const urlPath = new URL(req.url, 'http://localhost').pathname;
    const filePath = resolve(ROOT, urlPath.replace(/^\//, ''));
    try {
      const d = readFileSync(filePath);
      const ext = filePath.split('.').pop();
      const mime = { html: 'text/html', json: 'application/json', js: 'text/javascript' }[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': mime }); res.end(d);
    } catch { res.writeHead(404); res.end('Not found'); }
  });
  await new Promise(r => server.listen(8773, r));

  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--disable-gpu-sandbox'],
  });
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 800 } });
  const page = await ctx.newPage();
  await page.goto('http://localhost:8773/simamoc/index.html', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(5000);
  try { await page.evaluate(() => { document.getElementById('btn-start-exploring')?.click(); }); } catch {}
  await page.waitForTimeout(1000);

  const backend = await page.evaluate(() => document.getElementById('backend-badge')?.textContent);
  console.log(`Backend: ${backend}`);

  // Map param keys for lab API
  const labParams = { ...userParams };
  if ('r_friction' in labParams) { labParams.r = labParams.r_friction; delete labParams.r_friction; }
  if ('A_visc' in labParams) { labParams.A = labParams.A_visc; delete labParams.A_visc; }

  // Inject params and run
  // Inject params on running sim (no reset — avoids instability)
  if (Object.keys(labParams).length > 0) {
    console.log(`Injecting params: ${JSON.stringify(userParams)}`);
    await page.evaluate((p) => { lab.setParams(p); }, labParams);
  }
  await page.evaluate(() => {
    lab.setParams({ stepsPerFrame: 50, yearSpeed: 1 });
    showField = 'temp';
  });

  console.log(`Spinning up for ${SPINUP_SECS}s...`);
  await page.waitForTimeout(SPINUP_SECS * 1000);

  // Extract diagnostics
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

    let gSum = 0, gCount = 0, gMin = 999, gMax = -999, nanCount = 0;
    for (let k = 0; k < nx * ny; k++) {
      if (m[k]) {
        if (isNaN(t[k]) || !isFinite(t[k])) { nanCount++; continue; }
        gSum += t[k]; gCount++; gMin = Math.min(gMin, t[k]); gMax = Math.max(gMax, t[k]);
      }
    }

    // Current params
    const params = {
      S_solar, A_olr, B_olr, kappa_diff, alpha_T,
      r_friction, A_visc, gamma_mix, gamma_deep_form, kappa_deep,
      F_couple_s, F_couple_d, r_deep, windStrength, H_surface, H_deep,
      kappa_atm, gamma_oa, gamma_ao, gamma_la,
    };

    return {
      zonalTemp, zonalDeep, params,
      globalMean: gSum / gCount, globalMin: gMin, globalMax: gMax,
      nanCount, oceanCells: gCount,
      step: totalSteps, simTime, simYears: simTime / T_YEAR,
      amoc: typeof amocStrength !== 'undefined' ? amocStrength : 0,
    };
  });

  if (!data) {
    console.log('ERROR: No simulation data');
    await browser.close(); server.close();
    return;
  }

  // Print diagnostics
  console.log(`\nStep: ${data.step} | SimYears: ${data.simYears?.toFixed(1)} | NaN cells: ${data.nanCount}`);
  console.log(`Global mean: ${data.globalMean?.toFixed(1)}°C | Range: [${data.globalMin?.toFixed(1)}, ${data.globalMax?.toFixed(1)}]°C`);
  console.log(`AMOC: ${data.amoc?.toFixed(4)}\n`);

  console.log('Current params:');
  console.log(JSON.stringify(data.params, null, 2));

  // Zonal comparison
  console.log('\nZonal SST comparison:');
  console.log('  Lat    Sim     Obs    Error');
  let totalSE = 0, count = 0;
  const latBins = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70];
  for (const lat of latBins) {
    const sim = data.zonalTemp[lat];
    const obs = refData.refSST[lat];
    if (sim != null && obs != null) {
      const err = sim - obs;
      totalSE += err * err; count++;
      const sign = err >= 0 ? '+' : '';
      console.log(`  ${String(lat).padStart(4)}° ${sim.toFixed(1).padStart(7)} ${obs.toFixed(1).padStart(7)} ${(sign + err.toFixed(1)).padStart(7)}`);
    }
  }
  const rmse = Math.sqrt(totalSE / Math.max(count, 1));
  console.log(`\n  RMSE: ${rmse.toFixed(2)}°C`);

  // Deep temps
  console.log('\nZonal Deep Temp:');
  for (const lat of latBins) {
    const d = data.zonalDeep[lat];
    if (d != null) console.log(`  ${String(lat).padStart(4)}° ${d.toFixed(1).padStart(7)}°C`);
  }

  // Screenshots
  const prefix = `${OUT}/${ITER_LABEL}`;
  await page.screenshot({ path: `${prefix}-temp.png` });
  await page.evaluate(() => { showField = 'psi'; });
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${prefix}-psi.png` });
  await page.evaluate(() => { showField = 'speed'; });
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${prefix}-speed.png` });
  await page.evaluate(() => { showField = 'deeptemp'; });
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${prefix}-deep.png` });
  await page.evaluate(() => { showField = 'clouds'; });
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${prefix}-clouds.png` });
  await page.evaluate(() => { showField = 'airtemp'; });
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${prefix}-airtemp.png` });

  console.log(`\nScreenshots saved to ${OUT}/${ITER_LABEL}-*.png`);

  // Save report
  const report = { ...data, rmse, refSST: refData.refSST, userParams };
  writeFileSync(`${prefix}-report.json`, JSON.stringify(report, null, 2));

  await browser.close();
  server.close();
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
