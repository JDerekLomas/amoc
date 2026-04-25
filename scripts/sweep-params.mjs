#!/usr/bin/env node
/**
 * Parameter sweep using the full GPU simulation (Playwright + Chrome WebGPU)
 *
 * Usage:
 *   node scripts/sweep-params.mjs                          # default sweep
 *   node scripts/sweep-params.mjs --param=evapScale --values="0,0.2,0.4,0.6,0.8"
 *   node scripts/sweep-params.mjs --param=S_solar --values="5,6,6.2,7,8" --spinup=20
 *
 * Runs each parameter value headlessly, measures RMSE, prints comparison table.
 */

import { chromium } from 'playwright';
import { readFileSync, existsSync } from 'fs';
import { createServer } from 'http';
import { resolve } from 'path';

const ROOT = '/Users/dereklomas/lukebarrington/amoc';

// Parse args
const getArg = (name, def) => {
  const a = process.argv.find((_, i, arr) => arr[i - 1] === `--${name}`);
  return a ?? process.argv.find(x => x.startsWith(`--${name}=`))?.split('=')[1] ?? def;
};

const PARAM = getArg('param', 'evapScale');
const VALUES = getArg('values', '0,0.2,0.4,0.6,0.8,1.0').split(',').map(Number);
const SPINUP = parseInt(getArg('spinup', '20'));

// Reference SST
function loadReferenceSST() {
  const sstPath = resolve(ROOT, 'sst_global_1deg.json');
  if (!existsSync(sstPath)) return { '0': 27.6, '30': 22.2, '60': 5.2 };
  const raw = JSON.parse(readFileSync(sstPath, 'utf8'));
  const nx = 360, ny = 160;
  const latBins = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70];
  const ref = {};
  for (const lat of latBins) {
    const j = Math.round((lat + 79.5) / 1.0);
    if (j < 0 || j >= ny) continue;
    let sum = 0, count = 0;
    for (let i = 0; i < nx; i++) {
      const v = raw.sst?.[j * nx + i];
      if (v != null && !isNaN(v) && v > -90) { sum += v; count++; }
    }
    if (count > 0) ref[String(lat)] = +(sum / count).toFixed(1);
  }
  return ref;
}

const REF_SST = loadReferenceSST();

// Static file server
const server = createServer((req, res) => {
  const urlPath = new URL(req.url, 'http://localhost').pathname;
  const filePath = resolve(ROOT, urlPath.replace(/^\//, ''));
  try {
    const data = readFileSync(filePath);
    const ext = filePath.split('.').pop();
    const mime = { html: 'text/html', json: 'application/json', js: 'text/javascript', bin: 'application/octet-stream', png: 'image/png' }[ext] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  } catch { res.writeHead(404); res.end('Not found'); }
});

async function runWithParam(browser, paramName, paramValue) {
  const ctx = await browser.newContext({ viewport: { width: 1200, height: 800 } });
  const page = await ctx.newPage();

  await page.goto('http://localhost:8776/simamoc/index.html', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(4000);
  try { await page.evaluate(() => document.getElementById('btn-start-exploring')?.click()); } catch {}
  await page.waitForTimeout(1000);

  // Set param
  const params = { stepsPerFrame: 100, yearSpeed: 3 };
  params[paramName] = paramValue;
  await page.evaluate((p) => {
    for (const [k, v] of Object.entries(p)) {
      if (k in window) window[k] = v;
      if (typeof lab !== 'undefined') lab.setParams({ [k]: v });
    }
  }, params);

  // Spinup
  await page.waitForTimeout(SPINUP * 1000);

  // Measure RMSE
  const result = await page.evaluate((refSST) => {
    const nx = NX, ny = NY, m = mask, t = temp;
    if (!t || !m) return null;
    let sqSum = 0, nBands = 0, biasSum = 0;
    for (const [latStr, refT] of Object.entries(refSST)) {
      const lat = parseFloat(latStr);
      const j = Math.round((lat - (-80)) / 160 * (ny - 1));
      if (j < 0 || j >= ny) continue;
      let sum = 0, count = 0;
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (m[k] && isFinite(t[k])) { sum += t[k]; count++; }
      }
      if (count > 0) {
        const err = sum / count - refT;
        sqSum += err * err;
        biasSum += err;
        nBands++;
      }
    }

    // Global mean
    let gSum = 0, gCnt = 0;
    for (let k = 0; k < nx * ny; k++) {
      if (m[k] && isFinite(t[k])) { gSum += t[k]; gCnt++; }
    }

    return {
      rmse: nBands > 0 ? Math.sqrt(sqSum / nBands) : 999,
      bias: nBands > 0 ? biasSum / nBands : 999,
      globalSST: gCnt > 0 ? gSum / gCnt : NaN,
      step: totalSteps,
      simYears: +(simTime / T_YEAR).toFixed(2),
    };
  }, REF_SST);

  await ctx.close();
  return result;
}

async function main() {
  console.log(`Parameter sweep: ${PARAM}`);
  console.log(`Values: [${VALUES.join(', ')}]`);
  console.log(`Spinup: ${SPINUP}s per run`);
  console.log('');

  await new Promise(r => server.listen(8776, r));

  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
  });

  const results = [];

  for (const val of VALUES) {
    process.stdout.write(`  ${PARAM}=${val} ... `);
    try {
      const r = await runWithParam(browser, PARAM, val);
      if (r) {
        results.push({ value: val, ...r });
        console.log(`RMSE=${r.rmse.toFixed(2)}°C  bias=${r.bias.toFixed(2)}°C  globalSST=${r.globalSST.toFixed(1)}°C  (${r.simYears}yr, ${r.step} steps)`);
      } else {
        console.log('FAILED (null result)');
      }
    } catch (e) {
      console.log(`ERROR: ${e.message}`);
    }
  }

  await browser.close();
  server.close();

  // Summary table
  console.log('\n' + '='.repeat(70));
  console.log(`SWEEP RESULTS: ${PARAM}`);
  console.log('='.repeat(70));
  console.log(`${'Value'.padStart(8)}  ${'RMSE'.padStart(8)}  ${'Bias'.padStart(8)}  ${'GlobalSST'.padStart(10)}  ${'Steps'.padStart(8)}`);
  console.log('-'.repeat(70));
  let bestRMSE = Infinity, bestVal = null;
  for (const r of results) {
    const marker = r.rmse < bestRMSE ? ' *' : '';
    if (r.rmse < bestRMSE) { bestRMSE = r.rmse; bestVal = r.value; }
    console.log(`${String(r.value).padStart(8)}  ${r.rmse.toFixed(2).padStart(8)}  ${r.bias.toFixed(2).padStart(8)}  ${r.globalSST.toFixed(1).padStart(10)}  ${String(r.step).padStart(8)}${marker}`);
  }
  console.log('-'.repeat(70));
  if (bestVal !== null) console.log(`Best: ${PARAM}=${bestVal} → RMSE=${bestRMSE.toFixed(2)}°C`);
}

main().catch(e => { console.error(e); process.exit(1); });
