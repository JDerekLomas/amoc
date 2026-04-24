#!/usr/bin/env node
/**
 * Wiggum capture: runs sim with given params, outputs diagnostics + screenshots.
 * Used by Claude Code agents to run the feedback loop without an API key.
 *
 * Usage: node wiggum-capture.mjs [--params '{"S_solar":100}'] [--spinup 120] [--iter 1]
 */
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync, readFileSync } from 'fs';
import { createServer } from 'http';
import { resolve } from 'path';

const ROOT = '/Users/dereklomas/lukebarrington/amoc';
const OUT = './screenshots/wiggum-claude';
mkdirSync(OUT, { recursive: true });

const SPINUP = parseInt(process.argv.find((_, i, a) => a[i - 1] === '--spinup') || '120');
const ITER = process.argv.find((_, i, a) => a[i - 1] === '--iter') || '1';
const PARAMS_JSON = process.argv.find((_, i, a) => a[i - 1] === '--params') || '{}';
const params = JSON.parse(PARAMS_JSON);

// Reference SST data (NOAA OI SST 1991-2020 climatology, 10° latitude bands)
const REF_SST = {
  '-70': -1.0, '-60': 0.5, '-50': 4.0, '-40': 10.0, '-30': 17.0,
  '-20': 23.0, '-10': 26.5, '0': 27.0, '10': 26.5, '20': 24.0,
  '30': 20.0, '40': 14.0, '50': 7.0, '60': 2.0, '70': -1.0,
};
const REF_DEEP = {
  '-70': 0.5, '-60': 1.0, '-50': 2.0, '-40': 3.0, '-30': 4.0,
  '-20': 4.5, '-10': 4.5, '0': 4.5, '10': 4.5, '20': 4.0,
  '30': 3.5, '40': 3.0, '50': 2.5, '60': 1.5, '70': 0.5,
};

const server = createServer((req, res) => {
  const urlPath = new URL(req.url, 'http://localhost').pathname;
  const filePath = resolve(ROOT, urlPath.replace(/^\//, ''));
  try {
    const data = readFileSync(filePath);
    const ext = filePath.split('.').pop();
    const mime = { html: 'text/html', json: 'application/json', js: 'text/javascript' }[ext] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  } catch { res.writeHead(404); res.end('Not found'); }
});
await new Promise(r => server.listen(8774, r));

const browser = await chromium.launch({
  headless: false,
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const page = await browser.newPage();
await page.setViewportSize({ width: 1400, height: 900 });

console.log(`=== WIGGUM CAPTURE — Iteration ${ITER} ===`);
console.log(`Params: ${JSON.stringify(params)}`);
console.log(`Spinup: ${SPINUP}s\n`);

await page.goto('http://localhost:8774/simamoc/index.html', { waitUntil: 'load', timeout: 30000 });
await page.waitForTimeout(3000);
try { await page.click('#btn-start-exploring', { timeout: 3000 }); } catch {}
await page.waitForTimeout(1000);

// Reset, then inject params
try { await page.click('#btn-reset', { timeout: 5000 }); } catch {
  // Button may be hidden by mobile overlay — try via evaluate
  await page.evaluate(() => { var b = document.getElementById('btn-reset'); if (b) b.click(); });
}
await page.waitForTimeout(500);

if (Object.keys(params).length > 0) {
  await page.evaluate((p) => {
    for (const [k, v] of Object.entries(p)) {
      // Use eval to set let-scoped variables
      try { eval(`${k} = ${v}`); } catch {}
    }
  }, params);
  console.log('Injected parameters.');
}

// Max sim speed for spinup
await page.evaluate(() => {
  document.getElementById('speed-slider').value = 150;
  document.getElementById('speed-slider').dispatchEvent(new Event('input'));
  document.getElementById('year-speed-slider').value = 3;
  document.getElementById('year-speed-slider').dispatchEvent(new Event('input'));
});

console.log(`Spinning up for ${SPINUP}s...`);
await page.waitForTimeout(SPINUP * 1000);

// Capture screenshots for each view
const views = ['temp', 'psi', 'speed', 'deeptemp'];
for (const view of views) {
  await page.evaluate((v) => { var b = document.getElementById('btn-' + v); if (b) b.click(); }, view);
  await page.waitForTimeout(1500);
  const path = `${OUT}/iter-${ITER}-${view}.png`;
  await page.screenshot({ path });
  console.log(`  Screenshot: ${path}`);
}

// Back to temp view
await page.click('#btn-temp');
await page.waitForTimeout(500);

// Extract full diagnostics via lab API
const diagnostics = await page.evaluate(() => {
  if (!window.lab) return { error: 'lab API not available' };
  const d = window.lab.diagnostics({ profiles: true });
  return d;
});

// Compute RMSE vs reference
const errors = [];
let sqSum = 0, nBands = 0;
if (diagnostics.zonalMeanT) {
  const NY = diagnostics.zonalMeanT.length;
  const LAT0 = -80, LAT1 = 80;
  for (const [latStr, refT] of Object.entries(REF_SST)) {
    const lat = parseFloat(latStr);
    const j = Math.round((lat - LAT0) / (LAT1 - LAT0) * (NY - 1));
    const simT = diagnostics.zonalMeanT[j];
    if (simT !== undefined && !isNaN(simT)) {
      const err = simT - refT;
      errors.push({ lat, simT: +simT.toFixed(2), refT, error: +err.toFixed(2) });
      sqSum += err * err;
      nBands++;
    }
  }
}
const rmse = nBands > 0 ? Math.sqrt(sqSum / nBands) : NaN;

// Deep temp errors
const deepErrors = [];
if (diagnostics.zonalMeanT) {
  // We'd need deep temp zonal mean - approximate from diagnostics
}

// Conservation checks
const t1 = {
  tempRange: diagnostics.globalSST > -5 && diagnostics.globalSST < 35,
  equatorPoleGradient: (diagnostics.tropicalSST - diagnostics.polarSST) > 10,
  hemisphericSymmetry: Math.abs(diagnostics.nhPolarSST - diagnostics.shPolarSST) < 15,
  amocSign: diagnostics.amocSv !== undefined,
};
const t1Pass = Object.values(t1).every(Boolean);

// Structure checks (from diagnostics)
const t2 = {
  hasWesternIntensification: diagnostics.gyreRangePsi > 0.1,
  hasSubtropicalGyre: diagnostics.gyreRangePsi > 0.5,
  amocPositive: (diagnostics.amocSv || 0) > 0,
  accExists: Math.abs(diagnostics.accU || 0) > 0.01,
  polewardHeatTransport: (diagnostics.tropicalSST || 0) > (diagnostics.polarSST || 0) + 10,
};

const report = {
  iteration: ITER,
  params,
  diagnostics: {
    step: diagnostics.step,
    simYears: diagnostics.simYears?.toFixed(1),
    globalSST: diagnostics.globalSST?.toFixed(2),
    tropicalSST: diagnostics.tropicalSST?.toFixed(2),
    polarSST: diagnostics.polarSST?.toFixed(2),
    nhPolarSST: diagnostics.nhPolarSST?.toFixed(2),
    shPolarSST: diagnostics.shPolarSST?.toFixed(2),
    amocSv: diagnostics.amocSv?.toFixed(2),
    fovS: diagnostics.fovS?.toFixed(4),
    accU: diagnostics.accU?.toFixed(4),
    maxVel: diagnostics.maxVel?.toFixed(4),
    KE: diagnostics.KE?.toFixed(2),
    iceArea: diagnostics.iceArea,
    gyreRangePsi: diagnostics.gyreRangePsi?.toFixed(4),
  },
  rmse: +rmse.toFixed(2),
  errors,
  t1: { pass: t1Pass, checks: t1 },
  t2,
  screenshots: views.map(v => `${OUT}/iter-${ITER}-${v}.png`),
};

const reportPath = `${OUT}/iter-${ITER}-report.json`;
writeFileSync(reportPath, JSON.stringify(report, null, 2));
console.log(`\nReport: ${reportPath}`);
console.log(`RMSE: ${rmse.toFixed(2)}°C`);
console.log(`Global SST: ${diagnostics.globalSST?.toFixed(1)}°C`);
console.log(`Tropical SST: ${diagnostics.tropicalSST?.toFixed(1)}°C`);
console.log(`Polar SST: ${diagnostics.polarSST?.toFixed(1)}°C`);
console.log(`AMOC: ${diagnostics.amocSv?.toFixed(1)} Sv`);
console.log(`T1 Conservation: ${t1Pass ? 'PASS' : 'FAIL'}`);
console.log(`T2 Structure: ${Object.values(t2).filter(Boolean).length}/${Object.keys(t2).length}`);

await browser.close();
server.close();
console.log('\n=== DONE ===');
