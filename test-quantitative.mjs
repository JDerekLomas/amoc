import { chromium } from 'playwright';
import { mkdirSync, writeFileSync } from 'fs';
import { createServer } from 'http';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const root = '/Users/dereklomas/lukebarrington/amoc';
const OUT = './screenshots/quant';
mkdirSync(OUT, { recursive: true });

const server = createServer((req, res) => {
  const urlPath = new URL(req.url, 'http://localhost').pathname;
  const filePath = resolve(root, urlPath.replace(/^\//, ''));
  try {
    const data = readFileSync(filePath);
    const ext = filePath.split('.').pop();
    const mime = { html: 'text/html', json: 'application/json', js: 'text/javascript' }[ext] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  } catch { res.writeHead(404); res.end('Not found'); }
});
await new Promise(r => server.listen(8771, r));

const browser = await chromium.launch({
  headless: true,
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--enable-unsafe-webgpu'],
});
const page = await browser.newPage();
await page.setViewportSize({ width: 1400, height: 900 });

await page.goto('http://localhost:8771/v4-physics/index.html', { waitUntil: 'load', timeout: 30000 });
await page.waitForTimeout(5000);
try { await page.click('#btn-start-exploring', { timeout: 5000 }); } catch {}
await page.waitForTimeout(1000);

const backend = await page.evaluate(() => document.getElementById('backend-badge')?.textContent);
console.log(`Backend: ${backend}\n`);

await page.click('#btn-reset');
await page.waitForTimeout(500);

await page.evaluate(() => {
  document.getElementById('speed-slider').value = 200;
  document.getElementById('speed-slider').dispatchEvent(new Event('input'));
  document.getElementById('year-speed-slider').value = 3;
  document.getElementById('year-speed-slider').dispatchEvent(new Event('input'));
});

await page.click('#btn-temp');

// Target equilibrium temperatures by latitude
const targets = {};
for (let lat = -80; lat <= 80; lat += 10) {
  const latR = lat * Math.PI / 180;
  let total = 0;
  for (let m = 0; m < 12; m++) {
    const phase = 2 * Math.PI * ((m - 3) / 12);
    const decl = 23.44 * Math.sin(phase) * Math.PI / 180;
    const cosZ = Math.cos(latR) * Math.cos(decl) + Math.sin(latR) * Math.sin(decl);
    total += Math.max(0, cosZ);
  }
  const avgCosZ = total / 12;
  // Using current params: S=50, A=20, B=1.0
  targets[lat] = (50 * avgCosZ - 20) / 1.0;
}

console.log('=== QUANTITATIVE CONVERGENCE TEST ===');
console.log('Target equilibrium (S=50, A=20, B=1.0):');
console.log('  Lat   Target');
for (const [lat, t] of Object.entries(targets)) {
  console.log(`  ${String(lat).padStart(4)}°  ${t.toFixed(1)}°C`);
}
console.log('');

const CHECKPOINTS = [10, 30, 60, 120, 180];
let elapsed = 0;

for (const t of CHECKPOINTS) {
  const waitFor = t - elapsed;
  if (waitFor > 0) await page.waitForTimeout(waitFor * 1000);
  elapsed = t;

  // Extract zonal mean temperatures from the sim
  const data = await page.evaluate(() => {
    const nx = typeof NX !== 'undefined' ? NX : 360;
    const ny = typeof NY !== 'undefined' ? NY : 180;
    const m = typeof mask !== 'undefined' ? mask : null;
    const t = typeof temp !== 'undefined' ? temp : null;
    const dt = typeof deepTemp !== 'undefined' ? deepTemp : null;
    if (!t || !m) return null;

    // Zonal means at specific latitudes
    const zonalTemp = {};
    const zonalDeep = {};
    const latBins = [-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70];
    for (const targetLat of latBins) {
      // Find closest j
      const j = Math.round((targetLat - (-80)) / 160 * (ny - 1));
      if (j < 0 || j >= ny) continue;
      let sum = 0, dsum = 0, count = 0;
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (m[k]) {
          sum += t[k];
          if (dt) dsum += dt[k];
          count++;
        }
      }
      if (count > 0) {
        zonalTemp[targetLat] = sum / count;
        zonalDeep[targetLat] = dt ? dsum / count : null;
      }
    }

    // Global stats
    let gSum = 0, gCount = 0, gMin = 999, gMax = -999;
    for (let k = 0; k < nx * ny; k++) {
      if (m[k]) {
        gSum += t[k]; gCount++;
        if (t[k] < gMin) gMin = t[k];
        if (t[k] > gMax) gMax = t[k];
      }
    }

    return {
      zonalTemp,
      zonalDeep,
      globalMean: gSum / gCount,
      globalMin: gMin,
      globalMax: gMax,
      step: document.getElementById('stat-step')?.textContent,
      amoc: document.getElementById('stat-amoc')?.textContent,
    };
  });

  if (!data) { console.log(`T=${t}s: Could not read temp array`); continue; }

  await page.screenshot({ path: `${OUT}/temp-${String(t).padStart(3,'0')}s.png` });

  console.log(`--- T=${t}s | step=${data.step} | AMOC=${data.amoc} ---`);
  console.log(`  Global: mean=${data.globalMean.toFixed(1)}°C min=${data.globalMin.toFixed(1)} max=${data.globalMax.toFixed(1)}`);
  console.log(`  ${'Lat'.padStart(5)} ${'Sim'.padStart(7)} ${'Target'.padStart(7)} ${'Bias'.padStart(7)} ${'Deep'.padStart(7)}`);

  for (const lat of [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70]) {
    const simT = data.zonalTemp[lat];
    const deepT = data.zonalDeep[lat];
    const tgt = targets[lat];
    if (simT !== undefined && tgt !== undefined) {
      const bias = simT - tgt;
      console.log(`  ${String(lat).padStart(4)}° ${simT.toFixed(1).padStart(7)} ${tgt.toFixed(1).padStart(7)} ${(bias > 0 ? '+' : '') + bias.toFixed(1).padStart(6)} ${deepT !== null ? deepT.toFixed(1).padStart(7) : '   N/A'}`);
    }
  }
  console.log('');
}

// Final views
for (const view of ['psi', 'speed', 'deeptemp']) {
  await page.click(`#btn-${view}`);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${OUT}/final-${view}.png` });
}
console.log('Final view screenshots captured.');

await browser.close();
server.close();
