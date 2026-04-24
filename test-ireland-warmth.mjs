import { chromium } from 'playwright';
import { createServer } from 'http';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import { mkdirSync } from 'fs';

const root = '/Users/dereklomas/lukebarrington/amoc';
const OUT = './screenshots/ireland-test';
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
await new Promise(r => server.listen(8772, r));

const browser = await chromium.launch({
  headless: false,
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const page = await browser.newPage();
await page.setViewportSize({ width: 1400, height: 900 });

await page.goto('http://localhost:8772/simamoc/index.html', { waitUntil: 'load', timeout: 30000 });
await page.waitForTimeout(3000);
try { await page.click('#btn-start-exploring', { timeout: 3000 }); } catch {}
await page.waitForTimeout(1000);

// Reset and crank sim speed
await page.click('#btn-reset');
await page.evaluate(() => {
  document.getElementById('speed-slider').value = 150;
  document.getElementById('speed-slider').dispatchEvent(new Event('input'));
  document.getElementById('year-speed-slider').value = 3;
  document.getElementById('year-speed-slider').dispatchEvent(new Event('input'));
});

// Helper: read SST at specific lat/lon locations
async function sampleTemps() {
  return page.evaluate(() => {
    const NX = 360, NY = 180;
    const LON0 = -180, LON1 = 180, LAT0 = -80, LAT1 = 80;
    function lonToI(lon) { return Math.round((lon - LON0) / (LON1 - LON0) * (NX - 1)); }
    function latToJ(lat) { return Math.round((lat - LAT0) / (LAT1 - LAT0) * (NY - 1)); }
    function sst(lon, lat) {
      const i = lonToI(lon), j = latToJ(lat);
      const k = j * NX + i;
      if (!window.lab) return null;
      // Access temp array directly via lab
      const d = window.lab.diagnostics({ profiles: false });
      return null; // can't get point temps this way
    }
    // Use direct field access — temp is a global
    const points = {
      ireland:     { lon: -8,  lat: 53, label: 'Ireland (53N, 8W)' },
      labrador:    { lon: -60, lat: 53, label: 'Labrador (53N, 60W)' },
      iceland:     { lon: -20, lat: 64, label: 'Iceland (64N, 20W)' },
      norway:      { lon: 10,  lat: 60, label: 'Norway (60N, 10E)' },
      hudsonBay:   { lon: -85, lat: 58, label: 'Hudson Bay (58N, 85W)' },
      gulfStream:  { lon: -70, lat: 38, label: 'Gulf Stream (38N, 70W)' },
      scotland:    { lon: -5,  lat: 56, label: 'Scotland (56N, 5W)' },
      newfoundland:{ lon: -53, lat: 47, label: 'Newfoundland (47N, 53W)' },
    };
    const results = {};
    for (const [name, p] of Object.entries(points)) {
      const i = Math.round((p.lon - LON0) / (LON1 - LON0) * (NX - 1));
      const j = Math.round((p.lat - LAT0) / (LAT1 - LAT0) * (NY - 1));
      const k = j * NX + i;
      const isOcean = mask[k] === 1;
      results[name] = {
        label: p.label,
        sst: isOcean ? temp[k].toFixed(1) : 'LAND',
        isOcean,
      };
    }
    // Also grab season and AMOC
    results._meta = {
      season: document.getElementById('stat-season')?.textContent,
      amoc: document.getElementById('stat-amoc')?.textContent,
      step: document.getElementById('stat-step')?.textContent,
      fw: document.getElementById('fw-val')?.textContent,
    };
    return results;
  });
}

console.log('=== IRELAND WARMTH TEST: Is the AMOC keeping Ireland warm? ===\n');

// Phase 1: Spinup — let AMOC establish (90s at high speed)
console.log('Phase 1: Spinup (90s at speed=150)...');
await page.waitForTimeout(90000);

// Wait for winter (Dec/Jan/Feb)
console.log('Waiting for winter...');
for (let i = 0; i < 30; i++) {
  await page.waitForTimeout(1000);
  const s = await page.evaluate(() => document.getElementById('stat-season')?.textContent);
  if (s === 'Jan' || s === 'Feb' || s === 'Dec') {
    console.log(`  Got winter: ${s}`);
    break;
  }
}

await page.click('#btn-temp');
await page.waitForTimeout(500);

let temps = await sampleTemps();
console.log('\n--- HEALTHY AMOC (Winter) ---');
console.log(`  Season: ${temps._meta.season}, AMOC: ${temps._meta.amoc}, Step: ${temps._meta.step}`);
for (const [name, data] of Object.entries(temps)) {
  if (name === '_meta') continue;
  console.log(`  ${data.label}: ${data.sst}°C`);
}
const irelandBefore = parseFloat(temps.ireland?.sst) || 0;
const labradorBefore = parseFloat(temps.labrador?.sst) || 0;
const warmthAdvantage = irelandBefore - labradorBefore;
console.log(`\n  >> Ireland-Labrador difference: ${warmthAdvantage.toFixed(1)}°C`);
console.log(`  >> ${warmthAdvantage > 3 ? 'YES' : 'MARGINAL'} — AMOC is ${warmthAdvantage > 3 ? 'clearly' : 'possibly'} warming Ireland`);

await page.screenshot({ path: `${OUT}/01-healthy-amoc-winter.png` });

// Also capture summer for comparison
console.log('\nWaiting for summer...');
for (let i = 0; i < 30; i++) {
  await page.waitForTimeout(1000);
  const s = await page.evaluate(() => document.getElementById('stat-season')?.textContent);
  if (s === 'Jul' || s === 'Aug') {
    console.log(`  Got summer: ${s}`);
    break;
  }
}
let summerTemps = await sampleTemps();
console.log('\n--- HEALTHY AMOC (Summer) ---');
console.log(`  Season: ${summerTemps._meta.season}`);
for (const [name, data] of Object.entries(summerTemps)) {
  if (name === '_meta') continue;
  console.log(`  ${data.label}: ${data.sst}°C`);
}
await page.screenshot({ path: `${OUT}/02-healthy-amoc-summer.png` });

// Phase 2: Collapse the AMOC
console.log('\n\nPhase 2: Collapsing AMOC (fw=2.0 instant for speed)...');
await page.evaluate(() => {
  document.getElementById('fw-slider').value = 2.0;
  document.getElementById('fw-slider').dispatchEvent(new Event('input'));
});

// Wait for collapse (60s)
for (const sec of [15, 30, 45, 60]) {
  await page.waitForTimeout(15000);
  const s = await page.evaluate(() => document.getElementById('stat-amoc')?.textContent);
  console.log(`  +${sec}s: AMOC = ${s}`);
}

// Wait for winter again
console.log('\nWaiting for post-collapse winter...');
for (let i = 0; i < 30; i++) {
  await page.waitForTimeout(1000);
  const s = await page.evaluate(() => document.getElementById('stat-season')?.textContent);
  if (s === 'Jan' || s === 'Feb' || s === 'Dec') {
    console.log(`  Got winter: ${s}`);
    break;
  }
}

let postTemps = await sampleTemps();
console.log('\n--- COLLAPSED AMOC (Winter) ---');
console.log(`  Season: ${postTemps._meta.season}, AMOC: ${postTemps._meta.amoc}`);
for (const [name, data] of Object.entries(postTemps)) {
  if (name === '_meta') continue;
  console.log(`  ${data.label}: ${data.sst}°C`);
}

const irelandAfter = parseFloat(postTemps.ireland?.sst) || 0;
const labradorAfter = parseFloat(postTemps.labrador?.sst) || 0;
const irelandCooling = irelandBefore - irelandAfter;
const postAdvantage = irelandAfter - labradorAfter;

console.log(`\n  >> Ireland cooled by: ${irelandCooling.toFixed(1)}°C`);
console.log(`  >> Ireland-Labrador difference now: ${postAdvantage.toFixed(1)}°C (was ${warmthAdvantage.toFixed(1)}°C)`);

if (irelandCooling > 2) {
  console.log(`  >> CONFIRMED: AMOC collapse cooled Ireland by ${irelandCooling.toFixed(1)}°C`);
} else if (irelandCooling > 0) {
  console.log(`  >> PARTIAL: Ireland cooled slightly (${irelandCooling.toFixed(1)}°C) — AMOC effect may be modest in this model`);
} else {
  console.log(`  >> UNEXPECTED: Ireland did not cool — model may need tuning`);
}

await page.screenshot({ path: `${OUT}/03-collapsed-amoc-winter.png` });

// Summary table
console.log('\n\n=== SUMMARY TABLE ===');
console.log('Location              | Healthy Winter | Collapsed Winter | Change');
console.log('----------------------|----------------|------------------|-------');
for (const name of ['ireland', 'scotland', 'norway', 'iceland', 'labrador', 'newfoundland', 'gulfStream']) {
  const before = temps[name]?.sst || '--';
  const after = postTemps[name]?.sst || '--';
  const bv = parseFloat(before), av = parseFloat(after);
  const change = (!isNaN(bv) && !isNaN(av)) ? (av - bv).toFixed(1) : '--';
  const label = (temps[name]?.label || name).padEnd(22);
  console.log(`${label}| ${String(before).padEnd(15)}| ${String(after).padEnd(17)}| ${change}`);
}

await browser.close();
server.close();
console.log('\n=== DONE ===');
console.log(`Screenshots in ${OUT}/`);
