import { chromium } from 'playwright';
import { createServer } from 'http';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const root = '/Users/dereklomas/lukebarrington/amoc';

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
await new Promise(r => server.listen(8773, r));

const browser = await chromium.launch({
  headless: false,
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const page = await browser.newPage();
await page.setViewportSize({ width: 1400, height: 900 });

await page.goto('http://localhost:8773/simamoc/index.html', { waitUntil: 'load', timeout: 30000 });
await page.waitForTimeout(3000);
try { await page.click('#btn-start-exploring', { timeout: 3000 }); } catch {}
await page.waitForTimeout(1000);

await page.click('#btn-reset');
await page.evaluate(() => {
  document.getElementById('speed-slider').value = 150;
  document.getElementById('speed-slider').dispatchEvent(new Event('input'));
  document.getElementById('year-speed-slider').value = 3;
  document.getElementById('year-speed-slider').dispatchEvent(new Event('input'));
});

// Sample SST at a point, searching nearby for ocean cells if needed
async function sampleRegion(lonCenter, latCenter, label) {
  return page.evaluate(({ lon, lat, label }) => {
    const NX = 360, NY = 180;
    const LON0 = -180, LON1 = 180, LAT0 = -80, LAT1 = 80;
    function lonToI(ln) { return Math.round((ln - LON0) / (LON1 - LON0) * (NX - 1)); }
    function latToJ(lt) { return Math.round((lt - LAT0) / (LAT1 - LAT0) * (NY - 1)); }

    // Search in expanding radius for ocean cells, average them
    let sum = 0, count = 0;
    for (let r = 0; r <= 5; r++) {
      for (let di = -r; di <= r; di++) {
        for (let dj = -r; dj <= r; dj++) {
          if (Math.abs(di) < r && Math.abs(dj) < r) continue; // only perimeter
          const i = (lonToI(lon) + di + NX) % NX;
          const j = latToJ(lat) + dj;
          if (j < 0 || j >= NY) continue;
          const k = j * NX + i;
          if (mask[k]) { sum += temp[k]; count++; }
        }
      }
      if (count >= 3) break; // enough samples
    }
    return {
      label,
      sst: count > 0 ? (sum / count) : null,
      oceanCells: count,
    };
  }, { lon: lonCenter, lat: latCenter, label });
}

// Known winter SST values (NOAA OI SST v2, January climatology)
const KNOWN_DATA = {
  // North Atlantic — warmed by AMOC/Gulf Stream/NAC
  'W of Ireland (53N, 12W)':        { lon: -12, lat: 53, realJan: 10.0, realJul: 14.5 },
  'SW of Iceland (63N, 22W)':       { lon: -22, lat: 63, realJan: 5.0,  realJul: 9.0 },
  'Norwegian Sea (65N, 5E)':        { lon: 5,   lat: 65, realJan: 5.5,  realJul: 11.0 },
  'N Scotland (58N, 6W)':           { lon: -6,  lat: 58, realJan: 8.5,  realJul: 12.5 },
  'Bay of Biscay (46N, 5W)':        { lon: -5,  lat: 46, realJan: 11.5, realJul: 18.0 },

  // Same latitudes, western Atlantic — cold Labrador Current
  'Grand Banks (47N, 50W)':         { lon: -50, lat: 47, realJan: 3.0,  realJul: 12.0 },
  'S Labrador Sea (53N, 50W)':      { lon: -50, lat: 53, realJan: 1.5,  realJul: 6.0 },
  'Labrador coast (58N, 56W)':      { lon: -56, lat: 58, realJan: -1.0, realJul: 4.0 },

  // Gulf Stream
  'Gulf Stream (38N, 70W)':         { lon: -70, lat: 38, realJan: 18.0, realJul: 26.0 },
  'Mid-Atlantic (45N, 40W)':        { lon: -40, lat: 45, realJan: 11.0, realJul: 16.0 },

  // Tropics (sanity check)
  'Tropical Atlantic (10N, 30W)':   { lon: -30, lat: 10, realJan: 26.0, realJul: 27.5 },

  // Pacific comparison at same lat
  'N Pacific (53N, 170W)':          { lon: -170, lat: 53, realJan: 4.0, realJul: 9.0 },
};

console.log('=== IRELAND SST COMPARISON: SimAMOC vs NOAA Observed ===\n');
console.log('Spinup: 90s at speed=150...');
await page.waitForTimeout(90000);

// Sample in January
console.log('Waiting for January...');
for (let i = 0; i < 60; i++) {
  await page.waitForTimeout(500);
  const s = await page.evaluate(() => document.getElementById('stat-season')?.textContent);
  if (s === 'Jan') { console.log('  Got January'); break; }
}

console.log('\n--- JANUARY SST COMPARISON ---');
console.log('Location                        | Sim (°C) | Real (°C) | Bias (°C)');
console.log('--------------------------------|----------|-----------|----------');
const janResults = {};
for (const [name, data] of Object.entries(KNOWN_DATA)) {
  const result = await sampleRegion(data.lon, data.lat, name);
  janResults[name] = result;
  const simStr = result.sst !== null ? result.sst.toFixed(1).padStart(6) : '  LAND';
  const realStr = data.realJan.toFixed(1).padStart(6);
  const bias = result.sst !== null ? (result.sst - data.realJan).toFixed(1).padStart(7) : '     --';
  console.log(`${name.padEnd(32)}| ${simStr}   | ${realStr}    | ${bias}`);
}

// Key comparison: Ireland vs Labrador at same latitude
const irelandJan = janResults['W of Ireland (53N, 12W)']?.sst;
const labradorJan = janResults['S Labrador Sea (53N, 50W)']?.sst;
console.log(`\n>> Ireland-Labrador January difference: ${irelandJan !== null && labradorJan !== null ? (irelandJan - labradorJan).toFixed(1) : '--'}°C`);
console.log(`>> Real-world difference: ${(10.0 - 1.5).toFixed(1)}°C`);
console.log(`>> This cross-Atlantic gradient is THE signature of AMOC heat transport to Europe.\n`);

// Wait for July
console.log('Waiting for July...');
for (let i = 0; i < 60; i++) {
  await page.waitForTimeout(500);
  const s = await page.evaluate(() => document.getElementById('stat-season')?.textContent);
  if (s === 'Jul') { console.log('  Got July'); break; }
}

console.log('\n--- JULY SST COMPARISON ---');
console.log('Location                        | Sim (°C) | Real (°C) | Bias (°C)');
console.log('--------------------------------|----------|-----------|----------');
for (const [name, data] of Object.entries(KNOWN_DATA)) {
  const result = await sampleRegion(data.lon, data.lat, name);
  const simStr = result.sst !== null ? result.sst.toFixed(1).padStart(6) : '  LAND';
  const realStr = data.realJul.toFixed(1).padStart(6);
  const bias = result.sst !== null ? (result.sst - data.realJul).toFixed(1).padStart(7) : '     --';
  console.log(`${name.padEnd(32)}| ${simStr}   | ${realStr}    | ${bias}`);
}

// AMOC diagnostics
const amocText = await page.evaluate(() => document.getElementById('stat-amoc')?.textContent);
console.log(`\nCurrent AMOC strength: ${amocText}`);

// Zonal mean SST at 53N across Atlantic
console.log('\n--- ZONAL SST TRANSECT at 53°N (Ireland latitude) ---');
console.log('This should show warm east, cold west if AMOC is working:\n');
const transect = await page.evaluate(() => {
  const NX = 360, NY = 180;
  const LON0 = -180, LAT0 = -80, LAT1 = 80;
  const j = Math.round((53 - LAT0) / (LAT1 - LAT0) * (NY - 1));
  const pts = [];
  // Atlantic only: -80W to +10E
  for (let lon = -80; lon <= 10; lon += 5) {
    const i = Math.round((lon - LON0) / 360 * (NX - 1));
    const k = j * NX + i;
    pts.push({ lon, sst: mask[k] ? temp[k].toFixed(1) : 'land' });
  }
  return pts;
});

let line = '';
for (const p of transect) {
  line += `${p.lon}°: ${p.sst}  `;
}
console.log(line);

await browser.close();
server.close();
console.log('\n=== DONE ===');
