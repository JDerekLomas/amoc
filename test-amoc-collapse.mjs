import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
import { createServer } from 'http';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const root = '/Users/dereklomas/lukebarrington/amoc';
const OUT = './screenshots/amoc-collapse';
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
  headless: false, // visible browser for WebGPU
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const page = await browser.newPage();
await page.setViewportSize({ width: 1400, height: 900 });

console.log('=== AMOC COLLAPSE EXPERIMENT ===\n');
await page.goto('http://localhost:8771/simamoc/index.html', { waitUntil: 'load', timeout: 30000 });
await page.waitForTimeout(3000);

// Dismiss onboarding
try { await page.click('#btn-start-exploring', { timeout: 3000 }); } catch {}
await page.waitForTimeout(1000);

// Reset and set high sim speed for fast spinup
await page.click('#btn-reset');
await page.evaluate(() => {
  document.getElementById('speed-slider').value = 100;
  document.getElementById('speed-slider').dispatchEvent(new Event('input'));
  document.getElementById('year-speed-slider').value = 3;
  document.getElementById('year-speed-slider').dispatchEvent(new Event('input'));
});

async function getStats() {
  return page.evaluate(() => ({
    step: document.getElementById('stat-step')?.textContent,
    season: document.getElementById('stat-season')?.textContent,
    amoc: document.getElementById('stat-amoc')?.textContent,
    fw: document.getElementById('fw-val')?.textContent,
  }));
}

// Phase 1: Spinup (60s) — let AMOC establish
console.log('Phase 1: Spinup (60s)...');
await page.waitForTimeout(60000);

// Ensure temp view
await page.click('#btn-temp');
await page.waitForTimeout(1000);

var stats = await getStats();
console.log(`  Pre-hosing: AMOC=${stats.amoc}, Season=${stats.season}, Step=${stats.step}`);
await page.screenshot({ path: `${OUT}/01-pre-hosing-temp.png` });

await page.click('#btn-sal');
await page.waitForTimeout(500);
await page.screenshot({ path: `${OUT}/01-pre-hosing-sal.png` });
await page.click('#btn-temp');

// Phase 2: Hosing ramp — freshwater levels 0.5, 1.0, 1.5, 2.0
const FW_LEVELS = [0.5, 1.0, 1.5, 2.0];
for (const fw of FW_LEVELS) {
  console.log(`\nPhase 2: Setting freshwater = ${fw}...`);
  await page.evaluate((val) => {
    document.getElementById('fw-slider').value = val;
    document.getElementById('fw-slider').dispatchEvent(new Event('input'));
  }, fw);

  // Wait 30s for adjustment
  await page.waitForTimeout(30000);

  stats = await getStats();
  console.log(`  fw=${fw}: AMOC=${stats.amoc}, Season=${stats.season}`);

  // Capture temp and salinity views
  await page.click('#btn-temp');
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/02-hosing-fw${fw.toFixed(1)}-temp.png` });

  await page.click('#btn-sal');
  await page.waitForTimeout(500);
  await page.screenshot({ path: `${OUT}/02-hosing-fw${fw.toFixed(1)}-sal.png` });

  await page.click('#btn-temp');
}

// Phase 3: Full screenshot at max hosing
console.log('\nPhase 3: Full collapse screenshots...');
await page.waitForTimeout(15000);

stats = await getStats();
console.log(`  Post-collapse: AMOC=${stats.amoc}`);

const VIEWS = ['temp', 'psi', 'deeptemp', 'speed', 'sal', 'density'];
for (const view of VIEWS) {
  await page.click(`#btn-${view}`);
  await page.waitForTimeout(1000);
  await page.screenshot({ path: `${OUT}/03-collapsed-${view}.png` });
}

// Phase 4: Recovery test — remove freshwater
console.log('\nPhase 4: Recovery test (fw=0)...');
await page.evaluate(() => {
  document.getElementById('fw-slider').value = 0;
  document.getElementById('fw-slider').dispatchEvent(new Event('input'));
});
await page.click('#btn-temp');

// Wait 60s for potential recovery
for (const sec of [15, 30, 60]) {
  await page.waitForTimeout(15000);
  stats = await getStats();
  console.log(`  Recovery +${sec}s: AMOC=${stats.amoc}`);
  await page.screenshot({ path: `${OUT}/04-recovery-${sec}s-temp.png` });
}

// Final full-page screenshot showing timeseries
await page.screenshot({ path: `${OUT}/05-final-with-timeseries.png` });
stats = await getStats();
console.log(`\nFinal: AMOC=${stats.amoc}`);

await browser.close();
server.close();

console.log(`\n=== DONE ===`);
console.log(`Screenshots saved to ${OUT}/`);
console.log('Key files:');
console.log('  01-pre-hosing-temp.png  — healthy AMOC baseline');
console.log('  02-hosing-fw*.png       — progressive freshwater forcing');
console.log('  03-collapsed-*.png      — post-collapse state');
console.log('  04-recovery-*.png       — hysteresis test');
console.log('  05-final-with-timeseries.png — full UI with AMOC graph');
