import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
import { createServer } from 'http';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const root = '/Users/dereklomas/lukebarrington/amoc';
const OUT = './screenshots/convergence-v2';
mkdirSync(OUT, { recursive: true });

// Local server for the modified v4-physics
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
await new Promise(r => server.listen(8770, r));

const CHECKPOINTS = [5, 15, 30, 60, 90, 120, 180];

const browser = await chromium.launch({
  headless: true,
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
});
const page = await browser.newPage();
await page.setViewportSize({ width: 1400, height: 900 });

console.log('=== CONVERGENCE TEST (S=25, A=10, B=0.5) ===\n');
await page.goto('http://localhost:8770/simamoc/index.html', { waitUntil: 'load', timeout: 30000 });
// Give time for JS to initialize
await page.waitForTimeout(5000);

// Debug: check page state
const title = await page.title();
const bodyLen = await page.evaluate(() => document.body.innerHTML.length);
console.log(`Page title: "${title}", body length: ${bodyLen}`);

// Wait for page to fully initialize (CPU or GPU)
await page.waitForTimeout(3000);

// Check if onboarding is visible
try {
  await page.click('#btn-start-exploring', { timeout: 5000 });
  console.log('Dismissed onboarding');
} catch (e) {
  console.log('No onboarding overlay');
}

await page.waitForTimeout(1000);

// Check backend
const backend = await page.evaluate(() => document.getElementById('backend-badge')?.textContent);
console.log(`Backend: ${backend}`);

await page.click('#btn-reset');
await page.waitForTimeout(500);

// Max sim speed
await page.evaluate(() => {
  document.getElementById('speed-slider').value = 200;
  document.getElementById('speed-slider').dispatchEvent(new Event('input'));
  document.getElementById('year-speed-slider').value = 3;
  document.getElementById('year-speed-slider').dispatchEvent(new Event('input'));
});

await page.click('#btn-temp');
await page.waitForTimeout(500);

let elapsed = 0;
for (const t of CHECKPOINTS) {
  const waitFor = t - elapsed;
  if (waitFor > 0) {
    console.log(`  Waiting ${waitFor}s (total ${t}s)...`);
    await page.waitForTimeout(waitFor * 1000);
  }
  elapsed = t;

  const stats = await page.evaluate(() => ({
    step: document.getElementById('stat-step')?.textContent,
    season: document.getElementById('stat-season')?.textContent,
    amoc: document.getElementById('stat-amoc')?.textContent,
    vel: document.getElementById('stat-vel')?.textContent,
    ke: document.getElementById('stat-ke')?.textContent,
  }));

  await page.screenshot({ path: `${OUT}/temp-${String(t).padStart(3, '0')}s.png` });
  console.log(`  T=${t}s | step=${stats.step} season=${stats.season} AMOC=${stats.amoc} vel=${stats.vel} KE=${stats.ke}`);
}

// Also grab streamfunction and speed at the end
for (const view of ['psi', 'speed', 'deeptemp']) {
  await page.click(`#btn-${view}`);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${OUT}/final-${view}.png` });
  console.log(`  Captured final ${view}`);
}

await browser.close();
server.close();
console.log('\n=== DONE ===');
