import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const URL = 'https://amoc-sim.vercel.app/v4-physics/';
const OUT = './screenshots';
mkdirSync(OUT, { recursive: true });

const HYPOTHESES = [
  {
    name: '01-init',
    waitSec: 5,
    view: 'temp',
    hypothesis: 'Warm tropics (yellow/orange ~25-28C), cold poles (blue/white ice), clear latitude gradient. Land shows seasonal coloring.',
  },
  {
    name: '02-spinup-30s',
    waitSec: 30,
    view: 'temp',
    hypothesis: 'Western boundary currents visible (warm tongues at Gulf Stream, Kuroshio). Tropics approaching equilibrium (~25C+). Antarctic ice holds. Caribbean warmer than open ocean (shallow + land heating).',
  },
  {
    name: '03-mature-60s',
    waitSec: 30,
    view: 'temp',
    hypothesis: 'Established temperature pattern: ~28C tropics (orange/red), ~10C mid-lats (green), ice at poles (blue/white). Seasonal ice edge visible.',
  },
  {
    name: '04-streamfunction',
    waitSec: 0,
    view: 'psi',
    hypothesis: 'Subtropical gyres (red=anticyclonic), subpolar gyres (blue=cyclonic). Western intensification visible. Antarctic Circumpolar Current as strong zonal flow.',
  },
  {
    name: '05-deep-temp',
    waitSec: 0,
    view: 'deeptemp',
    hypothesis: 'Deep ocean ~1-5C. Southern hemisphere colder (Antarctic Bottom Water). Slowly warming from surface exchange.',
  },
  {
    name: '06-deep-flow',
    waitSec: 0,
    view: 'deepflow',
    hypothesis: 'Deep circulation developing: weaker than surface, driven by interfacial coupling. Own gyres forming.',
  },
  {
    name: '07-depth',
    waitSec: 0,
    view: 'depth',
    hypothesis: 'Bathymetry: light blue (shallow) continental shelves near coasts. Dark navy deep basins in open ocean. Caribbean/Mediterranean visibly shallow.',
  },
  {
    name: '08-speed',
    waitSec: 0,
    view: 'speed',
    hypothesis: 'Current speed: bright streaks at western boundaries (Gulf Stream, Kuroshio, Agulhas). ACC as fast band around Antarctica.',
  },
];

async function run() {
  console.log('=== OCEAN SIMULATION VISUAL TEST ===\n');

  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1400, height: 900 });

  console.log('Loading simulation...');
  await page.goto(URL, { waitUntil: 'networkidle', timeout: 30000 });

  // Dismiss onboarding
  try {
    await page.click('#btn-start-exploring', { timeout: 3000 });
  } catch (e) { /* no onboarding */ }

  await page.waitForTimeout(1000);

  // Reset to start fresh
  await page.click('#btn-reset');
  await page.waitForTimeout(500);

  // Set sim speed to 50 and year speed to 3x
  await page.evaluate(() => {
    const simSlider = document.getElementById('speed-slider');
    simSlider.value = 50;
    simSlider.dispatchEvent(new Event('input'));

    const yearSlider = document.getElementById('year-speed-slider');
    yearSlider.value = 3;
    yearSlider.dispatchEvent(new Event('input'));
  });

  console.log('Running simulation with sim speed 50, year speed 3x...\n');

  for (const h of HYPOTHESES) {
    if (h.waitSec > 0) {
      console.log(`  Waiting ${h.waitSec}s...`);
      await page.waitForTimeout(h.waitSec * 1000);
    }

    // Switch view
    await page.click(`#btn-${h.view}`);
    await page.waitForTimeout(1500);

    // Get stats
    const stats = await page.evaluate(() => ({
      step: document.getElementById('stat-step')?.textContent,
      season: document.getElementById('stat-season')?.textContent,
      amoc: document.getElementById('stat-amoc')?.textContent,
      vel: document.getElementById('stat-vel')?.textContent,
      ke: document.getElementById('stat-ke')?.textContent,
    }));

    const filename = `${OUT}/${h.name}.png`;
    await page.screenshot({ path: filename });

    console.log(`--- ${h.name} ---`);
    console.log(`  Stats: step=${stats.step} season=${stats.season} AMOC=${stats.amoc} vel=${stats.vel} KE=${stats.ke}`);
    console.log(`  HYPOTHESIS: ${h.hypothesis}`);
    console.log(`  Screenshot: ${filename}`);
    console.log('');
  }

  // Capture reference SST for comparison (serve locally)
  console.log('--- Capturing reference SST (NOAA OI SST v2 1991-2020) ---');
  const { createServer } = await import('http');
  const { readFileSync } = await import('fs');
  const { resolve } = await import('path');
  const root = resolve('.');
  const server = createServer((req, res) => {
    const filePath = resolve(root, req.url.replace(/^\//, ''));
    try {
      const data = readFileSync(filePath);
      const ext = filePath.split('.').pop();
      const mime = { html: 'text/html', json: 'application/json', js: 'text/javascript' }[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': mime });
      res.end(data);
    } catch { res.writeHead(404); res.end('Not found'); }
  });
  await new Promise(r => server.listen(8765, r));
  await page.goto('http://localhost:8765/reference-sst.html', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);
  await page.screenshot({ path: `${OUT}/00-reference-sst.png` });
  server.close();
  console.log(`  Screenshot: ${OUT}/00-reference-sst.png\n`);

  await browser.close();

  console.log('=== DONE ===');
  console.log(`Screenshots saved to ${OUT}/`);
  console.log('Compare sim screenshots against 00-reference-sst.png');
}

run().catch(console.error);
