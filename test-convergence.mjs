import { chromium } from 'playwright';
import { mkdirSync } from 'fs';

const URL = 'https://amoc-sim.vercel.app/v4-physics/';
const OUT = './screenshots/convergence';
mkdirSync(OUT, { recursive: true });

// Capture temperature view at intervals to see if it converges
const CHECKPOINTS = [5, 15, 30, 60, 90, 120, 150, 180];

async function run() {
  console.log('=== CONVERGENCE TEST ===');
  console.log('Sim speed 200 (max), year speed 3x');
  console.log(`Checkpoints at: ${CHECKPOINTS.join(', ')} seconds\n`);

  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1400, height: 900 });

  await page.goto(URL, { waitUntil: 'networkidle', timeout: 30000 });

  // Dismiss onboarding
  try {
    await page.click('#btn-start-exploring', { timeout: 3000 });
  } catch (e) {}

  await page.waitForTimeout(1000);
  await page.click('#btn-reset');
  await page.waitForTimeout(500);

  // Max out sim speed for faster convergence
  await page.evaluate(() => {
    const simSlider = document.getElementById('speed-slider');
    simSlider.value = 200;
    simSlider.dispatchEvent(new Event('input'));

    const yearSlider = document.getElementById('year-speed-slider');
    yearSlider.value = 3;
    yearSlider.dispatchEvent(new Event('input'));
  });

  // Make sure we're on temperature view
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

    const filename = `${OUT}/temp-${String(t).padStart(3, '0')}s.png`;
    await page.screenshot({ path: filename });

    console.log(`  T=${t}s | step=${stats.step} season=${stats.season} AMOC=${stats.amoc} vel=${stats.vel} KE=${stats.ke}`);
  }

  await browser.close();
  console.log('\n=== DONE ===');
}

run().catch(console.error);
