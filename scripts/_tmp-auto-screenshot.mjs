// Auto-screenshot: loads the sim, waits, captures, saves to screenshots/
import puppeteer from 'puppeteer';
import { existsSync, mkdirSync } from 'fs';

const dir = '/Users/dereklomas/lukebarrington/amoc/screenshots';
if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

const WAIT_SECS = parseInt(process.argv[2] || '15');
const LABEL = process.argv[3] || 'auto';

async function run() {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--disable-gpu'] // no WebGPU in headless, will use CPU fallback
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 800 });

  // Navigate without waiting for all resources (data files are large)
  page.goto('http://localhost:3333/v4-physics/').catch(() => {});

  // Wait for page to load and sim to run
  await new Promise(r => setTimeout(r, 5000));

  // Dismiss onboarding modal
  await page.evaluate(() => {
    document.querySelectorAll('button').forEach(b => {
      if (b.textContent.includes('Exploring') || b.textContent.includes('Start')) b.click();
    });
  }).catch(() => {});

  // Wait for simulation to run
  console.log(`Waiting ${WAIT_SECS}s for sim to evolve...`);
  await new Promise(r => setTimeout(r, WAIT_SECS * 1000));

  // Get simulation stats
  const stats = await page.evaluate(() => {
    return {
      steps: typeof totalSteps !== 'undefined' ? totalSteps : 0,
      maxVel: typeof psi !== 'undefined' ? Math.max(...Array.from(psi).slice(0, 1000).map(Math.abs)) : 0,
      useGPU: typeof useGPU !== 'undefined' ? useGPU : false,
      showField: typeof showField !== 'undefined' ? showField : 'unknown',
      NX: typeof NX !== 'undefined' ? NX : 0,
      NY: typeof NY !== 'undefined' ? NY : 0,
    };
  }).catch(() => ({ steps: 0 }));

  console.log('Stats:', JSON.stringify(stats));

  // Take main screenshot
  const mainFile = `${dir}/${LABEL}-main.png`;
  await page.screenshot({ path: mainFile });
  console.log('Saved:', mainFile);

  // Press D for diagnostic grid
  await page.keyboard.press('d');
  await new Promise(r => setTimeout(r, 2000));
  const diagFile = `${dir}/${LABEL}-diag.png`;
  await page.screenshot({ path: diagFile });
  console.log('Saved:', diagFile);

  await browser.close();
  console.log('Done');
}

run().catch(e => console.error(e.message));
