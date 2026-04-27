// Tune radiative balance: find S_solar/A_olr that match observed SST
import puppeteer from 'puppeteer';
import { writeFileSync } from 'fs';

const WAIT = 25; // seconds per run
const URL = 'http://localhost:3333/v4-physics/';
const dir = '/Users/dereklomas/lukebarrington/amoc/screenshots/tune';
import { existsSync, mkdirSync } from 'fs';
if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

// Target: observed SST zonal means
const OBS = { '-60': 1.0, '-30': 21.1, '0': 27.7, '30': 22.9, '60': 5.8 };

const tests = [
  { name: 'baseline',    S_solar: 5.0, A_olr: 2.0, B_olr: 0.1 },
  { name: 'S7_A2',       S_solar: 7.0, A_olr: 2.0, B_olr: 0.1 },   // more solar
  { name: 'S5_A1.5',     S_solar: 5.0, A_olr: 1.5, B_olr: 0.1 },   // less OLR
  { name: 'S6_A1.5',     S_solar: 6.0, A_olr: 1.5, B_olr: 0.1 },   // more solar + less OLR
  { name: 'S7_A1.5',     S_solar: 7.0, A_olr: 1.5, B_olr: 0.1 },   // even more
  { name: 'S6_A1.5_B05', S_solar: 6.0, A_olr: 1.5, B_olr: 0.05 },  // slower restoring
  { name: 'S8_A2_B05',   S_solar: 8.0, A_olr: 2.0, B_olr: 0.05 },  // more solar + slower
];

async function run() {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,UseSkiaRenderer']
  });

  console.log('Tuning radiative balance...\n');
  console.log('Target SST: eq=27.7  30N=22.9  60N=5.8  30S=21.1\n');

  const results = [];

  for (const test of tests) {
    const page = await browser.newPage();
    await page.setViewport({ width: 1400, height: 800 });
    page.goto(URL).catch(() => {});
    await new Promise(r => setTimeout(r, 5000));
    await page.evaluate(() => {
      document.querySelectorAll('button').forEach(b => {
        if (b.textContent.includes('Start') || b.textContent.includes('Exploring')) b.click();
      });
    }).catch(() => {});
    await new Promise(r => setTimeout(r, 1000));

    // Set parameters
    await page.evaluate((p) => {
      S_solar = p.S_solar;
      A_olr = p.A_olr;
      B_olr = p.B_olr;
    }, test);

    console.log(`  ${test.name}: S=${test.S_solar} A=${test.A_olr} B=${test.B_olr}`);
    await new Promise(r => setTimeout(r, WAIT * 1000));

    const stats = await page.evaluate(() => {
      const z = {};
      for (const latDeg of [-60, -30, 0, 30, 60]) {
        const j = Math.round((latDeg - LAT0) / (LAT1 - LAT0) * (NY - 1));
        let s = 0, c = 0;
        for (let i = 0; i < NX; i++) { const k = j*NX+i; if (mask[k] && temp[k]) { s += temp[k]; c++; } }
        z[latDeg] = c > 0 ? s / c : null;
      }
      // Equilibrium prediction: T_eq = (S_solar * cos(latRad) - A_olr) / B_olr
      const eq = {};
      for (const lat of [-60, -30, 0, 30, 60]) {
        const cosZ = Math.cos(lat * Math.PI / 180);
        eq[lat] = (S_solar * cosZ - A_olr) / B_olr;
      }
      return { steps: totalSteps, zonalSST: z, equilibrium: eq, maxVel: 0 };
    }).catch(() => ({}));

    // Compute RMSE vs observed
    let sse = 0, cnt = 0;
    for (const [lat, obs] of Object.entries(OBS)) {
      const sim = stats.zonalSST?.[lat];
      if (sim != null) { sse += (sim - obs) ** 2; cnt++; }
    }
    const rmse = Math.sqrt(sse / cnt);

    console.log(`    SST: eq=${stats.zonalSST?.[0]?.toFixed(1)} 30N=${stats.zonalSST?.[30]?.toFixed(1)} 60N=${stats.zonalSST?.[60]?.toFixed(1)}  RMSE=${rmse.toFixed(2)}`);
    console.log(`    Eq:  eq=${stats.equilibrium?.[0]?.toFixed(0)} 30N=${stats.equilibrium?.[30]?.toFixed(0)} 60N=${stats.equilibrium?.[60]?.toFixed(0)}`);

    await page.screenshot({ path: `${dir}/${test.name}.png` });
    results.push({ ...test, ...stats, rmse });
    await page.close();
  }

  // Sort by RMSE
  results.sort((a, b) => a.rmse - b.rmse);
  console.log('\n=== Ranked by RMSE ===');
  for (const r of results) {
    console.log(`${r.name.padEnd(20)} RMSE=${r.rmse.toFixed(2)}  eq=${r.zonalSST?.[0]?.toFixed(1)} 30N=${r.zonalSST?.[30]?.toFixed(1)} 60N=${r.zonalSST?.[60]?.toFixed(1)}`);
  }

  writeFileSync(`${dir}/results.json`, JSON.stringify(results, null, 2));
  await browser.close();
}

run().catch(e => console.error(e));
