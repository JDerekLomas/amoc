// Parametric testing: run sim with different parameter values, compare results
// Usage: node scripts/_tmp-param-test.mjs
import puppeteer from 'puppeteer';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';

const dir = '/Users/dereklomas/lukebarrington/amoc/screenshots/sweep';
if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

const WAIT_SECS = 30; // let each run evolve this long
const BASE_URL = 'http://localhost:3333/v4-physics/';

// Parameter tests to run
const tests = [
  { name: 'baseline', params: {} },
  { name: 'B_olr_0.03', params: { B_olr: 0.03 } },     // slower SST restoring
  { name: 'B_olr_0.01', params: { B_olr: 0.01 } },     // much slower restoring
  { name: 'wind_2x', params: { windStrength: 2.0 } },    // stronger wind
  { name: 'S_solar_3', params: { S_solar: 3.0 } },       // weaker solar
  { name: 'S_solar_8', params: { S_solar: 8.0 } },       // stronger solar
  { name: 'friction_low', params: { r_friction: 0.02 } }, // less friction
];

async function runTest(browser, test) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 800 });

  page.goto(BASE_URL).catch(() => {});
  await new Promise(r => setTimeout(r, 5000));

  // Dismiss modal
  await page.evaluate(() => {
    document.querySelectorAll('button').forEach(b => {
      if (b.textContent.includes('Start') || b.textContent.includes('Exploring')) b.click();
    });
  }).catch(() => {});

  await new Promise(r => setTimeout(r, 2000));

  // Apply parameter overrides
  if (Object.keys(test.params).length > 0) {
    await page.evaluate((params) => {
      for (const [key, val] of Object.entries(params)) {
        if (typeof window[key] !== 'undefined') {
          window[key] = val;
          console.log('Set ' + key + ' = ' + val);
        }
      }
    }, test.params);
  }

  // Let it evolve
  console.log(`  Running "${test.name}" for ${WAIT_SECS}s...`);
  await new Promise(r => setTimeout(r, WAIT_SECS * 1000));

  // Collect stats
  const stats = await page.evaluate(() => {
    // Compute SST zonal means
    const zonalSST = {};
    for (let latDeg = -60; latDeg <= 60; latDeg += 30) {
      const j = Math.round((latDeg - LAT0) / (LAT1 - LAT0) * (NY - 1));
      let sum = 0, cnt = 0;
      for (let i = 0; i < NX; i++) {
        const k = j * NX + i;
        if (mask[k] && temp[k]) { sum += temp[k]; cnt++; }
      }
      zonalSST[latDeg] = cnt > 0 ? (sum / cnt) : null;
    }

    // Max velocity
    let maxV = 0;
    for (let j = 1; j < NY - 1; j++) {
      for (let i = 0; i < NX; i++) {
        const k = j * NX + i;
        if (!mask[k]) continue;
        const ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;
        const u = -(psi[(j+1)*NX+i] - psi[(j-1)*NX+i]) * 0.5 * invDy;
        const v = (psi[j*NX+ip1] - psi[j*NX+im1]) * 0.5 * invDx;
        const spd = Math.sqrt(u*u + v*v);
        if (spd > maxV) maxV = spd;
      }
    }

    return {
      steps: totalSteps,
      amoc: amocStrength,
      maxVel: maxV,
      zonalSST,
      S_solar, A_olr, B_olr, r_friction, windStrength, alpha_T, beta_S
    };
  }).catch(() => ({ error: true }));

  // Screenshot
  await page.screenshot({ path: `${dir}/${test.name}.png` });

  await page.close();
  return { name: test.name, ...stats };
}

async function main() {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,UseSkiaRenderer']
  });

  console.log('=== Parameter Sweep ===\n');
  const results = [];

  for (const test of tests) {
    const result = await runTest(browser, test);
    results.push(result);
    console.log(`  ${test.name}: steps=${result.steps} maxVel=${result.maxVel?.toFixed(2)} AMOC=${result.amoc?.toFixed(4)}`);
    if (result.zonalSST) {
      console.log(`    SST zonal: eq=${result.zonalSST[0]?.toFixed(1)} 30N=${result.zonalSST[30]?.toFixed(1)} 60N=${result.zonalSST[60]?.toFixed(1)} 30S=${result.zonalSST[-30]?.toFixed(1)}`);
    }
    console.log('');
  }

  // Observed SST for comparison
  console.log('=== Observed SST zonal means ===');
  console.log('  eq=27.7  30N=22.9  60N=5.8  30S=21.1');

  // Summary
  console.log('\n=== Summary ===');
  for (const r of results) {
    console.log(`${r.name.padEnd(20)} steps=${String(r.steps).padStart(6)} maxVel=${(r.maxVel||0).toFixed(2).padStart(6)} AMOC=${(r.amoc||0).toFixed(4).padStart(8)}`);
  }

  writeFileSync(`${dir}/results.json`, JSON.stringify(results, null, 2));
  console.log('\nResults saved to ' + dir + '/results.json');

  await browser.close();
}

main().catch(e => console.error(e));
