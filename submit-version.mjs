#!/usr/bin/env node
/**
 * Submit a new version of SimAMOC to the leaderboard.
 *
 * Snapshots the current simamoc/index.html, runs evaluation,
 * records scores, and adds to the leaderboard.
 *
 * Usage:
 *   node submit-version.mjs --author "Derek" --name "Polar OLR + Brine" --description "Added polar OLR boost, brine rejection, Ekman transport"
 *   node submit-version.mjs --author "Luke" --name "Spectral winds" --description "ERA5-derived spectral wind forcing"
 */

import { chromium } from 'playwright';
import { mkdirSync, writeFileSync, readFileSync, copyFileSync, existsSync } from 'fs';
import { createServer } from 'http';
import { resolve } from 'path';

const ROOT = '/Users/dereklomas/lukebarrington/amoc';
const SPINUP = parseInt(process.argv.find((_, i, a) => a[i - 1] === '--spinup') || '150');
const AUTHOR = process.argv.find((_, i, a) => a[i - 1] === '--author') || 'Unknown';
const NAME = process.argv.find((_, i, a) => a[i - 1] === '--name') || 'Unnamed';
const DESC = process.argv.find((_, i, a) => a[i - 1] === '--description') || '';
const PARAMS_JSON = process.argv.find((_, i, a) => a[i - 1] === '--params') || '{}';
const params = JSON.parse(PARAMS_JSON);

// Generate version ID: author-YYYYMMDD-HHMMSS
const now = new Date();
const ts = now.toISOString().replace(/[-:T]/g, '').slice(0, 15);
const slug = AUTHOR.toLowerCase().replace(/[^a-z0-9]/g, '');
const VERSION_ID = `${slug}-${ts}`;
const VERSION_DIR = `versions/${VERSION_ID}`;

// Reference SST (NOAA OI SST 1991-2020, 10° bands)
const REF_SST = {
  '-70': -1.0, '-60': 0.5, '-50': 4.0, '-40': 10.0, '-30': 17.0,
  '-20': 23.0, '-10': 26.5, '0': 27.0, '10': 26.5, '20': 24.0,
  '30': 20.0, '40': 14.0, '50': 7.0, '60': 2.0, '70': -1.0,
};

async function main() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║  SimAMOC Version Submission                            ║');
  console.log('╚══════════════════════════════════════════════════════════╝');
  console.log(`  Author:  ${AUTHOR}`);
  console.log(`  Name:    ${NAME}`);
  console.log(`  Version: ${VERSION_ID}`);
  console.log(`  Params:  ${JSON.stringify(params)}\n`);

  // 1. Snapshot the current simulation
  mkdirSync(VERSION_DIR, { recursive: true });
  copyFileSync('simamoc/index.html', `${VERSION_DIR}/index.html`);
  copyFileSync('simamoc/model.js', `${VERSION_DIR}/model.js`);
  copyFileSync('simamoc/mask.json', `${VERSION_DIR}/mask.json`);
  copyFileSync('simamoc/coastlines.json', `${VERSION_DIR}/coastlines.json`);
  if (existsSync('simamoc/input-widget.js')) {
    copyFileSync('simamoc/input-widget.js', `${VERSION_DIR}/input-widget.js`);
  }
  console.log(`  Snapshot saved to ${VERSION_DIR}/`);

  // 2. Run evaluation
  console.log(`  Running evaluation (${SPINUP}s spinup)...`);

  const server = createServer((req, res) => {
    const urlPath = new URL(req.url, 'http://localhost').pathname;
    const filePath = resolve(ROOT, urlPath.replace(/^\//, ''));
    try {
      const data = readFileSync(filePath);
      const ext = filePath.split('.').pop();
      const mime = { html: 'text/html', json: 'application/json', js: 'text/javascript' }[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': mime });
      res.end(data);
    } catch { res.writeHead(404); res.end('Not found'); }
  });
  await new Promise(r => server.listen(8775, r));

  const browser = await chromium.launch({
    headless: false,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
  });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1400, height: 900 });

  // Load from the version directory (to evaluate the snapshot, not current main)
  await page.goto(`http://localhost:8775/${VERSION_DIR}/index.html`, { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(3000);
  try { await page.click('#btn-start-exploring', { timeout: 3000 }); } catch {}
  await page.waitForTimeout(1000);

  // Reset and inject params
  await page.click('#btn-reset');
  if (Object.keys(params).length > 0) {
    await page.evaluate((p) => {
      for (const [k, v] of Object.entries(p)) {
        try { eval(`${k} = ${v}`); } catch {}
      }
    }, params);
  }

  // Max sim speed
  await page.evaluate(() => {
    document.getElementById('speed-slider').value = 150;
    document.getElementById('speed-slider').dispatchEvent(new Event('input'));
    document.getElementById('year-speed-slider').value = 3;
    document.getElementById('year-speed-slider').dispatchEvent(new Event('input'));
  });

  console.log(`  Spinning up for ${SPINUP}s...`);
  await page.waitForTimeout(SPINUP * 1000);

  // Screenshots
  const views = ['temp', 'psi', 'speed', 'deeptemp'];
  for (const view of views) {
    await page.click(`#btn-${view}`);
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${VERSION_DIR}/${view}.png` });
  }
  await page.click('#btn-temp');

  // Full-page screenshot for leaderboard thumbnail
  await page.screenshot({ path: `${VERSION_DIR}/thumbnail.png` });

  // Diagnostics
  const diagnostics = await page.evaluate(() => {
    if (!window.lab) return {};
    return window.lab.diagnostics({ profiles: true });
  });

  // Compute RMSE
  let sqSum = 0, nBands = 0;
  const errors = [];
  if (diagnostics.zonalMeanT) {
    const NY = diagnostics.zonalMeanT.length;
    const LAT0 = -80, LAT1 = 80;
    for (const [latStr, refT] of Object.entries(REF_SST)) {
      const lat = parseFloat(latStr);
      const j = Math.round((lat - LAT0) / (LAT1 - LAT0) * (NY - 1));
      const simT = diagnostics.zonalMeanT[j];
      if (simT !== undefined && !isNaN(simT)) {
        const err = simT - refT;
        errors.push({ lat, simT: +simT.toFixed(2), refT, error: +err.toFixed(2) });
        sqSum += err * err;
        nBands++;
      }
    }
  }
  const rmse = nBands > 0 ? Math.sqrt(sqSum / nBands) : 999;

  // Quick performance benchmark (~3s)
  console.log('  Running performance benchmark...');
  let perf = {};
  try {
    perf = await page.evaluate(async () => {
      if (!window.lab || !window.lab.benchmark) return {};
      return await window.lab.benchmark();
    });
    console.log(`  FPS: ${perf.fps || '--'} | Steps/sec: ${perf.stepsPerSec || '--'} | Stable: ${perf.stable ?? '--'}`);
  } catch (e) {
    console.log(`  Benchmark skipped: ${e.message.slice(0, 80)}`);
  }

  await browser.close();
  server.close();

  // 3. Build version metadata
  const version = {
    id: VERSION_ID,
    author: AUTHOR,
    name: NAME,
    description: DESC,
    date: now.toISOString(),
    params,
    rmse: +rmse.toFixed(2),
    globalSST: +(diagnostics.globalSST || 0).toFixed(1),
    tropicalSST: +(diagnostics.tropicalSST || 0).toFixed(1),
    polarSST: +(diagnostics.polarSST || 0).toFixed(1),
    amocSv: +(diagnostics.amocSv || 0).toFixed(1),
    accU: +(diagnostics.accU || 0).toFixed(3),
    iceArea: diagnostics.iceArea || 0,
    perf: {
      fps: perf.fps || null,
      stepsPerSec: perf.stepsPerSec || null,
      avgFrameMs: perf.avgFrameMs || null,
      p95FrameMs: perf.p95FrameMs || null,
      jitterMs: perf.jitterMs || null,
      stable: perf.stable != null ? perf.stable : null,
    },
    errors,
    path: VERSION_DIR,
  };

  writeFileSync(`${VERSION_DIR}/metadata.json`, JSON.stringify(version, null, 2));

  // 4. Update leaderboard
  const scoresPath = 'versions/scores.json';
  let scores = [];
  if (existsSync(scoresPath)) {
    scores = JSON.parse(readFileSync(scoresPath, 'utf8'));
  }
  scores.push(version);
  scores.sort((a, b) => a.rmse - b.rmse);
  writeFileSync(scoresPath, JSON.stringify(scores, null, 2));

  // 5. Regenerate leaderboard HTML
  generateLeaderboard(scores);

  // 6. Print results
  console.log(`\n${'═'.repeat(60)}`);
  console.log('  SUBMISSION COMPLETE');
  console.log(`${'═'.repeat(60)}`);
  console.log(`  Version:  ${VERSION_ID}`);
  console.log(`  RMSE:     ${rmse.toFixed(2)}°C`);
  console.log(`  Global:   ${diagnostics.globalSST?.toFixed(1)}°C`);
  console.log(`  Tropical: ${diagnostics.tropicalSST?.toFixed(1)}°C`);
  console.log(`  Polar:    ${diagnostics.polarSST?.toFixed(1)}°C`);
  console.log(`  AMOC:     ${diagnostics.amocSv?.toFixed(1)} Sv`);
  if (perf.fps) {
    console.log(`  FPS:      ${perf.fps}`);
    console.log(`  Steps/s:  ${perf.stepsPerSec}`);
    console.log(`  P95 frame:${perf.p95FrameMs}ms`);
    console.log(`  Jitter:   ${perf.jitterMs}ms`);
    console.log(`  Stable:   ${perf.stable}`);
  }
  console.log(`\n  Leaderboard:`);
  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    const medal = i === 0 ? ' ***' : '';
    const me = s.id === VERSION_ID ? ' <-- NEW' : '';
    console.log(`    ${i + 1}. ${s.author.padEnd(10)} ${s.rmse.toFixed(2)}°C  ${s.name}${medal}${me}`);
  }
  console.log(`\n  View: http://localhost:8780/leaderboard/`);
  console.log(`  Play: http://localhost:8780/${VERSION_DIR}/`);
}

function generateLeaderboard(scores) {
  const rows = scores.map((s, i) => {
    const medal = i === 0 ? '&#129351;' : i === 1 ? '&#129352;' : i === 2 ? '&#129353;' : `${i + 1}.`;
    const p = s.perf || {};
    const fpsStr = p.fps ? `${p.fps}` : '--';
    const spsStr = p.stepsPerSec ? `${(p.stepsPerSec/1000).toFixed(0)}k` : '--';
    const stableStr = p.stable === true ? '&#10003;' : p.stable === false ? '&#10007;' : '--';
    return `
    <tr class="entry ${i === 0 ? 'winner' : ''}">
      <td class="rank">${medal}</td>
      <td class="author">${esc(s.author)}</td>
      <td class="name"><a href="../${s.path}/">${esc(s.name)}</a></td>
      <td class="rmse">${s.rmse.toFixed(2)}&deg;C</td>
      <td class="tropical">${s.tropicalSST.toFixed(1)}&deg;</td>
      <td class="polar">${s.polarSST.toFixed(1)}&deg;</td>
      <td class="amoc">${s.amocSv.toFixed(1)}</td>
      <td class="perf">${fpsStr} fps / ${spsStr} steps/s</td>
      <td class="stable">${stableStr}</td>
      <td class="date">${s.date.slice(0, 10)}</td>
    </tr>`;
  }).join('\n');

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SimAMOC Leaderboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#080e18;color:#c8d8e8;font-family:'Segoe UI',system-ui,sans-serif;line-height:1.7}
.container{max-width:1000px;margin:0 auto;padding:40px 24px}
h1{font-size:2em;font-weight:300;color:#7ab8de;margin-bottom:4px}
.sub{color:#5a7e98;font-size:.9em;margin-bottom:32px}
.sub a{color:#5a9ec8;text-decoration:none}
.sub a:hover{text-decoration:underline}
table{width:100%;border-collapse:collapse;margin-bottom:32px}
th{text-align:left;font-size:.7em;color:#4a6a82;text-transform:uppercase;letter-spacing:.08em;padding:8px 12px;border-bottom:2px solid #1a2838}
td{padding:10px 12px;border-bottom:1px solid #1a2838;font-size:.9em}
tr:hover{background:rgba(74,158,200,0.05)}
tr.winner{background:rgba(74,186,112,0.08)}
.rank{font-size:1.2em;text-align:center;width:40px}
.author{font-weight:600;color:#8cc8e8}
.name a{color:#a0c8e0;text-decoration:none}
.name a:hover{text-decoration:underline;color:#c0e8ff}
.rmse{font-weight:700;color:#4aba70;font-variant-numeric:tabular-nums;font-size:1.1em}
tr.winner .rmse{color:#4aba70}
.tropical,.polar,.amoc{color:#6a8aa4;font-variant-numeric:tabular-nums}
.date{color:#4a6a82;font-size:.8em}
.perf{color:#6a8aa4;font-size:.8em;font-variant-numeric:tabular-nums}
.stable{text-align:center;font-size:1.1em}
.info{background:#0c1420;border:1px solid #1a2838;border-radius:8px;padding:16px;margin-bottom:24px;font-size:.85em;color:#8aa8c0}
.info code{background:#1a2838;padding:2px 6px;border-radius:3px;font-size:.9em}
.target{color:#e0a050}
</style>
</head>
<body>
<div class="container">
  <h1>SimAMOC Leaderboard</h1>
  <p class="sub">Competitive ocean model development &mdash; who can get closest to observations?<br>
  <a href="../simamoc/">Launch Simulator</a> &nbsp;|&nbsp; <a href="../blog/">Blog</a> &nbsp;|&nbsp; <a href="https://github.com/JDerekLomas/amoc">GitHub</a></p>

  <div class="info">
    <strong>Scoring:</strong> RMSE against NOAA OI SST v2 (1991&ndash;2020 climatology) at 15 latitude bands.
    <span class="target">Target: &lt; 3.0&deg;C</span>.
    Observations: tropical SST ~27&deg;C, polar ~&minus;1&deg;C, AMOC ~17 Sv.<br>
    <strong>Submit:</strong> <code>node submit-version.mjs --author "Name" --name "Description" --params '{...}'</code>
  </div>

  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Author</th>
        <th>Version</th>
        <th>RMSE</th>
        <th>Trop</th>
        <th>Polar</th>
        <th>AMOC (Sv)</th>
        <th>Performance</th>
        <th>Stable</th>
        <th>Date</th>
      </tr>
    </thead>
    <tbody>
${rows}
    </tbody>
  </table>

  <div class="info">
    <strong>How it works:</strong> Each version is a snapshot of <code>simamoc/</code> (model.js + index.html) with specific physics changes.
    Submit your version, and the evaluation harness runs a ${scores[0]?.params ? '150' : '150'}s spinup, then compares zonal mean SST against NOAA observations.
    Click any version name to play that version live in your browser.
  </div>
</div>
</body>
</html>`;

  writeFileSync('leaderboard/index.html', html);
  console.log('  Leaderboard updated: leaderboard/index.html');
}

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
