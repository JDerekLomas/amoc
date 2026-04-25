#!/usr/bin/env node
/**
 * sim-control.mjs — long-lived Playwright + HTTP API for simamoc.
 *
 * Loads simamoc ONCE in headless Chromium, then exposes the `lab` API over
 * HTTP so Claude (or any client) can drive the sim with curl. Stays alive
 * between requests — no re-init cost.
 *
 * Usage:
 *   node sim-control.mjs                  # serves on :8775
 *   curl localhost:8775/status
 *   curl localhost:8775/step?n=200
 *   curl -X POST localhost:8775/params -d '{"windStrength":1.2}'
 *   curl 'localhost:8775/render?field=temp' -o /tmp/sim-frame.png
 *
 * Static files are served from the repo root on :8773 (or reuses an existing
 * server on that port if one is already running).
 */

import { chromium } from 'playwright';
import { createServer } from 'http';
import { readFileSync, mkdirSync, writeFileSync } from 'fs';
import { resolve } from 'path';

const ROOT = '/Users/dereklomas/lukebarrington/amoc';
const STATIC_PORT = 8773;
const CONTROL_PORT = 8775;
const FRAME_DIR = '/tmp/sim-frames';
const LATEST_FRAME = '/tmp/sim-frame.png';

mkdirSync(FRAME_DIR, { recursive: true });

// ---------- static file server (skip if 8773 already in use) ----------
function startStaticServer() {
  return new Promise((resolve_, reject) => {
    const srv = createServer((req, res) => {
      const p = new URL(req.url, 'http://localhost').pathname.replace(/^\//, '');
      try {
        const d = readFileSync(resolve(ROOT, p));
        const ext = p.split('.').pop();
        const mime = {
          html: 'text/html', json: 'application/json',
          js: 'text/javascript', css: 'text/css', png: 'image/png',
          bin: 'application/octet-stream',
        }[ext] || 'application/octet-stream';
        res.writeHead(200, { 'Content-Type': mime });
        res.end(d);
      } catch { res.writeHead(404); res.end(); }
    });
    srv.once('error', (e) => {
      if (e.code === 'EADDRINUSE') { console.log(`[sim-control] reusing existing static server on :${STATIC_PORT}`); resolve_(null); }
      else reject(e);
    });
    srv.listen(STATIC_PORT, () => { console.log(`[sim-control] static server on :${STATIC_PORT}`); resolve_(srv); });
  });
}

await startStaticServer();

// ---------- launch Playwright ----------
console.log('[sim-control] launching headless chromium...');
const browser = await chromium.launch({
  headless: true,
  args: ['--disable-gpu', '--disable-webgpu'],
});
const ctx = await browser.newContext({ viewport: { width: 1200, height: 800 } });
const page = await ctx.newPage();

page.on('pageerror', (e) => console.log(`[page-error] ${e.message}`));
page.on('console', (m) => {
  const t = m.text();
  // Filter noisy logs but keep errors and key milestones
  if (m.type() === 'error' || /AMOC|FFT|NaN|loaded|init|error/i.test(t)) {
    console.log(`[page-${m.type()}] ${t}`);
  }
});

console.log(`[sim-control] loading http://localhost:${STATIC_PORT}/simamoc/...`);
await page.goto(`http://localhost:${STATIC_PORT}/simamoc/index.html`, { waitUntil: 'load', timeout: 60000 });
// Sim doesn't initialize until user clicks "Start Exploring" — wait for that button to exist, then click.
await page.waitForTimeout(3000); // let data load
try { await page.evaluate(() => document.getElementById('btn-start-exploring')?.click()); } catch {}
await page.waitForFunction(() => typeof totalSteps !== 'undefined', { timeout: 60000 });
await page.waitForTimeout(1500);
// Pause the rAF loop — we control stepping via the API. Keeps `page.evaluate` responsive.
await page.evaluate(() => {
  if (typeof lab !== 'undefined') lab.setParams({ stepsPerFrame: 1, yearSpeed: 1 });
  if (typeof paused !== 'undefined') paused = true;
});
console.log('[sim-control] sim ready (paused — drive via /step)');

// ---------- helpers ----------
async function readBody(req) {
  return new Promise((res, rej) => {
    let d = ''; req.on('data', (c) => d += c); req.on('end', () => res(d)); req.on('error', rej);
  });
}
function send(res, code, obj) {
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(obj, null, 2));
}

// Reference SST for RMSE
let refSST = null;
function loadRef() {
  try {
    const raw = JSON.parse(readFileSync(resolve(ROOT, 'sst_global_1deg.json'), 'utf8'));
    const data = raw.sst || raw.data || raw;
    refSST = data;
    return data.length;
  } catch { return null; }
}
loadRef();

// ---------- control server ----------
const ctrl = createServer(async (req, res) => {
  let url, path;
  try {
    url = new URL(req.url, 'http://localhost');
    path = url.pathname;
  } catch { res.writeHead(400); res.end(); return; }

  try {
    if (path === '/status') {
      const s = await page.evaluate(() => {
        let nan = 0, oc = 0, gSum = 0, gMin = 999, gMax = -999;
        for (let k = 0; k < NX * NY; k++) {
          if (mask[k]) {
            oc++;
            if (!isFinite(temp[k])) nan++;
            else { gSum += temp[k]; if (temp[k] < gMin) gMin = temp[k]; if (temp[k] > gMax) gMax = temp[k]; }
          }
        }
        return {
          totalSteps, simYears: simTime / T_YEAR,
          amoc: typeof amocStrength !== 'undefined' ? amocStrength : null,
          oceanCells: oc, nanCells: nan,
          globalMeanSST: (gSum / (oc - nan)),
          rangeSST: [gMin, gMax],
          showField,
          paused,
        };
      });
      send(res, 200, s); return;
    }

    if (path === '/step') {
      const n = parseInt(url.searchParams.get('n') || '100');
      const snapEvery = parseInt(url.searchParams.get('snapshot') || '0');
      const field = url.searchParams.get('field') || null;
      // Chunk size: small enough that each chunk yields back to the server
      // event loop quickly, even on a 1024×512 grid.
      const CHUNK = parseInt(url.searchParams.get('chunk') || '20');
      if (field) await page.evaluate((f) => { showField = f; }, field);
      const frames = [];
      const t0 = Date.now();
      let done = 0;
      while (done < n) {
        const k = Math.min(CHUNK, n - done);
        await page.evaluate((m) => lab.step(m), k);
        done += k;
        if (snapEvery > 0 && done % snapEvery < CHUNK) {
          const png = await page.screenshot({ type: 'png' });
          const fp = `${FRAME_DIR}/${String(frames.length).padStart(5, '0')}.png`;
          writeFileSync(fp, png);
          frames.push(fp);
        }
      }
      const elapsed = (Date.now() - t0) / 1000;
      const s = await page.evaluate(() => ({
        totalSteps, simYears: simTime / T_YEAR,
        amoc: typeof amocStrength !== 'undefined' ? amocStrength : null,
      }));
      send(res, 200, { ...s, elapsedSec: elapsed, stepsPerSec: n / elapsed, chunkSize: CHUNK, frames });
      return;
    }

    if (path === '/params') {
      if (req.method === 'POST') {
        const body = await readBody(req);
        const p = JSON.parse(body);
        // Handle alias
        const labP = { ...p };
        if ('r_friction' in labP) { labP.r = labP.r_friction; delete labP.r_friction; }
        if ('A_visc' in labP) { labP.A = labP.A_visc; delete labP.A_visc; }
        await page.evaluate((q) => lab.setParams(q), labP);
      }
      const cur = await page.evaluate(() => ({
        S_solar, A_olr, B_olr, kappa_diff, alpha_T,
        r_friction, A_visc, gamma_mix, gamma_deep_form, kappa_deep,
        F_couple_s, F_couple_d, r_deep, windStrength, H_surface, H_deep,
        kappa_atm, gamma_oa, gamma_ao, gamma_la,
        stepsPerFrame: typeof stepsPerFrame !== 'undefined' ? stepsPerFrame : null,
        yearSpeed: typeof yearSpeed !== 'undefined' ? yearSpeed : null,
      }));
      send(res, 200, cur); return;
    }

    if (path === '/render') {
      const field = url.searchParams.get('field') || null;
      if (field) {
        await page.evaluate((f) => { showField = f; }, field);
        await page.waitForTimeout(80); // let one frame render
      }
      const png = await page.screenshot({ type: 'png', fullPage: false });
      writeFileSync(LATEST_FRAME, png);
      res.writeHead(200, { 'Content-Type': 'image/png', 'X-Frame-Path': LATEST_FRAME });
      res.end(png);
      return;
    }

    if (path === '/diagnostics') {
      const data = await page.evaluate(() => {
        const nx = NX, ny = NY;
        const latBins = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70];
        const zonal = {};
        for (const lat of latBins) {
          const j = Math.round((lat - (-79.5)) / 159 * (ny - 1));
          if (j < 0 || j >= ny) continue;
          let sum = 0, c = 0;
          for (let i = 0; i < nx; i++) {
            const k = j * nx + i;
            if (mask[k] && isFinite(temp[k])) { sum += temp[k]; c++; }
          }
          zonal[lat] = c ? sum / c : null;
        }
        return { zonal, totalSteps, simYears: simTime / T_YEAR, amoc: amocStrength };
      });

      // Compute RMSE vs reference if available
      if (refSST) {
        const refNX = 360, refNY = 160;
        const latBins = Object.keys(data.zonal).map(Number);
        const refZonal = {};
        for (const lat of latBins) {
          const j = Math.round((lat + 79.5) / 1.0);
          if (j < 0 || j >= refNY) continue;
          let sum = 0, c = 0;
          for (let i = 0; i < refNX; i++) {
            const v = refSST[j * refNX + i];
            if (v != null && isFinite(v) && v > -90) { sum += v; c++; }
          }
          refZonal[lat] = c ? sum / c : null;
        }
        const rows = [];
        let sqe = 0, n = 0;
        for (const lat of latBins) {
          const sim = data.zonal[lat], obs = refZonal[lat];
          if (sim != null && obs != null) {
            const err = sim - obs;
            sqe += err * err; n++;
            rows.push({ lat, sim: +sim.toFixed(2), obs: +obs.toFixed(2), err: +err.toFixed(2) });
          }
        }
        data.rmse = n ? +Math.sqrt(sqe / n).toFixed(3) : null;
        data.zonalCompare = rows;
      }
      send(res, 200, data); return;
    }

    if (path === '/reset') {
      await page.evaluate(() => { if (typeof resetSim === 'function') resetSim(); });
      send(res, 200, { ok: true, msg: 'sim reset' }); return;
    }

    if (path === '/eval') {
      const body = await readBody(req);
      const result = await page.evaluate((code) => {
        try { return { ok: true, value: JSON.stringify(eval(code)) }; }
        catch (e) { return { ok: false, error: String(e) }; }
      }, body);
      send(res, 200, result); return;
    }

    if (path === '/' || path === '/help') {
      send(res, 200, {
        endpoints: [
          'GET  /status',
          'GET  /step?n=N[&snapshot=K][&field=temp]',
          'GET  /params',
          'POST /params  body: {param:value,...}',
          'GET  /render?field=temp',
          'GET  /diagnostics',
          'POST /reset',
          'POST /eval  body: <js expression>',
        ],
        latestFrame: LATEST_FRAME,
        frameDir: FRAME_DIR,
      }); return;
    }

    res.writeHead(404); res.end('not found');
  } catch (e) {
    send(res, 500, { error: e.message, stack: e.stack });
  }
});

ctrl.listen(CONTROL_PORT, () => {
  console.log(`[sim-control] control API on :${CONTROL_PORT} — try curl localhost:${CONTROL_PORT}/help`);
});

// keep alive
process.on('SIGINT', async () => { await browser.close(); process.exit(0); });
process.on('unhandledRejection', (e) => { console.log('[unhandledRejection]', e); });
process.on('uncaughtException', (e) => { console.log('[uncaughtException]', e); });
process.on('exit', (code) => { console.log('[exit] code=', code); });
// Pin event loop with infinite interval
setInterval(() => {}, 1 << 30);
