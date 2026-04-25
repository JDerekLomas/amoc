// helm-lab — headless harness for the v4-physics ocean simulator.
//
// Drives the same sim-engine.js the browser uses, via Playwright Chromium
// in --headless=new mode (real Apple Metal WebGPU on macOS). Exposes the
// engine's window.lab API as a clean Node-side class.
//
// The browser is an implementation detail. Callers just see set / step /
// render / diag / fields / trajectory.

import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile, mkdir, appendFile, writeFile } from 'node:fs/promises';
import { existsSync, statSync, createReadStream } from 'node:fs';
import { extname, join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const MIME = {
  '.html': 'text/html', '.js': 'application/javascript', '.mjs': 'application/javascript',
  '.json': 'application/json', '.css': 'text/css', '.png': 'image/png', '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon', '.txt': 'text/plain', '.geojson': 'application/json',
  '.wasm': 'application/wasm', '.bin': 'application/octet-stream'
};

function startStaticServer(root, preferredPort = 8810) {
  return new Promise((res, rej) => {
    const server = createServer((req, resp) => {
      try {
        const url = new URL(req.url, 'http://localhost');
        let p = decodeURIComponent(url.pathname);
        if (p.endsWith('/')) p += 'index.html';
        const f = join(root, p);
        if (!f.startsWith(root)) { resp.statusCode = 403; return resp.end(); }
        if (!existsSync(f) || !statSync(f).isFile()) { resp.statusCode = 404; return resp.end('not found: ' + p); }
        resp.setHeader('content-type', MIME[extname(f)] || 'application/octet-stream');
        resp.setHeader('access-control-allow-origin', '*');
        createReadStream(f).pipe(resp);
      } catch (e) { resp.statusCode = 500; resp.end(e.message); }
    });
    server.on('error', rej);
    server.listen(preferredPort, '127.0.0.1', () => res({ server, port: preferredPort }));
  });
}

export class HelmLab {
  constructor({ root = PROJECT_ROOT, page = 'v4-physics/console.html', port = 8810, verbose = false } = {}) {
    this.root = root; this.pagePath = page; this.port = port; this.verbose = verbose;
    this.browser = null; this.page = null; this.server = null;
  }

  log(...a) { if (this.verbose) console.log('[helm-lab]', ...a); }

  async start() {
    const { server, port } = await startStaticServer(this.root, this.port);
    this.server = server; this.port = port;
    this.log(`server :${port} -> ${this.root}`);

    this.browser = await chromium.launch({
      // Headless:false + --headless=new is the magic combo that gets real
      // Apple Metal WebGPU instead of the SwiftShader CPU rasterizer.
      headless: false,
      args: [
        '--headless=new',
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,UseSkiaRenderer,WebGPU',
        '--no-sandbox',
        '--mute-audio',
      ],
    });
    this.page = await this.browser.newPage({ viewport: { width: 1280, height: 720 } });
    this.page.on('console', m => { if (this.verbose && m.type() !== 'log') this.log(`page ${m.type()}:`, m.text()); });
    this.page.on('pageerror', e => this.log('page error:', e.message));

    const url = `http://localhost:${port}/${this.pagePath}`;
    this.log('navigate', url);
    await this.page.goto(url, { waitUntil: 'domcontentloaded' });

    // Auto-dismiss onboarding so .viewport is unobstructed.
    await this.page.evaluate(() => {
      try { localStorage.setItem('amoc-onboarded', '1'); } catch (e) {}
      const ov = document.getElementById('onboarding-overlay');
      if (ov) ov.classList.add('hidden');
    });

    // Wait for window.lab AND for engine init (totalSteps moves or useGPU is set).
    await this.page.waitForFunction(() => !!(window.lab && window.lab._version), null, { timeout: 30000 });
    // Wait for engine init to finish — useGPU populated, mask loaded.
    const t0 = Date.now();
    while (Date.now() - t0 < 30000) {
      const ready = await this.page.evaluate(() => {
        try { const p = window.lab.getParams(); return { ok: typeof p.useGPU === 'boolean' && p.NX > 0, useGPU: p.useGPU, NX: p.NX }; }
        catch { return { ok: false }; }
      });
      if (ready.ok) { this.log('engine ready', ready); break; }
      await new Promise(r => setTimeout(r, 200));
    }
    return this;
  }

  async stop() {
    if (this.browser) { try { await this.browser.close(); } catch {} }
    if (this.server) {
      try { this.server.closeAllConnections?.(); } catch {}
      await new Promise(r => this.server.close(r));
    }
  }

  // ---- thin wrappers over window.lab ----
  getParams()       { return this.page.evaluate(() => window.lab.getParams()); }
  setParams(p)      { return this.page.evaluate(p => window.lab.setParams(p), p); }
  step(n)           { return this.page.evaluate(n => window.lab.step(n), n); }
  diag(opts = {})   { return this.page.evaluate(o => window.lab.diagnostics(o), opts); }
  reset()           { return this.page.evaluate(() => window.lab.reset()); }
  pause()           { return this.page.evaluate(() => window.lab.pause()); }
  resume()          { return this.page.evaluate(() => window.lab.resume()); }
  setView(v)        { return this.page.evaluate(v => window.lab.view(v), v); }
  scenario(name)    { return this.page.evaluate(n => window.lab.scenario(n), name); }

  /** Pull raw fields back to Node as plain Float32Arrays. Heavy — 1MB per field. */
  async fields(names = ['psi', 'temp', 'deepTemp', 'deepPsi']) {
    return this.page.evaluate(ns => {
      const f = window.lab.fields();
      const out = { NX: f.NX, NY: f.NY };
      for (const n of ns) if (f[n]) out[n] = Array.from(f[n]);
      return out;
    }, names);
  }

  /**
   * Render the current canvas as a clean PNG (1024x512 native, no HUD).
   * If view is given, switch first, give the renderer two animation frames
   * to settle, then capture.
   */
  async render(outPath, { view = null, includeOverlay = false } = {}) {
    if (view) {
      await this.setView(view);
      // Two frames + a pause so view-mode shader settles.
      await this.page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
      await this.page.waitForTimeout(120);
    }
    if (includeOverlay) {
      // Capture the .viewport block — includes HUD, brackets, compass.
      const sel = await this.page.$('.viewport');
      await sel.screenshot({ path: outPath });
    } else {
      // WebGPU canvases don't expose their pixels to 2D contexts (the
      // back-buffer is invalidated after present, and createImageBitmap
      // returns blank in this Chromium config). The reliable path is
      // Playwright's screenshot() which reads from the compositor.
      // Force a render frame so the GPU canvas has fresh content.
      await this.page.evaluate(async () => {
        try { if (typeof gpuRenderField === 'function') gpuRenderField(); } catch {}
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      });
      // Get the .viewport box and clip the page screenshot to it. Includes
      // both stacked canvases naturally; HUD overlays are inside this box too,
      // but we accept that as part of the "view" (lat/lon ticks are useful).
      // For pure-canvas captures, temporarily hide HUD/brackets/compass.
      await this.page.evaluate(() => {
        const els = document.querySelectorAll('.hud, .viewport-frame, .compass, .palette, .viewport-graticule, .brush-sz, #scenario-explanation');
        els.forEach(e => { e.dataset._hlVis = e.style.visibility || ''; e.style.visibility = 'hidden'; });
      });
      try {
        const box = await this.page.evaluate(() => {
          const v = document.querySelector('.viewport').getBoundingClientRect();
          return { x: v.x, y: v.y, w: v.width, h: v.height };
        });
        await mkdir(dirname(outPath), { recursive: true });
        await this.page.screenshot({
          path: outPath,
          clip: { x: Math.round(box.x), y: Math.round(box.y), width: Math.round(box.w), height: Math.round(box.h) },
        });
      } finally {
        await this.page.evaluate(() => {
          const els = document.querySelectorAll('.hud, .viewport-frame, .compass, .palette, .viewport-graticule, .brush-sz, #scenario-explanation');
          els.forEach(e => { e.style.visibility = e.dataset._hlVis || ''; delete e.dataset._hlVis; });
        });
      }
    }
    return outPath;
  }

  /**
   * Run a trajectory with periodic sampling. Writes individual frames
   * (per view), a JSONL of diagnostics, then composes a contact sheet
   * (one PNG showing all frames in a grid) and a sparkline summary.
   */
  async trajectory({ totalSteps, sampleEvery, views = ['temp'], outDir, includeFields = false }) {
    await mkdir(outDir, { recursive: true });
    await mkdir(join(outDir, 'frames'), { recursive: true });
    const jsonlPath = join(outDir, 'diag.jsonl');
    await writeFile(jsonlPath, ''); // truncate
    const samples = [];
    let elapsed = 0;
    // Initial sample at t=0.
    const sample = async (tag) => {
      const d = await this.diag({ profiles: true });
      const frames = {};
      for (const v of views) {
        const fp = join(outDir, 'frames', `t${tag}_${v}.png`);
        await this.render(fp, { view: v });
        frames[v] = fp;
      }
      const entry = { t: elapsed, simYears: d.simYears, frames, diag: d };
      samples.push(entry);
      // Strip array-heavy profile data from the JSONL line; keep separately if needed.
      const { zonalMeanT, zonalMeanPsi, zonalMeanU, latitudes, ...lean } = d;
      await appendFile(jsonlPath, JSON.stringify({ t: elapsed, ...lean }) + '\n');
      this.log(`sample t=${elapsed} simYr=${d.simYears.toFixed(2)} KE=${d.KE.toExponential(2)} amoc=${d.amoc.toFixed(4)}`);
      return entry;
    };
    await sample(String(elapsed).padStart(8, '0'));
    while (elapsed < totalSteps) {
      const n = Math.min(sampleEvery, totalSteps - elapsed);
      await this.step(n);
      elapsed += n;
      await sample(String(elapsed).padStart(8, '0'));
    }
    await this.composeContactSheet(samples, join(outDir, 'contact-sheet.png'));
    await this.composeSummary(samples, join(outDir, 'summary.png'));
    await writeFile(join(outDir, 'samples.json'), JSON.stringify(samples.map(s => ({ t: s.t, simYears: s.simYears, frames: s.frames, diag: stripArrays(s.diag) })), null, 2));
    return samples;
  }

  /** Run a parameter sweep: for each value, set param, spin up, sample. */
  async sweep({ param, values, spinupSteps = 20000, postSteps = 50000, views = ['temp'], outDir, resetEach = true }) {
    await mkdir(outDir, { recursive: true });
    const points = [];
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      this.log(`sweep ${param}=${v} (${i + 1}/${values.length})`);
      if (resetEach) await this.reset();
      await this.setParams({ [param]: v });
      if (spinupSteps) await this.step(spinupSteps);
      await this.step(postSteps);
      const d = await this.diag({ profiles: true });
      const frames = {};
      for (const view of views) {
        const fp = join(outDir, 'frames', `${param}_${formatVal(v)}_${view}.png`);
        await this.render(fp, { view });
        frames[view] = fp;
      }
      points.push({ value: v, frames, diag: d });
    }
    await this.composeSweep(points, param, join(outDir, 'sweep-sheet.png'));
    await writeFile(join(outDir, 'sweep.json'), JSON.stringify(points.map(p => ({ value: p.value, frames: p.frames, diag: stripArrays(p.diag) })), null, 2));
    return points;
  }

  // ---- in-page composition (no node-canvas needed) ----

  async composeContactSheet(samples, outPath) {
    const data = samples.map(s => ({
      t: s.t, simYears: s.simYears,
      frames: Object.fromEntries(Object.entries(s.frames).map(([k, p]) => [k, p])),
      diag: { KE: s.diag.KE, amoc: s.diag.amoc, globalSST: s.diag.globalSST, tropicalSST: s.diag.tropicalSST, polarSST: s.diag.polarSST, maxVel: s.diag.maxVel, iceArea: s.diag.iceArea }
    }));
    // Read all frames into base64 — small set, fine to inline.
    for (const s of data) {
      for (const v in s.frames) {
        s.frames[v] = 'data:image/png;base64,' + (await readFile(s.frames[v])).toString('base64');
      }
    }
    const png = await this.page.evaluate((D) => composeContactSheetInPage(D), data);
    await writeFile(outPath, Buffer.from(png.split(',')[1], 'base64'));
    return outPath;
  }

  async composeSweep(points, paramName, outPath) {
    const data = [];
    for (const p of points) {
      const frames = {};
      for (const v in p.frames) {
        frames[v] = 'data:image/png;base64,' + (await readFile(p.frames[v])).toString('base64');
      }
      data.push({ value: p.value, frames, diag: { KE: p.diag.KE, amoc: p.diag.amoc, globalSST: p.diag.globalSST, tropicalSST: p.diag.tropicalSST, polarSST: p.diag.polarSST } });
    }
    const png = await this.page.evaluate(({ D, P }) => composeSweepInPage(D, P), { D: data, P: paramName });
    await writeFile(outPath, Buffer.from(png.split(',')[1], 'base64'));
    return outPath;
  }

  async composeSummary(samples, outPath) {
    const series = samples.map(s => ({
      t: s.t, simYears: s.simYears,
      KE: s.diag.KE, amoc: s.diag.amoc, maxVel: s.diag.maxVel,
      globalSST: s.diag.globalSST, tropicalSST: s.diag.tropicalSST, polarSST: s.diag.polarSST,
      iceArea: s.diag.iceArea, accU: s.diag.accU
    }));
    const png = await this.page.evaluate(s => composeSummaryInPage(s), series);
    await writeFile(outPath, Buffer.from(png.split(',')[1], 'base64'));
    return outPath;
  }

  /**
   * Bootstrap composition helpers into the page. Call once after start.
   * Idempotent.
   */
  async installComposers() {
    const composer = await readFile(join(dirname(fileURLToPath(import.meta.url)), 'composer.js'), 'utf8');
    await this.page.evaluate(c => { (0, eval)(c); }, composer);
  }
}

function stripArrays(d) {
  const { zonalMeanT, zonalMeanPsi, zonalMeanU, latitudes, ...rest } = d || {};
  return rest;
}
function formatVal(v) {
  if (typeof v !== 'number') return String(v);
  if (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e4) return v.toExponential(2).replace('+', '').replace('.', 'p');
  return v.toFixed(4).replace('.', 'p').replace(/p?0+$/, '');
}
