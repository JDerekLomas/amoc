#!/usr/bin/env node
/**
 * Headless sim capture — runs v4-physics in Puppeteer, extracts model state.
 *
 * Usage:
 *   node scripts/capture-sim.mjs [url] [steps] [output-dir]
 *   node scripts/capture-sim.mjs https://amoc-xxx.vercel.app/v4-physics/ 25000 notebooks/runs/
 *   node scripts/capture-sim.mjs local 25000       # uses localhost:8080
 *
 * Outputs to [output-dir]/[hash]-[steps].json:
 *   { params, diagnostics, zonalMeanT, sst (flat array), screenshot (base64) }
 */

import puppeteer from 'puppeteer';
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const url = process.argv[2] || 'https://amoc.vercel.app/v4-physics/';
const steps = parseInt(process.argv[3] || '25000');
const outDir = process.argv[4] || 'notebooks/runs';

mkdirSync(outDir, { recursive: true });

console.log(`Capturing: ${url}`);
console.log(`Steps: ${steps}`);

const browser = await puppeteer.launch({
  headless: 'new',
  args: ['--enable-unsafe-webgpu', '--enable-gpu', '--use-angle=metal',
         '--no-sandbox', '--disable-setuid-sandbox'],
});

const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720 });

// Suppress console noise
page.on('console', () => {});

console.log('Loading page...');
await page.goto(url, { waitUntil: 'networkidle0', timeout: 30000 });

// Wait for lab API
await page.waitForFunction('typeof window.lab !== "undefined" && window.lab.diagnostics', { timeout: 15000 });
console.log('Lab API ready');

// Pause, run steps, extract
console.log(`Running ${steps} steps...`);
const t0 = Date.now();

const result = await page.evaluate(async (nSteps) => {
  const lab = window.lab;
  lab.pause();
  await lab.step(nSteps);

  const params = lab.getParams();
  const diag = lab.diagnostics({ profiles: true });
  const f = lab.fields();

  // RMSE computed in notebook (obsSSTData is module-scoped, not on window)

  // Extract SST as array (downsample to 360x180 for manageable size)
  const dsNX = 360, dsNY = 180;
  const sst = new Float32Array(dsNX * dsNY);
  for (let dj = 0; dj < dsNY; dj++) {
    const j = Math.round(dj / (dsNY - 1) * (f.NY - 1));
    for (let di = 0; di < dsNX; di++) {
      const i = Math.round(di / (dsNX - 1) * (f.NX - 1));
      sst[dj * dsNX + di] = f.temp[j * f.NX + i];
    }
  }

  return {
    params,
    diagnostics: diag,
    rmse: null, // computed in notebook
    sst: Array.from(sst),
    sstShape: [dsNY, dsNX],
    grid: { NX: f.NX, NY: f.NY, LAT0: f.LAT0, LAT1: f.LAT1, LON0: f.LON0, LON1: f.LON1 }
  };
}, steps);

const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
console.log(`Done in ${elapsed}s — ${result.diagnostics.step} steps, RMSE: ${result.rmse?.toFixed(2) || 'N/A'}°C`);

// Screenshot
const screenshot = await page.screenshot({ encoding: 'base64' });
result.screenshot = screenshot;
result.capturedAt = new Date().toISOString();
result.url = url;
result.elapsedSeconds = parseFloat(elapsed);

// Save
const hash = url.match(/amoc-([a-z0-9]+)-/)?.[1] || 'local';
const fname = `${hash}-${steps}.json`;
const outPath = join(outDir, fname);
writeFileSync(outPath, JSON.stringify(result));
console.log(`Saved: ${outPath} (${(JSON.stringify(result).length / 1024 / 1024).toFixed(1)} MB)`);

// Summary
console.log('\n--- Summary ---');
console.log(`Global SST: ${result.diagnostics.globalSST?.toFixed(1)}°C`);
console.log(`Tropical SST: ${result.diagnostics.tropicalSST?.toFixed(1)}°C`);
console.log(`Polar SST: ${result.diagnostics.polarSST?.toFixed(1)}°C`);
console.log(`AMOC: ${result.diagnostics.amoc?.toFixed(3)}`);
console.log(`Max velocity: ${result.diagnostics.maxVel?.toFixed(2)}`);
console.log(`RMSE vs obs: ${result.rmse?.toFixed(2) || 'N/A'}°C`);

await browser.close();
