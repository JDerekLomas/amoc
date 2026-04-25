#!/usr/bin/env node
/**
 * Browser test: loads the simulation, checks GPU atmosphere is running.
 * Requires: npx playwright (chromium with WebGPU)
 * Usage: node scripts/test-atmosphere-browser.mjs
 */
import { chromium } from 'playwright';
import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { join, extname } from 'path';

const PORT = 8399;
const SIMDIR = 'simamoc';

// Simple static file server
const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.mjs': 'application/javascript',
  '.css': 'text/css', '.json': 'application/json', '.bin': 'application/octet-stream', '.png': 'image/png' };

const server = createServer((req, res) => {
  let url = req.url.split('?')[0];
  if (url === '/') url = '/index.html';
  // Try simamoc dir first, then root
  let fp = join(SIMDIR, url);
  if (!existsSync(fp)) fp = join('.', url);
  if (!existsSync(fp)) { res.writeHead(404); res.end('Not found'); return; }
  const ct = MIME[extname(fp)] || 'application/octet-stream';
  res.writeHead(200, { 'Content-Type': ct });
  res.end(readFileSync(fp));
});

server.listen(PORT, async () => {
  console.log(`Server on http://localhost:${PORT}`);

  let browser;
  try {
    browser = await chromium.launch({
      args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
      headless: true,
    });
    const context = await browser.newContext();
    const page = await context.newPage();

    const errors = [];
    const logs = [];
    page.on('console', msg => {
      const text = msg.text();
      logs.push(`[${msg.type()}] ${text}`);
      if (msg.type() === 'error') errors.push(text);
    });
    page.on('pageerror', err => errors.push(err.message));

    console.log('Loading simulation...');
    await page.goto(`http://localhost:${PORT}`, { timeout: 30000 });

    // Wait for initialization
    await page.waitForTimeout(5000);

    // Check for WebGPU
    const hasGPU = await page.evaluate(() => typeof gpuDevice !== 'undefined' && gpuDevice !== null);
    console.log(`WebGPU: ${hasGPU ? 'YES' : 'NO (CPU fallback)'}`);

    // Check atmosphere variables exist
    const atmCheck = await page.evaluate(() => {
      return {
        hasAtmBuf: typeof gpuAtmBuf !== 'undefined' && gpuAtmBuf !== null,
        hasAtmPipeline: typeof gpuAtmospherePipeline !== 'undefined' && gpuAtmospherePipeline !== null,
        hasAirTemp: typeof airTemp !== 'undefined' && airTemp !== null,
        hasMoisture: typeof moisture !== 'undefined' && moisture !== null,
        airTempSample: airTemp ? [airTemp[256*1024+512], airTemp[128*1024+512], airTemp[384*1024+512]] : null,
        moistureSample: moisture ? [moisture[256*1024+512], moisture[128*1024+512], moisture[384*1024+512]] : null,
      };
    });

    console.log('\nAtmosphere state:');
    console.log(`  GPU atmosphere buffer: ${atmCheck.hasAtmBuf ? 'YES' : 'NO'}`);
    console.log(`  GPU atmosphere pipeline: ${atmCheck.hasAtmPipeline ? 'YES' : 'NO'}`);
    console.log(`  CPU airTemp array: ${atmCheck.hasAirTemp ? 'YES' : 'NO'}`);
    console.log(`  CPU moisture array: ${atmCheck.hasMoisture ? 'YES' : 'NO'}`);
    if (atmCheck.airTempSample) {
      console.log(`  airTemp samples (equator, 30S, 30N): ${atmCheck.airTempSample.map(v => v?.toFixed(1)).join(', ')}`);
    }
    if (atmCheck.moistureSample) {
      console.log(`  moisture samples (equator, 30S, 30N): ${atmCheck.moistureSample.map(v => v?.toExponential(2)).join(', ')}`);
    }

    // Let the sim run for a bit
    console.log('\nRunning simulation for 10 seconds...');
    await page.waitForTimeout(10000);

    // Check sim is stepping
    const simState = await page.evaluate(() => {
      return {
        totalSteps: totalSteps,
        simTime: simTime?.toFixed(4),
        paused: paused,
        NX: NX, NY: NY,
        useGPU: typeof gpuDevice !== 'undefined' && gpuDevice !== null,
      };
    });
    console.log(`\nSim state: ${simState.totalSteps} steps, simTime=${simState.simTime}, paused=${simState.paused}, GPU=${simState.useGPU}`);

    // Check atmosphere after running
    const atmAfter = await page.evaluate(() => {
      const N = NX * NY;
      let sumT = 0, sumQ = 0, cnt = 0;
      if (airTemp && moisture) {
        for (let j = 100; j < NY-100; j += 10) {
          for (let i = 0; i < NX; i += 10) {
            const k = j * NX + i;
            if (mask[k]) { sumT += airTemp[k]; sumQ += moisture[k]; cnt++; }
          }
        }
      }
      return {
        meanAirTemp: cnt > 0 ? sumT / cnt : null,
        meanMoisture: cnt > 0 ? sumQ / cnt : null,
        count: cnt,
        // SST for comparison
        meanSST: (() => {
          let s = 0, c = 0;
          for (let j = 100; j < NY-100; j += 10) {
            for (let i = 0; i < NX; i += 10) {
              const k = j * NX + i;
              if (mask[k] && temp[k]) { s += temp[k]; c++; }
            }
          }
          return c > 0 ? s / c : null;
        })(),
      };
    });

    console.log('\nAtmosphere diagnostics (mid-latitudes, sampled):');
    if (atmAfter.meanAirTemp !== null) {
      console.log(`  Mean air temp: ${atmAfter.meanAirTemp.toFixed(2)} °C`);
      console.log(`  Mean SST:      ${atmAfter.meanSST?.toFixed(2)} °C`);
      console.log(`  Mean moisture:  ${atmAfter.meanMoisture?.toExponential(3)} kg/kg`);

      // Sanity checks
      const checks = [];
      if (atmAfter.meanAirTemp < -10 || atmAfter.meanAirTemp > 40) checks.push(`air temp out of range: ${atmAfter.meanAirTemp.toFixed(1)}`);
      if (atmAfter.meanMoisture < 1e-5 || atmAfter.meanMoisture > 0.03) checks.push(`moisture out of range: ${atmAfter.meanMoisture.toExponential(2)}`);
      if (atmAfter.meanSST !== null && Math.abs(atmAfter.meanSST - atmAfter.meanAirTemp) > 20) checks.push(`air-SST gap too large: ${Math.abs(atmAfter.meanSST - atmAfter.meanAirTemp).toFixed(1)}°C`);

      if (checks.length === 0) {
        console.log('  Sanity checks: ALL PASS');
      } else {
        console.log('  Sanity checks: ISSUES');
        for (const c of checks) console.log(`    - ${c}`);
      }
    } else {
      console.log('  No atmosphere data available (GPU readback may not have fired yet)');
    }

    // Report errors
    if (errors.length > 0) {
      console.log('\nCONSOLE ERRORS:');
      for (const e of errors) console.log(`  ${e}`);
    } else {
      console.log('\nNo console errors.');
    }

    // Show relevant console logs
    const atmLogs = logs.filter(l => l.includes('atmos') || l.includes('Atmos') || l.includes('moisture') || l.includes('ERA5') || l.includes('cirrus'));
    if (atmLogs.length > 0) {
      console.log('\nRelevant console logs:');
      for (const l of atmLogs) console.log(`  ${l}`);
    }

  } catch (err) {
    console.error('Test error:', err.message);
  } finally {
    if (browser) await browser.close();
    server.close();
  }
});
