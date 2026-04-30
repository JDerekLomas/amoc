// SimAMOC v2 — main entry point
// Wires components together, loads data, runs simulation loop

import { Grid } from './grid.js';
import { Ocean } from './ocean.js';
import { Atmosphere } from './atmosphere.js';
import { Coupler } from './coupler.js';
import { Renderer } from './renderer.js';
import { UI } from './ui.js';
import { SIMULATION } from './params.js';
import { loadField1deg, loadWindStress, loadMask } from './data-loader.js';

async function init() {
  const grid = new Grid();
  console.log(`Grid: ${grid.nx}x${grid.ny}, ${grid.dlat.toFixed(2)} deg resolution`);

  const ocean = new Ocean(grid);
  const atmosphere = new Atmosphere(grid);
  const coupler = new Coupler(ocean, atmosphere, grid);

  // Build UI — must create UI first (builds viewport DOM), then append canvas, then create renderer
  const container = document.getElementById('app');
  const canvas = document.createElement('canvas');
  const ui = new UI(container, null, coupler);  // renderer set after creation
  ui.viewport.appendChild(canvas);
  const renderer = new Renderer(canvas, grid);
  ui.renderer = renderer;
  ui.running = false;
  ui.statusEl.textContent = 'Loading data...';

  // Load observational data (1deg JSON files at repo root)
  // Paths relative to src/index.html → parent directory
  const BASE = '..';
  try {
    console.log('Loading mask...');
    ocean.mask = await loadMask('mask.json', grid);

    console.log('Loading wind stress...');
    const wind = await loadWindStress(`${BASE}/wind_stress_1deg.json`, grid);
    atmosphere.tauX_obs = wind.tauX;
    atmosphere.tauY_obs = wind.tauY;

    console.log('Loading SST...');
    const sstObs = await loadField1deg(`${BASE}/sst_global_1deg.json`, 'sst', grid);
    ocean.initSST(sstObs);

    console.log('Loading cloud fraction...');
    atmosphere.cloudFraction = await loadField1deg(`${BASE}/cloud_fraction_1deg.json`, 'cloud_fraction', grid);

    console.log('Loading salinity...');
    const salObs = await loadField1deg(`${BASE}/salinity_1deg.json`, 'salinity', grid);
    ocean.initSalinity(salObs);

    console.log('Loading precipitation...');
    const precipRaw = await loadField1deg(`${BASE}/precipitation_1deg.json`, 'precipitation', grid);
    // Convert mm/year to m/s
    for (let k = 0; k < grid.size; k++) {
      atmosphere.precip_obs[k] = precipRaw[k] / (1000 * 365.25 * 86400);
    }

    console.log('Data loaded.');
  } catch (e) {
    console.error('Data load failed:', e);
    ui.statusEl.textContent = `Data load failed: ${e.message}`;
    return;
  }

  // Start simulation
  ui.running = true;
  let lastFrame = performance.now();
  let frameCount = 0;

  function loop(now) {
    requestAnimationFrame(loop);

    if (ui.running) {
      // Apply perturbation state from UI
      coupler.freshwaterHosing = ui.freshwaterHosing;
      coupler.co2Multiplier = ui.co2Multiplier;
      coupler.globalTempOffset = ui.globalTempOffset;

      const stepsThisFrame = SIMULATION.stepsPerFrame * ui.speedMultiplier;
      for (let s = 0; s < stepsThisFrame; s++) {
        coupler.step();
      }
    }

    renderer.render(ocean, atmosphere);
    ui.update();

    frameCount++;
    if (now - lastFrame > 3000) {
      const fps = (frameCount / (now - lastFrame)) * 1000;
      const stepMs = ui.running ? ((now - lastFrame) / frameCount).toFixed(1) : '-';
      console.log(`FPS: ${fps.toFixed(1)}, step: ${coupler.stepCount}, time: ${coupler.getTimeString()}, ms/frame: ${stepMs}`);
      lastFrame = now;
      frameCount = 0;
    }
  }

  requestAnimationFrame(loop);
}

init();
