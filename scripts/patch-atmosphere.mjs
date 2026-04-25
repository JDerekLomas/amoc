#!/usr/bin/env node
// Idempotent patch: adds atmosphere shader + wiring to model.js temperature shader
// Run after Luke's edits to re-apply atmosphere integration.
// Usage: node scripts/patch-atmosphere.mjs

import { readFileSync, writeFileSync } from 'fs';

const MODEL = 'simamoc/model.js';
let src = readFileSync(MODEL, 'utf8');
let changed = false;

function patch(label, find, replace) {
  if (src.includes(replace)) { console.log(`  [skip] ${label} (already applied)`); return; }
  if (!src.includes(find)) { console.log(`  [WARN] ${label}: pattern not found — may need manual fix`); return; }
  src = src.replace(find, replace);
  changed = true;
  console.log(`  [ok]   ${label}`);
}

console.log('Patching model.js for GPU atmosphere...');

// 1. Fix Params struct in temperature shader: add evapScale/peScale/snowAlbedo + atmosphere params
patch('temperature shader Params struct',
  "'  _padS0: u32, _padS1: u32, _padS2: u32, _padS3: u32,',\n'};',\n'',\n'@group(0) @binding(0) var<storage, read> psi: array<f32>;',\n'@group(0) @binding(1) var<storage, read> tempIn: array<f32>;',",
  "'  _padS0: u32, evapScale: f32, peScale: f32, snowAlbedo: f32,',\n'  kappaAtm: f32, gammaOA: f32, gammaAO: f32, evapCoeff: f32,',\n'};',\n'',\n'@group(0) @binding(0) var<storage, read> psi: array<f32>;',\n'@group(0) @binding(1) var<storage, read> tempIn: array<f32>;',"
);

// 2. Add atmosphere binding to temperature shader
patch('atmosphere binding(14)',
  "'@group(0) @binding(13) var<storage, read> precipRate: array<f32>;',\n'',\n'fn idx",
  "'@group(0) @binding(13) var<storage, read> precipRate: array<f32>;',\n'@group(0) @binding(14) var<storage, read> atmosphere: array<f32>;',\n'',\n'fn idx"
);

// 3. Replace humidity proxy with atmosphere moisture
patch('humidity from atmosphere',
  "'  // Humidity proxy: warm SST = more evaporation',\n'  let humidity = clamp((tempIn[k] - 5.0) / 25.0, 0.0, 1.0);',",
  "'  // Humidity from prognostic atmosphere moisture field',\n'  let N_atm = params.nx * params.ny;',\n'  let atm_q = atmosphere[k + N_atm];',\n'  let humidity = clamp(atm_q / 0.020, 0.0, 1.0);',"
);

// 4. Replace air temp estimate with actual
patch('airTempEst from atmosphere',
  "'  // Lower tropospheric stability: estimated air temp vs SST',\n'  // Warm air over cold water = inversion = stratocumulus',\n'  let airTempEst = 28.0 - 0.55 * absLat;',",
  "'  // Lower tropospheric stability: actual air temp vs SST',\n'  let airTempEst = atmosphere[k];',"
);

// 5. Replace water vapor greenhouse with actual moisture
patch('vaporGH from atmosphere moisture',
  "'  // Water vapor greenhouse: Clausius-Clapeyron moisture at 80% RH',\n'  let qSat = 3.75e-3 * exp(0.067 * tempIn[k]);',\n'  let vaporGH = 0.4 * clamp(0.8 * qSat / 0.015, 0.0, 1.0);',",
  "'  // Water vapor greenhouse: from prognostic atmospheric moisture',\n'  let vaporGH = 0.4 * clamp(atm_q / 0.015, 0.0, 1.0);',"
);

// 6. Replace static evapCool with dynamic atmosphere-derived
patch('dynamic evapCool',
  "'  let evapCool = params.evapScale * evapRate[k];',\n'  tempOut[k] = tempIn[k] + params.dt * (-advec + qNet + diff + landFlux - evapCool);',",
  "'  // Dynamic evaporative cooling from atmosphere moisture deficit',\n'  let qSat_sst = 3.75e-3 * exp(0.067 * tempIn[k]);',\n'  let dynEvap = params.evapCoeff * max(0.0, qSat_sst - atm_q);',\n'  let evapCool = dynEvap * 400.0;',\n'  tempOut[k] = tempIn[k] + params.dt * (-advec + qNet + diff + landFlux - evapCool);',\n'  // Atmosphere-ocean heat feedback',\n'  tempOut[k] += params.dt * params.gammaAO * (atmosphere[k] - tempIn[k]);',"
);

// 7. Replace static P-E with dynamic
patch('dynamic P-E flux',
  "'  let peFlux = params.peScale * (evapRate[k] - precipRate[k]) * tempIn[salK] / 35.0;',",
  "'  // Dynamic P-E from atmosphere moisture budget',\n'  let qSat_air = 3.75e-3 * exp(0.067 * atmosphere[k]);',\n'  let dynPrecip = max(0.0, atm_q - qSat_air);',\n'  let peFlux = params.peScale * (dynEvap - dynPrecip) * tempIn[salK] / 35.0;',"
);

if (changed) {
  writeFileSync(MODEL, src);
  console.log('Done — model.js patched for atmosphere.');
} else {
  console.log('No changes needed (all patches already applied).');
}
