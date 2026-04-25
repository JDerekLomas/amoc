#!/usr/bin/env node
/**
 * Headless SST RMSE validation tool
 *
 * Loads observed SST, runs the CPU physics for N steps, computes RMSE.
 * Usage:
 *   node scripts/validate-rmse.mjs                    # default 500 steps
 *   node scripts/validate-rmse.mjs --steps=2000
 *   node scripts/validate-rmse.mjs --evapScale=0.4    # override any param
 *
 * This does NOT use model.js directly (browser globals).
 * Instead it loads the binary data and runs a simplified physics loop.
 */

import { readFileSync, existsSync } from 'fs';

// ── Parse CLI args ──
const args = Object.fromEntries(
  process.argv.slice(2)
    .filter(a => a.startsWith('--'))
    .map(a => { const [k, v] = a.slice(2).split('='); return [k, parseFloat(v)]; })
);
const STEPS = args.steps || 500;

// ── Grid ──
const NX = 1024, NY = 512;
const LAT0 = -79.5, LAT1 = 79.5;
const N = NX * NY;

// ── Load binary data ──
function loadBin(name, field) {
  const metaPath = `data/bin/${name}.json`;
  if (!existsSync(metaPath)) return null;
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const info = meta.arrays?.[field];
  if (!info) return null;
  const buf = readFileSync(`data/bin/${info.file}`);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

console.log('Loading data...');
const obsSSTraw = loadBin('sst', 'sst');
const obsDeep = loadBin('deep_temp', 'temp');
const obsBathy = loadBin('bathymetry', 'depth');
const obsSal = loadBin('salinity', 'salinity');
const obsWindCurl = loadBin('wind_stress', 'wind_curl');
const obsTauX = loadBin('wind_stress', 'tau_x');
const obsTauY = loadBin('wind_stress', 'tau_y');
const obsEvap = loadBin('evaporation', 'evaporation');
const obsPrecip = loadBin('precipitation', 'precipitation');
const obsSeaIce = loadBin('sea_ice', 'ice_fraction');
const obsSnow = loadBin('snow_cover', 'snow_cover');

// Load mask
const maskMeta = JSON.parse(readFileSync('data/bin/mask.json', 'utf8'));
const maskBits = [];
for (const c of maskMeta.hex) {
  const v = parseInt(c, 16);
  maskBits.push((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1);
}
const mask = new Uint8Array(N);
for (let j = 0; j < NY; j++) {
  const sj = Math.min(j, NY - 1);
  for (let i = 0; i < NX; i++) {
    mask[j * NX + i] = maskBits[sj * NX + i] || 0;
  }
}
// Polar boundaries
for (let i = 0; i < NX; i++) { mask[i] = 0; mask[(NY - 1) * NX + i] = 0; }

if (!obsSSTraw) { console.error('No SST data found'); process.exit(1); }
console.log(`Grid: ${NX}x${NY}, mask ocean cells: ${mask.reduce((s, v) => s + v, 0)}`);

// ── Physics parameters ──
const P = {
  dt: 0.0005,
  S_solar: args.sSolar ?? 6.2,
  A_olr: args.aOlr ?? 1.8,
  B_olr: args.bOlr ?? 0.13,
  kappa_diff: args.kappaDiff ?? 2.5e-4,
  evapScale: args.evapScale ?? 0.8,
  peScale: args.peScale ?? 0.3,
  snowAlbedo: args.snowAlbedo ?? 0.45,
  co2_ppm: args.co2 ?? 420,
  yearSpeed: 1.0,
};

console.log('Params:', JSON.stringify(P, null, 2));

// ── Normalize evaporation and precipitation ──
let evapField = new Float32Array(N);
let precipField = new Float32Array(N);
if (obsEvap) {
  let sum = 0, cnt = 0;
  for (let k = 0; k < N; k++) { const v = obsEvap[k] || 0; if (v > 0 && mask[k]) { sum += v; cnt++; } }
  const scale = cnt > 0 ? 1.0 / (sum / cnt) : 0;
  for (let k = 0; k < N; k++) evapField[k] = Math.max(0, (obsEvap[k] || 0) * scale);
  if (obsPrecip) {
    for (let k = 0; k < N; k++) precipField[k] = Math.max(0, (obsPrecip[k] || 0) * scale);
  }
}

// Sea ice + snow
let seaIce = new Float32Array(N);
let snow = new Float32Array(N);
if (obsSeaIce) for (let k = 0; k < N; k++) seaIce[k] = Math.max(0, Math.min(1, obsSeaIce[k] || 0));
if (obsSnow) for (let k = 0; k < N; k++) snow[k] = Math.max(0, Math.min(1, (obsSnow[k] || 0) / 100));

// ── Init fields from observations ──
const temp = new Float64Array(N);
const sal = new Float64Array(N);
for (let k = 0; k < N; k++) {
  if (!mask[k]) continue;
  temp[k] = obsSSTraw[k] || 15;
  sal[k] = obsSal?.[k] || 35;
}

// ── Simplified timestep (radiation + diffusion only, no circulation) ──
const dx = 1.0 / (NX - 1), dy = 1.0 / (NY - 1);
let simTime = 0;

function step() {
  const tempNew = new Float64Array(N);
  const yearPhase = 2 * Math.PI * (simTime % 10) / 10;
  const decl = 23.44 * Math.sin(yearPhase) * Math.PI / 180;

  for (let j = 1; j < NY - 1; j++) {
    const lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    const latRad = lat * Math.PI / 180;
    const cosLat = Math.max(Math.cos(latRad), 0.087);
    const absLat = Math.abs(lat);
    const invDx = 1.0 / (dx * cosLat);
    const invDy = 1.0 / dy;

    for (let i = 0; i < NX; i++) {
      const k = j * NX + i;
      if (!mask[k]) continue;

      const T = temp[k];

      // Solar
      const cosZ = Math.cos(latRad) * Math.cos(decl) + Math.sin(latRad) * Math.sin(decl);
      let qSolar = P.S_solar * Math.max(0, cosZ);

      // Ice albedo (observed blend)
      const obsIce = seaIce[k];
      const sstIceT = Math.max(0, Math.min(1, (T + 2) / 10));
      const sstIceFrac = 1 - sstIceT * sstIceT * (3 - 2 * sstIceT);
      const iceFrac = obsIce > 0.001 ? obsIce : sstIceFrac;
      if (absLat > 45) {
        const iceStr = lat < 0 ? 0.65 : 0.50;
        qSolar *= 1 - iceStr * iceFrac * Math.max(0, Math.min(1, (absLat - 45) / 20));
      }

      // Snow albedo
      if (snow[k] > 0.01) qSolar *= 1 - P.snowAlbedo * snow[k];

      // Cloud (simplified: humidity proxy + storm track)
      const humidity = Math.max(0, Math.min(1, (T - 5) / 25));
      const stormTrack = 0.25 * Math.max(0, Math.min(1, (absLat - 35) / 10)) * Math.max(0, Math.min(1, (80 - absLat) / 15));
      const soCloud = lat < 0 ? 0.35 * Math.max(0, Math.min(1, (absLat - 45) / 10)) : 0;
      const nhCloud = lat > 0 ? 0.35 * Math.max(0, Math.min(1, (absLat - 40) / 12)) * Math.max(0, Math.min(1, (70 - absLat) / 10)) : 0;
      const cloudFrac = Math.max(0.05, Math.min(0.92, humidity * 0.3 + stormTrack + soCloud + nhCloud));
      const cloudAlbedo = cloudFrac * 0.30;
      qSolar *= 1 - cloudAlbedo;

      // OLR
      const olr = P.A_olr + P.B_olr * T;
      const qSat = 3.75e-3 * Math.exp(0.067 * T);
      const vaporGH = 0.4 * Math.max(0, Math.min(1, 0.8 * qSat / 0.015));
      const cloudGH = cloudFrac * 0.05;
      const co2GH = 5.35 * Math.log(P.co2_ppm / 280) / 240;
      const effectiveOlr = olr * (1 - cloudGH) * (1 - vaporGH) * (1 - co2GH);

      // Evaporative cooling
      const evapCool = P.evapScale * evapField[k];

      // Diffusion
      const ip1 = i === NX - 1 ? 0 : i + 1;
      const im1 = i === 0 ? NX - 1 : i - 1;
      const ke = j * NX + ip1, kw = j * NX + im1;
      const kn = (j + 1) * NX + i, ks = (j - 1) * NX + i;
      const tE = mask[ke] ? temp[ke] : T;
      const tW = mask[kw] ? temp[kw] : T;
      const tN = mask[kn] ? temp[kn] : T;
      const tS = mask[ks] ? temp[ks] : T;
      const lapT = invDx * invDx * (tE + tW - 2 * T) + invDy * invDy * (tN + tS - 2 * T);
      const diff = P.kappa_diff * lapT;

      tempNew[k] = T + P.dt * (qSolar - effectiveOlr + diff - evapCool);
      tempNew[k] = Math.max(-10, Math.min(40, tempNew[k]));
    }
  }

  for (let k = 0; k < N; k++) if (mask[k]) temp[k] = tempNew[k];
  simTime += P.dt;
}

// ── Run ──
console.log(`\nRunning ${STEPS} steps...`);
const t0 = Date.now();
for (let s = 0; s < STEPS; s++) {
  step();
  if ((s + 1) % 100 === 0) {
    // Compute RMSE
    let sse = 0, cnt = 0;
    for (let k = 0; k < N; k++) {
      if (!mask[k] || !obsSSTraw[k]) continue;
      const err = temp[k] - obsSSTraw[k];
      sse += err * err;
      cnt++;
    }
    const rmse = Math.sqrt(sse / cnt);
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`  step ${s + 1}: RMSE = ${rmse.toFixed(3)} °C  (${elapsed}s)`);
  }
}

// ── Final RMSE by latitude band ──
console.log('\nRMSE by latitude band:');
const bands = [
  { name: 'Tropics (0-20)', min: 0, max: 20 },
  { name: 'Subtropics (20-40)', min: 20, max: 40 },
  { name: 'Mid-lat (40-60)', min: 40, max: 60 },
  { name: 'High-lat (60-80)', min: 60, max: 80 },
];
for (const band of bands) {
  let sse = 0, cnt = 0;
  for (let j = 0; j < NY; j++) {
    const lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    const absLat = Math.abs(lat);
    if (absLat < band.min || absLat >= band.max) continue;
    for (let i = 0; i < NX; i++) {
      const k = j * NX + i;
      if (!mask[k] || !obsSSTraw[k]) continue;
      const err = temp[k] - obsSSTraw[k];
      sse += err * err;
      cnt++;
    }
  }
  if (cnt > 0) console.log(`  ${band.name}: ${Math.sqrt(sse / cnt).toFixed(3)} °C  (${cnt} cells)`);
}

// Global mean bias
let biasSum = 0, biasCnt = 0;
for (let k = 0; k < N; k++) {
  if (!mask[k] || !obsSSTraw[k]) continue;
  biasSum += temp[k] - obsSSTraw[k];
  biasCnt++;
}
console.log(`\nGlobal mean bias: ${(biasSum / biasCnt).toFixed(3)} °C`);
console.log(`Total time: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
