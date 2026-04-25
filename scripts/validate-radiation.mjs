#!/usr/bin/env node
/**
 * Radiation balance validation
 *
 * Checks: at observed SST, is the model in radiative equilibrium (qNet ≈ 0)?
 * A model with good energy balance will have small qNet everywhere.
 * Large positive qNet → model will warm. Large negative → model will cool.
 *
 * Also validates: zonal cloud fraction vs CERES/MODIS observations.
 *
 * Usage: node scripts/validate-radiation.mjs [--co2=420]
 */
import { readFileSync, existsSync } from 'fs';

const args = Object.fromEntries(
  process.argv.slice(2).filter(a => a.startsWith('--'))
    .map(a => { const [k, v] = a.slice(2).split('='); return [k, parseFloat(v)]; })
);

const NX = 1024, NY = 512;
const LAT0 = -79.5, LAT1 = 79.5;
const N = NX * NY;

function loadBin(name, field) {
  const metaPath = `data/bin/${name}.json`;
  if (!existsSync(metaPath)) return null;
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const info = meta.arrays?.[field];
  if (!info) return null;
  const buf = readFileSync(`data/bin/${info.file}`);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

const obsSST = loadBin('sst', 'sst');
const obsSeaIce = loadBin('sea_ice', 'ice_fraction');
const obsSnow = loadBin('snow_cover', 'snow_cover');
const obsCloud = loadBin('cloud_fraction', 'cloud_fraction');
const maskMeta = JSON.parse(readFileSync('data/bin/mask.json', 'utf8'));
const maskBits = [];
for (const c of maskMeta.hex) { const v = parseInt(c, 16); maskBits.push((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1); }
const mask = new Uint8Array(N);
for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) mask[j * NX + i] = maskBits[j * NX + i] || 0;
for (let i = 0; i < NX; i++) { mask[i] = 0; mask[(NY - 1) * NX + i] = 0; }

if (!obsSST) { console.error('No SST data'); process.exit(1); }

// Model parameters
const S_solar = args.sSolar ?? 6.5;
const A_olr = args.aOlr ?? 1.8;
const B_olr = args.bOlr ?? 0.13;
const snowAlbedo = args.snowAlbedo ?? 0.45;
const co2_ppm = args.co2 ?? 420;
const co2GH = 5.35 * Math.log(co2_ppm / 280) / 240;

console.log(`Radiation balance at ${co2_ppm} ppm CO2\n`);

const bands = [
  { name: 'Tropics (0-20°)',    min: 0,  max: 20, obs_cloud: 0.45 },
  { name: 'Subtropics (20-40°)',min: 20, max: 40, obs_cloud: 0.45 },
  { name: 'Mid-lat (40-60°)',   min: 40, max: 60, obs_cloud: 0.70 },
  { name: 'Polar (60-80°)',     min: 60, max: 80, obs_cloud: 0.75 },
];

for (const band of bands) {
  let sumNet = 0, sumSW = 0, sumOLR = 0, sumCloud = 0, sumObsCloud = 0;
  let cnt = 0, obsCnt = 0;

  for (let j = 1; j < NY - 1; j++) {
    const lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    const absLat = Math.abs(lat);
    if (absLat < band.min || absLat >= band.max) continue;
    const latRad = lat * Math.PI / 180;

    for (let i = 0; i < NX; i++) {
      const k = j * NX + i;
      if (!mask[k]) continue;
      const T = obsSST[k] || 15;

      // Solar (annual mean: average over declination cycle)
      let qSolar = S_solar * Math.max(0, Math.cos(latRad));

      // Ice albedo
      const iceFrac = obsSeaIce ? Math.max(0, Math.min(1, obsSeaIce[k] || 0)) : 0;
      if (absLat > 45) qSolar *= 1 - 0.50 * iceFrac * Math.max(0, Math.min(1, (absLat - 45) / 20));

      // Snow albedo
      const snowFrac = obsSnow ? Math.max(0, Math.min(1, (obsSnow[k] || 0) / 100)) : 0;
      if (snowFrac > 0.01) qSolar *= 1 - snowAlbedo * snowFrac;

      // Cloud model (matching GPU three-type)
      const humidity = Math.max(0, Math.min(1, (T - 5) / 25));
      const itczDist = lat / 10;
      const convCloud = 0.30 * Math.exp(-itczDist * itczDist) * humidity;
      const warmPool = 0.20 * Math.max(0, Math.min(1, (T - 26) / 4));
      const cirrus = 0.15 * humidity * Math.max(0.1, Math.min(1, 1 - absLat / 80));
      const thickConv = convCloud + warmPool;

      const airTempEst = 28 - 0.55 * absLat;
      const lts = Math.max(0, Math.min(1, (airTempEst - T) / 15));
      const stratocu = Math.min(0.20, 0.30 * lts * Math.max(0, Math.min(1, (35 - absLat) / 20)));
      const nhStormBoost = lat > 0 ? 0.22 * Math.max(0, Math.min(1, (absLat - 35) / 10)) * Math.max(0, Math.min(1, (58 - absLat) / 12)) : 0;
      const stormTrack = 0.30 * Math.max(0, Math.min(1, (absLat - 35) / 10)) * Math.max(0, Math.min(1, (80 - absLat) / 15)) + nhStormBoost;
      const soDist = (absLat - 62) / 7;
      const soCloud = lat < 0 ? (0.70 * Math.exp(-soDist * soDist) + 0.18 * Math.max(0, Math.min(1, (absLat - 53) / 8))) : 0;
      const nhCloud = lat > 0 ? 0.35 * Math.max(0, Math.min(1, (absLat - 40) / 12)) * Math.max(0, Math.min(1, (70 - absLat) / 10)) : 0;
      const polarCloud = 0.10 * Math.max(0, Math.min(1, (absLat - 60) / 10));
      const tradeDist = (absLat - 22) / 12;
      const tradeCu = 0.25 * Math.exp(-tradeDist * tradeDist) * humidity;
      const subDist = (absLat - 25) / 10;
      const subsidence = 0.15 * Math.exp(-subDist * subDist);
      const lowCloud = stratocu + tradeCu + stormTrack + soCloud + nhCloud + polarCloud;
      const cloudFrac = Math.max(0.05, Math.min(0.90, cirrus + thickConv + lowCloud - subsidence * (1 - humidity)));

      const w_total = Math.max(cirrus + thickConv + lowCloud, 0.01);
      const w_ci = cirrus / w_total, w_cv = thickConv / w_total, w_lo = lowCloud / w_total;
      const cloudAlbedo = cloudFrac * (w_ci * 0.05 + w_cv * 0.30 + w_lo * 0.35);
      qSolar *= 1 - cloudAlbedo;

      // OLR
      let olr = A_olr + B_olr * T;
      if (lat < -53) olr *= 1 + 0.55 * Math.max(0, Math.min(1, (absLat - 58) / 8));
      const cloudGHouse = cloudFrac * (w_ci * 0.20 + w_cv * 0.10 + w_lo * 0.03);
      const qSat = 3.75e-3 * Math.exp(0.067 * T);
      const vaporGH = 0.4 * Math.max(0, Math.min(1, 0.8 * qSat / 0.015));
      const effectiveOlr = olr * (1 - cloudGHouse) * (1 - vaporGH) * (1 - co2GH);

      const qNet = qSolar - effectiveOlr;
      sumNet += qNet;
      sumSW += qSolar;
      sumOLR += effectiveOlr;
      sumCloud += cloudFrac;
      cnt++;

      if (obsCloud && obsCloud[k] > 0) { sumObsCloud += obsCloud[k]; obsCnt++; }
    }
  }

  if (cnt === 0) continue;
  const meanNet = sumNet / cnt;
  const meanSW = sumSW / cnt;
  const meanOLR = sumOLR / cnt;
  const meanCloud = sumCloud / cnt;
  const meanObsCloud = obsCnt > 0 ? sumObsCloud / obsCnt : NaN;

  // qNet interpretation: positive = warming tendency, negative = cooling
  // At equilibrium qNet should be ~0. Small positive is OK (ocean heat uptake ~0.9 W/m²)
  const status = Math.abs(meanNet) < 0.3 ? 'BALANCED' : meanNet > 0 ? 'WARMING' : 'COOLING';

  console.log(`${band.name}:`);
  console.log(`  qNet = ${meanNet >= 0 ? '+' : ''}${meanNet.toFixed(3)} (model units)  [${status}]`);
  console.log(`  SW_in = ${meanSW.toFixed(3)}  OLR_out = ${meanOLR.toFixed(3)}`);
  console.log(`  Cloud: model ${(meanCloud * 100).toFixed(0)}% vs obs ${isNaN(meanObsCloud) ? 'N/A' : (meanObsCloud * 100).toFixed(0) + '%'} (CERES ~${(band.obs_cloud * 100).toFixed(0)}%)`);
  console.log();
}

// Global weighted balance
let gNet = 0, gW = 0;
for (let j = 1; j < NY - 1; j++) {
  const lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
  const latRad = lat * Math.PI / 180;
  const cosLat = Math.cos(latRad);
  for (let i = 0; i < NX; i++) {
    const k = j * NX + i;
    if (!mask[k]) continue;
    const T = obsSST[k] || 15;
    let qSolar = S_solar * Math.max(0, Math.cos(latRad));
    const humidity = Math.max(0, Math.min(1, (T - 5) / 25));
    const cloudFrac = Math.max(0.05, Math.min(0.90, 0.15 * humidity + 0.20));
    const cloudAlbedo = cloudFrac * 0.25;
    qSolar *= 1 - cloudAlbedo;
    let olr = A_olr + B_olr * T;
    const qSat = 3.75e-3 * Math.exp(0.067 * T);
    const vaporGH = 0.4 * Math.max(0, Math.min(1, 0.8 * qSat / 0.015));
    const cloudGH = cloudFrac * 0.08;
    olr = olr * (1 - cloudGH) * (1 - vaporGH) * (1 - co2GH);
    gNet += (qSolar - olr) * cosLat;
    gW += cosLat;
  }
}
console.log(`Global energy balance: qNet = ${(gNet / gW) >= 0 ? '+' : ''}${(gNet / gW).toFixed(4)} model units`);
console.log(`  (should be near 0 for stable climate; small positive = ocean heat uptake)`);
