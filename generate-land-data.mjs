#!/usr/bin/env node
/**
 * Generate surface albedo and precipitation maps for land physics.
 * Derives from geographic position (lat/lon) + elevation from bathymetry_1deg.json.
 *
 * Albedo: based on biome classification from lat/elev/aridity
 * Precipitation: based on ITCZ, monsoon, orographic, continental patterns
 *
 * Output: albedo_1deg.json, precipitation_1deg.json
 */

import { readFileSync, writeFileSync } from 'fs';

const NX = 360, NY = 160;
const LAT0 = -79.5, LAT1 = 79.5;
const LON0 = -179.5, LON1 = 179.5;

// Load bathymetry for mask + elevation
const bathy = JSON.parse(readFileSync('bathymetry_1deg.json', 'utf8'));
const mask = JSON.parse(readFileSync('simamoc/mask.json', 'utf8'));

// Decode mask bits
const maskBits = [];
for (const c of mask.hex) {
  const v = parseInt(c, 16);
  maskBits.push((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1);
}

function isOcean(i, j) {
  // Use mask at 360x180 resolution, nearest-neighbor to 360x160
  const sj = Math.min(Math.floor(j * 180 / NY), 179);
  const si = Math.min(i, 359);
  return maskBits[sj * 360 + si] === 1;
}

function getElev(i, j) {
  if (!bathy.elevation) return 0;
  const lat = LAT0 + j;
  const obsJ = Math.round(lat - (bathy.lat0 || -79.5));
  if (obsJ < 0 || obsJ >= (bathy.ny || 160)) return 0;
  return bathy.elevation[obsJ * (bathy.nx || 360) + i] || 0;
}

// ============================================================
// ALBEDO MODEL
// ============================================================
// Based on: lat → biome zone, elev → snow/ice, lon → continental position
// Real-world references:
//   Sahara/Arabia: 0.35-0.40
//   Tropical forest: 0.12-0.15
//   Temperate forest: 0.15-0.20
//   Grassland/savanna: 0.20-0.25
//   Ice sheets: 0.75-0.85
//   Tundra: 0.15-0.20 (summer), 0.60+ (winter)
//   Desert (cold): 0.25-0.30

function computeAlbedo(lat, lon, elev) {
  const absLat = Math.abs(lat);

  // Ice sheets: Greenland and Antarctica
  if (absLat > 70 && elev > 1500) return 0.80;
  if (absLat > 60 && elev > 2000) return 0.75;
  // Greenland specifically (lat 60-85, lon -60 to -15)
  if (lat > 60 && lon > -60 && lon < -15 && elev > 500) return 0.78;
  // Antarctic ice sheet (high elevation south of 70S)
  if (lat < -70 && elev > 200) return 0.82;

  // High mountain snow/glaciers
  if (elev > 5000) return 0.65;
  if (elev > 4000) return 0.45 + 0.15 * Math.max(0, (absLat - 20) / 40);
  if (elev > 3000) return 0.30 + 0.15 * Math.max(0, (absLat - 30) / 30);

  // Desert belts (15-35 deg, modified by longitude for specific deserts)
  if (absLat > 15 && absLat < 35) {
    // Sahara (lat 15-30N, lon -15 to 35)
    if (lat > 15 && lat < 33 && lon > -17 && lon < 35) return 0.37;
    // Arabian desert (lat 15-30N, lon 35-60)
    if (lat > 15 && lat < 32 && lon > 35 && lon < 60) return 0.35;
    // Thar desert (lat 24-30N, lon 68-75)
    if (lat > 22 && lat < 30 && lon > 66 && lon < 76) return 0.33;
    // Australian outback (lat 20-30S, lon 120-150)
    if (lat < -18 && lat > -32 && lon > 118 && lon < 150) return 0.30;
    // Kalahari/Namib (lat 15-30S, lon 12-28)
    if (lat < -15 && lat > -32 && lon > 10 && lon < 30) return 0.28;
    // Sonoran/Chihuahuan (lat 25-35N, lon -115 to -100)
    if (lat > 25 && lat < 35 && lon > -115 && lon < -100) return 0.30;
    // Atacama (lat 18-30S, lon -72 to -68)
    if (lat < -18 && lat > -30 && lon > -73 && lon < -68) return 0.32;
    // Generic subtropical: semi-arid
    return 0.25;
  }

  // Central Asian deserts (Gobi, Taklamakan) - 35-50N, 60-110E
  if (lat > 35 && lat < 50 && lon > 60 && lon < 110 && elev > 500) return 0.28;
  // Patagonian steppe
  if (lat < -40 && lat > -52 && lon > -72 && lon < -65) return 0.22;

  // Tropical forests (0-15 deg)
  if (absLat < 15) {
    // Amazon (lon -80 to -40)
    if (lon > -80 && lon < -40 && lat > -15 && lat < 5) return 0.13;
    // Congo (lon 10-30, lat -5 to 5)
    if (lon > 8 && lon < 32 && absLat < 8) return 0.13;
    // Maritime continent / SE Asia (lon 95-150)
    if (lon > 95 && lon < 150 && absLat < 10) return 0.14;
    // Other tropical
    return 0.16;
  }

  // Boreal/tundra (50-70 deg)
  if (absLat > 55) {
    if (absLat > 65) return 0.18; // Tundra (annual mean, partial snow)
    return 0.14; // Boreal forest (dark conifers)
  }

  // Temperate (35-55 deg)
  if (absLat > 35) {
    if (elev > 1500) return 0.22; // Mountain grassland
    return 0.18; // Temperate forest/mixed
  }

  // Savanna/grassland transition (15-35)
  return 0.20;
}

// ============================================================
// PRECIPITATION MODEL
// ============================================================
// Annual mean precipitation in mm/year
// Key patterns:
//   ITCZ (0-10 deg): 1500-3000 mm
//   Monsoon regions: 1000-2500 mm
//   Subtropical deserts (15-30): 50-250 mm
//   Mid-latitude westerlies: 500-1200 mm
//   Rain shadows: 100-300 mm
//   Polar: 100-300 mm

function computePrecip(lat, lon, elev) {
  const absLat = Math.abs(lat);

  // Base zonal pattern
  let precip;

  // ITCZ band (equatorial convergence)
  if (absLat < 10) {
    precip = 1800 + 800 * Math.exp(-absLat * absLat / 25);
  }
  // Subtropical dry zone
  else if (absLat < 35) {
    const dryCore = Math.exp(-(absLat - 25) * (absLat - 25) / 50);
    precip = 200 + 600 * (1 - dryCore);
  }
  // Mid-latitude wet zone (storm tracks)
  else if (absLat < 60) {
    const stormPeak = Math.exp(-(absLat - 45) * (absLat - 45) / 100);
    precip = 400 + 600 * stormPeak;
  }
  // Polar dry
  else {
    precip = 300 - 3 * (absLat - 60);
    if (precip < 50) precip = 50;
  }

  // ── Regional modifications ──

  // Amazon basin: extremely wet
  if (lon > -80 && lon < -40 && lat > -15 && lat < 5) {
    precip = Math.max(precip, 2000 + 500 * Math.exp(-absLat * absLat / 20));
  }

  // Congo basin: very wet
  if (lon > 8 && lon < 32 && absLat < 8) {
    precip = Math.max(precip, 1800);
  }

  // Maritime continent / SE Asia: extremely wet
  if (lon > 95 && lon < 155 && absLat < 12) {
    precip = Math.max(precip, 2200);
  }

  // Indian monsoon (lat 10-30N, lon 70-90)
  if (lat > 8 && lat < 30 && lon > 68 && lon < 92) {
    const monsoonStr = Math.max(0, 1 - (lat - 15) / 20);
    precip = Math.max(precip, 800 + 1500 * monsoonStr);
  }

  // Sahara: extremely dry
  if (lat > 15 && lat < 33 && lon > -17 && lon < 35) {
    precip = Math.min(precip, 50 + 30 * Math.max(0, (lat - 15) / 5));
  }

  // Arabian desert
  if (lat > 15 && lat < 32 && lon > 35 && lon < 60) {
    precip = Math.min(precip, 80);
  }

  // Australian interior
  if (lat < -18 && lat > -32 && lon > 118 && lon < 150) {
    precip = Math.min(precip, 200);
  }

  // Kalahari/Namib
  if (lat < -15 && lat > -32 && lon > 10 && lon < 28) {
    precip = Math.min(precip, 150);
  }

  // Atacama (driest place on Earth)
  if (lat < -18 && lat > -30 && lon > -73 && lon < -68) {
    precip = Math.min(precip, 20);
  }

  // Central Asian dry (Gobi, Taklamakan)
  if (lat > 35 && lat < 50 && lon > 60 && lon < 110) {
    const aridity = Math.exp(-(lon - 85) * (lon - 85) / 400);
    precip = Math.min(precip, 150 + 300 * (1 - aridity));
  }

  // Patagonian rain shadow
  if (lat < -40 && lat > -52 && lon > -72 && lon < -65) {
    precip = Math.min(precip, 200);
  }

  // Pacific Northwest / western Europe: wet westerlies
  if (lat > 40 && lat < 60) {
    // Western Europe
    if (lon > -10 && lon < 15) precip = Math.max(precip, 800);
    // Pacific NW
    if (lon > -130 && lon < -118) precip = Math.max(precip, 1200);
  }

  // Chilean/NZ west coast: orographic wet
  if (lat < -35 && lat > -55 && lon > -78 && lon < -70) {
    precip = Math.max(precip, 1500);
  }

  // Greenland/Antarctica ice caps: low precip
  if ((lat > 60 && lon > -60 && lon < -15 && elev > 500) ||
      (lat < -70 && elev > 200)) {
    precip = Math.min(precip, 200);
  }

  // Orographic enhancement: high mountains near coasts get more rain
  if (elev > 1000) {
    precip *= 1 + 0.3 * Math.min(1, (elev - 1000) / 3000);
  }

  // Continental drying: interior of large continents gets less
  // (crude: distance from coast would be better, but we approximate via longitude)

  return Math.max(10, Math.min(4000, precip));
}

// ============================================================
// GENERATE DATA
// ============================================================

const albedo = new Float32Array(NX * NY);
const precip = new Float32Array(NX * NY);

let landCount = 0;
let albedoSum = 0, precipSum = 0;

for (let j = 0; j < NY; j++) {
  const lat = LAT0 + j; // -79.5, -78.5, ..., 79.5
  for (let i = 0; i < NX; i++) {
    const lon = LON0 + i; // -179.5, -178.5, ..., 179.5
    const k = j * NX + i;

    if (isOcean(i, j)) {
      // Ocean: albedo ~0.06 (water), precip from evaporation
      albedo[k] = 0.06;
      precip[k] = 0; // not used for ocean cells
      continue;
    }

    const elev = getElev(i, j);
    albedo[k] = computeAlbedo(lat, lon, elev);
    precip[k] = computePrecip(lat, lon, elev);

    landCount++;
    albedoSum += albedo[k];
    precipSum += precip[k];
  }
}

console.log(`Land cells: ${landCount}/${NX * NY}`);
console.log(`Mean land albedo: ${(albedoSum / landCount).toFixed(3)}`);
console.log(`Mean land precip: ${(precipSum / landCount).toFixed(0)} mm/yr`);

// Verify key regions
function sampleRegion(name, latRange, lonRange) {
  let sum = 0, count = 0, aSum = 0;
  for (let j = 0; j < NY; j++) {
    const lat = LAT0 + j;
    if (lat < latRange[0] || lat > latRange[1]) continue;
    for (let i = 0; i < NX; i++) {
      const lon = LON0 + i;
      if (lon < lonRange[0] || lon > lonRange[1]) continue;
      const k = j * NX + i;
      if (isOcean(i, j)) continue;
      sum += precip[k];
      aSum += albedo[k];
      count++;
    }
  }
  if (count === 0) return;
  console.log(`  ${name}: albedo=${(aSum/count).toFixed(3)}, precip=${(sum/count).toFixed(0)} mm/yr (${count} cells)`);
}

console.log('\nRegional checks:');
sampleRegion('Sahara', [18, 30], [-10, 30]);
sampleRegion('Amazon', [-10, 2], [-70, -45]);
sampleRegion('Greenland', [65, 82], [-55, -20]);
sampleRegion('Australia interior', [-30, -20], [125, 145]);
sampleRegion('Congo', [-3, 5], [12, 28]);
sampleRegion('Siberia', [55, 70], [60, 140]);

// Save albedo
const albedoOut = {
  nx: NX, ny: NY,
  lat0: LAT0, lat1: LAT1,
  lon0: LON0, lon1: LON1,
  source: 'Derived from geographic position and elevation heuristics (biome-based)',
  albedo: Array.from(albedo).map(v => Math.round(v * 1000) / 1000) // 3 decimal places
};
writeFileSync('albedo_1deg.json', JSON.stringify(albedoOut));
console.log(`\nSaved albedo_1deg.json (${(JSON.stringify(albedoOut).length / 1024).toFixed(0)} KB)`);

// Save precipitation
const precipOut = {
  nx: NX, ny: NY,
  lat0: LAT0, lat1: LAT1,
  lon0: LON0, lon1: LON1,
  source: 'Derived from geographic position heuristics (zonal + regional patterns)',
  precipitation: Array.from(precip).map(v => Math.round(v)) // integer mm/year
};
writeFileSync('precipitation_1deg.json', JSON.stringify(precipOut));
console.log(`Saved precipitation_1deg.json (${(JSON.stringify(precipOut).length / 1024).toFixed(0)} KB)`);
