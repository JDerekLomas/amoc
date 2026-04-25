#!/usr/bin/env node
/**
 * Idempotent patch script: applies all data-wiring changes to model.js
 *
 * Safe to re-run after Luke's edits — checks if each change exists before applying.
 * Run: node scripts/patch-data-wiring.mjs
 *
 * What it adds:
 *   - evapScale, peScale, snowAlbedoScale JS variables
 *   - obsSnowData + snowLoadPromise data loading
 *   - Temperature shader: Params struct fields, @binding(10-13), observed ice/snow/evap/PE physics
 *   - Data generation functions: generateSnowField, generateSeaIceField, generateEvapField, generatePrecipField
 */

import { readFileSync, writeFileSync } from 'fs';

const FILE = 'simamoc/model.js';
let code = readFileSync(FILE, 'utf8');
let patches = 0;

function patch(name, check, apply) {
  if (code.includes(check)) {
    console.log(`  [skip] ${name} — already present`);
    return;
  }
  code = apply(code);
  if (!code.includes(check)) {
    console.error(`  [FAIL] ${name} — patch did not apply`);
    process.exit(1);
  }
  patches++;
  console.log(`  [applied] ${name}`);
}

console.log('Patching ' + FILE + '...\n');

// 1. Tunable params
patch('evapScale/peScale/snowAlbedoScale variables', 'let evapScale', c =>
  c.replace(
    'let freshwaterScale_pe = 0.5;',
    'let freshwaterScale_pe = 0.5;\n\n' +
    '// GPU physics scaling (tunable, uploaded to Params struct slots 33-35)\n' +
    'let evapScale = 0.8;           // evaporative cooling strength (0 = off, 0.8 ≈ 80 W/m² global mean)\n' +
    'let peScale = 0.3;             // P-E salinity flux strength (0 = off)\n' +
    'let snowAlbedoScale = 0.45;    // snow albedo boost (bare→snow, 0 = off, 0.45 ≈ 15%→60%)'
  )
);

// 2. Snow data loading
patch('obsSnowData + snowLoadPromise', 'obsSnowData', c =>
  c.replace(
    /let currentsLoadPromise[^\n]+\n/,
    m => m +
    'let obsSnowData = null;\n' +
    "let snowLoadPromise = loadBinData('snow_cover').then(function(d) { obsSnowData = d; });\n"
  )
);

// 3. Temperature shader Params struct
patch('Params: evapScale/peScale/snowAlbedo fields', 'evapScale: f32, peScale: f32, snowAlbedo: f32', c => {
  // Find the temperature shader's Params struct (the one right before @binding(0)...psi...tempIn)
  // It's the last occurrence of _padS1 before the tempIn binding
  const tempShaderParamsRe = /('  _padS0: u32, _padS1: u32, _padS2: u32, _padS3: u32,',\n'};',\n'',\n'@group\(0\) @binding\(0\) var<storage, read> psi: array<f32>;',\n'@group\(0\) @binding\(1\) var<storage, read> tempIn)/;
  return c.replace(tempShaderParamsRe,
    "'  _padS0: u32, evapScale: f32, peScale: f32, snowAlbedo: f32,',\n'};',\n'',\n'@group(0) @binding(0) var<storage, read> psi: array<f32>;',\n'@group(0) @binding(1) var<storage, read> tempIn"
  );
});

// 4. Shader bindings 10-13
patch('@binding(10-13) snow/ice/evap/precip', '@binding(10)', c =>
  c.replace(
    "'@group(0) @binding(9) var<storage, read> ekmanVel: array<f32>;',",
    "'@group(0) @binding(9) var<storage, read> ekmanVel: array<f32>;',\n" +
    "'@group(0) @binding(10) var<storage, read> snowCover: array<f32>;',\n" +
    "'@group(0) @binding(11) var<storage, read> seaIceFrac: array<f32>;',\n" +
    "'@group(0) @binding(12) var<storage, read> evapRate: array<f32>;',\n" +
    "'@group(0) @binding(13) var<storage, read> precipRate: array<f32>;',"
  )
);

// 5. Observed sea ice (replace analytical iceFrac with blended observed)
patch('observed sea ice blend', 'seaIceFrac[k]', c => {
  // Add observed ice variables after qSolar line
  c = c.replace(
    /(\/\/ Insolation with ice-albedo[^\n]*\n'  let cosZenith[^\n]+\n'  var qSolar[^\n]+\n)/,
    m => m +
    "'',\n" +
    "'  // Sea ice: blend observed NOAA ice fraction with SST-based fallback',\n" +
    "'  let obsIce = seaIceFrac[k];',\n" +
    "'  let sstIceT = clamp((tempIn[k] + 2.0) / 10.0, 0.0, 1.0);',\n" +
    "'  let sstIceFrac = 1.0 - sstIceT * sstIceT * (3.0 - 2.0 * sstIceT);',\n" +
    "'  let iceFrac = select(sstIceFrac, obsIce, obsIce > 0.001);',\n"
  );
  // Remove old analytical iceFrac lines if present
  c = c.replace(
    /'    let iceT = clamp\(\(tempIn\[k\] \+ 2\.0\) \/ 10\.0, 0\.0, 1\.0\);',\n'    let iceFrac = 1\.0 - iceT \* iceT \* \(3\.0 - 2\.0 \* iceT\);',\n/,
    ''
  );
  return c;
});

// 6. Snow-albedo
patch('snow-albedo feedback', 'snowCover[k]', c =>
  c.replace(
    /('  }',\n'',\n'  \/\/ ── CLOUD PARAM)/,
    "'  }',\n'',\n" +
    "'  // Snow-albedo on land',\n" +
    "'  let snowFrac = snowCover[k];',\n" +
    "'  if (snowFrac > 0.01) { qSolar *= 1.0 - params.snowAlbedo * snowFrac; }',\n" +
    "'',\n'  // ── CLOUD PARAM"
  )
);

// 7. Evaporative cooling
patch('evaporative cooling (evapCool)', 'params.evapScale * evapRate', c =>
  c.replace(
    /('  tempOut\[k\] = tempIn\[k\] \+ params\.dt \* \(-advec \+ qNet \+ diff \+ landFlux\);',)/,
    "'  let evapCool = params.evapScale * evapRate[k];',\n" +
    "'  tempOut[k] = tempIn[k] + params.dt * (-advec + qNet + diff + landFlux - evapCool);',"
  )
);

// 8. P-E salinity flux
patch('P-E salinity flux (peFlux)', 'params.peScale', c => {
  c = c.replace(
    /('  var fwSal: f32 = 0\.0;',)/,
    "'  let peFlux = params.peScale * (evapRate[k] - precipRate[k]) * tempIn[salK] / 35.0;',\n$1"
  );
  c = c.replace('salRestore + fwSal);', 'salRestore + fwSal + peFlux);');
  return c;
});

// 9. NH boundary layer clouds
patch('NH boundary layer clouds (nhCloud)', 'nhCloud', c => {
  c = c.replace(
    /('  \/\/ \d+\. Polar stratus \(both hemispheres\)')/,
    "'  // NH mid-latitude boundary layer clouds (observed 0.80-0.90 at 50-65N)',\n" +
    "'  let nhCloud = select(0.0, 0.35 * clamp((absLat - 40.0) / 12.0, 0.0, 1.0)',\n" +
    "'                                 * clamp((70.0 - absLat) / 10.0, 0.0, 1.0), lat > 0.0);',\n" +
    "'',\n$1"
  );
  // Add nhCloud to lowCloud sum
  c = c.replace(
    'stratocu + stormTrack + soCloud + polarCloud',
    'stratocu + stormTrack + soCloud + nhCloud + polarCloud'
  );
  // Bump cloud cap to 0.92
  c = c.replace("0.05, 0.85);'", "0.05, 0.92);'");
  return c;
});

// 10. Data generation functions
patch('generateSnowField function', 'function generateSnowField', c => {
  const funcs = `
// ============================================================
// SNOW / SEA ICE / EVAPORATION / PRECIPITATION FIELD GENERATORS
// ============================================================
var snowField = null;
function generateSnowField() {
  snowField = new Float32Array(NX * NY);
  if (obsSnowData && obsSnowData.snow_cover) {
    var src = obsSnowData.snow_cover;
    for (var k = 0; k < NX * NY; k++) snowField[k] = Math.max(0, Math.min(1, (src[k] || 0) / 100));
    console.log('Using MODIS observed snow cover');
  } else {
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var sf = Math.max(0, Math.min(0.6, (Math.abs(lat) - 50) / 30));
      for (var i = 0; i < NX; i++) snowField[j * NX + i] = sf;
    }
  }
}
var seaIceField = null;
function generateSeaIceField() {
  seaIceField = new Float32Array(NX * NY);
  if (obsSeaIceData && obsSeaIceData.ice_fraction) {
    var src = obsSeaIceData.ice_fraction;
    for (var k = 0; k < NX * NY; k++) seaIceField[k] = Math.max(0, Math.min(1, src[k] || 0));
    console.log('Using NOAA observed sea ice fraction');
  } else {
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var ice = Math.max(0, Math.min(1, (Math.abs(lat) - 60) / 15));
      for (var i = 0; i < NX; i++) seaIceField[j * NX + i] = ice;
    }
  }
}
var evapField = null;
function generateEvapField() {
  evapField = new Float32Array(NX * NY);
  if (obsEvapData && obsEvapData.evaporation) {
    var src = obsEvapData.evaporation;
    var sum = 0, cnt = 0;
    for (var k = 0; k < NX * NY; k++) { var v = src[k] || 0; if (v > 0 && mask[k]) { sum += v; cnt++; } }
    var meanEvap = cnt > 0 ? sum / cnt : 1000;
    var scale = 1.0 / meanEvap;
    for (var k = 0; k < NX * NY; k++) evapField[k] = Math.max(0, (src[k] || 0) * scale);
    console.log('Using ERA5 evaporation (mean=' + meanEvap.toFixed(0) + ' mm/yr)');
  } else {
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var e = Math.max(0, 1.0 - Math.pow((Math.abs(lat) - 20) / 40, 2));
      for (var i = 0; i < NX; i++) evapField[j * NX + i] = e;
    }
  }
}
var precipOceanField = null;
function generatePrecipField() {
  precipOceanField = new Float32Array(NX * NY);
  if (obsPrecipData && obsPrecipData.precipitation) {
    var src = obsPrecipData.precipitation;
    var evapMean = 1000;
    if (obsEvapData && obsEvapData.evaporation) {
      var es = 0, ec = 0;
      for (var k = 0; k < NX * NY; k++) { var v = obsEvapData.evaporation[k] || 0; if (v > 0 && mask[k]) { es += v; ec++; } }
      if (ec > 0) evapMean = es / ec;
    }
    var scale = 1.0 / evapMean;
    for (var k = 0; k < NX * NY; k++) precipOceanField[k] = Math.max(0, (src[k] || 0) * scale);
    console.log('Using observed precipitation for P-E flux');
  } else {
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var p = 0.5 * Math.exp(-lat * lat / 200) + 0.3 * Math.exp(-Math.pow((Math.abs(lat) - 45) / 15, 2));
      for (var i = 0; i < NX; i++) precipOceanField[j * NX + i] = p;
    }
  }
}
`;
  return c.replace(
    '// ============================================================\n// TEMPERATURE / SALINITY INITIALIZATION',
    funcs + '\n// ============================================================\n// TEMPERATURE / SALINITY INITIALIZATION'
  );
});

if (patches === 0) {
  console.log('\nNo changes needed — all patches already applied.');
} else {
  writeFileSync(FILE, code);
  console.log(`\nApplied ${patches} patches to ${FILE}`);
}
