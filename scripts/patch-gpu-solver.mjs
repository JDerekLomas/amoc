#!/usr/bin/env node
/**
 * Idempotent patch: applies data wiring changes to gpu-solver.js
 *
 * Safe to re-run. Adds: Snow/SeaIce/Evap/Precip GPU buffers, bind groups,
 * data upload calls, maxStorageBuffersPerShaderStage bump, uploadParams slots.
 *
 * Run: node scripts/patch-gpu-solver.mjs
 */

import { readFileSync, writeFileSync } from 'fs';

const FILE = 'simamoc/gpu-solver.js';
let code = readFileSync(FILE, 'utf8');
let patches = 0;

function patch(name, check, apply) {
  if (code.includes(check)) {
    console.log(`  [skip] ${name}`);
    return;
  }
  code = apply(code);
  if (!code.includes(check)) {
    console.error(`  [FAIL] ${name}`);
    process.exit(1);
  }
  patches++;
  console.log(`  [applied] ${name}`);
}

console.log('Patching ' + FILE + '...\n');

// 1. Buffer declarations
patch('gpuSnowBuf/gpuSeaIceBuf/gpuEvapBuf/gpuPrecipBuf declarations', 'gpuSnowBuf', c =>
  c.replace(
    'var gpuSalClimBuf, gpuWindCurlBuf, gpuEkmanBuf;',
    'var gpuSalClimBuf, gpuWindCurlBuf, gpuEkmanBuf;\nvar gpuSnowBuf, gpuSeaIceBuf, gpuEvapBuf, gpuPrecipBuf;'
  )
);

// 2. maxStorageBuffersPerShaderStage
patch('maxStorageBuffersPerShaderStage: 14', 'maxStorageBuffersPerShaderStage: 14', c =>
  c.replace(
    /maxStorageBuffersPerShaderStage: \d+/,
    'maxStorageBuffersPerShaderStage: 14'
  )
);

// 3. Buffer creation
patch('GPU buffer creation for snow/ice/evap/precip', 'gpuSnowBuf = gpuDevice.createBuffer', c =>
  c.replace(
    /(gpuEkmanBuf = gpuDevice\.createBuffer[^;]+;[^\n]*\n)/,
    '$1' +
    '  gpuSnowBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });\n' +
    '  gpuSeaIceBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });\n' +
    '  gpuEvapBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });\n' +
    '  gpuPrecipBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });\n'
  )
);

// 4. Data upload calls
patch('generateSnowField + upload', 'generateSnowField', c =>
  c.replace(
    /(gpuDevice\.queue\.writeBuffer\(gpuEkmanBuf, 0, ekmanField\);)/,
    '$1\n\n' +
    '  // Snow cover, sea ice, evaporation, precipitation for radiation + salinity\n' +
    '  generateSnowField();\n' +
    '  gpuDevice.queue.writeBuffer(gpuSnowBuf, 0, snowField);\n' +
    '  generateSeaIceField();\n' +
    '  gpuDevice.queue.writeBuffer(gpuSeaIceBuf, 0, seaIceField);\n' +
    '  generateEvapField();\n' +
    '  gpuDevice.queue.writeBuffer(gpuEvapBuf, 0, evapField);\n' +
    '  generatePrecipField();\n' +
    '  gpuDevice.queue.writeBuffer(gpuPrecipBuf, 0, precipOceanField);'
  )
);

// 5. Temperature bind group entries (normal)
patch('binding 10-13 in gpuTemperatureBindGroup', 'binding: 10, resource: { buffer: gpuSnowBuf }', c => {
  // Add to normal bind group
  c = c.replace(
    /(binding: 9, resource: \{ buffer: gpuEkmanBuf \} \},\n\s*\]\n\s*\}\);[\s\n]*\/\/ Swapped)/,
    'binding: 9, resource: { buffer: gpuEkmanBuf } },\n' +
    '      { binding: 10, resource: { buffer: gpuSnowBuf } },\n' +
    '      { binding: 11, resource: { buffer: gpuSeaIceBuf } },\n' +
    '      { binding: 12, resource: { buffer: gpuEvapBuf } },\n' +
    '      { binding: 13, resource: { buffer: gpuPrecipBuf } },\n' +
    '    ]\n  });\n\n  // Swapped'
  );
  // Add to swapped bind group
  c = c.replace(
    /(binding: 9, resource: \{ buffer: gpuEkmanBuf \} \},\n\s*\]\n\s*\}\);[\s\n]*\/\/ Deep timestep)/,
    'binding: 9, resource: { buffer: gpuEkmanBuf } },\n' +
    '      { binding: 10, resource: { buffer: gpuSnowBuf } },\n' +
    '      { binding: 11, resource: { buffer: gpuSeaIceBuf } },\n' +
    '      { binding: 12, resource: { buffer: gpuEvapBuf } },\n' +
    '      { binding: 13, resource: { buffer: gpuPrecipBuf } },\n' +
    '    ]\n  });\n  // Deep timestep'
  );
  return c;
});

// 6. uploadParams: write evapScale/peScale/snowAlbedoScale to slots 33-35
patch('uploadParams evapScale/peScale/snowAlbedoScale', 'f32[33] = evapScale', c =>
  c.replace(
    /u32\[32\] = 0; u32\[33\] = 0; u32\[34\] = 0; u32\[35\] = 0;[^\n]*/,
    'u32[32] = 0; // _padS0 (SOR color flag)\n  f32[33] = evapScale;\n  f32[34] = peScale;\n  f32[35] = snowAlbedoScale;'
  )
);

if (patches === 0) {
  console.log('\nNo changes needed.');
} else {
  writeFileSync(FILE, code);
  console.log(`\nApplied ${patches} patches to ${FILE}`);
}
