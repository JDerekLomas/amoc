#!/usr/bin/env node
/**
 * Validate all WGSL shader code strings in model.js
 * Checks: balanced braces/parens, required declarations, Params struct consistency
 */
import { readFileSync } from 'fs';

const src = readFileSync('simamoc/model.js', 'utf8');

const shaders = [
  'timestepShaderCode', 'poissonShaderCode', 'enforceBCShaderCode',
  'deepTimestepShaderCode', 'temperatureShaderCode', 'atmosphereShaderCode'
];

// Also extract from fft shaders
const fftShaders = [
  'fftButterflyShaderCode', 'fftBitRevShaderCode', 'fftTridiagShaderCode',
  'fftTransposeShaderCode', 'fftScaleMaskShaderCode'
];

let allOk = true;

for (const name of [...shaders, ...fftShaders]) {
  // Find the shader string array
  const start = src.indexOf(`var ${name} = [`);
  if (start === -1) { console.log(`  SKIP  ${name} (not found)`); continue; }
  const end = src.indexOf("].join('\\n')", start);
  if (end === -1) { console.log(`  SKIP  ${name} (no join found)`); continue; }

  const block = src.slice(start, end);

  // Extract string literals
  const lines = [];
  const re = /'((?:[^'\\]|\\.)*)'/g;
  let m;
  while ((m = re.exec(block)) !== null) {
    lines.push(m[1].replace(/\\'/g, "'").replace(/\\n/g, '\n'));
  }
  const wgsl = lines.join('\n');

  // Validate
  const issues = [];

  // Balanced delimiters
  let braceDepth = 0, parenDepth = 0;
  for (const ch of wgsl) {
    if (ch === '{') braceDepth++;
    else if (ch === '}') braceDepth--;
    else if (ch === '(') parenDepth++;
    else if (ch === ')') parenDepth--;
  }
  if (braceDepth !== 0) issues.push(`unbalanced braces (${braceDepth > 0 ? '+' : ''}${braceDepth})`);
  if (parenDepth !== 0) issues.push(`unbalanced parens (${parenDepth > 0 ? '+' : ''}${parenDepth})`);

  // Required elements
  if (name.includes('fft')) {
    if (!wgsl.includes('fn main')) issues.push('missing fn main');
  } else {
    if (!wgsl.includes('fn main')) issues.push('missing fn main');
    if (!wgsl.includes('struct Params')) issues.push('missing struct Params');
  }

  // Check Params struct fields for non-FFT shaders
  if (wgsl.includes('struct Params')) {
    const paramBlock = wgsl.match(/struct Params \{([\s\S]*?)\}/);
    if (paramBlock) {
      const fields = paramBlock[1].match(/\w+:\s*(u32|f32)/g) || [];
      if (fields.length < 36) issues.push(`only ${fields.length} Params fields (expected 40)`);
    }
  }

  // Check for undefined variable references in temperature shader
  if (name === 'temperatureShaderCode') {
    // Key variables that should be defined before use
    const defs = ['N_atm', 'atm_q', 'humidity', 'airTempEst', 'dynEvap', 'qSat_sst', 'qSat_air', 'dynPrecip'];
    for (const v of defs) {
      const defIdx = wgsl.indexOf(`let ${v}`);
      if (defIdx === -1 && wgsl.includes(v)) {
        issues.push(`${v} used but not defined with 'let'`);
      }
    }
    // Check atmosphere binding
    if (!wgsl.includes('binding(14)')) issues.push('missing atmosphere binding(14)');
    // Check three-type cloud model
    if (!wgsl.includes('w_ci')) issues.push('missing cirrus weighting (w_ci)');
    if (!wgsl.includes('cirrus')) issues.push('missing cirrus variable');
  }

  // Check atmosphere shader
  if (name === 'atmosphereShaderCode') {
    if (!wgsl.includes('fn qSat')) issues.push('missing qSat function');
    if (!wgsl.includes('ekmanVel')) issues.push('missing ekmanVel binding');
    if (!wgsl.includes('moistAdvec')) issues.push('missing moisture advection');
    if (!wgsl.includes('airAdvec')) issues.push('missing air temp advection');
  }

  const status = issues.length === 0 ? 'OK' : issues.join(', ');
  if (issues.length > 0) allOk = false;
  console.log(`  ${issues.length === 0 ? 'OK' : 'FAIL'}  ${name}: ${lines.length} lines — ${status}`);
}

// Check gpu-solver.js for atmosphere wiring
const gpuSrc = readFileSync('simamoc/gpu-solver.js', 'utf8');
const gpuChecks = [
  ['gpuAtmBuf', 'atmosphere buffer variable'],
  ['gpuAtmNewBuf', 'atmosphere double buffer'],
  ['gpuAtmospherePipeline', 'atmosphere pipeline'],
  ['gpuAtmosphereBindGroup', 'atmosphere bind group'],
  ['gpuSwapAtmosphereBindGroup', 'atmosphere swap bind group'],
  ['atmosphereShaderCode', 'atmosphere shader reference'],
  ['gpuAtmReadbackBuf', 'atmosphere readback buffer'],
  ['binding: 14', 'atmosphere binding in temperature BG'],
];

console.log('\ngpu-solver.js wiring:');
for (const [pat, desc] of gpuChecks) {
  const found = gpuSrc.includes(pat);
  if (!found) allOk = false;
  console.log(`  ${found ? 'OK' : 'FAIL'}  ${desc}`);
}

// Check uploadParams has atmosphere fields
const hasAtmParams = gpuSrc.includes('f32[36]') && gpuSrc.includes('f32[37]') &&
                     gpuSrc.includes('f32[38]') && gpuSrc.includes('f32[39]');
console.log(`  ${hasAtmParams ? 'OK' : 'FAIL'}  uploadParams atmosphere fields (slots 36-39)`);
if (!hasAtmParams) allOk = false;

// Check atmosphere step in gpuRunSteps
const hasAtmStep = gpuSrc.includes('gpuAtmospherePipeline') && gpuSrc.includes('atmPass');
console.log(`  ${hasAtmStep ? 'OK' : 'FAIL'}  atmosphere step in gpuRunSteps`);
if (!hasAtmStep) allOk = false;

// Check atmosphere readback
const hasAtmReadback = gpuSrc.includes('gpuAtmReadbackBuf.mapAsync');
console.log(`  ${hasAtmReadback ? 'OK' : 'FAIL'}  atmosphere readback`);
if (!hasAtmReadback) allOk = false;

// Check atmosphere init
const hasAtmInit = gpuSrc.includes('gpuAtmBuf, 0, atmInit');
console.log(`  ${hasAtmInit ? 'OK' : 'FAIL'}  atmosphere buffer initialization`);
if (!hasAtmInit) allOk = false;

console.log(`\n${allOk ? 'ALL CHECKS PASSED' : 'SOME CHECKS FAILED'}`);
process.exit(allOk ? 0 : 1);
