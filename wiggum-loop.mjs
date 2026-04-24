#!/usr/bin/env node
/**
 * Ralph Wiggum Loop for AMOC Climate Simulation
 * Multi-agent dialectic with physics-aligned evaluation
 *
 * Three Claude agents debate parameter changes:
 *   1. PHYSICIST — generates hypotheses about why the sim diverges
 *   2. TUNER — proposes parameter changes based on winning hypothesis
 *   3. VALIDATOR — checks if proposal violates physical constraints
 *
 * Four-tier evaluation:
 *   T1: Conservation (energy balance closes)
 *   T2: Structural (right features emerge for right reasons)
 *   T3: Sensitivity (correct response to perturbations)
 *   T4: Quantitative (SST matches observations)
 *
 * Usage:
 *   ANTHROPIC_API_KEY=sk-ant-... node wiggum-loop.mjs [--model sonnet] [--max-iters 10] [--spinup 120]
 *
 * Models: opus, sonnet (default), haiku
 */

import { chromium } from 'playwright';
import { mkdirSync, writeFileSync, readFileSync } from 'fs';
import { createServer } from 'http';
import { resolve } from 'path';
import Anthropic from '@anthropic-ai/sdk';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const ROOT = '/Users/dereklomas/lukebarrington/amoc';
const OUT = './screenshots/wiggum';
const MAX_ITERS = parseInt(process.argv.find((_, i, a) => a[i - 1] === '--max-iters') || '10');
const SPINUP_SECS = parseInt(process.argv.find((_, i, a) => a[i - 1] === '--spinup') || '120');

const MODEL_ARG = (process.argv.find((_, i, a) => a[i - 1] === '--model') || 'sonnet').toLowerCase();
const MODEL_MAP = {
  opus:   'claude-opus-4-6',
  sonnet: 'claude-sonnet-4-6',
  haiku:  'claude-haiku-4-5-20251001',
};
const CLAUDE_MODEL = MODEL_MAP[MODEL_ARG] || MODEL_MAP.sonnet;

const COST_LIMIT = parseFloat(process.argv.find((_, i, a) => a[i - 1] === '--cost-limit') || '50');
// Claude pricing (per million tokens) — approximate, varies by model
const PRICING = {
  'claude-opus-4-6':          { input: 15.0 / 1e6, output: 75.0 / 1e6 },
  'claude-sonnet-4-6':        { input: 3.0 / 1e6,  output: 15.0 / 1e6 },
  'claude-haiku-4-5-20251001': { input: 0.80 / 1e6, output: 4.0 / 1e6 },
};
const PRICE_INPUT = PRICING[CLAUDE_MODEL]?.input || 3.0 / 1e6;
const PRICE_OUTPUT = PRICING[CLAUDE_MODEL]?.output || 15.0 / 1e6;
let totalCostEstimate = 0;

const anthropic = new Anthropic();

mkdirSync(OUT, { recursive: true });

// ---------------------------------------------------------------------------
// Reference data — NOAA OI SST + WOA23 deep temps
// ---------------------------------------------------------------------------
function loadReferenceData() {
  const sstRaw = JSON.parse(readFileSync(resolve(ROOT, 'sst_global_1deg.json'), 'utf8'));
  const deepRaw = JSON.parse(readFileSync(resolve(ROOT, 'deep_temp_1deg.json'), 'utf8'));
  const nx = 360, ny = 160;
  const latBins = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70];
  const refSST = {}, refDeep = {};

  for (const targetLat of latBins) {
    const j = Math.round((targetLat + 79.5) / 1.0);
    if (j < 0 || j >= ny) continue;
    let sstSum = 0, sstCount = 0, deepSum = 0, deepCount = 0;
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const sv = sstRaw.sst?.[k] ?? sstRaw.data?.[k] ?? sstRaw[k];
      const dv = deepRaw.temp?.[k] ?? deepRaw.data?.[k] ?? deepRaw[k];
      if (sv != null && !isNaN(sv) && sv > -90) { sstSum += sv; sstCount++; }
      if (dv != null && !isNaN(dv) && dv > -90) { deepSum += dv; deepCount++; }
    }
    if (sstCount > 0) refSST[targetLat] = sstSum / sstCount;
    if (deepCount > 0) refDeep[targetLat] = deepSum / deepCount;
  }

  let gSum = 0, gCount = 0;
  const data = sstRaw.sst || sstRaw.data || sstRaw;
  for (let k = 0; k < nx * ny; k++) {
    const v = data[k];
    if (v != null && !isNaN(v) && v > -90) { gSum += v; gCount++; }
  }
  return { refSST, refDeep, globalMeanSST: gSum / gCount };
}

// ---------------------------------------------------------------------------
// Tunable parameters
// ---------------------------------------------------------------------------
const TUNABLE_PARAMS = {
  S_solar:        { default: 100.0,  min: 20,    max: 200,   desc: 'Solar heating amplitude' },
  A_olr:          { default: 40.0,   min: 10,    max: 80,    desc: 'OLR constant (higher = colder equilibrium)' },
  B_olr:          { default: 2.0,    min: 0.5,   max: 5.0,   desc: 'OLR linear coefficient (restoring strength)' },
  kappa_diff:     { default: 2.5e-4, min: 1e-5,  max: 1e-3,  desc: 'Thermal diffusion coefficient' },
  alpha_T:        { default: 0.05,   min: 0.001, max: 0.2,   desc: 'Buoyancy-vorticity coupling strength' },
  r_friction:     { default: 0.04,   min: 0.005, max: 0.15,  desc: 'Bottom friction (higher = slower currents)' },
  A_visc:         { default: 5e-4,   min: 5e-5,  max: 2e-3,  desc: 'Lateral eddy viscosity' },
  gamma_mix:      { default: 0.01,   min: 0.001, max: 0.1,   desc: 'Base vertical mixing rate' },
  gamma_deep_form:{ default: 0.5,    min: 0.05,  max: 2.0,   desc: 'Deep water formation rate' },
  kappa_deep:     { default: 2e-5,   min: 1e-6,  max: 1e-4,  desc: 'Deep horizontal diffusion' },
  F_couple_s:     { default: 0.5,    min: 0.05,  max: 2.0,   desc: 'Interfacial coupling (surface feels deep)' },
  r_deep:         { default: 0.1,    min: 0.01,  max: 0.5,   desc: 'Deep layer bottom friction' },
  windStrength:   { default: 1.0,    min: 0.2,   max: 2.0,   desc: 'Wind forcing multiplier' },
};

// ---------------------------------------------------------------------------
// Claude API
// ---------------------------------------------------------------------------
async function callClaude(prompt, { jsonMode = true, images = [] } = {}) {
  // Build content blocks: images first, then text
  const content = [];
  for (const img of images) {
    content.push({
      type: 'image',
      source: {
        type: 'base64',
        media_type: 'image/png',
        data: img.buffer.toString('base64'),
      },
    });
    content.push({ type: 'text', text: `[Above: ${img.label}]` });
  }

  // For JSON mode, wrap prompt with explicit JSON instruction
  const textPrompt = jsonMode
    ? prompt + '\n\nRespond with ONLY valid JSON, no markdown fences or extra text.'
    : prompt;
  content.push({ type: 'text', text: textPrompt });

  const resp = await anthropic.messages.create({
    model: CLAUDE_MODEL,
    max_tokens: 4096,
    temperature: 0.3,
    messages: [{ role: 'user', content }],
  });

  const text = resp.content?.[0]?.text;
  if (!text) throw new Error('Empty Claude response');

  // Track cost from usage
  const inputTokens = resp.usage?.input_tokens || 0;
  const outputTokens = resp.usage?.output_tokens || 0;
  const cost = inputTokens * PRICE_INPUT + outputTokens * PRICE_OUTPUT;
  totalCostEstimate += cost;
  console.log(`    [${CLAUDE_MODEL} | ${inputTokens}+${outputTokens} tok | $${cost.toFixed(4)} | total: $${totalCostEstimate.toFixed(4)} / $${COST_LIMIT}]`);
  if (totalCostEstimate >= COST_LIMIT) {
    throw new Error(`BUDGET LIMIT reached: $${totalCostEstimate.toFixed(2)} >= $${COST_LIMIT}`);
  }

  if (jsonMode) {
    // Extract JSON from response (handle markdown fences if model adds them)
    const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) || [null, text];
    return JSON.parse(jsonMatch[1].trim());
  }
  return text;
}

// ---------------------------------------------------------------------------
// TIER 1: Conservation checks (computed, not LLM-judged)
// ---------------------------------------------------------------------------
function checkConservation(simData) {
  const checks = [];
  const latBins = Object.keys(simData.zonalTemp).map(Number).sort((a, b) => a - b);

  // Energy balance: at equilibrium, global mean should be stable
  // We check if temperature range is physically reasonable
  const { globalMean, globalMin, globalMax } = simData;
  checks.push({
    name: 'temperature_range',
    pass: globalMin > -15 && globalMax < 40,
    value: `[${globalMin.toFixed(1)}, ${globalMax.toFixed(1)}]°C`,
    expected: '[-15, 40]°C',
    reason: 'Temperatures must stay within physically realizable ocean range',
  });

  // Equator-to-pole gradient should be positive (warm tropics, cold poles)
  const tropicalT = simData.zonalTemp[0] ?? simData.zonalTemp[10];
  const polarT = simData.zonalTemp[-70] ?? simData.zonalTemp[-60];
  const gradient = (tropicalT ?? 25) - (polarT ?? 0);
  checks.push({
    name: 'equator_pole_gradient',
    pass: gradient > 10 && gradient < 50,
    value: `${gradient.toFixed(1)}°C`,
    expected: '15-35°C',
    reason: 'Meridional temperature gradient must exist and be reasonable',
  });

  // Hemispheric asymmetry should be modest (not wildly different N vs S)
  const nhTemps = latBins.filter(l => l > 0).map(l => simData.zonalTemp[l]).filter(Boolean);
  const shTemps = latBins.filter(l => l < 0).map(l => simData.zonalTemp[l]).filter(Boolean);
  const nhMean = nhTemps.reduce((a, b) => a + b, 0) / nhTemps.length;
  const shMean = shTemps.reduce((a, b) => a + b, 0) / shTemps.length;
  const asymm = Math.abs(nhMean - shMean);
  checks.push({
    name: 'hemispheric_symmetry',
    pass: asymm < 8,
    value: `NH=${nhMean.toFixed(1)}°C, SH=${shMean.toFixed(1)}°C, diff=${asymm.toFixed(1)}°C`,
    expected: 'Difference < 8°C',
    reason: 'Ocean temperatures are roughly symmetric; large asymmetry suggests broken forcing',
  });

  // Deep ocean should be colder than surface everywhere
  let deepWarmerCount = 0, deepTotal = 0;
  for (const lat of latBins) {
    if (simData.zonalTemp[lat] != null && simData.zonalDeep[lat] != null) {
      deepTotal++;
      if (simData.zonalDeep[lat] > simData.zonalTemp[lat] + 2) deepWarmerCount++;
    }
  }
  checks.push({
    name: 'deep_colder_than_surface',
    pass: deepWarmerCount === 0,
    value: `${deepWarmerCount}/${deepTotal} bands have deep warmer than surface`,
    expected: '0 bands',
    reason: 'Deep ocean is always colder than surface (stable stratification)',
  });

  // AMOC should be positive (northward surface, southward deep in Atlantic)
  // Note: amocStrength is non-dimensional (psi-gradient units), not Sverdrups
  // Typical range: 0.0001-0.01 for active overturning
  const amocVal = simData.amoc || 0;
  checks.push({
    name: 'amoc_positive',
    pass: amocVal > 1e-5,
    value: `AMOC index: ${amocVal.toExponential(2)} (non-dim)`,
    expected: '> 1e-5 (active overturning)',
    reason: 'AMOC must be positive (thermally-driven overturning)',
  });

  return checks;
}

// ---------------------------------------------------------------------------
// TIER 2: Structural emergence checks (extracted from simulation state)
// ---------------------------------------------------------------------------
async function checkStructure(page) {
  return await page.evaluate(() => {
    // Grid: LON0=-180, LON1=180, LAT0=-80, LAT1=80
    // i=0 → -180°, i=180 → 0°, i=360 → 180°
    // j=0 → -80°, j=179 → 80°
    const nx = NX, ny = NY;
    const m = mask, p = psi;
    if (!m || !p) return [];

    // Helper: lon/lat to grid indices
    function lonToI(lon) { return Math.round((lon + 180) / 360 * (nx - 1)); }
    function latToJ(lat) { return Math.round((lat + 80) / 160 * (ny - 1)); }
    function speed(i, j) {
      const k = j * nx + i;
      if (!m[k]) return 0;
      const jn = Math.min(j + 1, ny - 1), js = Math.max(j - 1, 0);
      const ie = (i + 1) % nx, iw = (i - 1 + nx) % nx;
      const u = -(p[jn * nx + i] - p[js * nx + i]) * 0.5 * ny;
      const v = (p[j * nx + ie] - p[j * nx + iw]) * 0.5 * nx;
      return Math.sqrt(u * u + v * v);
    }

    const checks = [];

    // ── WESTERN BOUNDARY INTENSIFICATION ──
    // North Atlantic: lon -80° to -10°, lat 25° to 50°
    // Western boundary (Gulf Stream): lon -80° to -60° (near US east coast)
    // Eastern interior: lon -50° to -10°
    const jLo = latToJ(25), jHi = latToJ(50);
    const iWestLo = lonToI(-80), iWestHi = lonToI(-60);
    const iEastLo = lonToI(-50), iEastHi = lonToI(-10);

    let westMaxSpd = 0, eastMaxSpd = 0;
    for (let j = jLo; j <= jHi; j++) {
      for (let i = iWestLo; i <= iWestHi; i++) westMaxSpd = Math.max(westMaxSpd, speed(i, j));
      for (let i = iEastLo; i <= iEastHi; i++) eastMaxSpd = Math.max(eastMaxSpd, speed(i, j));
    }
    checks.push({
      name: 'western_intensification',
      pass: westMaxSpd > eastMaxSpd * 1.5,
      value: `Gulf Stream max: ${westMaxSpd.toFixed(3)}, NA interior max: ${eastMaxSpd.toFixed(3)}, ratio: ${(westMaxSpd / Math.max(eastMaxSpd, 1e-6)).toFixed(1)}x`,
      expected: 'Western boundary > 1.5x eastern interior',
      reason: 'Stommel: beta effect concentrates return flow on western boundary',
    });

    // ── SUBTROPICAL GYRE EXISTS ──
    // North Atlantic subtropical gyre: lon -70° to -20°, lat 20° to 40°
    const jGyLo = latToJ(20), jGyHi = latToJ(40);
    const iGyLo = lonToI(-70), iGyHi = lonToI(-20);
    let gyrePsiSum = 0, gyreN = 0, gyrePsiMin = Infinity, gyrePsiMax = -Infinity;
    for (let j = jGyLo; j <= jGyHi; j++) {
      for (let i = iGyLo; i <= iGyHi; i++) {
        const k = j * nx + i;
        if (m[k]) {
          gyrePsiSum += p[k]; gyreN++;
          if (p[k] < gyrePsiMin) gyrePsiMin = p[k];
          if (p[k] > gyrePsiMax) gyrePsiMax = p[k];
        }
      }
    }
    const gyrePsiMean = gyreN > 0 ? gyrePsiSum / gyreN : 0;
    const gyrePsiRange = gyrePsiMax - gyrePsiMin;
    // Gyre exists if psi has meaningful variation (not flat) within the basin
    checks.push({
      name: 'subtropical_gyre_exists',
      pass: gyrePsiRange > 0.005,
      value: `NA subtrop psi: mean=${gyrePsiMean.toFixed(4)}, range=${gyrePsiRange.toFixed(4)}, cells=${gyreN}`,
      expected: 'Psi range > 0.005 (organized recirculation)',
      reason: 'Wind-driven subtropical gyre from Sverdrup balance',
    });

    // ── ANTARCTIC CIRCUMPOLAR CURRENT ──
    // ACC flows eastward at ~55-60°S through Drake Passage
    const jACC = latToJ(-58);
    let accEastward = 0, accN = 0;
    for (let i = 0; i < nx; i++) {
      const k = jACC * nx + i;
      if (!m[k]) continue;
      const jn = Math.min(jACC + 1, ny - 1), js = Math.max(jACC - 1, 0);
      // u = -dpsi/dy (eastward velocity)
      const u = -(p[jn * nx + i] - p[js * nx + i]) * 0.5 * ny;
      accEastward += u;
      accN++;
    }
    const meanACC = accN > 0 ? accEastward / accN : 0;
    checks.push({
      name: 'acc_eastward_flow',
      pass: Math.abs(meanACC) > 0.01,
      value: `Mean zonal flow at 58°S: ${meanACC.toFixed(4)} (${accN} ocean cells)`,
      expected: 'Significant circumpolar flow',
      reason: 'Drake Passage open → wind drives eastward ACC',
    });

    // ── DEEP WATER FORMATION SIGNATURE ──
    if (typeof deepTemp !== 'undefined') {
      const dt = deepTemp;
      const jPolar = latToJ(65), jTropical = latToJ(0);
      let polarD = 0, polarN = 0, tropD = 0, tropN = 0;
      for (let i = 0; i < nx; i++) {
        const kp = jPolar * nx + i, kt = jTropical * nx + i;
        if (m[kp]) { polarD += dt[kp]; polarN++; }
        if (m[kt]) { tropD += dt[kt]; tropN++; }
      }
      const pMean = polarN > 0 ? polarD / polarN : 0;
      const tMean = tropN > 0 ? tropD / tropN : 10;
      checks.push({
        name: 'deep_water_formation',
        pass: pMean < tMean,
        value: `Polar deep: ${pMean.toFixed(1)}°C (${polarN} cells), Tropical deep: ${tMean.toFixed(1)}°C (${tropN} cells)`,
        expected: 'Polar deep < Tropical deep',
        reason: 'Thermohaline: cold dense water sinks at poles, fills deep basins',
      });
    }

    // ── MERIDIONAL HEAT TRANSPORT (new — most important AMOC diagnostic) ──
    // Compute v*T at 30°N across all longitudes — should be poleward (positive)
    const jHT = latToJ(30);
    let heatTransport = 0, htN = 0;
    for (let i = 1; i < nx - 1; i++) {
      const k = jHT * nx + i;
      if (!m[k]) continue;
      const ie = (i + 1) % nx, iw = (i - 1 + nx) % nx;
      const v = (p[k + 1] - p[k - 1]) * 0.5 * nx; // meridional velocity from psi
      const T = typeof temp !== 'undefined' ? temp[k] : 0;
      heatTransport += v * T;
      htN++;
    }
    const meanHT = htN > 0 ? heatTransport / htN : 0;
    checks.push({
      name: 'poleward_heat_transport',
      pass: meanHT > 0, // should be northward (poleward) at 30°N
      value: `Mean v*T at 30°N: ${meanHT.toFixed(4)} (${htN} cells)`,
      expected: 'Positive (poleward heat transport)',
      reason: 'Ocean must transport heat from tropics to poles — fundamental energy balance',
    });

    // ── BASIN ASYMMETRY: Atlantic warmer than Pacific at same latitude ──
    // Due to Gulf Stream / AMOC, Atlantic should be warmer than Pacific at ~40°N
    if (typeof temp !== 'undefined') {
      const j40N = latToJ(40);
      const iAtlLo = lonToI(-70), iAtlHi = lonToI(-10);
      const iPacLo = lonToI(140), iPacHi = lonToI(220 - 360); // Pacific: 140°E to 140°W
      let atlSum = 0, atlN = 0, pacSum = 0, pacN = 0;
      for (let i = iAtlLo; i <= iAtlHi; i++) {
        const k = j40N * nx + i;
        if (m[k]) { atlSum += temp[k]; atlN++; }
      }
      // Pacific wraps: 140°E (i≈320) to 220°E=140°W (i≈40, wrapping)
      for (let i = lonToI(140); i < nx; i++) {
        const k = j40N * nx + i;
        if (m[k]) { pacSum += temp[k]; pacN++; }
      }
      for (let i = 0; i <= lonToI(-140); i++) {
        const k = j40N * nx + i;
        if (m[k]) { pacSum += temp[k]; pacN++; }
      }
      const atlMean = atlN > 0 ? atlSum / atlN : 0;
      const pacMean = pacN > 0 ? pacSum / pacN : 0;
      checks.push({
        name: 'atlantic_warmer_than_pacific',
        pass: atlMean > pacMean,
        value: `40°N Atlantic: ${atlMean.toFixed(1)}°C (${atlN}), Pacific: ${pacMean.toFixed(1)}°C (${pacN})`,
        expected: 'Atlantic > Pacific (AMOC warms Atlantic)',
        reason: 'Gulf Stream and AMOC transport extra heat into North Atlantic',
      });
    }

    return checks;
  });
}

// ---------------------------------------------------------------------------
// TIER 3: Sensitivity (perturbation response)
// Run only on final best params to validate causal structure
// ---------------------------------------------------------------------------
async function checkSensitivity(page, baseParams, baseAMOC) {
  const checks = [];

  // Test: adding freshwater to North Atlantic should weaken AMOC
  console.log('  Sensitivity test: freshwater perturbation...');
  await page.click('#btn-reset');
  await page.waitForTimeout(500);
  await page.evaluate((p) => {
    for (const [k, v] of Object.entries(p)) {
      try { eval(k + ' = ' + JSON.stringify(v)); } catch(e) {}
    }
    try { eval('freshwaterForcing = 2.0'); } catch(e) {}
  }, baseParams);
  await page.evaluate(() => {
    document.getElementById('speed-slider').value = 200;
    document.getElementById('speed-slider').dispatchEvent(new Event('input'));
  });
  await page.waitForTimeout(60000);
  const fwData = await page.evaluate(() => ({
    amoc: typeof amocStrength !== 'undefined' ? amocStrength : 0,
  }));
  checks.push({
    name: 'freshwater_weakens_amoc',
    pass: fwData.amoc < baseAMOC,
    value: `Base AMOC: ${baseAMOC.toExponential(2)} → With freshwater: ${fwData.amoc.toExponential(2)} (non-dim)`,
    expected: 'AMOC decreases with freshwater forcing',
    reason: 'Stommel bifurcation: freshwater caps surface, prevents sinking, weakens overturning',
  });

  // Test: reducing solar (ice age) should cool everything
  console.log('  Sensitivity test: ice age perturbation...');
  await page.click('#btn-reset');
  await page.waitForTimeout(500);
  await page.evaluate((p) => {
    for (const [k, v] of Object.entries(p)) {
      try { eval(k + ' = ' + JSON.stringify(v)); } catch(e) {}
    }
    try { eval('freshwaterForcing = 0'); } catch(e) {}
    try { eval('globalTempOffset = -8.0'); } catch(e) {}
  }, baseParams);
  await page.evaluate(() => {
    document.getElementById('speed-slider').value = 200;
    document.getElementById('speed-slider').dispatchEvent(new Event('input'));
  });
  await page.waitForTimeout(60000);
  const iceData = await page.evaluate(() => {
    const nx = typeof NX !== 'undefined' ? NX : 360;
    const ny = typeof NY !== 'undefined' ? NY : 180;
    const m = mask, t = temp;
    let sum = 0, count = 0;
    for (let k = 0; k < nx * ny; k++) {
      if (m[k]) { sum += t[k]; count++; }
    }
    return { globalMean: sum / count };
  });
  checks.push({
    name: 'cooling_reduces_temperature',
    pass: iceData.globalMean < 15, // should be substantially cooler
    value: `Ice age global mean: ${iceData.globalMean.toFixed(1)}°C`,
    expected: '< 15°C (substantially cooler than present ~18°C)',
    reason: 'Negative temperature offset must produce global cooling — basic radiative balance',
  });

  // Reset to base state
  await page.click('#btn-reset');
  await page.waitForTimeout(500);
  await page.evaluate((p) => {
    for (const [k, v] of Object.entries(p)) {
      try { eval(k + ' = ' + JSON.stringify(v)); } catch(e) {}
    }
    try { eval('freshwaterForcing = 0'); } catch(e) {}
    try { eval('globalTempOffset = 0'); } catch(e) {}
  }, baseParams);

  return checks;
}

// ---------------------------------------------------------------------------
// TIER 4: Quantitative match
// ---------------------------------------------------------------------------
function checkQuantitative(simData, refData) {
  const errors = [];
  let totalSE = 0, count = 0;
  const latBins = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70];

  for (const lat of latBins) {
    const sim = simData.zonalTemp[lat];
    const obs = refData.refSST[lat];
    if (sim != null && obs != null) {
      const err = sim - obs;
      totalSE += err * err;
      count++;
      errors.push({ lat, sim: +sim.toFixed(1), obs: +obs.toFixed(1), error: +err.toFixed(1) });
    }
  }

  const rmse = Math.sqrt(totalSE / Math.max(count, 1));
  const globalBias = simData.globalMean - refData.globalMeanSST;
  const amocVal = simData.amoc || 0;

  return { errors, rmse, globalBias, amocVal };
}

// ---------------------------------------------------------------------------
// Composite score — weighted across tiers
// ---------------------------------------------------------------------------
function computeCompositeScore(t1Checks, t2Checks, t4Report) {
  // T1 (conservation): binary — all must pass or it's invalid
  const t1Pass = t1Checks.every(c => c.pass);
  const t1Score = t1Pass ? 1.0 : 0.0;

  // T2 (structure): fraction passing
  const t2Score = t2Checks.length > 0
    ? t2Checks.filter(c => c.pass).length / t2Checks.length
    : 0;

  // T4 (quantitative): RMSE-based, normalized so 0°C RMSE = 1.0, 15°C RMSE = 0.0
  const t4Score = Math.max(0, 1.0 - t4Report.rmse / 15.0);

  // AMOC bonus: in-range [5, 25] Sv
  // AMOC in non-dimensional units: healthy range ~0.001 to 0.1
  const amocInRange = t4Report.amocVal >= 0.0005 && t4Report.amocVal <= 0.5;
  const amocScore = amocInRange ? 1.0 : 0.5;

  // T1 as partial credit (fraction passing) rather than binary gate
  // This allows the score to reflect T2/T4 improvements even when some T1 checks fail
  const t1Frac = t1Checks.length > 0 ? t1Checks.filter(c => c.pass).length / t1Checks.length : 0;

  // Weighted composite: T1 conservation weighted by fraction passing
  const raw = 0.30 * t2Score + 0.30 * t4Score + 0.15 * amocScore + 0.25 * t1Frac;

  return {
    composite: raw,
    t1: { score: t1Frac, pass: t1Pass, checks: t1Checks },
    t2: { score: t2Score, checks: t2Checks },
    t4: { score: t4Score, rmse: t4Report.rmse },
    amoc: { score: amocScore, value: t4Report.amocVal },
  };
}

// ---------------------------------------------------------------------------
// Agent prompts
// ---------------------------------------------------------------------------
function physicistPrompt(params, scorecard, refData, history) {
  return `You are a PHYSICAL OCEANOGRAPHER analyzing why a 2-layer barotropic ocean simulation diverges from reality. Your job is to generate competing HYPOTHESES — not to tune parameters.

## Model Physics
- Barotropic vorticity equation with wind stress curl, beta effect, friction, viscosity
- Two-layer thermodynamics: solar forcing, linearized OLR, ice-albedo feedback
- Thermohaline coupling: buoyancy term drives overturning from temperature gradients
- Deep water formation enhanced at cold high latitudes

## Current Parameters
${JSON.stringify(params, null, 2)}

## Evaluation Scorecard
Conservation (T1): ${scorecard.t1.pass ? 'PASS' : 'FAIL'}
${scorecard.t1.checks.map(c => `  ${c.pass ? '✓' : '✗'} ${c.name}: ${c.value} (expected: ${c.expected})`).join('\n')}

Structure (T2): ${(scorecard.t2.score * 100).toFixed(0)}%
${scorecard.t2.checks.map(c => `  ${c.pass ? '✓' : '✗'} ${c.name}: ${c.value}`).join('\n')}

Quantitative (T4): RMSE = ${scorecard.t4.rmse.toFixed(2)}°C
AMOC: ${scorecard.amoc.value.toExponential(2)} (non-dim)

Composite score: ${(scorecard.composite * 100).toFixed(1)}%

## Zonal Temperature Errors
${scorecard.t4.checks ? '' : ''}Lat   Sim    Obs    Error
${Object.entries(refData.refSST).map(([lat, obs]) => {
  const sim = scorecard.simData?.zonalTemp?.[lat];
  if (sim == null) return null;
  const err = sim - obs;
  return `${String(lat).padStart(4)}° ${sim.toFixed(1).padStart(6)} ${obs.toFixed(1).padStart(6)} ${(err > 0 ? '+' : '') + err.toFixed(1).padStart(6)}`;
}).filter(Boolean).join('\n')}

## History
${history.length === 0 ? 'First iteration.' : history.slice(-5).map(h => `Iter ${h.iter}: score=${(h.composite * 100).toFixed(0)}% RMSE=${h.rmse.toFixed(1)}°C — hypothesis: ${h.hypothesis}`).join('\n')}

## Visual Evidence
You have been given 4 screenshots of the current simulation state:
1. **Temperature map** — SST field. Look for: realistic tropical warm pool, cold polar water, smooth gradients (no checkerboard artifacts), reasonable land-ocean contrast
2. **Streamfunction (ψ)** — circulation pattern. Look for: anticyclonic subtropical gyres (red in NH), western intensification (tighter contours on west side), ACC in Southern Ocean
3. **Speed field** — current velocity magnitude. Look for: bright western boundary currents (Gulf Stream off east NA, Kuroshio off Japan, ACC circumpolar), weak interior flow
4. **Deep temperature** — 1000m layer. Look for: cold water filling deep basins from polar source regions, warmer deep tropics

USE THESE IMAGES to identify spatial problems that zonal means would miss: boundary artifacts, missing or misplaced currents, unrealistic patterns, numerical instabilities.

## Your Task
Generate 2-3 competing hypotheses explaining the DOMINANT error pattern. For each hypothesis:
1. State the physical mechanism
2. Predict which parameters would need to change AND what side-effects that change would have
3. Rate your confidence (low/medium/high)

Focus on the LOWEST-SCORING TIER first:
- If T1 (conservation) fails → something is fundamentally broken
- If T2 (structure) is low → the dynamics are wrong, not just the temperatures
- If T4 (quantitative) is low but T1/T2 pass → parameter tuning can help

Return JSON:
{
  "dominant_error": "One-line summary of the biggest problem",
  "hypotheses": [
    {
      "id": "H1",
      "mechanism": "Physical explanation",
      "predicted_fix": { "param": "new_value" },
      "side_effects": "What else this would change",
      "confidence": "low|medium|high"
    }
  ],
  "recommended": "H1|H2|H3 — which hypothesis to test first and why"
}`;
}

function tunerPrompt(params, physicistOutput, scorecard, refData) {
  return `You are a PARAMETER TUNER for a 2-layer ocean model. A physical oceanographer has analyzed the errors and generated hypotheses. Your job is to translate the winning hypothesis into specific, bounded parameter changes.

## Physicist's Analysis
${JSON.stringify(physicistOutput, null, 2)}

## Current Parameters
${JSON.stringify(params, null, 2)}

## Parameter Ranges (HARD BOUNDS — do not exceed)
${Object.entries(TUNABLE_PARAMS).map(([k, v]) => `- ${k}: [${v.min}, ${v.max}] — ${v.desc}`).join('\n')}

## Rules
1. Only change parameters that the recommended hypothesis calls for
2. Maximum 3 parameter changes per iteration
3. Make CONSERVATIVE changes — move each parameter at most 30% toward its bound
4. If the physicist says structure (T2) is the problem, prioritize dynamical parameters (friction, viscosity, wind, coupling) over thermal parameters
5. If conservation (T1) is broken, prioritize the specific failing check

## Physical Consistency Checks
Before proposing a change, verify:
- S_solar / B_olr sets the equilibrium temperature range. Equilibrium T ≈ (S × avg_cosZ − A_olr) / B_olr
- Munk boundary layer width ∝ (A_visc / beta)^(1/3) — if you change A_visc, boundary currents widen
- Stommel boundary layer width ∝ r_friction / beta — if you reduce friction, WBC gets narrower
- H_surface / H_deep ratio affects coupling: F_couple_d should ≈ (H_surface/H_deep) × F_couple_s

Return JSON:
{
  "hypothesis_tested": "H1|H2|H3",
  "reasoning": "Why these specific values",
  "params": { "param_name": new_value },
  "predicted_effect": "What should improve and by roughly how much"
}`;
}

function validatorPrompt(params, tunerOutput, physicistOutput, scorecard) {
  return `You are a PHYSICS VALIDATOR. A tuner has proposed parameter changes for a 2-layer ocean model. Your job is to check whether the proposal is physically consistent — NOT whether it will reduce RMSE.

## Proposed Changes
${JSON.stringify(tunerOutput, null, 2)}

## Current Parameters (before change)
${JSON.stringify(params, null, 2)}

## Physicist's Reasoning
${JSON.stringify(physicistOutput, null, 2)}

## Current Scorecard
Conservation: ${scorecard.t1.pass ? 'PASS' : 'FAIL'}
Structure: ${(scorecard.t2.score * 100).toFixed(0)}%
Quantitative: RMSE ${scorecard.t4.rmse.toFixed(2)}°C

## Validation Checks — apply each:

1. EQUILIBRIUM SANITY: With proposed S_solar, A_olr, B_olr:
   - Equatorial T ≈ (S × 0.9 − A) / B  — should be 25-30°C
   - Polar T ≈ (S × 0.2 − A) / B  — should be -5 to 5°C
   - If either is out of range → REJECT

2. BOUNDARY LAYER CONSISTENCY: If friction or viscosity changed:
   - Munk width ∝ (A_visc)^(1/3) — should be ~2-5 grid cells
   - Stommel width ∝ r_friction — should be ~1-3 grid cells
   - If either produces sub-grid or domain-scale features → REJECT

3. COUPLING CONSISTENCY:
   - F_couple_d should ≈ (H_surface / H_deep) × F_couple_s = (100/4000) × F_couple_s
   - If they're proposing to change one without the other → FLAG

4. NO COMPENSATING ERRORS:
   - If they're simultaneously raising S_solar AND A_olr to keep the same equilibrium → REJECT
   - The point is to change the physics, not play whack-a-mole

5. CAUSAL COHERENCE:
   - Does the proposed change address the mechanism in the hypothesis?
   - Or is it just curve-fitting dressed up as physics?

Return JSON:
{
  "verdict": "APPROVE|MODIFY|REJECT",
  "issues": ["list of specific physical problems, if any"],
  "modifications": { "param": "corrected_value" },
  "explanation": "Why this verdict"
}`;
}

// ---------------------------------------------------------------------------
// ON-DEMAND AGENTS — called only when specific conditions trigger them
// ---------------------------------------------------------------------------

// NUMERICAL ANALYST: Triggered when T1 fails or screenshots show artifacts
function numericsPrompt(params, scorecard) {
  return `You are a COMPUTATIONAL FLUID DYNAMICS expert reviewing a WebGPU ocean simulation for NUMERICAL issues (not physics).

## Simulation Setup
- Grid: 360×180 (1° resolution), two layers
- Timestep: dt = ${params.dt || 5e-5}
- Poisson solver: 60 Jacobi iterations (surface), 20 (deep)
- Advection: Arakawa Jacobian (conserves energy + enstrophy)
- Diffusion: explicit forward Euler

## Current Parameters
${JSON.stringify(params, null, 2)}

## Scorecard
Conservation: ${scorecard.t1.pass ? 'PASS' : 'FAIL'}
${scorecard.t1.checks.map(c => `  ${c.pass ? '✓' : '✗'} ${c.name}: ${c.value}`).join('\n')}

## Screenshots are attached. Look for:
1. **Checkerboard patterns** — sign of odd-even decoupling, needs more viscosity or different stencil
2. **Ringing near coastlines** — Gibbs-like artifacts from sharp land/ocean transitions
3. **Blow-up signatures** — extreme values at specific grid points
4. **Poisson solver under-convergence** — streamfunction not smooth, noisy velocity field
5. **CFL violation** — max velocity × dt / dx > 1 means timestep is too large
   - dx ≈ 111km × cos(lat) at equator ≈ 111km. In non-dim units dx = 1/360
   - CFL = max_vel × dt × 360. With dt=${params.dt || 5e-5}, CFL safe if max_vel < ${(1/((params.dt||5e-5)*360)).toFixed(0)}

Return JSON:
{
  "numerical_health": "HEALTHY|WARNING|CRITICAL",
  "issues": [
    { "type": "checkerboard|ringing|blowup|cfl|poisson", "location": "where", "severity": "low|medium|high", "fix": "what to do" }
  ],
  "recommended_param_changes": { "param": value },
  "recommended_code_changes": "description of any code-level fixes needed, or null"
}`;
}

// SKEPTIC: Triggered every N iterations to audit the trajectory
function skepticPrompt(params, history, scorecard, refData) {
  return `You are a RED TEAM SKEPTIC auditing an AI parameter tuning loop for a climate simulation. Your job is to find ways the tuning might be CHEATING — producing right numbers for wrong reasons.

## Tuning History
${history.map(h => `Iter ${h.iter}: score=${(h.composite * 100).toFixed(0)}% RMSE=${h.rmse.toFixed(1)}°C — ${h.hypothesis} — changes: ${h.changes}`).join('\n')}

## Current Parameters
${JSON.stringify(params, null, 2)}

## Default Parameters (starting point)
${JSON.stringify(Object.fromEntries(Object.entries(TUNABLE_PARAMS).map(([k, v]) => [k, v.default])), null, 2)}

## Current Scorecard
Conservation: ${scorecard.t1.pass ? 'PASS' : 'FAIL'}
Structure: ${(scorecard.t2.score * 100).toFixed(0)}%
RMSE: ${scorecard.t4.rmse.toFixed(2)}°C

## Your Audit Checks

1. **PARAMETER DRIFT** — How far have params moved from defaults? Large drift in multiple params simultaneously suggests overfitting, not physics.

2. **COMPENSATING ERRORS** — Are S_solar and A_olr both elevated (or both reduced)? That's curve-fitting: changing both to keep equilibrium while losing physical meaning.

3. **DIMENSIONAL ANALYSIS** — Do the current parameter values make physical sense?
   - r_friction = ${params.r_friction}: damping timescale ≈ 1/r = ${(1/params.r_friction).toFixed(0)} timesteps. Reasonable?
   - A_visc = ${params.A_visc}: Munk width ��� (A/β)^(1/3) ≈ ${Math.pow(params.A_visc, 1/3).toFixed(3)} ≈ ${(Math.pow(params.A_visc, 1/3) * 360).toFixed(1)} grid cells
   - kappa_diff = ${params.kappa_diff}: diffusive scale ≈ √(κ × T_year) ≈ ${Math.sqrt(params.kappa_diff * 10).toFixed(4)}

4. **TRAJECTORY COHERENCE** — Is the tuning following a consistent physical narrative, or zigzagging between contradictory hypotheses?

5. **SCORE GAMING** — Could the composite score be gamed? E.g., passing T1 trivially while T2/T4 aren't improving.

Return JSON:
{
  "verdict": "CLEAN|SUSPICIOUS|COMPROMISED",
  "red_flags": ["specific concerns"],
  "parameter_drift_score": 0.0-1.0,
  "recommendation": "what to do — rollback? reset specific params? change evaluation?"
}`;
}

// LITERATURE AGENT: Triggered when params hit bounds or physicist is low-confidence
function literaturePrompt(paramName, currentValue, bounds) {
  return `You are a climate modeling literature expert. A parameter tuning loop needs a reality check on a specific parameter value.

Parameter: ${paramName}
Current value: ${currentValue}
Allowed bounds: [${bounds.min}, ${bounds.max}]
Description: ${bounds.desc}

This is for a SIMPLIFIED 2-layer barotropic ocean model at 1° resolution (educational/research tool, not a full GCM).

What values do published ocean models use for equivalent parameters?
- Consider: MOM6, NEMO, POP, MITgcm, and idealized models from textbooks (Vallis, Pedlosky, Cushman-Roisin)
- Account for the fact that this is a SIMPLIFIED model — effective parameters may differ from resolved GCMs

Return JSON:
{
  "parameter": "${paramName}",
  "published_ranges": [
    { "source": "model/paper name", "value_or_range": "X-Y", "context": "resolution, model type" }
  ],
  "recommended_bounds": { "min": X, "max": Y },
  "physical_reasoning": "Why these values make sense for a 1° 2-layer model",
  "current_value_assessment": "reasonable|high|low|extreme"
}`;
}

// OBSERVATIONAL SCIENTIST: Triggered when tuning targets specific latitude bands
function obsScientistPrompt(refData, problematicLats) {
  return `You are an observational oceanographer assessing the RELIABILITY of reference data being used to tune a climate simulation.

## Reference Dataset
- SST: NOAA OI SST v2 Long-Term Mean (1991-2020), 1�� resolution
- Deep: WOA23 annual mean at 1000m depth

## Problematic Latitude Bands
The tuning loop is struggling most at these latitudes:
${problematicLats.map(l => `  ${l.lat}°: sim=${l.sim}°C, obs=${l.obs}°C, error=${l.error}°C`).join('\n')}

## Your Assessment
For each problematic latitude, evaluate:
1. **Data quality** — How reliable is the NOAA/WOA data at this latitude? Sparse sampling? Sea ice contamination? Interpolation artifacts?
2. **Representativeness** — Is the annual mean meaningful here? (e.g., seasonal ice zones have bimodal SST)
3. **Model-obs comparison fairness** — Our sim is a 2-layer barotropic model with idealized forcing. At this latitude, what biases are EXPECTED from the model structure (not tunable)?
4. **Recommended uncertainty** — What error bar should we put on the reference value?

Return JSON:
{
  "assessments": [
    {
      "lat": N,
      "data_quality": "high|medium|low",
      "known_issues": "description",
      "expected_model_bias": "description of structural model limitations at this latitude",
      "recommended_uncertainty_C": N,
      "should_downweight": true/false
    }
  ],
  "overall_recommendation": "Should we adjust our error weighting by latitude?"
}`;
}

// ---------------------------------------------------------------------------
// On-demand agent dispatcher
// ---------------------------------------------------------------------------
async function callOnDemandAgent(agentName, prompt, images = []) {
  console.log(`\n  [${agentName.toUpperCase()}] (on-demand specialist)...`);
  try {
    const result = await callClaude(prompt, { images });
    return result;
  } catch (err) {
    console.log(`  ${agentName} error: ${err.message}`);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Run simulation and extract full diagnostics
// ---------------------------------------------------------------------------
async function runSimulation(page, params, iterNum) {
  // Reset first, THEN inject params (reset overwrites params to defaults)
  await page.click('#btn-reset');
  await page.waitForTimeout(500);

  // Inject params using eval() because top-level `let` vars aren't on `window`
  await page.evaluate((p) => {
    for (const [key, val] of Object.entries(p)) {
      try { eval(key + ' = ' + JSON.stringify(val)); } catch(e) {}
    }
    try { eval('freshwaterForcing = 0'); } catch(e) {}
    try { eval('globalTempOffset = 0'); } catch(e) {}
  }, params);
  await page.waitForTimeout(500);
  await page.evaluate(() => {
    document.getElementById('speed-slider').value = 200;
    document.getElementById('speed-slider').dispatchEvent(new Event('input'));
    document.getElementById('year-speed-slider').value = 3;
    document.getElementById('year-speed-slider').dispatchEvent(new Event('input'));
  });
  await page.click('#btn-temp');

  console.log(`  Spinning up for ${SPINUP_SECS}s...`);
  await page.waitForTimeout(SPINUP_SECS * 1000);

  const data = await page.evaluate(() => {
    const nx = typeof NX !== 'undefined' ? NX : 360;
    const ny = typeof NY !== 'undefined' ? NY : 180;
    const m = typeof mask !== 'undefined' ? mask : null;
    const t = typeof temp !== 'undefined' ? temp : null;
    const dt = typeof deepTemp !== 'undefined' ? deepTemp : null;
    if (!t || !m) return null;

    const zonalTemp = {}, zonalDeep = {};
    const latBins = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70];
    for (const targetLat of latBins) {
      const j = Math.round((targetLat - (-80)) / 160 * (ny - 1));
      if (j < 0 || j >= ny) continue;
      let sum = 0, dsum = 0, count = 0;
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (m[k]) { sum += t[k]; if (dt) dsum += dt[k]; count++; }
      }
      if (count > 0) {
        zonalTemp[targetLat] = sum / count;
        zonalDeep[targetLat] = dt ? dsum / count : null;
      }
    }

    let gSum = 0, gCount = 0, gMin = 999, gMax = -999;
    for (let k = 0; k < nx * ny; k++) {
      if (m[k]) { gSum += t[k]; gCount++; gMin = Math.min(gMin, t[k]); gMax = Math.max(gMax, t[k]); }
    }

    return {
      zonalTemp, zonalDeep,
      globalMean: gSum / gCount, globalMin: gMin, globalMax: gMax,
      step: document.getElementById('stat-step')?.textContent,
      amoc: typeof amocStrength !== 'undefined' ? amocStrength : 0,
    };
  });

  // Capture screenshots as both files and buffers for multimodal agent input
  const screenshots = {};
  if (data) {
    const prefix = `${OUT}/iter-${String(iterNum).padStart(2, '0')}`;

    // Temperature view (already active)
    const tempBuf = await page.screenshot({ path: `${prefix}-temp.png` });
    screenshots.temp = { buffer: tempBuf, label: 'Sea Surface Temperature — blue (cold) to red (warm)' };

    // Streamfunction — shows gyre structure and western boundary currents
    await page.click('#btn-psi');
    await page.waitForTimeout(300);
    const psiBuf = await page.screenshot({ path: `${prefix}-psi.png` });
    screenshots.psi = { buffer: psiBuf, label: 'Streamfunction (ψ) — red=anticyclonic gyres, blue=cyclonic. Shows wind-driven circulation structure' };

    // Speed — highlights western boundary currents
    await page.click('#btn-speed');
    await page.waitForTimeout(300);
    const spdBuf = await page.screenshot({ path: `${prefix}-speed.png` });
    screenshots.speed = { buffer: spdBuf, label: 'Current Speed — bright=fast. Western boundary currents (Gulf Stream, Kuroshio, ACC) should be brightest' };

    // Deep temperature — thermohaline signature
    await page.click('#btn-deeptemp');
    await page.waitForTimeout(300);
    const deepBuf = await page.screenshot({ path: `${prefix}-deeptemp.png` });
    screenshots.deep = { buffer: deepBuf, label: 'Deep Ocean Temperature (1000m) — cold polar water should fill deep basins' };

    await page.click('#btn-temp');
  }

  return { data, screenshots };
}

// ---------------------------------------------------------------------------
// Print scorecard
// ---------------------------------------------------------------------------
function printScorecard(scorecard) {
  console.log('\n  ┌─────────────────────────────────────────────┐');
  console.log(`  │  COMPOSITE SCORE: ${(scorecard.composite * 100).toFixed(1).padStart(5)}%                      │`);
  console.log('  ├─────────────────────────────────────────────┤');
  console.log(`  │  T1 Conservation: ${scorecard.t1.pass ? 'PASS ✓' : 'FAIL ✗'}                       │`);
  for (const c of scorecard.t1.checks) {
    console.log(`  │    ${c.pass ? '✓' : '✗'} ${c.name.padEnd(30)} ${c.pass ? '' : '← FIX'}│`);
  }
  console.log(`  │  T2 Structure:    ${(scorecard.t2.score * 100).toFixed(0).padStart(3)}%                        │`);
  for (const c of scorecard.t2.checks) {
    console.log(`  │    ${c.pass ? '✓' : '✗'} ${c.name.padEnd(30)} ${c.pass ? '' : '← FIX'}│`);
  }
  console.log(`  │  T4 Quantitative: RMSE ${scorecard.t4.rmse.toFixed(1)}°C                │`);
  console.log(`  │  AMOC:            ${scorecard.amoc.value.toFixed(1)} Sv                      │`);
  console.log('  └─────────────────────────────────────────────┘');
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
async function main() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║  RALPH WIGGUM LOOP: Physics-Aligned Parameter Tuning   ║');
  console.log('║  3-Agent Dialectic × 4-Tier Evaluation                 ║');
  console.log('╚══════════════════════════════════════════════════════════╝');
  console.log(`Model: ${CLAUDE_MODEL}`);
  console.log(`Max iterations: ${MAX_ITERS} | Spinup: ${SPINUP_SECS}s\n`);

  const refData = loadReferenceData();
  console.log(`Reference: global mean SST ${refData.globalMeanSST.toFixed(1)}°C`);
  console.log(`           range ${Math.min(...Object.values(refData.refSST)).toFixed(1)}°C to ${Math.max(...Object.values(refData.refSST)).toFixed(1)}°C\n`);

  // Local server
  const server = createServer((req, res) => {
    const urlPath = new URL(req.url, 'http://localhost').pathname;
    const filePath = resolve(ROOT, urlPath.replace(/^\//, ''));
    try {
      const d = readFileSync(filePath);
      const ext = filePath.split('.').pop();
      const mime = { html: 'text/html', json: 'application/json', js: 'text/javascript' }[ext] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': mime }); res.end(d);
    } catch { res.writeHead(404); res.end('Not found'); }
  });
  await new Promise(r => server.listen(8772, r));

  // Browser
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan'],
  });
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.goto('http://localhost:8772/simamoc/index.html', { waitUntil: 'load', timeout: 30000 });
  await page.waitForTimeout(5000);
  try { await page.click('#btn-start-exploring', { timeout: 5000 }); } catch {}
  await page.waitForTimeout(1000);
  const backend = await page.evaluate(() => document.getElementById('backend-badge')?.textContent);
  console.log(`Backend: ${backend}\n`);

  // Init params
  let currentParams = {};
  for (const [k, v] of Object.entries(TUNABLE_PARAMS)) currentParams[k] = v.default;

  const history = [];
  let bestScore = 0;
  let bestParams = { ...currentParams };

  for (let iter = 1; iter <= MAX_ITERS; iter++) {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`  ITERATION ${iter}/${MAX_ITERS}`);
    console.log(`${'═'.repeat(60)}`);

    // --- RUN SIMULATION ---
    const simResult = await runSimulation(page, currentParams, iter);
    if (!simResult?.data) {
      console.log('  ERROR: no simulation data. Reverting to best params.');
      currentParams = { ...bestParams };
      continue;
    }
    const simData = simResult.data;
    const screenshots = simResult.screenshots;

    // --- EVALUATE ALL TIERS ---
    const t1Checks = checkConservation(simData);
    const t2Checks = await checkStructure(page);
    const t4Report = checkQuantitative(simData, refData);

    const scorecard = computeCompositeScore(t1Checks, t2Checks, t4Report);
    scorecard.simData = simData; // attach for prompt building
    printScorecard(scorecard);

    // Print zonal errors
    console.log(`\n  Lat   Sim    Obs   Error`);
    for (const e of t4Report.errors) {
      console.log(`  ${String(e.lat).padStart(4)}° ${e.sim.toFixed(1).padStart(6)} ${e.obs.toFixed(1).padStart(6)} ${(e.error > 0 ? '+' : '') + e.error.toFixed(1).padStart(5)}`);
    }

    // Track best
    if (scorecard.composite > bestScore) {
      bestScore = scorecard.composite;
      bestParams = { ...currentParams };
      console.log(`\n  ★ New best composite score: ${(bestScore * 100).toFixed(1)}%`);
    }

    // --- AGENT 1: PHYSICIST (with visual context) ---
    console.log('\n  [PHYSICIST] Generating hypotheses (with screenshots)...');
    let physicistOutput;
    // Send all 4 views so the physicist can see spatial patterns, not just zonal means
    const physicistImages = Object.values(screenshots || {});
    try {
      physicistOutput = await callClaude(
        physicistPrompt(currentParams, scorecard, refData, history),
        { images: physicistImages }
      );
      console.log(`  Dominant error: ${physicistOutput.dominant_error}`);
      for (const h of physicistOutput.hypotheses || []) {
        console.log(`    ${h.id}: ${h.mechanism} [${h.confidence}]`);
      }
      console.log(`  Recommended: ${physicistOutput.recommended}`);
    } catch (err) {
      console.log(`  Physicist error: ${err.message}`);
      history.push({ iter, composite: scorecard.composite, rmse: t4Report.rmse, hypothesis: 'ERROR' });
      continue;
    }

    // --- AGENT 2: TUNER ---
    console.log('\n  [TUNER] Proposing parameter changes...');
    let tunerOutput;
    try {
      tunerOutput = await callClaude(tunerPrompt(currentParams, physicistOutput, scorecard, refData));
      console.log(`  Testing: ${tunerOutput.hypothesis_tested}`);
      console.log(`  Reasoning: ${tunerOutput.reasoning}`);
      console.log(`  Proposed: ${JSON.stringify(tunerOutput.params)}`);
    } catch (err) {
      console.log(`  Tuner error: ${err.message}`);
      history.push({ iter, composite: scorecard.composite, rmse: t4Report.rmse, hypothesis: physicistOutput.dominant_error });
      continue;
    }

    // --- AGENT 3: VALIDATOR ---
    console.log('\n  [VALIDATOR] Checking physical consistency...');
    let validatorOutput;
    try {
      validatorOutput = await callClaude(validatorPrompt(currentParams, tunerOutput, physicistOutput, scorecard));
      console.log(`  Verdict: ${validatorOutput.verdict}`);
      if (validatorOutput.issues?.length) {
        for (const issue of validatorOutput.issues) console.log(`    ⚠ ${issue}`);
      }
      console.log(`  Explanation: ${validatorOutput.explanation}`);
    } catch (err) {
      console.log(`  Validator error: ${err.message} — proceeding with tuner proposal`);
      validatorOutput = { verdict: 'APPROVE', modifications: {} };
    }

    // --- ON-DEMAND SPECIALISTS (triggered by conditions) ---

    // NUMERICAL ANALYST: if T1 conservation fails, check for numerical artifacts
    if (!scorecard.t1.pass) {
      const numericsResult = await callOnDemandAgent(
        'Numerical Analyst',
        numericsPrompt(currentParams, scorecard),
        Object.values(screenshots || {})
      );
      if (numericsResult) {
        console.log(`  Numerical health: ${numericsResult.numerical_health}`);
        for (const issue of numericsResult.issues || []) {
          console.log(`    ${issue.severity.toUpperCase()}: ${issue.type} at ${issue.location} — ${issue.fix}`);
        }
        // If critical numerical issues, apply fixes before physics tuning
        if (numericsResult.numerical_health === 'CRITICAL' && numericsResult.recommended_param_changes) {
          console.log('  Applying numerical fixes before physics tuning...');
          for (const [k, v] of Object.entries(numericsResult.recommended_param_changes)) {
            if (TUNABLE_PARAMS[k]) currentParams[k] = Math.max(TUNABLE_PARAMS[k].min, Math.min(TUNABLE_PARAMS[k].max, v));
          }
          history.push({ iter, composite: scorecard.composite, rmse: t4Report.rmse, hypothesis: 'NUMERICS FIX: ' + (numericsResult.issues?.[0]?.type || 'unknown') });
          continue; // re-run with numerical fix before trying physics tuning
        }
      }
    }

    // SKEPTIC: every 4th iteration, audit the trajectory for curve-fitting
    if (iter > 1 && iter % 4 === 0 && history.length >= 2) {
      const skepticResult = await callOnDemandAgent(
        'Skeptic',
        skepticPrompt(currentParams, history, scorecard, refData)
      );
      if (skepticResult) {
        console.log(`  Audit verdict: ${skepticResult.verdict} (drift: ${skepticResult.parameter_drift_score?.toFixed(2)})`);
        for (const flag of skepticResult.red_flags || []) {
          console.log(`    🚩 ${flag}`);
        }
        if (skepticResult.verdict === 'COMPROMISED') {
          console.log('  ✗ Skeptic says tuning is COMPROMISED. Rolling back to defaults.');
          for (const [k, v] of Object.entries(TUNABLE_PARAMS)) currentParams[k] = v.default;
          history.push({ iter, composite: scorecard.composite, rmse: t4Report.rmse, hypothesis: 'SKEPTIC ROLLBACK' });
          continue;
        }
      }
    }

    // OBSERVATIONAL SCIENTIST: if biggest errors are at polar latitudes, check data quality
    const worstErrors = t4Report.errors.sort((a, b) => Math.abs(b.error) - Math.abs(a.error)).slice(0, 3);
    const polarErrorDominant = worstErrors.some(e => Math.abs(e.lat) >= 60);
    if (polarErrorDominant && iter > 1 && !history.some(h => h.hypothesis?.includes('OBS_QUALITY'))) {
      const obsResult = await callOnDemandAgent(
        'Observational Scientist',
        obsScientistPrompt(refData, worstErrors)
      );
      if (obsResult) {
        for (const a of obsResult.assessments || []) {
          console.log(`  ${a.lat}°: data quality=${a.data_quality}, uncertainty=±${a.recommended_uncertainty_C}°C${a.should_downweight ? ' [DOWNWEIGHT]' : ''}`);
          if (a.known_issues) console.log(`    Known issues: ${a.known_issues}`);
        }
        // Store that we've consulted the obs scientist so we don't repeat
        history.push({ iter: iter + 0.1, composite: scorecard.composite, rmse: t4Report.rmse, hypothesis: 'OBS_QUALITY check', changes: obsResult.overall_recommendation || '' });
      }
    }

    // LITERATURE AGENT: if any param has hit its bound, check published values
    const paramsAtBound = Object.entries(currentParams).filter(([k, v]) => {
      const b = TUNABLE_PARAMS[k];
      return b && (v <= b.min * 1.05 || v >= b.max * 0.95);
    });
    if (paramsAtBound.length > 0 && !history.some(h => h.hypothesis?.includes('LIT_CHECK'))) {
      for (const [paramName, val] of paramsAtBound.slice(0, 1)) { // check one at a time
        const litResult = await callOnDemandAgent(
          'Literature',
          literaturePrompt(paramName, val, TUNABLE_PARAMS[paramName])
        );
        if (litResult) {
          console.log(`  ${paramName}: assessment=${litResult.current_value_assessment}`);
          for (const ref of (litResult.published_ranges || []).slice(0, 2)) {
            console.log(`    ${ref.source}: ${ref.value_or_range} (${ref.context})`);
          }
          // If literature says our bounds are wrong, widen them
          if (litResult.recommended_bounds) {
            const oldMin = TUNABLE_PARAMS[paramName].min;
            const oldMax = TUNABLE_PARAMS[paramName].max;
            if (litResult.recommended_bounds.min < oldMin) TUNABLE_PARAMS[paramName].min = litResult.recommended_bounds.min;
            if (litResult.recommended_bounds.max > oldMax) TUNABLE_PARAMS[paramName].max = litResult.recommended_bounds.max;
            if (TUNABLE_PARAMS[paramName].min !== oldMin || TUNABLE_PARAMS[paramName].max !== oldMax) {
              console.log(`  Updated bounds for ${paramName}: [${TUNABLE_PARAMS[paramName].min}, ${TUNABLE_PARAMS[paramName].max}]`);
            }
          }
          history.push({ iter: iter + 0.2, composite: scorecard.composite, rmse: t4Report.rmse, hypothesis: `LIT_CHECK: ${paramName}`, changes: litResult.physical_reasoning || '' });
        }
      }
    }

    // --- APPLY CHANGES ---
    if (validatorOutput.verdict === 'REJECT') {
      console.log('\n  ✗ Validator REJECTED proposal. Keeping current params.');
      history.push({ iter, composite: scorecard.composite, rmse: t4Report.rmse, hypothesis: physicistOutput.dominant_error + ' [REJECTED]' });
      continue;
    }

    // Merge tuner params with validator modifications
    const finalChanges = { ...tunerOutput.params, ...(validatorOutput.modifications || {}) };
    const changes = [];
    for (const [key, val] of Object.entries(finalChanges)) {
      if (TUNABLE_PARAMS[key]) {
        const clamped = Math.max(TUNABLE_PARAMS[key].min, Math.min(TUNABLE_PARAMS[key].max, val));
        if (clamped !== currentParams[key]) {
          changes.push(`${key}: ${currentParams[key]} → ${clamped}`);
          currentParams[key] = clamped;
        }
      }
    }
    console.log(`\n  Applied: ${changes.join(', ') || 'no changes'}`);

    history.push({
      iter,
      composite: scorecard.composite,
      rmse: t4Report.rmse,
      hypothesis: physicistOutput.dominant_error,
      changes: changes.join(', '),
    });

    // --- STALL DETECTION → ESCALATE TO CLAUDE ---
    // If the last 3 iterations show < 5% improvement, parameter tuning alone
    // isn't enough. Escalate to Claude for structural code review.
    if (history.length >= 3) {
      const recent = history.slice(-3);
      const scoreImprovement = recent[recent.length - 1].composite - recent[0].composite;
      const isStalled = Math.abs(scoreImprovement) < 0.05;
      const isRejectionLoop = recent.filter(h => h.hypothesis?.includes('REJECTED')).length >= 2;

      if (isStalled || isRejectionLoop) {
        console.log('\n  ╔════════════════════════���══════════════════════════╗');
        console.log('  ║  STALL DETECTED — Escalating to Claude            ║');
        console.log('  ╚═════���═══════════════���═════════════════════════��═══╝');

        const escalationReport = {
          message: 'Parameter tuning has stalled. The Claude agents have been unable to improve the composite score for 3 iterations.',
          currentScore: scorecard.composite,
          tier1: scorecard.t1,
          tier2: scorecard.t2,
          tier4: { rmse: t4Report.rmse, errors: t4Report.errors },
          recentHistory: recent,
          currentParams,
          physicistLastHypothesis: physicistOutput,
          suggestion: 'Consider whether structural changes to the physics code (index.html) are needed — new forcing terms, different boundary conditions, modified advection scheme — rather than just parameter tuning.',
        };

        const reportPath = `${OUT}/escalation-iter${iter}.json`;
        writeFileSync(reportPath, JSON.stringify(escalationReport, null, 2));

        // Write screenshot paths for easy Claude review
        const screenshotPaths = Object.entries(screenshots || {}).map(
          ([view, s]) => `${OUT}/iter-${String(iter).padStart(2, '0')}-${view}.png`
        );

        console.log(`  Escalation report: ${reportPath}`);
        console.log(`  Screenshots: ${screenshotPaths.join(', ')}`);
        console.log('');
        console.log('  To bring Claude in for code-level physics review, run:');
        console.log(`    claude "Review the AMOC simulation escalation report at ${reportPath}`);
        console.log(`    and the screenshots. The Wiggum loop has stalled at ${(scorecard.composite * 100).toFixed(0)}%.`);
        console.log('    Analyze whether structural changes to simamoc/index.html are needed');
        console.log('    (not just parameter tuning) to improve physical realism."');
        console.log('');

        // Also write a ready-to-use Claude prompt file
        const claudePromptPath = `${OUT}/claude-review-iter${iter}.md`;
        writeFileSync(claudePromptPath, `# AMOC Wiggum Loop — Claude Escalation (Iteration ${iter})

## Situation
The Ralph Wiggum loop has run ${iter} iterations of physics-aligned parameter tuning
using Claude (${CLAUDE_MODEL}), but has stalled at a composite score of ${(scorecard.composite * 100).toFixed(1)}%.

Parameter tuning alone cannot close the gap. We need your help reviewing the
**physics code** in \`simamoc/index.html\` for structural improvements.

## Current Scorecard
- T1 Conservation: ${scorecard.t1.pass ? 'PASS' : 'FAIL'}
${scorecard.t1.checks.map(c => `  - ${c.pass ? 'OK' : 'FAIL'}: ${c.name} — ${c.value}`).join('\n')}
- T2 Structure: ${(scorecard.t2.score * 100).toFixed(0)}%
${scorecard.t2.checks.map(c => `  - ${c.pass ? 'OK' : 'FAIL'}: ${c.name} — ${c.value}`).join('\n')}
- T4 RMSE: ${t4Report.rmse.toFixed(2)}°C
- AMOC: ${t4Report.amocVal.toFixed(1)} Sv

## Physicist's Last Hypothesis
${physicistOutput.dominant_error}
${(physicistOutput.hypotheses || []).map(h => `- ${h.id}: ${h.mechanism} [${h.confidence}]`).join('\n')}

## What To Look For
1. Are there physics terms missing from the equations that would fix the structural (T2) failures?
2. Is the boundary condition handling causing artifacts?
3. Is the wind forcing pattern realistic enough?
4. Could the ice-albedo feedback be improved?
5. Is the deep water formation parameterization too simple?

## Screenshots
Review the iteration ${iter} screenshots in \`${OUT}/\` — temperature, streamfunction, speed, and deep temperature views.

## Parameters (current best)
\`\`\`json
${JSON.stringify(currentParams, null, 2)}
\`\`\`

After reviewing, please make code changes to \`simamoc/index.html\` that would
improve the physics, then we'll re-run the Wiggum loop with the updated code.
`);
        console.log(`  Claude review prompt saved to: ${claudePromptPath}`);
        console.log('  Continuing loop with parameter tuning...\n');
      }
    }
  }

  // --- FINAL EVALUATION WITH SENSITIVITY ---
  console.log(`\n${'═'.repeat(60)}`);
  console.log('  FINAL EVALUATION');
  console.log(`${'═'.repeat(60)}`);

  // Run best params one more time
  const finalResult = await runSimulation(page, bestParams, 99);
  if (finalResult?.data) {
    const finalSim = finalResult.data;
    const t1 = checkConservation(finalSim);
    const t2 = await checkStructure(page);
    const t4 = checkQuantitative(finalSim, refData);
    const finalScore = computeCompositeScore(t1, t2, t4);
    printScorecard(finalScore);

    // Sensitivity tests (T3)
    console.log('\n  Running sensitivity tests (T3)...');
    const baseAMOC = finalSim.amoc || 0;
    const t3 = await checkSensitivity(page, bestParams, baseAMOC);
    console.log('\n  T3 Sensitivity:');
    for (const c of t3) {
      console.log(`    ${c.pass ? '✓' : '✗'} ${c.name}: ${c.value}`);
    }
    const t3Score = t3.filter(c => c.pass).length / Math.max(t3.length, 1);

    // Full composite including T3
    const fullComposite = finalScore.composite * 0.8 + t3Score * 0.2;
    console.log(`\n  FULL COMPOSITE (with T3): ${(fullComposite * 100).toFixed(1)}%`);

    // Save everything
    const results = {
      timestamp: new Date().toISOString(),
      model: CLAUDE_MODEL,
      iterations: history.length,
      finalScore: { composite: fullComposite, t1: t1, t2: t2, t3: t3, t4: t4 },
      bestParams,
      history,
      referenceData: { globalMeanSST: refData.globalMeanSST, zonalSST: refData.refSST },
    };
    writeFileSync(`${OUT}/results.json`, JSON.stringify(results, null, 2));
    console.log(`\n  Results saved to ${OUT}/results.json`);
  }

  console.log(`\n  Best parameters:`);
  console.log(JSON.stringify(bestParams, null, 2));

  await browser.close();
  server.close();
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
