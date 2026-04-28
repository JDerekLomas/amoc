/**
 * SimAMOC v2 — Initialization
 *
 * Load observational data, build mask, initialize state.
 * Data files are in ../data/ (shared with v4-physics).
 */

const DATA_BASE = '../data/';

async function loadJSON(file) {
  try {
    const r = await fetch(DATA_BASE + file);
    return await r.json();
  } catch (e) {
    console.warn(`Failed to load ${file}:`, e.message);
    return null;
  }
}

/**
 * Load mask and resample to target grid.
 * Mask source is 1024x512, hex-encoded, north-first.
 * We resample to NX×NY with south-first convention.
 */
function resampleMask(maskData, grid) {
  const { nx, ny } = grid;
  const srcNX = maskData.nx || 1024;
  const srcNY = maskData.ny || 512;

  // Decode hex
  const bits = [];
  for (let c = 0; c < maskData.hex.length; c++) {
    const v = parseInt(maskData.hex[c], 16);
    bits.push((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1);
  }

  // Resample with south-first output (flip rows)
  const mask = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) {
    const lat = grid.latAt(j);
    // Source is north-first: srcJ = 0 at 79.5°N
    const srcJ = Math.round((79.5 - lat) / 159 * (srcNY - 1));
    for (let i = 0; i < nx; i++) {
      const lon = grid.lonAt(i);
      const srcI = Math.round((lon + 180) / 360 * (srcNX - 1)) % srcNX;
      mask[j * nx + i] = bits[srcJ * srcNX + srcI] || 0;
    }
  }

  // Force polar walls
  for (let i = 0; i < nx; i++) {
    mask[i] = 0;                    // south wall
    mask[(ny - 1) * nx + i] = 0;   // north wall
  }

  return mask;
}

/**
 * Resample a flat data array from source grid to model grid.
 * srcData: flat array, srcNX × srcNY, north-first.
 * Returns: Float32Array, NX × NY, south-first.
 */
function resampleField(srcData, srcNX, srcNY, grid, fallback = 0) {
  const { nx, ny } = grid;
  const out = new Float32Array(nx * ny);

  for (let j = 0; j < ny; j++) {
    const lat = grid.latAt(j);
    const srcJ = Math.round((79.5 - lat) / 159 * (srcNY - 1));
    if (srcJ < 0 || srcJ >= srcNY) continue;

    for (let i = 0; i < nx; i++) {
      const lon = grid.lonAt(i);
      const srcI = Math.round((lon + 180) / 360 * (srcNX - 1)) % srcNX;
      const val = srcData[srcJ * srcNX + srcI];
      out[j * nx + i] = (val !== undefined && val !== null && val > -90) ? val : fallback;
    }
  }
  return out;
}

/**
 * Initialize model state from observational data.
 */
export async function initialize(state, grid) {
  console.log('[init] Loading data...');

  const [maskData, sstData, salData, windData] = await Promise.all([
    loadJSON('mask.json'),
    loadJSON('sst.json'),
    loadJSON('salinity.json'),
    loadJSON('wind_stress.json'),
  ]);

  // Mask
  if (maskData) {
    state.mask.set(resampleMask(maskData, grid));
    console.log(`[init] Mask: ${state.mask.reduce((a, b) => a + b, 0)} ocean cells / ${grid.n}`);
  }

  // SST
  if (sstData && sstData.sst) {
    const srcNX = sstData.nx || 1024, srcNY = sstData.ny || 512;
    const sst = resampleField(sstData.sst, srcNX, srcNY, grid, 15);
    // Initialize temp only on ocean cells
    for (let k = 0; k < grid.n; k++) {
      state.temp[k] = state.mask[k] ? sst[k] : 0;
      state.airTemp[k] = sst[k]; // air starts at SST
    }
    state.obsSSTTarget.set(sst);
    console.log('[init] SST loaded');
  }

  // Salinity
  if (salData && salData.salinity) {
    const srcNX = salData.nx || 1024, srcNY = salData.ny || 512;
    const sal = resampleField(salData.salinity, srcNX, srcNY, grid, 35);
    for (let k = 0; k < grid.n; k++) {
      state.sal[k] = state.mask[k] ? sal[k] : 0;
    }
    state.obsSalTarget.set(sal);
    console.log('[init] Salinity loaded');
  }

  // Wind stress curl
  if (windData && windData.curl) {
    const srcNX = windData.nx || 1024, srcNY = windData.ny || 512;
    state.windCurlTau.set(resampleField(windData.curl, srcNX, srcNY, grid, 0));
    console.log('[init] Wind curl loaded from observations');
  } else {
    // Analytical wind: 3-belt pattern (trades/westerlies/polar easterlies)
    console.log('[init] Using analytical wind curl');
    for (let j = 0; j < grid.ny; j++) {
      const lat = grid.latAt(j);
      const latRad = lat * Math.PI / 180;
      const shBoost = lat < 0 ? 2.0 : 1.0;
      const polarDamp = Math.abs(lat) > 60 ? 0.7 : 1.0;
      const curl = -Math.cos(3 * latRad) * shBoost * polarDamp * 2.0;
      for (let i = 0; i < grid.nx; i++) {
        const k = j * grid.nx + i;
        if (state.mask[k]) state.windCurlTau[k] = curl;
      }
    }
  }

  // Initialize humidity from SST (Clausius-Clapeyron at 80% RH)
  for (let k = 0; k < grid.n; k++) {
    if (state.mask[k]) {
      const qSat = 3.75e-3 * Math.exp(0.067 * state.temp[k]);
      state.humidity[k] = 0.8 * qSat;
    }
  }

  // Deep temperature: rough estimate (surface - gradient)
  for (let k = 0; k < grid.n; k++) {
    state.deepTemp[k] = state.mask[k] ? Math.max(1, state.temp[k] * 0.3) : 0;
    state.deepSal[k] = state.sal[k];
  }

  console.log('[init] Done');
}
