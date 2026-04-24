// ============================================================
// OCEAN CIRCULATION MODEL — Physics & Simulation Core
// ============================================================
// Extracted from index.html (Phase 1 of model/UI separation)
// No DOM dependencies. All rendering-related code stays in index.html.

// --- Simulation parameters (shared by GPU and CPU paths) ---
let beta = 1.0;
let r_friction = 0.04;         // increased friction for stability
let A_visc = 2e-4;             // viscosity
let windStrength = 1.0;
let doubleGyre = true;
let stepsPerFrame = 1;         // FFT Poisson solver: ~36ms/call × 2 = ~72ms/step
let paused = false;
let dt = 1.5e-5;               // scaled for 0.35° resolution (CFL)
let dtBase = 1.5e-5;
let totalSteps = 0;
let showField = 'temp';
let showParticles = true;

// Temperature / thermohaline parameters
let S_solar = 6.2;            // solar heating amplitude (tuned for regime-based clouds)
let A_olr = 1.8;              // OLR constant
let B_olr = 0.13;             // OLR linear coefficient
let kappa_diff = 2.5e-4;      // thermal diffusion
let alpha_T = 0.05;            // buoyancy coupling
// Two-layer ocean
let H_surface = 100;           // surface layer depth (m)
let H_deep = 4000;             // deep layer depth (m)
let gamma_mix = 0.001;         // base vertical mixing rate
let gamma_deep_form = 0.05;    // enhanced mixing for deep water formation
let kappa_deep = 2e-5;         // deep layer horizontal diffusion
// Two-layer circulation coupling
let F_couple_s = 0.5;          // interfacial coupling felt by surface layer
let F_couple_d = 0.0125;       // interfacial coupling felt by deep layer
let r_deep = 0.1;              // deep layer bottom friction
let yearSpeed = 1.0;          // seasonal cycle speed
let freshwaterForcing = 0.0;  // freshwater flux to northern box
let globalTempOffset = 0.0;   // global temperature offset in degrees C
let simTime = 0;              // continuous time for seasonal cycle
let T_YEAR = 10.0;            // simulation time units per year
let temp;                     // surface temperature field
let deepTemp;                 // deep ocean temperature field
let cpuDeepTempNew;           // CPU deep temp scratch buffer
let sal;                      // surface salinity field (PSU)
let deepSal;                  // deep salinity field (PSU)
let cpuSalNew;                // CPU salinity scratch buffer
let cpuDeepSalNew;            // CPU deep salinity scratch buffer
let beta_S = 0.8;             // haline contraction
let kappa_sal = 2.5e-4;       // salinity diffusion
let kappa_deep_sal = 2e-5;    // deep salinity diffusion
let salRestoringRate = 0.005; // surface salinity restoring toward climatology
let deepPsi;                  // deep ocean streamfunction
let deepZeta;                 // deep ocean vorticity
let cpuDeepZetaNew;           // CPU deep vorticity scratch buffer
let depth;                    // ocean depth field (meters)
let amocStrength = 0;         // diagnostic

// Atmosphere (1-layer energy balance, two-way coupled)
let airTemp;                  // atmospheric temperature field (degrees C)
let cloudField;               // cloud fraction field (0-1), updated each readback
let obsCloudField;            // observed cloud fraction (MODIS), static
let ekmanField;               // Ekman velocity (u_ek, v_ek) stacked, nondimensional
let kappa_atm = 3e-3;        // atmospheric heat diffusion
let gamma_oa = 0.005;        // ocean→atmosphere heat exchange rate
let gamma_ao = 0.001;        // atmosphere→ocean feedback (gentler — ocean has much more thermal inertia)
let gamma_la = 0.01;         // land→atmosphere heat exchange rate

// Grid sizes (NX=512 power-of-2 for radix-2 FFT, NY=160 matches data)
const GPU_NX = 512, GPU_NY = 160;
const CPU_NX = 512, CPU_NY = 160;
let NX, NY, dx, dy, invDx, invDy, invDx2, invDy2;
let cellW, cellH;             // rendering cell dimensions (set by init functions)

// Mask source dimensions (mask.json is 360x160, upscaled to model grid)
const MASK_SRC_NX = 360, MASK_SRC_NY = 160;
const LON0 = -180, LON1 = 180, LAT0 = -79.5, LAT1 = 79.5;

// Buffers (set during init)
let psi, zeta, zetaNew, mask;
let useGPU = false;
let cpuTempNew; // CPU temp scratch buffer

// Particles
const NP = 3000;
let px = new Float64Array(NP), py = new Float64Array(NP), page_ = new Float64Array(NP);
const MAX_AGE = 400;

// Coastline polygons (loaded from coastlines.json, used by rendering)
let LAND_POLYS = [];

// Mask source data (decoded from mask.json)
let maskSrcBits = null;

// ============================================================
// LOAD DATA
// ============================================================
// Load highest-res mask available (1024x512 > 360x160 fallback)
var maskFileName = (GPU_NX > 512) ? 'mask_1024x512.json' : 'mask.json';
let maskLoadPromise = fetch(maskFileName).then(function(r) { return r.json(); }).then(function(d) {
  var bits = [];
  for (var c = 0; c < d.hex.length; c++) {
    var v = parseInt(d.hex[c], 16);
    bits.push((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1);
  }
  maskSrcBits = bits;
}).catch(function() { maskSrcBits = null; });

let coastLoadPromise = fetch('coastlines.json').then(function(r) { return r.json(); }).then(function(p) {
  LAND_POLYS = p;
}).catch(function() {});

// Observational data for realistic initialization
let obsSSTData = null;   // NOAA OI SST v2 (1991-2020 annual mean)
let obsDeepData = null;  // WOA23 at 1000m depth
let sstLoadPromise = fetch('../sst_global_1deg.json').then(function(r) { return r.json(); }).then(function(d) {
  obsSSTData = d;
}).catch(function() { obsSSTData = null; });
let deepLoadPromise = fetch('../deep_temp_1deg.json').then(function(r) { return r.json(); }).then(function(d) {
  obsDeepData = d;
}).catch(function() { obsDeepData = null; });

// Real bathymetry from ETOPO1
let obsBathyData = null;
let bathyLoadPromise = fetch('../bathymetry_1deg.json').then(function(r) { return r.json(); }).then(function(d) {
  obsBathyData = d;
}).catch(function() { obsBathyData = null; });

// Surface albedo and precipitation maps for land physics
let obsSalinityData = null;
let salinityLoadPromise = fetch('../salinity_1deg.json').then(function(r) { return r.json(); }).then(function(d) {
  obsSalinityData = d;
}).catch(function() { obsSalinityData = null; });

let obsWindData = null;
let windLoadPromise = fetch('../wind_stress_1deg.json').then(function(r) { return r.json(); }).then(function(d) {
  obsWindData = d;
}).catch(function() { obsWindData = null; });

let obsAlbedoData = null;
let obsPrecipData = null;
let albedoLoadPromise = fetch('../albedo_1deg.json').then(function(r) { return r.json(); }).then(function(d) {
  obsAlbedoData = d;
}).catch(function() { obsAlbedoData = null; });
let precipLoadPromise = fetch('../precipitation_1deg.json').then(function(r) { return r.json(); }).then(function(d) {
  obsPrecipData = d;
}).catch(function() { obsPrecipData = null; });

let obsCloudData = null;
let cloudLoadPromise = fetch('../cloud_fraction_1deg.json').then(function(r) { return r.json(); }).then(function(d) {
  obsCloudData = d;
}).catch(function() { obsCloudData = null; });

// ============================================================
// MASK HELPERS
// ============================================================
// Resample observation data (360 columns) to model grid (NX columns).
// Latitude rows match 1:1 (both NY=160). Longitude is bilinearly interpolated.
var OBS_NX = 360; // all observation data files have 360 longitude columns
function resampleToModelGrid(srcArray) {
  if (NX === OBS_NX) return srcArray; // no-op if grids match
  var out = new Float32Array(NX * NY);
  for (var j = 0; j < NY; j++) {
    for (var i = 0; i < NX; i++) {
      var srcI = i * OBS_NX / NX; // fractional source column
      var i0 = Math.floor(srcI), i1 = (i0 + 1) % OBS_NX;
      var t = srcI - i0;
      var v0 = srcArray[j * OBS_NX + i0] || 0;
      var v1 = srcArray[j * OBS_NX + i1] || 0;
      out[j * NX + i] = v0 + t * (v1 - v0);
    }
  }
  return out;
}

var remappedElevation = null;
var remappedAlbedo = null;
var remappedPrecip = null;

function buildRemappedFields() {
  if (obsBathyData && obsBathyData.elevation) remappedElevation = resampleToModelGrid(obsBathyData.elevation);
  if (obsAlbedoData && obsAlbedoData.albedo) remappedAlbedo = resampleToModelGrid(obsAlbedoData.albedo);
  if (obsPrecipData && obsPrecipData.precipitation) remappedPrecip = resampleToModelGrid(obsPrecipData.precipitation);
}

function buildMask(nx, ny) {
  var m = new Uint8Array(nx * ny);
  if (!maskSrcBits) {
    // Fallback: simple rectangular ocean
    for (var j = 1; j < ny - 1; j++)
      for (var i = 0; i < nx; i++)
        m[j * nx + i] = 1;
    return m;
  }
  // Nearest-neighbor upscale from MASK_SRC_NX x MASK_SRC_NY
  for (var j = 0; j < ny; j++) {
    var sj = Math.min(Math.floor(j * MASK_SRC_NY / ny), MASK_SRC_NY - 1);
    for (var i = 0; i < nx; i++) {
      var si = Math.min(Math.floor(i * MASK_SRC_NX / nx), MASK_SRC_NX - 1);
      m[j * nx + i] = maskSrcBits[sj * MASK_SRC_NX + si] || 0;
    }
  }
  // Ensure polar boundaries are land (j=0, j=ny-1)
  for (var i = 0; i < nx; i++) { m[i] = 0; m[(ny - 1) * nx + i] = 0; }
  return m;
}

function buildMaskU32(mask8, nx, ny) {
  var m = new Uint32Array(nx * ny);
  for (var k = 0; k < nx * ny; k++) m[k] = mask8[k];
  return m;
}

// ============================================================
// WebGPU SHADERS (WGSL)
// ============================================================

var timestepShaderCode = [
'struct Params {',
'  nx: u32, ny: u32,',
'  dx: f32, dy: f32,',
'  dt: f32, beta: f32,',
'  r: f32, A: f32,',
'  windStrength: f32, doubleGyre: u32,',
'  alphaT: f32, simTime: f32,',
'  yearSpeed: f32, freshwater: f32,',
'  globalTempOffset: f32, gammaMix: f32,',
'  gammaDeepForm: f32, kappaDeep: f32,',
'  hSurface: f32, hDeep: f32,',
'  fCoupleS: f32, fCoupleD: f32,',
'  sSolar: f32, aOlr: f32,',
'  bOlr: f32, kappaDiff: f32,',
'  rDeep: f32, landHeatK: f32,',
'  betaS: f32, kappaSal: f32,',
'  kappaDeepSal: f32, salRestoring: f32,',
'  _padS0: u32, _padS1: u32, _padS2: u32, _padS3: u32,',
'};',
'',
'@group(0) @binding(0) var<storage, read> psi: array<f32>;',
'@group(0) @binding(1) var<storage, read> zeta: array<f32>;',
'@group(0) @binding(2) var<storage, read_write> zetaNew: array<f32>;',
'@group(0) @binding(3) var<storage, read> mask: array<u32>;',
'@group(0) @binding(4) var<uniform> params: Params;',
'@group(0) @binding(5) var<storage, read> tempIn: array<f32>;',
'@group(0) @binding(6) var<storage, read> deepPsiIn: array<f32>;',
'@group(0) @binding(7) var<storage, read> windCurlField: array<f32>;',
'',
'fn idx(i: u32, j: u32) -> u32 { return j * params.nx + i; }',
'',
'@compute @workgroup_size(8, 8)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let i = id.x;',
'  let j = id.y;',
'  let nx = params.nx;',
'  let ny = params.ny;',
'  if (i >= nx || j < 1u || j >= ny - 1u) { return; }',
'  let k = idx(i, j);',
'  if (mask[k] == 0u) { zetaNew[k] = 0.0; return; }',
'',
'  // Periodic wrapping in x (longitude)',
'  let ip1 = select(i + 1u, 0u, i == nx - 1u);',
'  let im1 = select(i - 1u, nx - 1u, i == 0u);',
'',
'  // Check cardinal neighbors are ocean',
'  let ke = idx(ip1, j); let kw = idx(im1, j);',
'  let kn = idx(i, j + 1u); let ks = idx(i, j - 1u);',
'  if (mask[ke] == 0u || mask[kw] == 0u || mask[kn] == 0u || mask[ks] == 0u) {',
'    zetaNew[k] = zeta[k] * 0.9;',
'    return;',
'  }',
'  // Check diagonal neighbors for Arakawa Jacobian',
'  let kne = idx(ip1, j + 1u); let knw = idx(im1, j + 1u);',
'  let kse = idx(ip1, j - 1u); let ksw = idx(im1, j - 1u);',
'  if (mask[kne] == 0u || mask[knw] == 0u || mask[kse] == 0u || mask[ksw] == 0u) {',
'    zetaNew[k] = zeta[k] * 0.95;',
'    return;',
'  }',
'',
'  // Latitude for this cell — needed for metric correction',
'  let lat = -79.5 + f32(j) / f32(ny - 1u) * 159.0;',
'  let latRad = lat * 3.14159265 / 180.0;',
'  let cosLat = max(cos(latRad), 0.087); // clamp at ~5° to avoid singularity',
'',
'  // Grid derivatives with cos(lat) metric correction',
'  let invDxRaw = 1.0 / params.dx;',
'  let invDx = invDxRaw / cosLat;  // zonal: physical = grid / cos(lat)',
'  let invDy = 1.0 / params.dy;     // meridional: unchanged',
'  let invDx2 = invDx * invDx;',
'  let invDy2 = invDy * invDy;',
'',
'  // Arakawa Jacobian J(psi, zeta) with metric correction',
'  let mDx = params.dx * cosLat;  // physical dx',
'  let mDy = params.dy;',
'  let J1 = (psi[ke] - psi[kw]) * (zeta[kn] - zeta[ks])',
'         - (psi[kn] - psi[ks]) * (zeta[ke] - zeta[kw]);',
'  let J2 = psi[ke] * (zeta[kne] - zeta[kse])',
'         - psi[kw] * (zeta[knw] - zeta[ksw])',
'         - psi[kn] * (zeta[kne] - zeta[knw])',
'         + psi[ks] * (zeta[kse] - zeta[ksw]);',
'  let J3 = zeta[ke] * (psi[kne] - psi[kse])',
'         - zeta[kw] * (psi[knw] - psi[ksw])',
'         - zeta[kn] * (psi[kne] - psi[knw])',
'         + zeta[ks] * (psi[kse] - psi[ksw]);',
'  let jac = (J1 + J2 + J3) / (12.0 * mDx * mDy);',
'',
'  // Beta term: varies with latitude (beta ~ cos(lat) in real ocean)',
'  let betaLocal = params.beta * cos(latRad);',
'  let betaV = betaLocal * (psi[ke] - psi[kw]) * 0.5 * invDx;',
'',
'  // Wind forcing from pre-scaled field (observed NCEP or analytical fallback)',
'  let F = params.windStrength * windCurlField[k];',
'',
'  // Friction',
'  let fric = -params.r * zeta[k];',
'',
'  // Viscosity: A * laplacian(zeta) with metric correction',
'  let lapZeta = invDx2 * (zeta[ke] + zeta[kw] - 2.0 * zeta[k])',
'             + invDy2 * (zeta[kn] + zeta[ks] - 2.0 * zeta[k]);',
'  let visc = params.A * lapZeta;',
'',
'  // Buoyancy coupling: density gradient from T AND S',
'  let N_off = params.nx * params.ny;',
'  let dRhodx = -params.alphaT * (tempIn[ke] - tempIn[kw]) + params.betaS * (tempIn[ke + N_off] - tempIn[kw + N_off]);',
'  let buoyancy = -dRhodx * 0.5 * invDx;',
'',
'  // Interfacial coupling to deep layer',
'  let coupling = params.fCoupleS * (deepPsiIn[k] - psi[k]);',
'',
'  zetaNew[k] = clamp(zeta[k] + params.dt * (-jac - betaV + F + fric + visc + buoyancy + coupling), -500.0, 500.0);',
'}'
].join('\n');

var poissonShaderCode = [
'struct Params {',
'  nx: u32, ny: u32,',
'  dx: f32, dy: f32,',
'  dt: f32, beta: f32,',
'  r: f32, A: f32,',
'  windStrength: f32, doubleGyre: u32,',
'  alphaT: f32, simTime: f32,',
'  yearSpeed: f32, freshwater: f32,',
'  globalTempOffset: f32, gammaMix: f32,',
'  gammaDeepForm: f32, kappaDeep: f32,',
'  hSurface: f32, hDeep: f32,',
'  fCoupleS: f32, fCoupleD: f32,',
'  sSolar: f32, aOlr: f32,',
'  bOlr: f32, kappaDiff: f32,',
'  rDeep: f32, landHeatK: f32,',
'  betaS: f32, kappaSal: f32,',
'  kappaDeepSal: f32, salRestoring: f32,',
'  _padS0: u32, _padS1: u32, _padS2: u32, _padS3: u32,',
'};',
'',
'@group(0) @binding(0) var<storage, read_write> psi: array<f32>;',
'@group(0) @binding(1) var<storage, read> zeta: array<f32>;',
'@group(0) @binding(2) var<storage, read> mask: array<u32>;',
'@group(0) @binding(3) var<uniform> params: Params;',
'',
'fn idx(i: u32, j: u32) -> u32 { return j * params.nx + i; }',
'',
'@compute @workgroup_size(8, 8)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let i = id.x;',
'  let j = id.y;',
'  let nx = params.nx;',
'  if (i >= nx || j < 1u || j >= params.ny - 1u) { return; }',
'',
'  // Red-black SOR: _padS0 holds color (0=red, 1=black)',
'  let color = params._padS0;',
'  if ((i + j) % 2u != color) { return; }',
'',
'  let k = idx(i, j);',
'  if (mask[k] == 0u) { return; }',
'',
'  // Periodic wrapping in x',
'  let ip1 = select(i + 1u, 0u, i == nx - 1u);',
'  let im1 = select(i - 1u, nx - 1u, i == 0u);',
'',
'  // Poisson uses computational (grid) Laplacian — ζ = ∇²_grid ψ',
'  // cos(lat) correction is in the physics operators, not the Poisson inversion',
'  let invDx2 = 1.0 / (params.dx * params.dx);',
'  let invDy2 = 1.0 / (params.dy * params.dy);',
'  let cx = invDx2;',
'  let cy = invDy2;',
'  let cc = -2.0 * (cx + cy);',
'',
'  let rhs = zeta[k];',
'  let neighbor_sum = cx * (psi[idx(ip1, j)] + psi[idx(im1, j)])',
'                   + cy * (psi[idx(i, j + 1u)] + psi[idx(i, j - 1u)]);',
'  let psiNew = (rhs - neighbor_sum) / cc;',
'  let omega = 1.92;',
'  psi[k] = psi[k] + omega * (psiNew - psi[k]);',
'}'
].join('\n');

var enforceBCShaderCode = [
'struct Params {',
'  nx: u32, ny: u32,',
'  dx: f32, dy: f32,',
'  dt: f32, beta: f32,',
'  r: f32, A: f32,',
'  windStrength: f32, doubleGyre: u32,',
'  alphaT: f32, simTime: f32,',
'  yearSpeed: f32, freshwater: f32,',
'  globalTempOffset: f32, gammaMix: f32,',
'  gammaDeepForm: f32, kappaDeep: f32,',
'  hSurface: f32, hDeep: f32,',
'  fCoupleS: f32, fCoupleD: f32,',
'  sSolar: f32, aOlr: f32,',
'  bOlr: f32, kappaDiff: f32,',
'  rDeep: f32, landHeatK: f32,',
'  betaS: f32, kappaSal: f32,',
'  kappaDeepSal: f32, salRestoring: f32,',
'  _padS0: u32, _padS1: u32, _padS2: u32, _padS3: u32,',
'};',
'',
'@group(0) @binding(0) var<storage, read_write> psi: array<f32>;',
'@group(0) @binding(1) var<storage, read_write> zeta: array<f32>;',
'@group(0) @binding(2) var<storage, read> mask: array<u32>;',
'@group(0) @binding(3) var<uniform> params: Params;',
'',
'@compute @workgroup_size(64)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let k = id.x;',
'  if (k >= params.nx * params.ny) { return; }',
'  let j = k / params.nx;',
'  if (mask[k] == 0u || j == 0u || j == params.ny - 1u) {',
'    psi[k] = 0.0;',
'    zeta[k] = 0.0;',
'  }',
'}'
].join('\n');

// Deep layer vorticity compute shader
var deepTimestepShaderCode = [
'struct Params {',
'  nx: u32, ny: u32,',
'  dx: f32, dy: f32,',
'  dt: f32, beta: f32,',
'  r: f32, A: f32,',
'  windStrength: f32, doubleGyre: u32,',
'  alphaT: f32, simTime: f32,',
'  yearSpeed: f32, freshwater: f32,',
'  globalTempOffset: f32, gammaMix: f32,',
'  gammaDeepForm: f32, kappaDeep: f32,',
'  hSurface: f32, hDeep: f32,',
'  fCoupleS: f32, fCoupleD: f32,',
'  sSolar: f32, aOlr: f32,',
'  bOlr: f32, kappaDiff: f32,',
'  rDeep: f32, landHeatK: f32,',
'  betaS: f32, kappaSal: f32,',
'  kappaDeepSal: f32, salRestoring: f32,',
'  _padS0: u32, _padS1: u32, _padS2: u32, _padS3: u32,',
'};',
'',
'@group(0) @binding(0) var<storage, read> deepPsi: array<f32>;',
'@group(0) @binding(1) var<storage, read> deepZeta: array<f32>;',
'@group(0) @binding(2) var<storage, read_write> deepZetaNew: array<f32>;',
'@group(0) @binding(3) var<storage, read> mask: array<u32>;',
'@group(0) @binding(4) var<uniform> params: Params;',
'@group(0) @binding(5) var<storage, read> surfacePsi: array<f32>;',
'@group(0) @binding(6) var<storage, read> deepTempIn: array<f32>;',
'',
'fn idx(i: u32, j: u32) -> u32 { return j * params.nx + i; }',
'',
'@compute @workgroup_size(8, 8)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let i = id.x;',
'  let j = id.y;',
'  let nx = params.nx;',
'  let ny = params.ny;',
'  if (i >= nx || j < 1u || j >= ny - 1u) { return; }',
'  let k = idx(i, j);',
'  if (mask[k] == 0u) { deepZetaNew[k] = 0.0; return; }',
'',
'  let ip1 = select(i + 1u, 0u, i == nx - 1u);',
'  let im1 = select(i - 1u, nx - 1u, i == 0u);',
'  let ke = idx(ip1, j); let kw = idx(im1, j);',
'  let kn = idx(i, j + 1u); let ks = idx(i, j - 1u);',
'',
'  // Coastal damping',
'  if (mask[ke] == 0u || mask[kw] == 0u || mask[kn] == 0u || mask[ks] == 0u) {',
'    deepZetaNew[k] = deepZeta[k] * 0.9;',
'    return;',
'  }',
'',
'  // Latitude and metric correction',
'  let lat = -79.5 + f32(j) / f32(ny - 1u) * 159.0;',
'  let latRad = lat * 3.14159265 / 180.0;',
'  let cosLat = max(cos(latRad), 0.087);',
'  let invDx = 1.0 / (params.dx * cosLat);',
'  let invDy = 1.0 / params.dy;',
'  let invDx2 = invDx * invDx;',
'  let invDy2 = invDy * invDy;',
'',
'  // Simplified Jacobian J(deepPsi, deepZeta)',
'  let dPdx = (deepPsi[ke] - deepPsi[kw]) * 0.5 * invDx;',
'  let dPdy = (deepPsi[kn] - deepPsi[ks]) * 0.5 * invDy;',
'  let dZdx = (deepZeta[ke] - deepZeta[kw]) * 0.5 * invDx;',
'  let dZdy = (deepZeta[kn] - deepZeta[ks]) * 0.5 * invDy;',
'  let jac = dPdx * dZdy - dPdy * dZdx;',
'',
'  // Beta term: varies with latitude',
'  let betaV = params.beta * cos(latRad) * (deepPsi[ke] - deepPsi[kw]) * 0.5 * invDx;',
'',
'  // Bottom friction (stronger than surface)',
'  let fric = -params.rDeep * deepZeta[k];',
'',
'  // Viscosity',
'  let lapZeta = invDx2 * (deepZeta[ke] + deepZeta[kw] - 2.0 * deepZeta[k])',
'             + invDy2 * (deepZeta[kn] + deepZeta[ks] - 2.0 * deepZeta[k]);',
'  let visc = params.A * lapZeta;',
'',
'  // Interfacial coupling: deep layer pulled toward surface flow',
'  let coupling = params.fCoupleD * (surfacePsi[k] - deepPsi[k]);',
'',
'  // Deep buoyancy forcing: density-driven overturning from deep temperature gradients',
'  let N_doff = params.nx * params.ny;',
'  let dRhodxDeep = -params.alphaT * (deepTempIn[ke] - deepTempIn[kw]) + params.betaS * (deepTempIn[ke + N_doff] - deepTempIn[kw + N_doff]);',
'  let deepBuoyancy = dRhodxDeep * 0.5 * invDx;',
'',
'  // Meridional overturning: density gradient drives deep equatorward flow',
'  let deepTN = select(deepTempIn[k], deepTempIn[kn], mask[kn] != 0u);',
'  let deepTS = select(deepTempIn[k], deepTempIn[ks], mask[ks] != 0u);',
'  let dTdyDeep = (deepTN - deepTS) * 0.5 * invDy;',
'  let motTendency = 0.05 * dTdyDeep;',
'',
'  deepZetaNew[k] = clamp(deepZeta[k] + params.dt * (-jac - betaV + fric + visc + coupling + deepBuoyancy + motTendency), -500.0, 500.0);',
'}'
].join('\n');

// Temperature compute shader
var temperatureShaderCode = [
'struct Params {',
'  nx: u32, ny: u32,',
'  dx: f32, dy: f32,',
'  dt: f32, beta: f32,',
'  r: f32, A: f32,',
'  windStrength: f32, doubleGyre: u32,',
'  alphaT: f32, simTime: f32,',
'  yearSpeed: f32, freshwater: f32,',
'  globalTempOffset: f32, gammaMix: f32,',
'  gammaDeepForm: f32, kappaDeep: f32,',
'  hSurface: f32, hDeep: f32,',
'  fCoupleS: f32, fCoupleD: f32,',
'  sSolar: f32, aOlr: f32,',
'  bOlr: f32, kappaDiff: f32,',
'  rDeep: f32, landHeatK: f32,',
'  betaS: f32, kappaSal: f32,',
'  kappaDeepSal: f32, salRestoring: f32,',
'  _padS0: u32, _padS1: u32, _padS2: u32, _padS3: u32,',
'};',
'',
'@group(0) @binding(0) var<storage, read> psi: array<f32>;',
'@group(0) @binding(1) var<storage, read> tempIn: array<f32>;',
'@group(0) @binding(2) var<storage, read_write> tempOut: array<f32>;',
'@group(0) @binding(3) var<storage, read> mask: array<u32>;',
'@group(0) @binding(4) var<uniform> params: Params;',
'@group(0) @binding(5) var<storage, read> deepTempIn: array<f32>;',
'@group(0) @binding(6) var<storage, read_write> deepTempOut: array<f32>;',
'@group(0) @binding(7) var<storage, read> depthField: array<f32>;',
'@group(0) @binding(8) var<storage, read> salClimatology: array<f32>;',
'@group(0) @binding(9) var<storage, read> ekmanVel: array<f32>;',
'',
'fn idx(i: u32, j: u32) -> u32 { return j * params.nx + i; }',
'',
'@compute @workgroup_size(8, 8)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let i = id.x;',
'  let j = id.y;',
'  let nx = params.nx;',
'  let ny = params.ny;',
'  if (i >= nx) { return; }',
'  let k = idx(i, j);',
'  if (j < 1u || j >= ny - 1u || mask[k] == 0u) { tempOut[k] = 0.0; deepTempOut[k] = 0.0; return; }',
'',
'  // Periodic wrapping in x',
'  let ip1 = select(i + 1u, 0u, i == nx - 1u);',
'  let im1 = select(i - 1u, nx - 1u, i == 0u);',
'',
'  let ke = idx(ip1, j); let kw = idx(im1, j);',
'  let kn = idx(i, j + 1u); let ks = idx(i, j - 1u);',
'  // One-sided stencil: use self temp/psi for land neighbors (zero-gradient BC)',
'  let tE = select(tempIn[k], tempIn[ke], mask[ke] != 0u);',
'  let tW = select(tempIn[k], tempIn[kw], mask[kw] != 0u);',
'  let tN = select(tempIn[k], tempIn[kn], mask[kn] != 0u);',
'  let tS = select(tempIn[k], tempIn[ks], mask[ks] != 0u);',
'  let pE = select(psi[k], psi[ke], mask[ke] != 0u);',
'  let pW = select(psi[k], psi[kw], mask[kw] != 0u);',
'  let pN = select(psi[k], psi[kn], mask[kn] != 0u);',
'  let pS = select(psi[k], psi[ks], mask[ks] != 0u);',
'',
'  // Metric correction for spherical geometry',
'  let lat = -79.5 + f32(j) / f32(ny - 1u) * 159.0;',
'  let latRad = lat * 3.14159265 / 180.0;',
'  let cosLat = max(cos(latRad), 0.087);',
'  let invDx = 1.0 / (params.dx * cosLat);',
'  let invDy = 1.0 / params.dy;',
'  let invDx2 = invDx * invDx;',
'  let invDy2 = invDy * invDy;',
'',
'  // Advection: J(psi, T) + Ekman transport',
'  let dPdx = (pE - pW) * 0.5 * invDx;',
'  let dPdy = (pN - pS) * 0.5 * invDy;',
'  let dTdx = (tE - tW) * 0.5 * invDx;',
'  let dTdy = (tN - tS) * 0.5 * invDy;',
'  let geoAdvec = dPdx * dTdy - dPdy * dTdx;',
'',
'  // Ekman heat transport: u_ek * dT/dx + v_ek * dT/dy',
'  let N_ek = params.nx * params.ny;',
'  let u_ek = ekmanVel[k] * params.windStrength;',
'  let v_ek = ekmanVel[k + N_ek] * params.windStrength;',
'  let ekmanAdvec = u_ek * dTdx + v_ek * dTdy;',
'  let advec = geoAdvec + ekmanAdvec;',
'',
'  // Seasonal solar declination (lat, latRad already computed above)',
'  let yearPhase = 2.0 * 3.14159265 * (params.simTime % 10.0) / 10.0;',
'  let declination = 23.44 * sin(yearPhase) * 3.14159265 / 180.0;',
'',
'  // Insolation with ice-albedo feedback',
'  let cosZenith = cos(latRad) * cos(declination) + sin(latRad) * sin(declination);',
'  var qSolar = params.sSolar * max(0.0, cosZenith);',
'  if (abs(lat) > 45.0) {',
'    let iceT = clamp((tempIn[k] + 2.0) / 10.0, 0.0, 1.0);',
'    let iceFrac = 1.0 - iceT * iceT * (3.0 - 2.0 * iceT);',
'    let latRamp = clamp((abs(lat) - 45.0) / 20.0, 0.0, 1.0);',
'    qSolar *= 1.0 - 0.50 * iceFrac * latRamp;',
'  }',
'',
'  // ── CLOUD PARAMETERIZATION ──',
'  // Physical regime-based clouds: ITCZ convection, subtropical subsidence,',
'  // marine stratocumulus, mid-latitude storm tracks, polar stratus',
'  let absLat = abs(lat);',
'',
'  // Humidity proxy: warm SST = more evaporation',
'  let humidity = clamp((tempIn[k] - 5.0) / 25.0, 0.0, 1.0);',
'',
'  // Lower tropospheric stability: estimated air temp vs SST',
'  // Warm air over cold water = inversion = stratocumulus',
'  let airTempEst = 28.0 - 0.55 * absLat;',
'  let lts = clamp((airTempEst - tempIn[k]) / 15.0, 0.0, 1.0);',
'',
'  // Seasonal ITCZ position (migrates ~5 deg with seasons)',
'  let itczLat = 5.0 * sin(yearPhase);',
'',
'  // 1. ITCZ deep convection',
'  let itczDist = (lat - itczLat) / 10.0;',
'  let convCloud = 0.30 * exp(-itczDist * itczDist) * humidity;',
'',
'  // 2. Warm-pool convection (SST > 26C threshold)',
'  let warmPool = 0.20 * clamp((tempIn[k] - 26.0) / 4.0, 0.0, 1.0);',
'',
'  // 3. Subtropical subsidence (Hadley descent ~25 deg, suppresses clouds)',
'  let subDist = (absLat - 25.0) / 10.0;',
'  let subsidence = 0.25 * exp(-subDist * subDist);',
'',
'  // 4. Marine stratocumulus (cold SST + stable air, subtropics)',
'  let stratocu = 0.30 * lts * clamp((35.0 - absLat) / 20.0, 0.0, 1.0);',
'',
'  // 5. Mid-latitude storm track (40-75 deg, extended for Southern Ocean)',
'  let stormTrack = 0.25 * clamp((absLat - 35.0) / 10.0, 0.0, 1.0)',
'                       * clamp((80.0 - absLat) / 15.0, 0.0, 1.0);',
'',
'  // 6. Southern Ocean boundary layer clouds (0.80-0.90 observed)',
'  let soCloud = select(0.0, 0.35 * clamp((absLat - 45.0) / 10.0, 0.0, 1.0), lat < 0.0);',
'',
'  // 7. Polar stratus (both hemispheres)',
'  let polarCloud = 0.15 * clamp((absLat - 55.0) / 15.0, 0.0, 1.0);',
'',
'  // Combine: high clouds (convective) + low clouds (stratiform) - subsidence',
'  let highCloud = convCloud + warmPool;',
'  let lowCloud = stratocu + stormTrack + soCloud + polarCloud;',
'  let cloudFrac = clamp(highCloud + lowCloud - subsidence * (1.0 - humidity), 0.05, 0.85);',
'',
'  // Convective fraction determines radiative properties',
'  let convFrac = select(0.0, clamp(highCloud / (highCloud + lowCloud + 0.01), 0.0, 1.0), cloudFrac > 0.05);',
'',
'  // SW albedo: low clouds reflect more (0.35) than high clouds (0.20)',
'  let cloudAlbedo = cloudFrac * (0.35 * (1.0 - convFrac) + 0.20 * convFrac);',
'  qSolar *= 1.0 - cloudAlbedo;',
'',
'  // Outgoing longwave: A + B*T (global heat balance)',
'  let olr = params.aOlr - params.bOlr * params.globalTempOffset + params.bOlr * tempIn[k];',
'  // LW greenhouse: high clouds trap more (0.12) than low clouds (0.03)',
'  let cloudGreenhouse = cloudFrac * (0.03 * (1.0 - convFrac) + 0.12 * convFrac);',
'  let effectiveOlr = olr * (1.0 - cloudGreenhouse);',
'',
'  // Net radiative heating',
'  let qNet = qSolar - effectiveOlr;',
'',
'  let y = f32(j) / f32(ny - 1u);',
'',
'  // Diffusion with one-sided stencil',
'  let lapT = invDx2 * (tE + tW - 2.0 * tempIn[k])',
'           + invDy2 * (tN + tS - 2.0 * tempIn[k]);',
'  let diff = params.kappaDiff * lapT;',
'',
'  // Land-ocean heat exchange',
'  var landFlux: f32 = 0.0;',
'  let nLand = f32(select(0u, 1u, mask[ke] == 0u)) + f32(select(0u, 1u, mask[kw] == 0u))',
'            + f32(select(0u, 1u, mask[kn] == 0u)) + f32(select(0u, 1u, mask[ks] == 0u));',
'  let nOcean = 4.0 - nLand;',
'  if (nLand > 0.0 && nOcean > 0.0) {',
'    let landT = 50.0 * max(0.0, cosZenith) - 20.0;',
'    let rawFlux = params.landHeatK * (landT - tempIn[k]) * (nOcean / 4.0);',
'    landFlux = clamp(rawFlux, -0.5, 0.5);',
'  }',
'',
'  tempOut[k] = tempIn[k] + params.dt * (-advec + qNet + diff + landFlux);',
'',
'  // Variable mixed layer depth: deep in Southern Ocean + subpolar NH, shallow in tropics',
'  let mldBase = 30.0 + 70.0 * pow(absLat / 80.0, 1.5);',
'  let accDist = (lat + 50.0) / 12.0;',
'  let mldACC = select(0.0, 250.0 * exp(-accDist * accDist), lat < -35.0 && lat > -65.0);',
'  let subpDist = (lat - 62.0) / 8.0;',
'  let mldSubpolar = select(0.0, 150.0 * exp(-subpDist * subpDist), lat > 50.0 && lat < 75.0);',
'  let mixedLayerDepth = mldBase + mldACC + mldSubpolar;',
'',
'  // Two-layer vertical exchange',
'  let localDepth = depthField[k];',
'  let hSurf = min(mixedLayerDepth, localDepth);',
'  let hDeep = max(1.0, localDepth - mixedLayerDepth);',
'  let hasDeepLayer = select(0.0, 1.0, localDepth > mixedLayerDepth);',
'',
'  // ── SALINITY (stacked at offset N in the same buffers) ──',
'  let N = params.nx * params.ny;',
'  let salK = k + N;',
'',
'  let sE = select(tempIn[salK], tempIn[idx(ip1,j) + N], mask[ke] != 0u);',
'  let sW = select(tempIn[salK], tempIn[idx(im1,j) + N], mask[kw] != 0u);',
'  let sN = select(tempIn[salK], tempIn[idx(i,j+1u) + N], mask[kn] != 0u);',
'  let sS = select(tempIn[salK], tempIn[idx(i,j-1u) + N], mask[ks] != 0u);',
'',
'  let dSdx = (sE - sW) * 0.5 * invDx;',
'  let dSdy = (sN - sS) * 0.5 * invDy;',
'  let salAdvec = dPdx * dSdy - dPdy * dSdx + u_ek * dSdx + v_ek * dSdy;',
'',
'  let lapS = invDx2 * (sE + sW - 2.0 * tempIn[salK])',
'           + invDy2 * (sN + sS - 2.0 * tempIn[salK]);',
'  let salDiff = params.kappaSal * lapS;',
'',
'  // Salinity restoring: use observed WOA23 climatology if available, else zonal formula',
'  let salClimObs = salClimatology[k];',
'  let salClim = select(34.0 + 2.0 * cos(2.0 * latRad) - 0.5 * cos(4.0 * latRad), salClimObs, salClimObs > 1.0);',
'  let salRestore = params.salRestoring * (salClim - tempIn[salK]);',
'',
'  var fwSal: f32 = 0.0;',
'  if (y > 0.75) {',
'    fwSal = -params.freshwater * 3.0 * (y - 0.75) * 4.0;',
'  }',
'',
'  tempOut[salK] = tempIn[salK] + params.dt * (-salAdvec + salDiff + salRestore + fwSal);',
'',
'  // ── DENSITY-BASED DEEP WATER FORMATION ──',
'  let rhoSurf = -params.alphaT * tempIn[k] + params.betaS * tempIn[salK];',
'  let rhoDeep = -params.alphaT * deepTempIn[k] + params.betaS * deepTempIn[k + N];',
'',
'  var gamma = params.gammaMix;',
'  if (abs(lat) > 40.0 && rhoSurf > rhoDeep) { gamma = params.gammaDeepForm; }',
'',
'  let vertExchangeT = gamma * (tempIn[k] - deepTempIn[k]) * hasDeepLayer;',
'  tempOut[k] = clamp(tempOut[k] - params.dt * vertExchangeT / hSurf, -10.0, 40.0);',
'',
'  let vertExchangeS = gamma * (tempIn[salK] - deepTempIn[k + N]) * hasDeepLayer;',
'  tempOut[salK] = clamp(tempOut[salK] - params.dt * vertExchangeS / hSurf, 28.0, 40.0);',
'',
'  // ── DEEP LAYER: temperature + salinity ──',
'  let dE = select(deepTempIn[k], deepTempIn[ke], mask[ke] != 0u);',
'  let dW = select(deepTempIn[k], deepTempIn[kw], mask[kw] != 0u);',
'  let dN = select(deepTempIn[k], deepTempIn[kn], mask[kn] != 0u);',
'  let dS = select(deepTempIn[k], deepTempIn[ks], mask[ks] != 0u);',
'  let lapDeep = invDx2 * (dE + dW - 2.0 * deepTempIn[k])',
'             + invDy2 * (dN + dS - 2.0 * deepTempIn[k]);',
'  let deepDiff = params.kappaDeep * lapDeep;',
'  deepTempOut[k] = clamp(deepTempIn[k] + params.dt * (vertExchangeT / hDeep + deepDiff) * hasDeepLayer, -5.0, 30.0);',
'',
'  // Deep salinity',
'  let dsE = select(deepTempIn[k+N], deepTempIn[ke+N], mask[ke] != 0u);',
'  let dsW = select(deepTempIn[k+N], deepTempIn[kw+N], mask[kw] != 0u);',
'  let dsN = select(deepTempIn[k+N], deepTempIn[kn+N], mask[kn] != 0u);',
'  let dsS = select(deepTempIn[k+N], deepTempIn[ks+N], mask[ks] != 0u);',
'  let lapDeepSal = invDx2 * (dsE + dsW - 2.0 * deepTempIn[k+N])',
'                 + invDy2 * (dsN + dsS - 2.0 * deepTempIn[k+N]);',
'  let deepSalDiff = params.kappaDeepSal * lapDeepSal;',
'  deepTempOut[k+N] = clamp(deepTempIn[k+N] + params.dt * (vertExchangeS / hDeep + deepSalDiff) * hasDeepLayer, 33.0, 37.0);',
'}'
].join('\n');

// ============================================================
// DEPTH FIELD GENERATION
// ============================================================
function generateDepthField() {
  depth = new Float32Array(NX * NY);

  // Use real ETOPO1 bathymetry when available
  if (obsBathyData && obsBathyData.depth) {
    var remapped = resampleToModelGrid(obsBathyData.depth);
    for (var k = 0; k < NX * NY; k++) {
      if (!mask[k]) { depth[k] = 0; continue; }
      var d = remapped[k];
      if (d > 0) {
        depth[k] = Math.min(5500, Math.max(50, d));
      } else {
        depth[k] = 200;
      }
    }
    console.log('Using real ETOPO1 bathymetry');
    return;
  }

  // Fallback: BFS from all land cells to compute distance-to-coast
  console.log('Using BFS distance-to-coast bathymetry (fallback)');
  var dist = new Float32Array(NX * NY);
  for (var k = 0; k < NX * NY; k++) dist[k] = 9999;

  var queue = [];
  for (var j = 0; j < NY; j++) for (var i = 0; i < NX; i++) {
    var k = j * NX + i;
    if (!mask[k]) { dist[k] = 0; continue; }
    var ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;
    var hasLand = false;
    if (j > 0 && !mask[(j-1)*NX+i]) hasLand = true;
    if (j < NY-1 && !mask[(j+1)*NX+i]) hasLand = true;
    if (!mask[j*NX+ip1]) hasLand = true;
    if (!mask[j*NX+im1]) hasLand = true;
    if (hasLand) { dist[k] = 1; queue.push(k); }
  }

  var head = 0;
  while (head < queue.length) {
    var ck = queue[head++];
    var ci = ck % NX, cj = Math.floor(ck / NX);
    var cd = dist[ck];
    var neighbors = [
      [ci, cj-1], [ci, cj+1],
      [(ci+1)%NX, cj], [(ci-1+NX)%NX, cj]
    ];
    for (var n = 0; n < 4; n++) {
      var ni = neighbors[n][0], nj = neighbors[n][1];
      if (nj < 0 || nj >= NY) continue;
      var nk = nj * NX + ni;
      if (mask[nk] && dist[nk] > cd + 1) {
        dist[nk] = cd + 1;
        queue.push(nk);
      }
    }
  }

  for (var k2 = 0; k2 < NX * NY; k2++) {
    if (!mask[k2]) { depth[k2] = 0; continue; }
    var d = dist[k2];
    var t = Math.min(1, Math.max(0, (d - 1) / 5));
    t = t * t * (3 - 2 * t); // smoothstep
    depth[k2] = 200 + 3800 * t;
  }
}

// ============================================================
// SALINITY CLIMATOLOGY FIELD (WOA23 or zonal formula fallback)
// ============================================================
var salClimatologyField = null;
function generateSalClimatologyField() {
  salClimatologyField = new Float32Array(NX * NY);
  if (obsSalinityData && obsSalinityData.salinity) {
    var srcSal = obsSalinityData.salinity;
    var sNX = obsSalinityData.nx || 360, sNY = obsSalinityData.ny || 160;
    var sLat0 = obsSalinityData.lat0 || -79.5, sLat1 = obsSalinityData.lat1 || 79.5;
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var fj = (lat - sLat0) / (sLat1 - sLat0) * (sNY - 1);
      var j0s = Math.max(0, Math.min(sNY - 1, Math.floor(fj)));
      var j1s = Math.min(sNY - 1, j0s + 1);
      var ts = Math.max(0, Math.min(1, fj - j0s));
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        var fi = i / NX * sNX;
        var i0s = Math.floor(fi) % sNX, i1s = (i0s + 1) % sNX, si = fi - Math.floor(fi);
        var v00 = srcSal[j0s*sNX+i0s]||0, v01 = srcSal[j0s*sNX+i1s]||0;
        var v10 = srcSal[j1s*sNX+i0s]||0, v11 = srcSal[j1s*sNX+i1s]||0;
        var s = (1-ts)*((1-si)*v00+si*v01) + ts*((1-si)*v10+si*v11);
        if (s > 1) {
          salClimatologyField[k] = s;
        } else {
          var latRad = lat * Math.PI / 180;
          salClimatologyField[k] = 34.0 + 2.0 * Math.cos(2 * latRad) - 0.5 * Math.cos(4 * latRad);
        }
      }
    }
    console.log('Using WOA23 observed salinity climatology');
  } else {
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var latRad = lat * Math.PI / 180;
      var sc = 34.0 + 2.0 * Math.cos(2 * latRad) - 0.5 * Math.cos(4 * latRad);
      for (var i = 0; i < NX; i++) salClimatologyField[j * NX + i] = sc;
    }
  }
}

// ============================================================
// WIND STRESS CURL FIELD (NCEP or analytical fallback)
// ============================================================
var windCurlFieldData = null;
function generateWindCurlField() {
  windCurlFieldData = new Float32Array(NX * NY);
  if (obsWindData && obsWindData.wind_curl) {
    // Compute RMS of observed zonal-mean curl and analytical to find scaling factor
    var obsNX = obsWindData.nx || 360, obsNY = obsWindData.ny || 160;
    var obsLat0 = obsWindData.lat0 || -79.5, obsLat1 = obsWindData.lat1 || 79.5;
    var rmsObs2 = 0, rmsAnal2 = 0, nLats = 0;
    for (var jj = 0; jj < obsNY; jj++) {
      var lat = obsLat0 + jj * (obsLat1 - obsLat0) / (obsNY - 1);
      if (Math.abs(lat) > 75) continue; // skip polar edges
      var zonalSum = 0, zonalCnt = 0;
      for (var ii = 0; ii < obsNX; ii++) {
        var v = obsWindData.wind_curl[jj * obsNX + ii];
        if (v !== 0) { zonalSum += v; zonalCnt++; }
      }
      if (zonalCnt < 10) continue;
      var zonalMean = zonalSum / zonalCnt;
      var latRad = lat * Math.PI / 180;
      var shBoost = lat < 0 ? 2.0 : 1.0;
      var polarDamp = Math.abs(lat) > 60 ? 0.7 : 1.0;
      var analytical = (-Math.cos(3 * latRad) * shBoost * polarDamp) * 2.0;
      rmsObs2 += zonalMean * zonalMean;
      rmsAnal2 += analytical * analytical;
      nLats++;
    }
    var windCurlScale = Math.sqrt(rmsAnal2 / rmsObs2);
    console.log('Wind curl scaling: ' + windCurlScale.toExponential(4) + ' (from ' + nLats + ' latitude bands)');

    // Bilinear interpolate observed curl to model grid, pre-scaled to model units
    var obsNX = obsWindData.nx || 360, obsNY = obsWindData.ny || 160;
    var obsLat0w = obsWindData.lat0 || -79.5, obsLat1w = obsWindData.lat1 || 79.5;
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var fracJ = (lat - obsLat0w) / (obsLat1w - obsLat0w) * (obsNY - 1);
      var j0w = Math.floor(fracJ), j1w = j0w + 1, tw = fracJ - j0w;
      if (j0w < 0) { j0w = 0; j1w = 0; tw = 0; }
      if (j1w >= obsNY) { j1w = obsNY - 1; j0w = j1w; tw = 0; }
      for (var i = 0; i < NX; i++) {
        var fracI = i / NX * obsNX;
        var i0w = Math.floor(fracI) % obsNX, i1w = (i0w + 1) % obsNX;
        var si = fracI - Math.floor(fracI);
        var v00 = obsWindData.wind_curl[j0w * obsNX + i0w] || 0;
        var v01 = obsWindData.wind_curl[j0w * obsNX + i1w] || 0;
        var v10 = obsWindData.wind_curl[j1w * obsNX + i0w] || 0;
        var v11 = obsWindData.wind_curl[j1w * obsNX + i1w] || 0;
        var val = (1-tw)*((1-si)*v00 + si*v01) + tw*((1-si)*v10 + si*v11);
        windCurlFieldData[j * NX + i] = val * windCurlScale;
      }
    }
    console.log('Using NCEP observed wind stress curl (pre-scaled to model units)');
  } else {
    // Analytical fallback: populate field so shader uses same code path
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var latRad = lat * Math.PI / 180;
      var shBoost = lat < 0 ? 2.0 : 1.0;
      var polarDamp = Math.abs(lat) > 60 ? 0.7 : 1.0;
      var F = (-Math.cos(3 * latRad) * shBoost * polarDamp) * 2.0;
      for (var i = 0; i < NX; i++) {
        windCurlFieldData[j * NX + i] = F;
      }
    }
  }
}

// ============================================================
// OBSERVED CLOUD FRACTION FIELD (MODIS)
// ============================================================
function generateObsCloudField() {
  if (!obsCloudData || !obsCloudData.cloud_fraction) return;
  obsCloudField = resampleToModelGrid(obsCloudData.cloud_fraction);
  console.log('Loaded MODIS observed cloud fraction');
}

// ============================================================
// EKMAN VELOCITY FIELD (from wind stress)
// ============================================================
function generateEkmanField() {
  // Ekman transport: v_e = -tau_x/(rho*f*H_ek), u_e = tau_y/(rho*f*H_ek)
  // Stacked layout: first NX*NY = u_ekman, second NX*NY = v_ekman
  ekmanField = new Float32Array(NX * NY * 2);
  var OMEGA = 7.292e-5;   // Earth rotation (rad/s)
  var RHO = 1025;          // seawater density (kg/m^3)
  var H_EK = 50;           // Ekman layer depth (m)

  if (obsWindData && obsWindData.tau_x && obsWindData.tau_y) {
    var tauX = obsWindData.tau_x;
    var tauY = obsWindData.tau_y;
    var eNX = obsWindData.nx || 360, eNY = obsWindData.ny || 160;
    var eLat0 = obsWindData.lat0 || -79.5, eLat1 = obsWindData.lat1 || 79.5;

    // Helper: bilinear sample from obs grid
    function sampleObs(arr, lat, lon_i) {
      var fj = (lat - eLat0) / (eLat1 - eLat0) * (eNY - 1);
      var j0 = Math.floor(fj), j1 = j0 + 1, tj = fj - j0;
      if (j0 < 0) { j0 = 0; j1 = 0; tj = 0; }
      if (j1 >= eNY) { j1 = eNY - 1; j0 = j1; tj = 0; }
      var fi = lon_i / NX * eNX;
      var i0 = Math.floor(fi) % eNX, i1 = (i0 + 1) % eNX, ti = fi - Math.floor(fi);
      var v00 = arr[j0 * eNX + i0] || 0, v01 = arr[j0 * eNX + i1] || 0;
      var v10 = arr[j1 * eNX + i0] || 0, v11 = arr[j1 * eNX + i1] || 0;
      return (1-tj)*((1-ti)*v00 + ti*v01) + tj*((1-ti)*v10 + ti*v11);
    }

    // Compute RMS of Ekman velocity to find nondimensional scaling
    var ekRms2 = 0, ekN = 0;
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var latRad = lat * Math.PI / 180;
      var f = 2 * OMEGA * Math.sin(latRad);
      if (Math.abs(lat) < 5) continue;
      for (var i = 0; i < NX; i++) {
        var tx = sampleObs(tauX, lat, i), ty = sampleObs(tauY, lat, i);
        var ue = ty / (RHO * f * H_EK);
        var ve = -tx / (RHO * f * H_EK);
        ekRms2 += ue * ue + ve * ve;
        ekN++;
      }
    }
    var ekRms = Math.sqrt(ekRms2 / Math.max(ekN, 1));
    var targetRms = 0.3;
    var ekScale = ekRms > 0 ? targetRms / ekRms : 0;

    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var latRad = lat * Math.PI / 180;
      var f = 2 * OMEGA * Math.sin(latRad);
      if (Math.abs(lat) < 5) f = 2 * OMEGA * Math.sin(5 * Math.PI / 180) * (lat >= 0 ? 1 : -1);
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        var tx = sampleObs(tauX, lat, i), ty = sampleObs(tauY, lat, i);
        var ue = ty / (RHO * f * H_EK) * ekScale;
        var ve = -tx / (RHO * f * H_EK) * ekScale;
        ekmanField[k] = ue;
        ekmanField[k + NX * NY] = ve;
      }
    }
    console.log('Ekman velocities from NCEP wind stress (scale: ' + ekScale.toExponential(3) + ', RMS: ' + (ekRms * ekScale).toFixed(4) + ')');
  } else {
    // Analytical fallback: v_ekman from zonal wind τ_x ∝ -cos(3*lat)
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var latRad = lat * Math.PI / 180;
      var sinLat = Math.sin(latRad);
      if (Math.abs(sinLat) < Math.sin(5 * Math.PI / 180)) sinLat = Math.sin(5 * Math.PI / 180) * (lat >= 0 ? 1 : -1);
      // τ_x ∝ -cos(3*lat) → v_e = τ_x / f ∝ cos(3*lat) / sin(lat)
      var ve_raw = Math.cos(3 * latRad) / sinLat;
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        ekmanField[k] = 0; // u_ekman ≈ 0 for zonal winds
        ekmanField[k + NX * NY] = ve_raw * 0.15; // scaled for reasonable magnitude
      }
    }
  }
}

// ============================================================
// TEMPERATURE / SALINITY INITIALIZATION
// ============================================================
function initTemperatureField() {
  var useObs = obsSSTData && obsSSTData.sst;
  var useDeepObs = obsDeepData && obsDeepData.temp;
  var sstRemapped = useObs ? resampleToModelGrid(obsSSTData.sst) : null;
  var deepRemapped = useDeepObs ? resampleToModelGrid(obsDeepData.temp) : null;

  for (var j = 0; j < NY; j++) {
    var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    for (var i = 0; i < NX; i++) {
      var k = j * NX + i;
      if (!mask[k]) { temp[k] = 0; deepTemp[k] = 0; continue; }

      // Surface temperature
      var gotSST = false;
      if (sstRemapped) {
        var sv = sstRemapped[k];
        if (sv > -90) {
          temp[k] = sv;
          gotSST = true;
        }
      }
      if (!gotSST) {
        var tBase = 28 - 0.55 * Math.abs(lat) - 0.0003 * Math.pow(Math.abs(lat), 2);
        temp[k] = Math.max(-2, Math.min(30, tBase));
      }

      // Deep temperature
      var gotDeep = false;
      if (deepRemapped) {
        var dv = deepRemapped[k];
        if (dv > -90) {
          deepTemp[k] = dv;
          gotDeep = true;
        }
      }
      if (!gotDeep) {
        var yFrac = j / (NY - 1);
        deepTemp[k] = 0.5 + 3.0 * yFrac;
      }

      // Surface salinity
      var latRad = lat * Math.PI / 180;
      sal[k] = 34.0 + 2.0 * Math.cos(2 * latRad) - 0.5 * Math.cos(4 * latRad);

      // Deep salinity
      deepSal[k] = 34.7 + 0.2 * Math.cos(2 * latRad);
    }
  }
}

function initStommelSolution() {
  for (var k = 0; k < NX * NY; k++) {
    psi[k] = 0;
    zeta[k] = 0;
  }
}

// ============================================================
// CPU FALLBACK SOLVER
// ============================================================
var cpuZetaNew;

function initCPU() {
  NX = CPU_NX; NY = CPU_NY;
  dx = 1.0 / (NX - 1); dy = 1.0 / (NY - 1);
  invDx = 1 / dx; invDy = 1 / dy;
  invDx2 = invDx * invDx; invDy2 = invDy * invDy;
  cellW = W / NX; cellH = H / NY;

  mask = buildMask(NX, NY);
  psi = new Float64Array(NX * NY);
  zeta = new Float64Array(NX * NY);
  cpuZetaNew = new Float64Array(NX * NY);
  temp = new Float64Array(NX * NY);
  cpuTempNew = new Float64Array(NX * NY);
  deepTemp = new Float64Array(NX * NY);
  cpuDeepTempNew = new Float64Array(NX * NY);
  sal = new Float64Array(NX * NY);
  cpuSalNew = new Float64Array(NX * NY);
  deepSal = new Float64Array(NX * NY);
  cpuDeepSalNew = new Float64Array(NX * NY);
  deepPsi = new Float64Array(NX * NY);
  deepZeta = new Float64Array(NX * NY);
  cpuDeepZetaNew = new Float64Array(NX * NY);
  airTemp = new Float64Array(NX * NY);
  buildRemappedFields();
  generateDepthField();
  generateWindCurlField();
  generateObsCloudField();
  generateEkmanField();
  initTemperatureField();
  // Initialize air temp from surface: over ocean use SST, over land use radiative equilibrium
  for (var ai = 0; ai < NX * NY; ai++) {
    if (mask[ai]) airTemp[ai] = temp[ai];
    else {
      var aj = Math.floor(ai / NX);
      var alat = LAT0 + (aj / (NY - 1)) * (LAT1 - LAT0);
      airTemp[ai] = 28 - 0.55 * Math.abs(alat);
    }
  }
}

function cpuI(i, j) { return j * NX + i; }

function cpuWindCurl(i, j) {
  // Read from pre-scaled field (observed or analytical)
  return windStrength * windCurlFieldData[j * NX + i];
}

function cpuCosLat(j) {
  var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
  return Math.max(Math.cos(lat * Math.PI / 180), 0.087);
}

function cpuBeta(j) {
  var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
  var latRad = lat * Math.PI / 180;
  return beta * Math.cos(latRad);
}

var rhoGS, omegaSOR;

// Compute Poisson residual: ||∇²ψ - ζ||₂ / ||ζ||₂
function poissonResidual() {
  var resSum = 0, zetaSum = 0;
  var cx = invDx2, cy = invDy2, cc = -2 * (cx + cy);
  for (var j = 1; j < NY - 1; j++) {
    for (var i = 0; i < NX; i++) {
      var k = cpuI(i, j);
      if (!mask[k]) continue;
      var ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;
      var res = cx * (psi[cpuI(ip1, j)] + psi[cpuI(im1, j)])
              + cy * (psi[cpuI(i, j + 1)] + psi[cpuI(i, j - 1)])
              + cc * psi[k] - zeta[k];
      resSum += res * res;
      zetaSum += zeta[k] * zeta[k];
    }
  }
  return { residual: Math.sqrt(resSum), zetaNorm: Math.sqrt(zetaSum), relResidual: Math.sqrt(resSum / Math.max(zetaSum, 1e-30)) };
}

function initSOR() {
  rhoGS = Math.cos(Math.PI / NX) + Math.cos(Math.PI / NY);
  omegaSOR = 2 / (1 + Math.sqrt(1 - rhoGS * rhoGS / 4));
  initFFTPoisson();
}

// ============================================================
// FFT POISSON SOLVER — exact, O(N log N)
// Radix-2 FFT in x (NX must be power of 2), tridiagonal Thomas in y
// ============================================================
function fftRadix2(re, im, n, inv) {
  for (var i = 1, j = 0; i < n; i++) {
    var bit = n >> 1; while (j & bit) { j ^= bit; bit >>= 1; } j ^= bit;
    if (i < j) { var t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  var sgn = inv ? 1 : -1;
  for (var len = 2; len <= n; len <<= 1) {
    var ang = sgn * 2 * Math.PI / len, wR = Math.cos(ang), wI = Math.sin(ang);
    for (var i = 0; i < n; i += len) {
      var cR = 1, cI = 0;
      for (var j = 0; j < (len >> 1); j++) {
        var uR = re[i+j], uI = im[i+j];
        var vR = re[i+j+(len>>1)]*cR - im[i+j+(len>>1)]*cI;
        var vI = re[i+j+(len>>1)]*cI + im[i+j+(len>>1)]*cR;
        re[i+j] = uR+vR; im[i+j] = uI+vI;
        re[i+j+(len>>1)] = uR-vR; im[i+j+(len>>1)] = uI-vI;
        var tR = cR*wR - cI*wI; cI = cR*wI + cI*wR; cR = tR;
      }
    }
  }
  if (inv) { for (var i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
}

function initFFTPoisson() {
  // Verify NX is power of 2
  if (NX & (NX - 1)) console.warn('FFT Poisson: NX=' + NX + ' is not power of 2!');
  console.log('FFT Poisson solver initialized: NX=' + NX + ' (radix-2), NY=' + NY);
}

function cpuSolveFFT(psiArr, zetaArr) {
  var tmpR = new Float64Array(NX), tmpI = new Float64Array(NX);
  // Forward FFT each row of zeta
  var hatR = new Float64Array(NX * NY), hatI = new Float64Array(NX * NY);
  for (var j = 0; j < NY; j++) {
    for (var i = 0; i < NX; i++) { tmpR[i] = zetaArr[j*NX+i]; tmpI[i] = 0; }
    fftRadix2(tmpR, tmpI, NX, false);
    for (var m = 0; m < NX; m++) { hatR[m*NY+j] = tmpR[m]; hatI[m*NY+j] = tmpI[m]; }
  }
  // Tridiagonal solve per Fourier mode
  var pHR = new Float64Array(NX * NY), pHI = new Float64Array(NX * NY);
  for (var m = 0; m < NX; m++) {
    var km2 = invDx2 * 2 * (Math.cos(2 * Math.PI * m / NX) - 1);
    var b = new Float64Array(NY), dR = new Float64Array(NY), dI = new Float64Array(NY);
    b[0] = 1; b[NY-1] = 1;
    for (var j = 1; j < NY-1; j++) {
      var cl = cpuCosLat(j);
      b[j] = km2 / (cl * cl) - 2 * invDy2;
      dR[j] = hatR[m*NY+j]; dI[j] = hatI[m*NY+j];
    }
    // Thomas forward elimination
    for (var j = 1; j < NY; j++) {
      var cp = (j-1 > 0 && j-1 < NY-1) ? invDy2 : 0;
      var w = invDy2 / b[j-1];
      b[j] -= w * cp;
      dR[j] -= w * dR[j-1]; dI[j] -= w * dI[j-1];
    }
    // Thomas back substitution
    pHR[m*NY+(NY-1)] = dR[NY-1] / b[NY-1];
    pHI[m*NY+(NY-1)] = dI[NY-1] / b[NY-1];
    for (var j = NY-2; j >= 0; j--) {
      var c = (j > 0 && j < NY-1) ? invDy2 : 0;
      pHR[m*NY+j] = (dR[j] - c * pHR[m*NY+(j+1)]) / b[j];
      pHI[m*NY+j] = (dI[j] - c * pHI[m*NY+(j+1)]) / b[j];
    }
  }
  // Inverse FFT each row
  for (var j = 0; j < NY; j++) {
    for (var m = 0; m < NX; m++) { tmpR[m] = pHR[m*NY+j]; tmpI[m] = pHI[m*NY+j]; }
    fftRadix2(tmpR, tmpI, NX, true);
    for (var i = 0; i < NX; i++) psiArr[j*NX+i] = tmpR[i];
  }
  // Zero out land cells
  for (var k = 0; k < NX * NY; k++) if (!mask[k]) psiArr[k] = 0;
}

function cpuSolveSOR(nIter) {
  // Poisson uses grid Laplacian (no cos(lat)) — consistent with ζ = ∇²_grid ψ
  var cx = invDx2, cy = invDy2, cc = -2 * (cx + cy), invCC = 1 / cc;
  for (var iter = 0; iter < nIter; iter++) {
    for (var j = 1; j < NY - 1; j++) {
      for (var i = 0; i < NX; i++) {
        var k = cpuI(i, j);
        if (!mask[k]) continue;
        var ip1 = (i + 1) % NX;
        var im1 = (i - 1 + NX) % NX;
        var res = cx * (psi[cpuI(ip1, j)] + psi[cpuI(im1, j)])
                + cy * (psi[cpuI(i, j + 1)] + psi[cpuI(i, j - 1)])
                + cc * psi[k] - zeta[k];
        psi[k] -= omegaSOR * res * invCC;
      }
    }
  }
}

function cpuSolveDeepSOR(nIter) {
  var cx = invDx2, cy = invDy2, cc = -2 * (cx + cy), invCC = 1 / cc;
  for (var iter = 0; iter < nIter; iter++) {
    for (var j = 1; j < NY - 1; j++) {
      for (var i = 0; i < NX; i++) {
        var k = cpuI(i, j);
        if (!mask[k]) continue;
        var ip1 = (i + 1) % NX;
        var im1 = (i - 1 + NX) % NX;
        var res = cx * (deepPsi[cpuI(ip1, j)] + deepPsi[cpuI(im1, j)])
                + cy * (deepPsi[cpuI(i, j + 1)] + deepPsi[cpuI(i, j - 1)])
                + cc * deepPsi[k] - deepZeta[k];
        deepPsi[k] -= omegaSOR * res * invCC;
      }
    }
  }
}

function cpuJacobian(i, j) {
  var cl = cpuCosLat(j);
  var ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;
  var e = cpuI(ip1, j), w = cpuI(im1, j), n = cpuI(i, j + 1), s = cpuI(i, j - 1);
  var ne = cpuI(ip1, j + 1), nw = cpuI(im1, j + 1), se = cpuI(ip1, j - 1), sw = cpuI(im1, j - 1);
  var J1 = (psi[e] - psi[w]) * (zeta[n] - zeta[s]) - (psi[n] - psi[s]) * (zeta[e] - zeta[w]);
  var J2 = psi[e] * (zeta[ne] - zeta[se]) - psi[w] * (zeta[nw] - zeta[sw]) - psi[n] * (zeta[ne] - zeta[nw]) + psi[s] * (zeta[se] - zeta[sw]);
  var J3 = zeta[e] * (psi[ne] - psi[se]) - zeta[w] * (psi[nw] - psi[sw]) - zeta[n] * (psi[ne] - psi[nw]) + zeta[s] * (psi[se] - psi[sw]);
  return (J1 + J2 + J3) / (12 * dx * cl * dy);
}

function cpuLaplacian(f, i, j) {
  var cl = cpuCosLat(j);
  var ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;
  var k = cpuI(i, j);
  return invDx2 / (cl * cl) * (f[cpuI(ip1, j)] + f[cpuI(im1, j)] - 2 * f[k])
       + invDy2 * (f[cpuI(i, j + 1)] + f[cpuI(i, j - 1)] - 2 * f[k]);
}

function cpuTimestep() {
  // Vorticity timestep with buoyancy coupling — periodic in x
  for (var j = 1; j < NY - 1; j++) for (var i = 0; i < NX; i++) {
    var ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;
    var k = cpuI(i, j);
    if (!mask[k]) { cpuZetaNew[k] = 0; cpuTempNew[k] = 0; cpuDeepTempNew[k] = 0; continue; }
    // Vorticity update
    if (!mask[cpuI(ip1, j)] || !mask[cpuI(im1, j)] || !mask[cpuI(i, j + 1)] || !mask[cpuI(i, j - 1)]) {
      cpuZetaNew[k] = zeta[k] * 0.9;
    } else if (!mask[cpuI(ip1, j + 1)] || !mask[cpuI(im1, j + 1)] || !mask[cpuI(ip1, j - 1)] || !mask[cpuI(im1, j - 1)]) {
      cpuZetaNew[k] = zeta[k] * 0.95;
    } else {
      var jac = cpuJacobian(i, j);
      var cl = cpuCosLat(j);
      var invDxPhys = invDx / cl;
      var betaV = cpuBeta(j) * (psi[cpuI(ip1, j)] - psi[cpuI(im1, j)]) * 0.5 * invDxPhys;
      var F = cpuWindCurl(i, j);
      var fric = -r_friction * zeta[k];
      var visc = A_visc * cpuLaplacian(zeta, i, j);
      var dRhodx_cpu = -alpha_T * (temp[cpuI(ip1, j)] - temp[cpuI(im1, j)]) + beta_S * (sal[cpuI(ip1, j)] - sal[cpuI(im1, j)]);
      var buoyancy = -dRhodx_cpu * 0.5 * invDxPhys;
      var coupling = F_couple_s * (deepPsi[k] - psi[k]);
      cpuZetaNew[k] = zeta[k] + dt * (-jac - betaV + F + fric + visc + buoyancy + coupling);
    }

    // Temperature equation
    var ke = cpuI(ip1, j), kw = cpuI(im1, j), kn = cpuI(i, j + 1), ks = cpuI(i, j - 1);
    var tE = mask[ke] ? temp[ke] : temp[k];
    var tW = mask[kw] ? temp[kw] : temp[k];
    var tN = mask[kn] ? temp[kn] : temp[k];
    var tS = mask[ks] ? temp[ks] : temp[k];
    var pE = mask[ke] ? psi[ke] : psi[k];
    var pW = mask[kw] ? psi[kw] : psi[k];
    var pN = mask[kn] ? psi[kn] : psi[k];
    var pS = mask[ks] ? psi[ks] : psi[k];

    var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    var y = j / (NY - 1);
    var clT = cpuCosLat(j);
    var invDxT = invDx / clT;
    var dPdx = (pE - pW) * 0.5 * invDxT;
    var dPdy = (pN - pS) * 0.5 * invDy;
    var dTdx = (tE - tW) * 0.5 * invDxT;
    var dTdy = (tN - tS) * 0.5 * invDy;
    var geoAdvec = dPdx * dTdy - dPdy * dTdx;
    var u_ek = ekmanField ? ekmanField[k] * windStrength : 0;
    var v_ek = ekmanField ? ekmanField[k + NX * NY] * windStrength : 0;
    var advec = geoAdvec + u_ek * dTdx + v_ek * dTdy;
    var latRad = lat * Math.PI / 180;
    var yearPhase = 2 * Math.PI * simTime / T_YEAR;
    var declination = 23.44 * Math.sin(yearPhase) * Math.PI / 180;
    var cosZenith = Math.cos(latRad) * Math.cos(declination) + Math.sin(latRad) * Math.sin(declination);
    var qSolar = S_solar * Math.max(0, cosZenith);
    if (Math.abs(lat) > 45) {
      var iceT2 = Math.max(0, Math.min(1, (temp[k] + 2) / 10));
      var iceFrac2 = 1 - iceT2 * iceT2 * (3 - 2 * iceT2);
      var latRamp = Math.max(0, Math.min(1, (Math.abs(lat) - 45) / 20));
      qSolar *= 1.0 - 0.50 * iceFrac2 * latRamp;
    }
    // Cloud parameterization (regime-based)
    var absLat = Math.abs(lat);
    var humidity = Math.max(0, Math.min(1, (temp[k] - 5) / 25));
    var airTempEst = 28 - 0.55 * absLat;
    var lts = Math.max(0, Math.min(1, (airTempEst - temp[k]) / 15));
    var itczLat = 5 * Math.sin(yearPhase);
    var itczDist = (lat - itczLat) / 10;
    var convCloud = 0.30 * Math.exp(-itczDist * itczDist) * humidity;
    var warmPool = 0.20 * Math.max(0, Math.min(1, (temp[k] - 26) / 4));
    var subDist = (absLat - 25) / 10;
    var subsidence = 0.25 * Math.exp(-subDist * subDist);
    var stratocu = 0.30 * lts * Math.max(0, Math.min(1, (35 - absLat) / 20));
    var stormTrack = 0.25 * Math.max(0, Math.min(1, (absLat - 35) / 10)) * Math.max(0, Math.min(1, (80 - absLat) / 15));
    var soCloud = lat < 0 ? 0.35 * Math.max(0, Math.min(1, (absLat - 45) / 10)) : 0;
    var polarCloud = 0.15 * Math.max(0, Math.min(1, (absLat - 55) / 15));
    var highCloud = convCloud + warmPool;
    var lowCloud = stratocu + stormTrack + soCloud + polarCloud;
    var cloudFrac = Math.max(0.05, Math.min(0.85, highCloud + lowCloud - subsidence * (1 - humidity)));
    var convFrac = cloudFrac > 0.05 ? Math.max(0, Math.min(1, highCloud / (highCloud + lowCloud + 0.01))) : 0;
    var cloudAlbedo = cloudFrac * (0.35 * (1 - convFrac) + 0.20 * convFrac);
    qSolar *= 1 - cloudAlbedo;
    var olr = A_olr - B_olr * globalTempOffset + B_olr * temp[k];
    var cloudGreenhouse = cloudFrac * (0.03 * (1 - convFrac) + 0.12 * convFrac);
    var qNet = qSolar - olr * (1 - cloudGreenhouse);
    var lapT = invDx2 / (clT * clT) * (tE + tW - 2 * temp[k]) + invDy2 * (tN + tS - 2 * temp[k]);
    var diff = kappa_diff * lapT;
    var nLand = (!mask[ke] ? 1 : 0) + (!mask[kw] ? 1 : 0) + (!mask[kn] ? 1 : 0) + (!mask[ks] ? 1 : 0);
    var nOcean = 4 - nLand;
    var landFlux = 0;
    if (nLand > 0 && nOcean > 0) {
      // Use seasonal land temp (includes albedo + lapse rate) if available
      var landT = (landTempField && landTempField[k] !== 0) ? landTempField[k] : (50 * Math.max(0, cosZenith) - 20);
      var rawFlux = 0.02 * (landT - temp[k]) * (nOcean / 4);
      landFlux = Math.max(-0.5, Math.min(0.5, rawFlux));
    }
    cpuTempNew[k] = temp[k] + dt * (-advec + qNet + diff + landFlux);

    // ── CPU SALINITY ──
    var sE = mask[ke] ? sal[ke] : sal[k];
    var sW = mask[kw] ? sal[kw] : sal[k];
    var sN2 = mask[kn] ? sal[kn] : sal[k];
    var sS2 = mask[ks] ? sal[ks] : sal[k];
    var dSdx = (sE - sW) * 0.5 * invDxT;
    var dSdy = (sN2 - sS2) * 0.5 * invDy;
    var salAdvec = dPdx * dSdy - dPdy * dSdx + u_ek * dSdx + v_ek * dSdy;
    var lapS = invDx2 / (clT * clT) * (sE + sW - 2 * sal[k]) + invDy2 * (sN2 + sS2 - 2 * sal[k]);
    var salDiff = kappa_sal * lapS;
    var latRad = lat * Math.PI / 180;
    var salClimObs = salClimatologyField ? salClimatologyField[k] : 0;
    var salClim = (salClimObs > 1) ? salClimObs : (34.0 + 2.0 * Math.cos(2 * latRad) - 0.5 * Math.cos(4 * latRad));
    var salRestore = salRestoringRate * (salClim - sal[k]);
    var fwSal = 0;
    if (y > 0.75) fwSal = -freshwaterForcing * 3.0 * (y - 0.75) * 4.0;
    cpuSalNew[k] = sal[k] + dt * (-salAdvec + salDiff + salRestore + fwSal);

    // Variable mixed layer depth
    var cabsLat2 = Math.abs(lat);
    var mldBase = 30 + 70 * Math.pow(cabsLat2 / 80, 1.5);
    var mldACC = 0, mldSub = 0;
    if (lat < -35 && lat > -65) { var d = (lat + 50) / 12; mldACC = 250 * Math.exp(-d * d); }
    if (lat > 50 && lat < 75) { var d = (lat - 62) / 8; mldSub = 150 * Math.exp(-d * d); }
    var mixedLayerDepth = mldBase + mldACC + mldSub;

    // Two-layer vertical exchange
    var localDepth = depth ? depth[k] : 4000;
    var hSurf = Math.min(mixedLayerDepth, localDepth);
    var hDeep = Math.max(1, localDepth - mixedLayerDepth);
    var hasDeep = localDepth > mixedLayerDepth ? 1 : 0;

    var rhoSurf = -alpha_T * temp[k] + beta_S * sal[k];
    var rhoDeep = -alpha_T * deepTemp[k] + beta_S * deepSal[k];
    var gamma = gamma_mix;
    if (Math.abs(lat) > 40 && rhoSurf > rhoDeep) gamma = gamma_deep_form;

    var vertExchangeT = gamma * (temp[k] - deepTemp[k]) * hasDeep;
    cpuTempNew[k] -= dt * vertExchangeT / hSurf;
    var vertExchangeS = gamma * (sal[k] - deepSal[k]) * hasDeep;
    cpuSalNew[k] -= dt * vertExchangeS / hSurf;

    // Deep layer
    var dE = mask[ke] ? deepTemp[ke] : deepTemp[k];
    var dW = mask[kw] ? deepTemp[kw] : deepTemp[k];
    var dN = mask[kn] ? deepTemp[kn] : deepTemp[k];
    var dS = mask[ks] ? deepTemp[ks] : deepTemp[k];
    var lapDeep = invDx2 * (dE + dW - 2 * deepTemp[k]) + invDy2 * (dN + dS - 2 * deepTemp[k]);
    var deepDiff = kappa_deep * lapDeep;
    cpuDeepTempNew[k] = deepTemp[k] + dt * (vertExchangeT / hDeep + deepDiff) * hasDeep;

    var dsE = mask[ke] ? deepSal[ke] : deepSal[k];
    var dsW = mask[kw] ? deepSal[kw] : deepSal[k];
    var dsN = mask[kn] ? deepSal[kn] : deepSal[k];
    var dsS = mask[ks] ? deepSal[ks] : deepSal[k];
    var lapDeepSal = invDx2 * (dsE + dsW - 2 * deepSal[k]) + invDy2 * (dsN + dsS - 2 * deepSal[k]);
    var deepSalDiff = kappa_deep_sal * lapDeepSal;
    cpuDeepSalNew[k] = deepSal[k] + dt * (vertExchangeS / hDeep + deepSalDiff) * hasDeep;
  }
  var tmp = zeta; zeta = cpuZetaNew; cpuZetaNew = tmp;
  var tmpT = temp; temp = cpuTempNew; cpuTempNew = tmpT;
  var tmpD = deepTemp; deepTemp = cpuDeepTempNew; cpuDeepTempNew = tmpD;
  var tmpS = sal; sal = cpuSalNew; cpuSalNew = tmpS;
  var tmpDS = deepSal; deepSal = cpuDeepSalNew; cpuDeepSalNew = tmpDS;
  for (var k = 0; k < NX * NY; k++) { if (!mask[k]) { psi[k] = 0; zeta[k] = 0; temp[k] = 0; deepTemp[k] = 0; sal[k] = 0; deepSal[k] = 0; deepPsi[k] = 0; deepZeta[k] = 0; } }

  // ── ATMOSPHERE: 1-layer energy balance (two-way coupled) ──
  // Air temp evolves via: exchange with ocean/land surface + meridional diffusion
  // Then feeds back to ocean SST (atmosphere→ocean, gentle)
  if (airTemp) {
    var airNew = new Float64Array(NX * NY);
    for (var aj = 1; aj < NY - 1; aj++) {
      for (var ai = 0; ai < NX; ai++) {
        var ak = aj * NX + ai;
        var aip = (ai + 1) % NX, aim = (ai - 1 + NX) % NX;
        var aE = airTemp[aj * NX + aip], aW = airTemp[aj * NX + aim];
        var aN = airTemp[(aj + 1) * NX + ai], aS = airTemp[(aj - 1) * NX + ai];
        // Meridional diffusion (represents Hadley/Ferrel/polar cells)
        var lapAir = invDx2 * (aE + aW - 2 * airTemp[ak]) + invDy2 * (aN + aS - 2 * airTemp[ak]);
        var airDiff = kappa_atm * lapAir;
        // Exchange with surface below — use seasonal land temp if available
        var surfT;
        if (mask[ak]) {
          surfT = temp[ak];
        } else if (landTempField && landTempField[ak] !== 0) {
          surfT = landTempField[ak];
        } else {
          var alat = LAT0 + (aj / (NY - 1)) * (LAT1 - LAT0);
          surfT = 28 - 0.55 * Math.abs(alat);
        }
        var gamma = mask[ak] ? gamma_oa : gamma_la;
        var exchange = gamma * (surfT - airTemp[ak]);
        airNew[ak] = airTemp[ak] + dt * (airDiff + exchange);
      }
    }
    // Polar boundaries: copy from neighbor
    for (var ai = 0; ai < NX; ai++) { airNew[ai] = airNew[NX + ai]; airNew[(NY-1)*NX+ai] = airNew[(NY-2)*NX+ai]; }
    for (var ak = 0; ak < NX * NY; ak++) airTemp[ak] = airNew[ak];
    // Two-way feedback: atmosphere warms/cools ocean surface
    // Gentle effect — ocean has ~1000x more thermal inertia than atmosphere
    for (var ak = 0; ak < NX * NY; ak++) {
      if (mask[ak]) {
        temp[ak] += dt * gamma_ao * (airTemp[ak] - temp[ak]);
      }
    }
  }

  cpuSolveFFT(psi, zeta);

  // Deep layer vorticity
  for (var j = 1; j < NY - 1; j++) for (var i = 0; i < NX; i++) {
    var ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;
    var k2 = cpuI(i, j);
    if (!mask[k2]) { cpuDeepZetaNew[k2] = 0; continue; }
    var ke2 = cpuI(ip1, j), kw2 = cpuI(im1, j), kn2 = cpuI(i, j+1), ks2 = cpuI(i, j-1);
    if (!mask[ke2] || !mask[kw2] || !mask[kn2] || !mask[ks2]) {
      cpuDeepZetaNew[k2] = deepZeta[k2] * 0.9; continue;
    }
    var dPdx2 = (deepPsi[ke2] - deepPsi[kw2]) * 0.5 * invDx;
    var dPdy2 = (deepPsi[kn2] - deepPsi[ks2]) * 0.5 * invDy;
    var dZdx2 = (deepZeta[ke2] - deepZeta[kw2]) * 0.5 * invDx;
    var dZdy2 = (deepZeta[kn2] - deepZeta[ks2]) * 0.5 * invDy;
    var jac2 = dPdx2 * dZdy2 - dPdy2 * dZdx2;
    var betaV2 = cpuBeta(j) * (deepPsi[ke2] - deepPsi[kw2]) * 0.5 * invDx;
    var fric2 = -r_deep * deepZeta[k2];
    var lapZ2 = invDx2 * (deepZeta[ke2] + deepZeta[kw2] - 2 * deepZeta[k2])
              + invDy2 * (deepZeta[kn2] + deepZeta[ks2] - 2 * deepZeta[k2]);
    var visc2 = A_visc * lapZ2;
    var coupling2 = F_couple_d * (psi[k2] - deepPsi[k2]);
    var dRhodxDeep2 = -alpha_T * (deepTemp[ke2] - deepTemp[kw2]) + beta_S * (deepSal[ke2] - deepSal[kw2]);
    var deepBuoyancy2 = dRhodxDeep2 * 0.5 * invDx;
    var deepTN2 = mask[kn2] ? deepTemp[kn2] : deepTemp[k2];
    var deepTS2 = mask[ks2] ? deepTemp[ks2] : deepTemp[k2];
    var dTdyDeep2 = (deepTN2 - deepTS2) * 0.5 * invDy;
    var motTendency2 = 0.05 * dTdyDeep2;
    cpuDeepZetaNew[k2] = deepZeta[k2] + dt * (-jac2 - betaV2 + fric2 + visc2 + coupling2 + deepBuoyancy2 + motTendency2);
  }
  var tmpDZ = deepZeta; deepZeta = cpuDeepZetaNew; cpuDeepZetaNew = tmpDZ;
  for (var k3 = 0; k3 < NX * NY; k3++) { if (!mask[k3]) { deepPsi[k3] = 0; deepZeta[k3] = 0; } }
  cpuSolveFFT(deepPsi, deepZeta);

  totalSteps++;
  simTime += dt * yearSpeed;
}

function cpuReset() {
  psi = new Float64Array(NX * NY);
  zeta = new Float64Array(NX * NY);
  cpuZetaNew = new Float64Array(NX * NY);
  temp = new Float64Array(NX * NY);
  cpuTempNew = new Float64Array(NX * NY);
  deepTemp = new Float64Array(NX * NY);
  cpuDeepTempNew = new Float64Array(NX * NY);
  sal = new Float64Array(NX * NY);
  cpuSalNew = new Float64Array(NX * NY);
  deepSal = new Float64Array(NX * NY);
  cpuDeepSalNew = new Float64Array(NX * NY);
  deepPsi = new Float64Array(NX * NY);
  deepZeta = new Float64Array(NX * NY);
  cpuDeepZetaNew = new Float64Array(NX * NY);
  initTemperatureField();
  totalSteps = 0;
  simTime = 0;
}

// ============================================================
// VELOCITY / PARTICLES (shared, CPU-side from readback data)
// ============================================================
function getVel(fi, fj) {
  var i = Math.floor(fi);
  var j = Math.min(Math.max(Math.floor(fj), 1), NY - 2);
  i = ((i % NX) + NX) % NX;
  var ip1 = (i + 1) % NX;
  var im1 = (i - 1 + NX) % NX;
  var cl = cpuCosLat(j);
  return [
    -(psi[(j + 1) * NX + i] - psi[(j - 1) * NX + i]) * 0.5 * invDy,
    (psi[j * NX + ip1] - psi[j * NX + im1]) * 0.5 * invDx / cl
  ];
}

function spawnInOcean() {
  var x, y, tries = 0;
  do {
    x = Math.random() * NX;
    y = 2 + Math.random() * (NY - 4);
    tries++;
  } while (!mask[Math.floor(y) * NX + (Math.floor(x) % NX)] && tries < 100);
  return [x, y];
}

function initParticles() {
  for (var p = 0; p < NP; p++) {
    var pos = spawnInOcean();
    px[p] = pos[0]; py[p] = pos[1];
    page_[p] = Math.floor(Math.random() * MAX_AGE);
  }
}

function resetParticle(p) {
  var pos = spawnInOcean();
  px[p] = pos[0]; py[p] = pos[1]; page_[p] = 0;
}

function advectParticles() {
  var dtA = dt * stepsPerFrame;
  for (var p = 0; p < NP; p++) {
    var vel = getVel(px[p], py[p]);
    px[p] += vel[0] * dtA * invDx;
    py[p] += vel[1] * dtA * invDy;
    if (px[p] >= NX) px[p] -= NX;
    if (px[p] < 0) px[p] += NX;
    page_[p]++;
    var gi = Math.floor(px[p]), gj = Math.floor(py[p]);
    gi = ((gi % NX) + NX) % NX;
    if (gj < 1 || gj >= NY - 1 || !mask[gj * NX + gi] || page_[p] > MAX_AGE)
      resetParticle(p);
  }
}

// ============================================================
// STABILITY CHECK
// ============================================================
function stabilityCheck() {
  var maxV = 0;
  var maxZeta = 0;
  var blownUp = false;
  var N = NX * NY;

  for (var k = 0; k < N; k++) {
    if (!mask[k]) continue;

    var az = Math.abs(zeta[k]);
    if (az > maxZeta) maxZeta = az;

    if (az > 500) { zeta[k] = zeta[k] > 0 ? 500 : -500; blownUp = true; }

    if (temp[k] > 40) temp[k] = 40;
    else if (temp[k] < -10) temp[k] = -10;
    if (deepTemp[k] > 30) deepTemp[k] = 30;
    else if (deepTemp[k] < -2) deepTemp[k] = -2;

    if (zeta[k] !== zeta[k] || psi[k] !== psi[k] || temp[k] !== temp[k]) {
      zeta[k] = 0; psi[k] = 0; temp[k] = 0; deepTemp[k] = 0;
      blownUp = true;
    }
  }

  for (var j = 1; j < NY - 1; j += 2) for (var ii = 1; ii < NX - 1; ii += 4) {
    if (!mask[j * NX + ii]) continue;
    var vel = getVel(ii, j);
    var s2 = vel[0] * vel[0] + vel[1] * vel[1];
    if (s2 > maxV) maxV = s2;
  }
  maxV = Math.sqrt(maxV);

  var dx_ = 1.0 / (NX - 1);
  var dtTarget = maxV > 0 ? Math.min(dtBase, 0.3 * dx_ / maxV) : dtBase;
  dt = 0.7 * dt + 0.3 * dtTarget;

  if (maxZeta > 200) {
    var damp = 200 / maxZeta;
    for (var k3 = 0; k3 < N; k3++) { if (mask[k3]) zeta[k3] *= damp; }
    blownUp = true;
  }

  return blownUp;
}
