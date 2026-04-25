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
let stepsPerFrame = 3;         // CPU+FFT at 512x160: ~19ms/step → ~17fps
let paused = false;
let dt = 5e-5;                 // conservative dt (was 1.5e-5 for 1024x512, 5e-4 blew up)
let dtBase = 5e-5;
let totalSteps = 0;
let showField = 'temp';
let showParticles = true;

// Temperature / thermohaline parameters
let S_solar = 6.5;            // solar heating amplitude (tuned for regime-based clouds)
let A_olr = 1.8;              // OLR constant
let B_olr = 0.13;             // OLR linear coefficient (increased for stronger radiation feedback)
let kappa_diff = 3.0e-4;      // thermal diffusion (increased for poleward heat transport)
let alpha_T = 0.05;            // buoyancy coupling
// Two-layer ocean
let H_surface = 100;           // surface layer depth (m)
let H_deep = 4000;             // deep layer depth (m)
let gamma_mix = 0.0007;        // base vertical mixing rate (reduced to decrease upwelling cooling in mid-lats)
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
let moisture;                 // atmospheric specific humidity field (kg/kg, ~0-0.025)
let precipField;              // precipitation rate field (mm/day equivalent, diagnostic)
let cloudField;               // cloud fraction field (0-1), updated each readback
let obsCloudField;            // observed cloud fraction (MODIS), static
let ekmanField;               // Ekman velocity (u_ek, v_ek) stacked, nondimensional
let kappa_atm = 3e-3;        // atmospheric heat diffusion
let gamma_oa = 0.005;        // ocean→atmosphere heat exchange rate
let gamma_ao = 0.001;        // atmosphere→ocean feedback (gentler — ocean has much more thermal inertia)
let gamma_la = 0.01;         // land→atmosphere heat exchange rate

// Moisture parameters
let E0 = 0.003;              // evaporation rate coefficient (kg/kg per timestep, tunable)
let greenhouse_q = 0.4;      // water vapor greenhouse (matched to GPU shader)
let q_ref = 0.015;           // reference specific humidity for greenhouse scaling
let freshwaterScale_pe = 0.5; // P-E salinity flux strength (PSU per unit precip)

// GPU physics scaling (tunable, uploaded to Params struct slots 33-35)
let evapScale = 0.8;           // evaporative cooling strength (0 = off, 0.8 ≈ 80 W/m² global mean)
let peScale = 0.3;             // P-E salinity flux strength (0 = off)
let snowAlbedoScale = 0.45;    // snow albedo boost (bare→snow, 0 = off, 0.45 ≈ 15%→60%)

// Grid sizes (both power-of-2 for radix-2 FFT Poisson solver)
const GPU_NX = 1024, GPU_NY = 512;
const CPU_NX = 1024, CPU_NY = 512;
let NX, NY, dx, dy, invDx, invDy, invDx2, invDy2;
let cellW, cellH;             // rendering cell dimensions (set by init functions)

// Mask source dimensions (mask_1024x512.json for high-res, mask.json for ≤512)
const MASK_SRC_NX = GPU_NX, MASK_SRC_NY = GPU_NY;
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
// LOAD DATA — binary Float32 from data/bin/ (fast ArrayBuffer, no JSON.parse)
// ============================================================
var DATA_BASE = '../data/bin/';

function flipVerticalFloat32(arr,nx,ny){var t=new Float32Array(nx);for(var j=0;j<(ny>>1);j++){var T=j*nx,B=(ny-1-j)*nx;for(var i=0;i<nx;i++)t[i]=arr[T+i];for(var i=0;i<nx;i++)arr[T+i]=arr[B+i];for(var i=0;i<nx;i++)arr[B+i]=t[i];}}

function loadBinData(baseName) {
  // Load metadata JSON (tiny) + binary Float32 arrays
  return fetch(DATA_BASE + baseName + '.json').then(function(r) { return r.json(); }).then(function(meta) {
    if (!meta || !meta.arrays) return null;
    var promises = [];
    var keys = Object.keys(meta.arrays);
    for (var i = 0; i < keys.length; i++) {
      (function(key) {
        promises.push(
          fetch(DATA_BASE + meta.arrays[key].file)
            .then(function(r) { return r.arrayBuffer(); })
            .then(function(buf) { var a=new Float32Array(buf); if(meta.nx&&meta.ny&&a.length===meta.nx*meta.ny) flipVerticalFloat32(a,meta.nx,meta.ny); meta[key]=a; })
        );
      })(keys[i]);
    }
    return Promise.all(promises).then(function() { return meta; });
  }).catch(function() { return null; });
}

let maskLoadPromise = fetch(DATA_BASE + 'mask.json').then(function(r) { return r.json(); }).then(function(d) {
  if (!d) return;
  var bits = [];
  for (var c = 0; c < d.hex.length; c++) {
    var v = parseInt(d.hex[c], 16);
    bits.push((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1);
  }
  if(d.nx&&d.ny&&bits.length>=d.nx*d.ny){var rt=new Array(d.nx);for(var j=0;j<(d.ny>>1);j++){var T=j*d.nx,B=(d.ny-1-j)*d.nx;for(var i=0;i<d.nx;i++)rt[i]=bits[T+i];for(var i=0;i<d.nx;i++)bits[T+i]=bits[B+i];for(var i=0;i<d.nx;i++)bits[B+i]=rt[i];}}
maskSrcBits = bits;
}).catch(function() {});

let coastLoadPromise = fetch('coastlines.json').then(function(r) { return r.json(); }).then(function(p) {
  LAND_POLYS = p;
}).catch(function() {});

// Observational data (all 1024x512, loaded as Float32Array from binary)
let obsSSTData = null;
let obsDeepData = null;
let obsBathyData = null;
let obsSalinityData = null;
let obsWindData = null;
let obsAlbedoData = null;
let obsPrecipData = null;
let obsCloudData = null;
let obsSeaIceData = null;
let obsAirTempData = null;
let obsLSTData = null;
let obsEvapData = null;
let obsCurrentsData = null;
let sstLoadPromise = loadBinData('sst').then(function(d) { obsSSTData = d; });
let deepLoadPromise = loadBinData('deep_temp').then(function(d) { obsDeepData = d; });
let bathyLoadPromise = loadBinData('bathymetry').then(function(d) { obsBathyData = d; });
let salinityLoadPromise = loadBinData('salinity').then(function(d) { obsSalinityData = d; });
let windLoadPromise = loadBinData('wind_stress').then(function(d) { obsWindData = d; });
let albedoLoadPromise = loadBinData('albedo').then(function(d) { obsAlbedoData = d; });
let precipLoadPromise = loadBinData('precipitation').then(function(d) { obsPrecipData = d; });
let cloudLoadPromise = loadBinData('cloud_fraction').then(function(d) { obsCloudData = d; });
let seaIceLoadPromise = loadBinData('sea_ice').then(function(d) { obsSeaIceData = d; });
let airTempLoadPromise = loadBinData('air_temp').then(function(d) { obsAirTempData = d; });
let lstLoadPromise = loadBinData('land_surface_temp').then(function(d) { obsLSTData = d; });
let evapLoadPromise = loadBinData('evaporation').then(function(d) { obsEvapData = d; });
let currentsLoadPromise = loadBinData('ocean_currents').then(function(d) { obsCurrentsData = d; });
let obsSnowData = null;
let snowLoadPromise = loadBinData('snow_cover').then(function(d) { obsSnowData = d; });

// ============================================================
// MASK HELPERS
// ============================================================
// Convert a data array to Float32Array. Data is already at 1024x512 — no resampling needed.
// Kept as a function for legacy compatibility and type conversion.
function toFloat32(srcArray) {
  if (!srcArray) return null;
  if (srcArray instanceof Float32Array) return srcArray;
  var out = new Float32Array(srcArray.length);
  for (var k = 0; k < srcArray.length; k++) out[k] = srcArray[k] || 0;
  return out;
}

var remappedElevation = null;
var remappedAlbedo = null;
var remappedPrecip = null;
var remappedLST = null;
var remappedSeaIce = null;
var remappedEvap = null;

function buildRemappedFields() {
  if (obsBathyData && obsBathyData.elevation) remappedElevation = toFloat32(obsBathyData.elevation);
  if (obsAlbedoData && obsAlbedoData.albedo) remappedAlbedo = toFloat32(obsAlbedoData.albedo);
  if (obsPrecipData && obsPrecipData.precipitation) remappedPrecip = toFloat32(obsPrecipData.precipitation);
  if (obsLSTData && obsLSTData.lst) remappedLST = toFloat32(obsLSTData.lst);
  if (obsSeaIceData && obsSeaIceData.ice_fraction) remappedSeaIce = toFloat32(obsSeaIceData.ice_fraction);
  if (obsEvapData && obsEvapData.evaporation) remappedEvap = toFloat32(obsEvapData.evaporation);
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
  // Mask data is at 1024x512 matching model grid — direct copy or nearest-neighbor if different
  var srcNX = MASK_SRC_NX, srcNY = MASK_SRC_NY;
  for (var j = 0; j < ny; j++) {
    var sj = Math.min(Math.floor(j * srcNY / ny), srcNY - 1);
    for (var i = 0; i < nx; i++) {
      var si = Math.min(Math.floor(i * srcNX / nx), srcNX - 1);
      m[j * nx + i] = maskSrcBits[sj * srcNX + si] || 0;
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
'  _padS0: u32, evapScale: f32, peScale: f32, snowAlbedo: f32,',
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

// ============================================================
// GPU FFT POISSON SOLVER SHADERS
// Three stages: FFT rows → tridiagonal solve per mode → inverse FFT rows
// ============================================================

// Stage 1 & 3: Radix-2 butterfly pass (one pass per dispatch, 9 dispatches for N=512)
// Params via uniform: passStride (1,2,4,...,256), direction (+1=forward, -1=inverse), nx, ny
var fftButterflyShaderCode = [
'struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };',
'@group(0) @binding(0) var<storage, read_write> re: array<f32>;',
'@group(0) @binding(1) var<storage, read_write> im: array<f32>;',
'@group(0) @binding(2) var<uniform> p: FFTParams;',
'',
'@compute @workgroup_size(64)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let row = id.x / (p.nx / 2u);',  // which row (0..NY-1)
'  let halfIdx = id.x % (p.nx / 2u);',  // which butterfly in this row
'  if (row >= p.ny) { return; }',
'',
'  let stride = p.passStride;',
'  let halfLen = stride;',
'  let fullLen = stride * 2u;',
'',
'  // Which butterfly group and position within it',
'  let group = halfIdx / halfLen;',
'  let j = halfIdx % halfLen;',
'  let base = row * p.nx + group * fullLen;',
'  let i0 = base + j;',
'  let i1 = base + j + halfLen;',
'',
'  // Twiddle factor: exp(direction * -2πi * j / fullLen)',
'  let ang = p.direction * 2.0 * 3.14159265358979 * f32(j) / f32(fullLen);',
'  let wR = cos(ang);',
'  let wI = sin(ang);',
'',
'  let uR = re[i0]; let uI = im[i0];',
'  let vR = re[i1] * wR - im[i1] * wI;',
'  let vI = re[i1] * wI + im[i1] * wR;',
'  re[i0] = uR + vR; im[i0] = uI + vI;',
'  re[i1] = uR - vR; im[i1] = uI - vI;',
'}'
].join('\n');

// Bit-reversal permutation (run once before butterfly passes)
var fftBitRevShaderCode = [
'struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };',
'@group(0) @binding(0) var<storage, read_write> re: array<f32>;',
'@group(0) @binding(1) var<storage, read_write> im: array<f32>;',
'@group(0) @binding(2) var<uniform> p: FFTParams;',
'',
'@compute @workgroup_size(64)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let row = id.x / p.nx;',
'  let i = id.x % p.nx;',
'  if (row >= p.ny) { return; }',
'',
'  // Compute bit-reversed index for log2(nx) bits',
'  var rev = 0u;',
'  var val = i;',
'  var bits = 0u;',
'  var tmp = p.nx >> 1u;',
'  while (tmp > 0u) { bits++; tmp >>= 1u; }',
'  for (var b = 0u; b < bits; b++) {',
'    rev = (rev << 1u) | (val & 1u);',
'    val >>= 1u;',
'  }',
'',
'  if (i < rev) {',
'    let base = row * p.nx;',
'    let tR = re[base + i]; let tI = im[base + i];',
'    re[base + i] = re[base + rev]; im[base + i] = im[base + rev];',
'    re[base + rev] = tR; im[base + rev] = tI;',
'  }',
'}'
].join('\n');

// Stage 2: Tridiagonal solve per Fourier mode (Thomas algorithm)
// One workgroup per mode, sequential in j within each workgroup
var fftTridiagShaderCode = [
'struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };',
'@group(0) @binding(0) var<storage, read_write> reIn: array<f32>;',   // zeta_hat real (mode-major: [m*NY+j])
'@group(0) @binding(1) var<storage, read_write> imIn: array<f32>;',   // zeta_hat imag
'@group(0) @binding(2) var<storage, read_write> reOut: array<f32>;',  // psi_hat real
'@group(0) @binding(3) var<storage, read_write> imOut: array<f32>;',  // psi_hat imag
'@group(0) @binding(4) var<uniform> p: FFTParams;',
'@group(0) @binding(5) var<storage, read> cosLatArr: array<f32>;',    // precomputed cos(lat) per row
'',
'@compute @workgroup_size(1)',  // one workgroup per mode (sequential Thomas)
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let m = id.x;',
'  if (m >= p.nx) { return; }',
'  let ny = p.ny;',
'  let dx = 1.0 / f32(p.nx - 1u);',
'  let dy = 1.0 / f32(ny - 1u);',
'  let invDy2 = 1.0 / (dy * dy);',
'  let invDx2 = 1.0 / (dx * dx);',
'',
'  // Eigenvalue for mode m: km2 = invDx2 * 2 * (cos(2πm/NX) - 1)',
'  let km2 = invDx2 * 2.0 * (cos(2.0 * 3.14159265358979 * f32(m) / f32(p.nx)) - 1.0);',
'',
'  // Build tridiagonal system: a*ψ_{j-1} + b_j*ψ_j + c*ψ_{j+1} = ζ̂_j',
'  // a = c = invDy2, b_j = km2/cos²(lat_j) - 2*invDy2',
'  // Boundary: ψ(0) = ψ(NY-1) = 0',
'',
'  // Thomas algorithm — forward elimination',
'  // We reuse the input arrays as scratch (overwrite diagonal and RHS)',
'  // Store b in reOut[m*ny+j], dR in reIn[m*ny+j], dI in imIn[m*ny+j]',
'  var b_prev = 1.0;',  // b[0] = 1 (boundary)
'  var dR_prev = 0.0;',
'  var dI_prev = 0.0;',
'',
'  // Store b values in reOut temporarily',
'  reOut[m * ny] = 1.0;',  // b[0]
'  imOut[m * ny] = 0.0;',
'',
'  for (var j = 1u; j < ny; j++) {',
'    var b_j: f32;',
'    var rhs_r: f32;',
'    var rhs_i: f32;',
'    if (j < ny - 1u) {',
'      // Grid Laplacian: no cos(lat) — consistent with ζ = ∇²_grid ψ',
'      let _cl = cosLatArr[j];', // keep binding alive to avoid WGSL validation error
'      b_j = km2 - 2.0 * invDy2;',
'      rhs_r = reIn[m * ny + j];',
'      rhs_i = imIn[m * ny + j];',
'    } else {',
'      b_j = 1.0;',  // boundary
'      rhs_r = 0.0;',
'      rhs_i = 0.0;',
'    }',
'    let a = invDy2;',
'    let cp = select(0.0, invDy2, j - 1u > 0u && j - 1u < ny - 1u);',
'    let w = a / b_prev;',
'    b_j -= w * cp;',
'    rhs_r -= w * dR_prev;',
'    rhs_i -= w * dI_prev;',
'    reOut[m * ny + j] = b_j;',  // store b for back-sub
'    b_prev = b_j;',
'    dR_prev = rhs_r;',
'    dI_prev = rhs_i;',
'    reIn[m * ny + j] = rhs_r;',  // overwrite with modified RHS
'    imIn[m * ny + j] = rhs_i;',
'  }',
'',
'  // Back substitution',
'  let last = ny - 1u;',
'  reOut[m * ny + last] = reIn[m * ny + last] / reOut[m * ny + last];',
'  imOut[m * ny + last] = imIn[m * ny + last] / reOut[m * ny + last];',
'  for (var jj = 1u; jj < ny; jj++) {',
'    let j = last - jj;',
'    let c = select(0.0, invDy2, j > 0u && j < ny - 1u);',
'    let b_j = reOut[m * ny + j];',
'    reOut[m * ny + j] = (reIn[m * ny + j] - c * reOut[m * ny + j + 1u]) / b_j;',
'    imOut[m * ny + j] = (imIn[m * ny + j] - c * imOut[m * ny + j + 1u]) / b_j;',
'  }',
'}'
].join('\n');

// Transpose: row-major [j*NX+i] ↔ mode-major [m*NY+j] (for tridiagonal stage)
var fftTransposeShaderCode = [
'struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };',
'@group(0) @binding(0) var<storage, read> src: array<f32>;',
'@group(0) @binding(1) var<storage, read_write> dst: array<f32>;',
'@group(0) @binding(2) var<uniform> p: FFTParams;',
'',
'@compute @workgroup_size(64)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let idx = id.x;',
'  if (idx >= p.nx * p.ny) { return; }',
'  // Row-major to mode-major: src[j*nx+i] → dst[i*ny+j]',
'  let j = idx / p.nx;',
'  let i = idx % p.nx;',
'  dst[i * p.ny + j] = src[j * p.nx + i];',
'}'
].join('\n');

// Scale + copy back: apply 1/N for inverse FFT and copy to psi buffer, zeroing land
var fftScaleMaskShaderCode = [
'struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };',
'@group(0) @binding(0) var<storage, read> re: array<f32>;',
'@group(0) @binding(1) var<storage, read_write> psi: array<f32>;',
'@group(0) @binding(2) var<storage, read> mask: array<u32>;',
'@group(0) @binding(3) var<uniform> p: FFTParams;',
'',
'@compute @workgroup_size(64)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let k = id.x;',
'  if (k >= p.nx * p.ny) { return; }',
'  psi[k] = select(0.0, clamp(re[k] / f32(p.nx), -50.0, 50.0), mask[k] != 0u);',
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
'@group(0) @binding(10) var<storage, read> snowCover: array<f32>;',
'@group(0) @binding(11) var<storage, read> seaIceFrac: array<f32>;',
'@group(0) @binding(12) var<storage, read> evapRate: array<f32>;',
'@group(0) @binding(13) var<storage, read> precipRate: array<f32>;',
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
'',
'  // Sea ice: blend observed NOAA ice fraction with SST-based fallback',
'  let obsIce = seaIceFrac[k];',
'  let sstIceT = clamp((tempIn[k] + 2.0) / 10.0, 0.0, 1.0);',
'  let sstIceFrac = 1.0 - sstIceT * sstIceT * (3.0 - 2.0 * sstIceT);',
'  let iceFrac = select(sstIceFrac, obsIce, obsIce > 0.001);',
'  if (abs(lat) > 45.0) {',
'    let latRamp = clamp((abs(lat) - 45.0) / 20.0, 0.0, 1.0);',
'    qSolar *= 1.0 - 0.50 * iceFrac * latRamp;',
'  }',
'',
'  // Snow-albedo on land',
'  let snowFrac = snowCover[k];',
'  if (snowFrac > 0.01) { qSolar *= 1.0 - params.snowAlbedo * snowFrac; }',
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
'  // SH subtropics need stronger subsidence suppression — 30-40S is too cold, more sun needed',
'  let subDist = (absLat - 25.0) / 10.0;',
'  let subsidenceBase = 0.25 * exp(-subDist * subDist);',
'  // Extra subsidence suppression in SH subtropics 25-40S (anticyclonic belt)',
'  let shSubDist = (lat + 32.0) / 10.0;',
'  let shSubExtra = select(0.0, 0.12 * exp(-shSubDist * shSubDist), lat < 0.0 && lat > -50.0);',
'  let subsidence = subsidenceBase + shSubExtra;',
'',
'  // 4. Marine stratocumulus (cold SST + stable air, subtropics)',
'  let stratocu = 0.30 * lts * clamp((35.0 - absLat) / 20.0, 0.0, 1.0);',
'',
'  // 5. Mid-latitude storm track (40-75 deg)',
'  // NH gets an extra boost at 35-55N (observed 0.65-0.75 in N Atlantic/Pacific storm tracks)',
'  let nhStormBoost = select(0.0, 0.22 * clamp((absLat - 35.0) / 10.0, 0.0, 1.0)',
'                                     * clamp((58.0 - absLat) / 12.0, 0.0, 1.0), lat > 0.0);',
'  let stormTrack = 0.30 * clamp((absLat - 35.0) / 10.0, 0.0, 1.0)',
'                       * clamp((80.0 - absLat) / 15.0, 0.0, 1.0) + nhStormBoost;',
'',
'  // 6. Southern Ocean boundary layer clouds (observed ~0.85 at 55-65S)',
'  // Gaussian peak centered at 62S — ACC forcing drives persistent low cloud deck',
'  // Note: do NOT let this spread too far north into 30-45S (already too cold there)',
'  let soDist = (absLat - 62.0) / 7.0;',
'  let soCloud = select(0.0,',
'    0.70 * exp(-soDist * soDist) + 0.18 * clamp((absLat - 53.0) / 8.0, 0.0, 1.0),',
'    lat < 0.0);',
'',
'  // 7. Polar stratus (both hemispheres)',
'  let polarCloud = 0.10 * clamp((absLat - 60.0) / 10.0, 0.0, 1.0);',
'',
'  // Combine: high clouds (convective) + low clouds (stratiform) - subsidence',
'  // Stratocu is capped to prevent cold-SST runaway in SH subtropics (30-40S):',
'  // excessive LTS-driven clouds over cold model SST cause a cold-amplifying feedback.',
'  let stratocuCapped = clamp(stratocu, 0.0, 0.20);',
'  let highCloud = convCloud + warmPool;',
'  let lowCloud = stratocuCapped + stormTrack + soCloud + polarCloud;',
'  let cloudFrac = clamp(highCloud + lowCloud - subsidence * (1.0 - humidity), 0.05, 0.90);',
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
'  // Southern Ocean OLR enhancement: extra cooling south of 58S (dry polar air, less greenhouse)',
'  // Only apply at high southern latitudes — 30-50S is already too cold, don\'t add cooling there.',
'  let soOlrMult = select(1.0, 1.0 + 0.55 * clamp((absLat - 58.0) / 8.0, 0.0, 1.0), lat < -53.0);',
'  olr *= soOlrMult;',
'  // LW greenhouse: high clouds trap more (0.12) than low clouds (0.03)',
'  let cloudGreenhouse = cloudFrac * (0.03 * (1.0 - convFrac) + 0.12 * convFrac);',
'  // Water vapor greenhouse: Clausius-Clapeyron moisture at 80% RH',
'  let qSat = 3.75e-3 * exp(0.067 * tempIn[k]);',
'  let vaporGH = 0.4 * clamp(0.8 * qSat / 0.015, 0.0, 1.0);',
'  let effectiveOlr = olr * (1.0 - cloudGreenhouse) * (1.0 - vaporGH);',
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
'  let evapCool = params.evapScale * evapRate[k];',
'  tempOut[k] = tempIn[k] + params.dt * (-advec + qNet + diff + landFlux - evapCool);',
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
'  let peFlux = params.peScale * (evapRate[k] - precipRate[k]) * tempIn[salK] / 35.0;',
'  var fwSal: f32 = 0.0;',
'  if (y > 0.75) {',
'    fwSal = -params.freshwater * 3.0 * (y - 0.75) * 4.0;',
'  }',
'',
'  tempOut[salK] = tempIn[salK] + params.dt * (-salAdvec + salDiff + salRestore + fwSal + peFlux);',
'',
'  // ── DENSITY-BASED DEEP WATER FORMATION ──',
'  let rhoSurf = -params.alphaT * tempIn[k] + params.betaS * tempIn[salK];',
'  let rhoDeep = -params.alphaT * deepTempIn[k] + params.betaS * deepTempIn[k + N];',
'',
'  var gamma = params.gammaMix;',
'  // Deep water formation:',
'  //   NH: NADW forms at >40N (N Atlantic deep water, physically correct)',
'  //   SH: AABW forms only in Weddell/Ross seas (>62S). Do NOT trigger at 40-62S',
'  //   which is SH mid-lats already too cold in model.',
'  let isDeepFormRegion = (lat > 40.0 && rhoSurf > rhoDeep)',
'                       || (lat < -62.0 && rhoSurf > rhoDeep);',
'  if (isDeepFormRegion) { gamma = params.gammaDeepForm; }',
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
    var remapped = toFloat32(obsBathyData.depth);
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
    // Data is already at 1024x512 — direct copy
    var srcSal = obsSalinityData.salinity;
    for (var k = 0; k < NX * NY; k++) {
      var s = srcSal[k] || 0;
      if (s > 1) {
        salClimatologyField[k] = s;
      } else {
        var j = Math.floor(k / NX);
        var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
        var latRad = lat * Math.PI / 180;
        salClimatologyField[k] = 34.0 + 2.0 * Math.cos(2 * latRad) - 0.5 * Math.cos(4 * latRad);
      }
    }
    console.log('Using WOA23 observed salinity climatology (1024x512)');
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
    // Data is now 1024x512 matching model grid
    var obsNX = NX, obsNY = NY;
    var obsLat0 = LAT0, obsLat1 = LAT1;
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

    // Data is at 1024x512 matching model grid — direct copy with scaling
    for (var k = 0; k < NX * NY; k++) {
      windCurlFieldData[k] = (obsWindData.wind_curl[k] || 0) * windCurlScale;
    }
    console.log('Using ERA5 observed wind stress curl (1024x512, pre-scaled to model units)');
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
  obsCloudField = toFloat32(obsCloudData.cloud_fraction);
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

    // Data is at 1024x512 matching model grid — direct read
    // Compute RMS of Ekman velocity to find nondimensional scaling
    var ekRms2 = 0, ekN = 0;
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var latRad = lat * Math.PI / 180;
      var f = 2 * OMEGA * Math.sin(latRad);
      if (Math.abs(lat) < 5) continue;
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        var tx = tauX[k] || 0, ty = tauY[k] || 0;
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
        var tx = tauX[k] || 0, ty = tauY[k] || 0;
        var ue = ty / (RHO * f * H_EK) * ekScale;
        var ve = -tx / (RHO * f * H_EK) * ekScale;
        ekmanField[k] = ue;
        ekmanField[k + NX * NY] = ve;
      }
    }
    console.log('Ekman velocities from ERA5 wind stress (scale: ' + ekScale.toExponential(3) + ', RMS: ' + (ekRms * ekScale).toFixed(4) + ')');
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

// ============================================================
// TEMPERATURE / SALINITY INITIALIZATION
// ============================================================
function initTemperatureField() {
  var useObs = obsSSTData && obsSSTData.sst;
  var useDeepObs = obsDeepData && obsDeepData.temp;
  var sstRemapped = useObs ? toFloat32(obsSSTData.sst) : null;
  var deepRemapped = useDeepObs ? toFloat32(obsDeepData.temp) : null;

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
  moisture = new Float64Array(NX * NY);
  precipField = new Float64Array(NX * NY);
  buildRemappedFields();
  generateDepthField();
  generateWindCurlField();
  generateObsCloudField();
  generateEkmanField();
  initTemperatureField();
  // Initialize air temp from ERA5 observed 2m temperature if available
  var useObsAirTemp = obsAirTempData && obsAirTempData.air_temp;
  for (var ai = 0; ai < NX * NY; ai++) {
    if (useObsAirTemp) {
      airTemp[ai] = obsAirTempData.air_temp[ai] || 15;
    } else if (mask[ai]) {
      airTemp[ai] = temp[ai];
    } else {
      var aj = Math.floor(ai / NX);
      var alat = LAT0 + (aj / (NY - 1)) * (LAT1 - LAT0);
      airTemp[ai] = 28 - 0.55 * Math.abs(alat);
    }
    moisture[ai] = 0.80 * qSat(airTemp[ai]);
  }
  if (useObsAirTemp) console.log('Air temp initialized from ERA5 2m temperature');
}

// Clausius-Clapeyron: saturation specific humidity as function of temperature (°C)
// Linearized form: q_sat(T) = q0 * exp(L * T / (Rv * T0^2))
// With q0=3.75e-3 kg/kg at 0°C, L/Rv/T0^2 ≈ 0.067 per °C
function qSat(T) {
  return 3.75e-3 * Math.exp(0.067 * T);
}

// Initialize streamfunction from GODAS observed surface currents
// Computes vorticity from u,v, scales to model units, solves Poisson for psi
function initCirculationFromObs() {
  if (!obsCurrentsData || !obsCurrentsData.u || !obsCurrentsData.v) return;
  var u = obsCurrentsData.u, v = obsCurrentsData.v;
  if (u.length !== NX * NY) return; // grid mismatch

  // Compute vorticity: zeta = dv/dx - du/dy (on the model grid)
  var obsZeta = new Float64Array(NX * NY);
  for (var j = 1; j < NY - 1; j++) {
    for (var i = 0; i < NX; i++) {
      var k = j * NX + i;
      if (!mask[k]) continue;
      var ip = (i + 1) % NX, im = (i - 1 + NX) % NX;
      var ke = j * NX + ip, kw = j * NX + im;
      var kn = (j + 1) * NX + i, ks = (j - 1) * NX + i;
      if (!mask[ke] || !mask[kw] || !mask[kn] || !mask[ks]) continue;
      var dvdx = (v[ke] - v[kw]) * 0.5 * invDx;
      var dudy = (u[kn] - u[ks]) * 0.5 * invDy;
      obsZeta[k] = dvdx - dudy;
    }
  }

  // Scale observed vorticity to match model's nondimensional units
  // RMS-match: find scale factor so RMS(obsZeta * scale) ≈ RMS(windCurlField)
  var rmsObs = 0, rmsWind = 0, nObs = 0;
  for (var k = 0; k < NX * NY; k++) {
    if (!mask[k]) continue;
    rmsObs += obsZeta[k] * obsZeta[k];
    if (windCurlFieldData) rmsWind += windCurlFieldData[k] * windCurlFieldData[k];
    nObs++;
  }
  if (nObs === 0 || rmsObs === 0) return;
  rmsObs = Math.sqrt(rmsObs / nObs);
  rmsWind = Math.sqrt(rmsWind / nObs);
  // Scale observed vorticity to model units
  // Use ~1x wind curl magnitude (circulation builds from forcing, not imposed at full strength)
  var scale = rmsWind > 0 ? rmsWind / rmsObs : 0.0002 / rmsObs;

  for (var k = 0; k < NX * NY; k++) {
    zeta[k] = mask[k] ? obsZeta[k] * scale : 0;
  }

  // Solve Poisson to get psi from zeta
  cpuSolveFFT(psi, zeta);

  // Deep layer: initialize with weaker, opposite-sign flow (thermohaline return)
  for (var k = 0; k < NX * NY; k++) {
    if (mask[k]) {
      deepPsi[k] = -0.15 * psi[k];
      deepZeta[k] = -0.15 * zeta[k];
    }
  }
  cpuSolveFFT(deepPsi, deepZeta);

  console.log('Circulation initialized from GODAS observed currents (scale=' + scale.toExponential(2) + ')');
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
      b[j] = km2 - 2 * invDy2; // grid Laplacian (no cos(lat)) — consistent with ζ = ∇²_grid ψ
      dR[j] = hatR[m*NY+j]; dI[j] = hatI[m*NY+j];
    }
    // Thomas forward elimination (skip boundary rows 0 and NY-1)
    for (var j = 1; j < NY - 1; j++) {
      var cp = (j-1 > 0) ? invDy2 : 0; // c[j-1]: 0 for boundary row 0, invDy2 for interior
      var w = invDy2 / b[j-1];          // a[j] = invDy2 (sub-diagonal)
      b[j] -= w * cp;
      dR[j] -= w * dR[j-1]; dI[j] -= w * dI[j-1];
    }
    // Thomas back substitution
    pHR[m*NY+(NY-1)] = 0; // boundary: ψ = 0
    pHI[m*NY+(NY-1)] = 0;
    for (var j = NY-2; j >= 1; j--) {
      var c = invDy2; // super-diagonal for interior rows
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
  // Note: psi over land is non-physical but harmless — vorticity equation skips land cells,
  // and velocity is only computed where mask=1. Zeroing land psi would create discontinuities
  // that corrupt the Laplacian at coastlines.
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
    var subsidenceBase = 0.25 * Math.exp(-subDist * subDist);
    var shSubDist = (lat + 32) / 10;
    var shSubExtra = (lat < 0 && lat > -50) ? 0.12 * Math.exp(-shSubDist * shSubDist) : 0;
    var subsidence = subsidenceBase + shSubExtra;
    var stratocu = 0.30 * lts * Math.max(0, Math.min(1, (35 - absLat) / 20));
    var nhStormBoost = lat > 0 ? 0.22 * Math.max(0, Math.min(1, (absLat - 35) / 10)) * Math.max(0, Math.min(1, (58 - absLat) / 12)) : 0;
    var stormTrack = 0.30 * Math.max(0, Math.min(1, (absLat - 35) / 10)) * Math.max(0, Math.min(1, (80 - absLat) / 15)) + nhStormBoost;
    var soDist = (absLat - 62) / 7;
    var soCloud = lat < 0 ? (0.70 * Math.exp(-soDist * soDist) + 0.18 * Math.max(0, Math.min(1, (absLat - 53) / 8))) : 0;
    var polarCloud = 0.10 * Math.max(0, Math.min(1, (absLat - 60) / 10));
    var stratocuCapped = Math.max(0, Math.min(0.20, stratocu));
    var highCloud = convCloud + warmPool;
    var lowCloud = stratocuCapped + stormTrack + soCloud + polarCloud;
    var cloudFrac = Math.max(0.05, Math.min(0.90, highCloud + lowCloud - subsidence * (1 - humidity)));
    var convFrac = cloudFrac > 0.05 ? Math.max(0, Math.min(1, highCloud / (highCloud + lowCloud + 0.01))) : 0;
    var cloudAlbedo = cloudFrac * (0.35 * (1 - convFrac) + 0.20 * convFrac);
    qSolar *= 1 - cloudAlbedo;
    var olr = A_olr - B_olr * globalTempOffset + B_olr * temp[k];
    // Southern Ocean OLR enhancement: extra cooling south of 58S only
    // 30-50S is already too cold — don't apply OLR enhancement there.
    var soOlrMult = lat < -53 ? (1.0 + 0.55 * Math.max(0, Math.min(1, (absLat - 58) / 8))) : 1.0;
    olr *= soOlrMult;
    var cloudGreenhouse = cloudFrac * (0.03 * (1 - convFrac) + 0.12 * convFrac);
    // Water vapor greenhouse: moist air traps more longwave (strongest feedback in real climate)
    var vaporGreenhouse = moisture ? greenhouse_q * Math.min(1, moisture[k] / q_ref) : 0;
    var qNet = qSolar - olr * (1 - cloudGreenhouse) * (1 - vaporGreenhouse);
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
    // Deep water formation:
    //   NH: NADW forms at >40N (physically correct for AMOC)
    //   SH: AABW forms only in Weddell/Ross seas (>62S). Do NOT trigger at 40-62S
    //   which is SH mid-lats already too cold in the model.
    var isDeepFormRegion = (lat > 40 && rhoSurf > rhoDeep) || (lat < -62 && rhoSurf > rhoDeep);
    if (isDeepFormRegion) gamma = gamma_deep_form;

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
  // Zero fields on land — but NOT psi/deepPsi (FFT solver needs continuous values across coastlines)
  for (var k = 0; k < NX * NY; k++) { if (!mask[k]) { zeta[k] = 0; temp[k] = 0; deepTemp[k] = 0; sal[k] = 0; deepSal[k] = 0; deepZeta[k] = 0; } }

  // ── ATMOSPHERE: 1-layer energy balance + moisture (two-way coupled) ──
  // Air temp evolves via: exchange with surface + diffusion + latent heat release
  // Moisture evolves via: evaporation from ocean + diffusion - condensation
  if (airTemp && moisture) {
    var airNew = new Float64Array(NX * NY);
    var qNew = new Float64Array(NX * NY);
    for (var aj = 1; aj < NY - 1; aj++) {
      for (var ai = 0; ai < NX; ai++) {
        var ak = aj * NX + ai;
        var aip = (ai + 1) % NX, aim = (ai - 1 + NX) % NX;
        var aE = airTemp[aj * NX + aip], aW = airTemp[aj * NX + aim];
        var aN = airTemp[(aj + 1) * NX + ai], aS = airTemp[(aj - 1) * NX + ai];
        // Meridional diffusion for temperature
        var lapAir = invDx2 * (aE + aW - 2 * airTemp[ak]) + invDy2 * (aN + aS - 2 * airTemp[ak]);
        var airDiff = kappa_atm * lapAir;
        // Moisture diffusion (same coefficient — carried by same large-scale motions)
        var qE = moisture[aj * NX + aip], qW = moisture[aj * NX + aim];
        var qN = moisture[(aj + 1) * NX + ai], qS = moisture[(aj - 1) * NX + ai];
        var lapQ = invDx2 * (qE + qW - 2 * moisture[ak]) + invDy2 * (qN + qS - 2 * moisture[ak]);
        var qDiff = kappa_atm * lapQ;
        // Exchange with surface below
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
        // Evaporation: only over ocean, driven by saturation deficit and wind
        var evap = 0;
        if (mask[ak]) {
          var qs = qSat(surfT);
          var deficit = Math.max(0, qs - moisture[ak]);
          evap = E0 * deficit;  // kg/kg per timestep
        }
        // Update moisture: diffusion + evaporation
        qNew[ak] = moisture[ak] + dt * (qDiff) + evap;
        // Condensation: if supersaturated, excess precipitates
        var qs_air = qSat(airTemp[ak]);
        var precip = 0;
        if (qNew[ak] > qs_air) {
          precip = qNew[ak] - qs_air;
          qNew[ak] = qs_air;
        }
        // Clamp moisture to physical range
        qNew[ak] = Math.max(1e-5, qNew[ak]);
        // Store precipitation for diagnostics and salinity feedback
        precipField[ak] = precip;
        // Latent heat release from condensation warms atmosphere
        // L/c_p ≈ 2500 K per kg/kg, but in nondimensional units scale down
        var latentHeat = 800 * precip;
        // Update air temperature: diffusion + surface exchange + latent heat
        airNew[ak] = airTemp[ak] + dt * (airDiff + exchange) + latentHeat;
      }
    }
    // Polar boundaries: copy from neighbor
    for (var ai = 0; ai < NX; ai++) {
      airNew[ai] = airNew[NX + ai]; airNew[(NY-1)*NX+ai] = airNew[(NY-2)*NX+ai];
      qNew[ai] = qNew[NX + ai]; qNew[(NY-1)*NX+ai] = qNew[(NY-2)*NX+ai];
    }
    for (var ak = 0; ak < NX * NY; ak++) { airTemp[ak] = airNew[ak]; moisture[ak] = qNew[ak]; }
    // Two-way feedback: atmosphere warms/cools ocean surface
    // Also: evaporative cooling removes latent heat from ocean, precipitation freshens salinity
    for (var ak = 0; ak < NX * NY; ak++) {
      if (mask[ak]) {
        temp[ak] += dt * gamma_ao * (airTemp[ak] - temp[ak]);
        // Evaporative cooling: ocean loses latent heat when water evaporates
        var qs = qSat(temp[ak]);
        var deficit = Math.max(0, qs - moisture[ak]);
        var evapCool = E0 * deficit * 400;  // nondimensional latent heat loss
        temp[ak] -= evapCool;
        // P-E salinity flux: precipitation freshens, evaporation concentrates
        var netFW = precipField[ak] - E0 * deficit;  // net freshwater (positive = freshening)
        sal[ak] -= dt * freshwaterScale_pe * netFW;
      }
    }
  }

  cpuSolveFFT(psi, zeta);   // exact solve on full rectangle (land psi is non-physical but harmless)

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
  for (var k3 = 0; k3 < NX * NY; k3++) { if (!mask[k3]) { deepZeta[k3] = 0; } }
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
  // Reset atmosphere fields
  if (airTemp) {
    for (var ai = 0; ai < NX * NY; ai++) {
      if (mask[ai]) airTemp[ai] = temp[ai];
      else {
        var aj = Math.floor(ai / NX);
        var alat = LAT0 + (aj / (NY - 1)) * (LAT1 - LAT0);
        airTemp[ai] = 28 - 0.55 * Math.abs(alat);
      }
    }
  }
  moisture = new Float64Array(NX * NY);
  precipField = new Float64Array(NX * NY);
  if (airTemp) {
    for (var ai = 0; ai < NX * NY; ai++) {
      moisture[ai] = 0.80 * qSat(airTemp[ai]);
    }
  }
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

    // Clamp psi to prevent FFT Poisson solver runaway
    var ap = Math.abs(psi[k]);
    if (ap > 50) { psi[k] = psi[k] > 0 ? 50 : -50; blownUp = true; }

    if (temp[k] > 40) temp[k] = 40;
    else if (temp[k] < -10) temp[k] = -10;
    if (deepTemp[k] > 30) deepTemp[k] = 30;
    else if (deepTemp[k] < -2) deepTemp[k] = -2;

    // Clamp deep layer too
    if (deepPsi && Math.abs(deepPsi[k]) > 50) { deepPsi[k] = deepPsi[k] > 0 ? 50 : -50; blownUp = true; }
    if (deepZeta && Math.abs(deepZeta[k]) > 500) { deepZeta[k] = deepZeta[k] > 0 ? 500 : -500; blownUp = true; }

    if (zeta[k] !== zeta[k] || psi[k] !== psi[k] || temp[k] !== temp[k]) {
      zeta[k] = 0; psi[k] = 0; temp[k] = 0; deepTemp[k] = 0;
      if (deepPsi) deepPsi[k] = 0;
      if (deepZeta) deepZeta[k] = 0;
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
