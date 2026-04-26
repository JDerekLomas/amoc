// ============================================================
// OCEAN CIRCULATION SIMULATION ENGINE
// ============================================================
//
// A real-time global ocean circulation model solving the barotropic vorticity
// equation with two-layer thermohaline dynamics, dynamic sea ice, and
// interactive land-ocean coupling. Runs on WebGPU with CPU fallback.
//
// PHYSICS OVERVIEW
// ────────────────
// The model solves four coupled systems on a 1024×512 equirectangular grid
// covering 360° longitude × 159° latitude (-79.5°S to 79.5°N):
//
// 1. BAROTROPIC VORTICITY EQUATION (surface + deep layers)
//    ∂ζ/∂t + J(ψ,ζ) + β·∂ψ/∂x = curl(τ) - r·ζ + A·∇²ζ + buoyancy + coupling
//    where ζ = ∇²ψ (solved via Poisson equation each timestep)
//    - Wind-driven gyres from 3-belt wind pattern (trades, westerlies, polar easterlies)
//    - Density-driven buoyancy: -α·∂T/∂x + β·∂S/∂x (thermohaline component)
//    - Two layers coupled by interfacial drag
//
// 2. TEMPERATURE (surface + deep layers)
//    ∂T/∂t = -J(ψ,T) + Q_solar - OLR(T) + κ·∇²T + vertical mixing + land flux
//    - Seasonal solar forcing with cos(zenith angle) and declination
//    - Linearized OLR: A + B·T (restoring toward radiative equilibrium)
//    - Ice-albedo feedback (ice reflects ~50% of incoming solar)
//    - Two-way land-ocean heat exchange using actual land temperatures
//
// 3. SALINITY (surface + deep layers)
//    ∂S/∂t = -J(ψ,S) + κ·∇²S + restoring + freshwater forcing
//    - Initialized from WOA observed salinity
//    - Weak restoring toward climatology prevents drift
//    - Brine rejection / meltwater from sea ice phase changes
//
// 4. SEA ICE (thermodynamic, no dynamics/transport)
//    Ice thickness h evolves via Stefan problem:
//    - Freezing: dh/dt = k·(T_freeze - T)·(1-h/h_max),  SST clamped at T_freeze
//    - Melting:  dh/dt = -k·(T - T_freeze)·h,            latent heat absorbs warmth
//    - Insulation: thick ice reduces ocean-atmosphere heat flux
//    - Phase change acts as latent heat buffer (SST holds at -1.8°C while ice grows/melts)
//
// POISSON SOLVER
// ──────────────
// The ζ = ∇²ψ inversion is the computational bottleneck. At 1024×512, iterative
// SOR doesn't converge (spectral radius 0.994, needs ~750 iterations). Solution:
//
//   COARSE-GRID POISSON: Downsample ζ to 512×256 via GPU shader, solve with
//   40 red-black SOR iterations (converges at this resolution), bilinear
//   upsample ψ back to 1024×512. Three GPU dispatches replace 160+ SOR dispatches.
//   Trade-off: western boundary currents slightly smoothed.
//
//   Future: GPU FFT Poisson solver (issue #43) — 12 dispatches, exact solution.
//
// LAND-OCEAN COUPLING
// ───────────────────
// - Ocean → Land: coastal land cells blend ~30% of adjacent ocean SST into their
//   equilibrium temperature (maritime climate effect: Gulf Stream warms Europe)
// - Land → Ocean: coastal ocean cells exchange heat with actual land temperature
//   field (hot Sahara warms Mediterranean, cold Siberia cools Sea of Okhotsk)
// - Land temperature: solar-driven with seasonal cycle, altitude lapse rate
//   (-6.5°C/km from ETOPO1 elevation), and thermal inertia
//
// DATA SOURCES
// ────────────
// All observational data at 1024×512 from data/ directory (north-first row order):
//   sst.json          — NOAA OI SST v2 (1991-2020 annual mean) → initial SST
//   deep_temp.json    — WOA23 at 1000m depth → initial deep temperature
//   bathymetry.json   — ETOPO1 ocean depth + land elevation
//   salinity.json     — WOA23 surface salinity → initial salinity
//   sea_ice.json      — NOAA observed ice fraction → initial ice thickness
//   mask.json         — 1024×512 land/ocean mask
//   wind_stress.json  — NCEP wind stress curl (loaded, not yet wired)
//   mixed_layer_depth.json — observed MLD (loaded, not yet wired)
//
// ARCHITECTURE
// ────────────
// GPU path (preferred): WebGPU compute shaders for timestep, temperature,
//   boundary conditions, Poisson. Fragment shader for field rendering.
//   CPU only handles: particles, ice overlay, land rendering, diagnostics.
//
// CPU path (fallback): Full physics in JavaScript. Slower but functional.
//
// Rendering: GPU fragment shader draws SST/vorticity/speed/streamfunction.
//   CPU canvas overlays: land with elevation coloring, sea ice, particles,
//   grid lines, contours, labels.
//
// CODE STRUCTURE (4178 lines)
// ──────────────────────────
//   Lines    1- 100: Parameters (documented with physical basis audit)
//   Lines  100- 200: Data loading (from data/ directory, north-first → flip)
//   Lines  200- 260: Mask building + strait opening
//   Lines  260- 330: Map underlay rendering
//   Lines  330- 920: WGSL compute shaders (timestep, Poisson, BC, temperature, deep)
//   Lines  920-1070: GPU state variables
//   Lines 1070-1200: initWebGPU() — buffers, pipelines, bind groups, coarse grid
//   Lines 1200-1400: Data field initialization (depth, temperature, salinity, ice)
//   Lines 1400-1700: GPU render pipeline + params upload
//   Lines 1700-2000: gpuRunSteps() — batched GPU dispatch with coarse Poisson
//   Lines 2000-2200: GPU readback + ice update
//   Lines 2200-2600: CPU physics (vorticity, temperature, salinity, ice, deep layer)
//   Lines 2600-2850: CPU rendering (field colormaps, land temperature)
//   Lines 2850-3050: draw() — CPU field rendering fallback
//   Lines 3050-3200: drawOverlay() — ice, particles, grid, contours
//   Lines 3200-3350: Profile/diagnostic panels
//   Lines 3350-3450: gpuTick() / cpuTick() — main loop
//   Lines 3450-4178: Stability checks, paint tools, initialization, event handlers
//
// ============================================================

// ============================================================
// SIMULATION PARAMETERS
// ============================================================
// This model solves the nondimensional barotropic vorticity equation on a
// unit square [0,1]×[0,1] mapped to the global ocean (LON0..LON1, LAT0..LAT1).
// The nondimensionalization uses L (basin width) as length scale and 1/(β₀L)
// as time scale. Physical correspondence of parameters is noted where known.
//
// PARAMETER AUDIT (2026-04-26):
//   ✓ = physically derived    ~ = reasonable order of magnitude    ✗ = tuned/ad hoc
// ============================================================

// --- Vorticity equation parameters ---
let beta = 1.0;               // ✓ Normalized Coriolis gradient (= 1 by definition of nondimensionalization)
let r_friction = 0.04;        // ~ Bottom friction (Rayleigh drag). Real: r ~ 10⁻⁷ s⁻¹.
                               //   Nondimensional r_nd = r_phys * L / U. Value is stability-tuned.
let A_visc = 2e-4;            // ~ Lateral viscosity. Real: A_H = 10²–10⁴ m²/s.
                               //   Nondimensional A_nd = A_H / (U*L). Value is stability-tuned.
let windStrength = 1.0;       //   Wind stress multiplier (1.0 = default analytical or observed)
let doubleGyre = true;
let stepsPerFrame = 10;        // Each step = ~90 GPU dispatches. 10 steps = 900 dispatches = smooth 60fps.
let paused = false;
let dt = 2e-4;                // ~ Timestep. CFL: u*dt/dx < 1. At max velocity ~5, dx ~0.001: need dt < 2e-4.
let dtBase = 2e-4;
let totalSteps = 0;
let showField = 'temp';
let showParticles = true;

// --- Radiative heat balance ---
// dT/dt = Q_solar(lat,t) - OLR(T) + diffusion + advection
// Q_solar = S_solar * max(0, cos(zenith_angle))
// OLR = A_olr + B_olr * T  (linearized Stefan-Boltzmann)
// Equilibrium: T_eq = (Q_solar - A_olr) / B_olr
//   Equator: (5-2)/0.1 = 30°C,  60°N: (2.5-2)/0.1 = 5°C  → reasonable SST range
//
// Real physics: mean solar at surface ~240 W/m², OLR feedback ~1.9 W/m²/K (CMIP5 consensus),
//   SST restoring timescale ~30-50 years. Our B_olr=0.1 gives ~1 model-year restoring — too fast.
//   This is a deliberate speedup for interactive simulation.
let S_solar = 5.0;            // ✗ Solar amplitude (nondimensional, tuned for equilibrium SSTs)
let A_olr = 2.0;              // ✗ OLR constant (nondimensional, tuned for ~28°C equatorial equilibrium)
let B_olr = 0.1;              // ✗ OLR feedback (nondimensional). Real: ~1.9 W/m²/K. Ours is ~10x too fast.

// --- Thermal diffusion ---
let kappa_diff = 2.5e-4;      // ~ Horizontal thermal diffusion. Real K_H ~ 10³ m²/s.
                               //   Stability-tuned to smooth grid-scale noise.

// --- Buoyancy coupling (thermohaline drive) ---
// Density: ρ = ρ₀(1 - α*T + β*S)
// Real seawater: α ≈ 2×10⁻⁴ K⁻¹, β ≈ 7.5×10⁻⁴ PSU⁻¹, ratio α/β ≈ 0.27
// Our ratio: alpha_T/beta_S should be ~0.27 for correct thermohaline dynamics.
// Previously: 0.05/0.8 = 0.0625 — salinity was 4x too strong vs temperature.
let alpha_T = 0.15;           // ~ Thermal expansion coupling. Increased from 0.05 to fix α/β ratio.
let beta_S = 0.55;            // ~ Haline contraction coupling. Reduced from 0.8 to fix α/β ratio.
                               //   New ratio: 0.15/0.55 = 0.27, matching observed α/β.

// --- Two-layer ocean ---
let H_surface = 100;          // ✓ Surface mixed layer depth (m). Real: 20–200m, mean ~80m. [100m reasonable]
let H_deep = 4000;            // ✓ Deep layer depth (m). Mean ocean depth ~3700m.
let gamma_mix = 0.001;        // ✗ Base vertical mixing rate. No clear physical derivation.
                               //   Real diapycnal diffusivity K_v ~ 10⁻⁵ m²/s → γ ~ K_v/H² ~ 10⁻⁸ s⁻¹.
let gamma_deep_form = 0.05;   // ✗ Enhanced mixing at deep water formation sites.
                               //   50x background. Real convective mixing: 100–1000x background.
let kappa_deep = 2e-5;        // ~ Deep horizontal diffusion (10x weaker than surface, physically reasonable).

// --- Two-layer circulation coupling ---
let F_couple_s = 0.5;         // ✗ Interfacial drag felt by surface layer. Ad hoc.
let F_couple_d = 0.0125;      // ✗ Interfacial drag felt by deep layer. = H_s/H_d * F_couple_s.
                               //   The ratio is physically motivated (momentum conservation).
let r_deep = 0.1;             // ✗ Deep bottom friction. Ad hoc, higher than surface r_friction.

// --- Seasonal cycle ---
let yearSpeed = 1.0;          //   Seasonal cycle speed multiplier
let freshwaterForcing = 0.0;  //   External freshwater flux (hosing experiment parameter)
let globalTempOffset = 0.0;   //   Radiative forcing offset (°C, for CO₂ experiments)
let simTime = 0;
let T_YEAR = 10.0;            //   Simulation time units per year (nondimensional)

// --- Field arrays ---
let temp;                     // Surface temperature (°C)
let deepTemp;                 // Deep ocean temperature (°C)
let cpuDeepTempNew;
let sal;                      // Surface salinity (PSU)
let deepSal;                  // Deep salinity (PSU)
let cpuSalNew;
let cpuDeepSalNew;

// --- Salinity parameters ---
let kappa_sal = 2.5e-4;       // ~ Salinity diffusion (same as thermal — physically they're similar)
let kappa_deep_sal = 2e-5;    // ~ Deep salinity diffusion
let salRestoringRate = 0.005; // ✗ Nudging toward climatology. Modeling convenience, not physics.
                               //   Prevents salinity from drifting in long runs.

// --- Circulation fields ---
let deepPsi;
let deepZeta;
let cpuDeepZetaNew;
let depth;                    // Ocean depth from ETOPO1 bathymetry (m)
let seaIce;                   // Sea ice thickness (m), 0 = open water
let windCurlField;            // Precomputed wind curl per cell (observed or analytical)
let airTemp;                  // Atmospheric temperature field (°C), diffuses between land and ocean
let amocStrength = 0;

// --- Atmosphere parameters ---
// Uses Jacobi smoothing (unconditionally stable) instead of explicit diffusion.
// See atmosphereStep() for the algorithm. Parameters defined there:
//   ATM_SMOOTH_PASSES = 5    — smoothing radius (~2-3 cells per readback)
//   ATM_SURFACE_RELAX = 0.15 — how fast air snaps to surface temp
//   ATM_FEEDBACK_SST  = 0.003 — how strongly air modifies SST (very gentle)
//   ATM_FEEDBACK_LAND = 0.05  — how strongly air modifies land temp

// --- Sea ice thermodynamics ---
// Stefan problem: dh/dt = k_ice * ΔT / (ρ_ice * L_fuse * h)
// At h=1m, ΔT=1°C: dh/dt = 2/(917×334000×1) ≈ 6.5×10⁻⁹ m/s ≈ 0.2 m/year
var T_FREEZE = -1.8;          // ✓ Seawater freezing point at ~35 PSU
var L_FUSE = 334000;          // ✓ Latent heat of fusion (J/kg)
var RHO_ICE = 917;            // ✓ Ice density (kg/m³)
var ICE_GROW_RATE = 0.2;      // ✓ Growth rate (m/year per °C undercooling). From Stefan problem.
var ICE_MELT_RATE = 0.4;      // ~ Melt rate (m/year per °C warming). Faster than growth (solar + ocean heat).
var ICE_INSULATION = 0.3;     // ✓ Insulation length scale (m). k_ice ≈ 2 W/mK, so 0.3m of ice cuts flux ~90%.
                               //   Previously 3.0 — was 10-30x too weak.
var ICE_MAX = 4.0;            // ✓ Max thickness (m). Multi-year Arctic ice is 3–5m.

// Grid sizes (power-of-2 for radix-2 FFT Poisson solver)
const GPU_NX = 1024, GPU_NY = 512;
const CPU_NX = 1024, CPU_NY = 512;
let NX, NY, dx, dy, invDx, invDy, invDx2, invDy2;

// Mask source dimensions
const MASK_SRC_NX = 1024, MASK_SRC_NY = 512;
const LON0 = -180, LON1 = 180, LAT0 = -79.5, LAT1 = 79.5;

// Buffers (set during init)
let psi, zeta, zetaNew, mask;
let useGPU = false;
let cpuTempNew; // CPU temp scratch buffer

// Particles
const NP = 3000;
let px = new Float64Array(NP), py = new Float64Array(NP), page_ = new Float64Array(NP);
const MAX_AGE = 400;

// Canvas refs
const simCanvas = document.getElementById('sim');
const ctx = simCanvas.getContext('2d');
const W = simCanvas.width, H = simCanvas.height;
let cellW, cellH;

// Map rendering
let LAND_POLYS = [];
const mapCanvas = document.createElement('canvas');
mapCanvas.width = W; mapCanvas.height = H;
const mapCtx = mapCanvas.getContext('2d');

// Mask source data (decoded from mask.json)
let maskSrcBits = null;

// ============================================================
// LOAD DATA — from data/ directory at 1024x512 native resolution
// ============================================================
var DATA_BASE = '../data/';
function loadJSON(file) {
  return fetch(DATA_BASE + file).then(function(r) { return r.json(); }).catch(function() { return null; });
}

let maskLoadPromise = loadJSON('mask.json').then(function(d) {
  if (!d) return;
  var bits = [];
  for (var c = 0; c < d.hex.length; c++) {
    var v = parseInt(d.hex[c], 16);
    bits.push((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1);
  }
  // Data files are stored north-first; flip rows so row 0 = south (sim grid convention)
  var srcNX = d.nx || MASK_SRC_NX, srcNY = d.ny || MASK_SRC_NY;
  var flipped = new Array(srcNX * srcNY);
  for (var j = 0; j < srcNY; j++) {
    var srcRow = srcNY - 1 - j;
    for (var i = 0; i < srcNX; i++) flipped[j * srcNX + i] = bits[srcRow * srcNX + i];
  }
  maskSrcBits = flipped;
});

let coastLoadPromise = fetch('coastlines.json').then(function(r) { return r.json(); }).then(function(p) {
  LAND_POLYS = p;
}).catch(function() {});

// Observational data (all 1024x512 from data/ directory)
let obsSSTData = null;
let obsDeepData = null;
let obsBathyData = null;
let obsSalinityData = null;
let obsWindData = null;
let obsSeaIceData = null;
let obsMldData = null;
let obsCurrentsData = null;
let sstLoadPromise = loadJSON('sst.json').then(function(d) { obsSSTData = d; });
let deepLoadPromise = loadJSON('deep_temp.json').then(function(d) { obsDeepData = d; });
let bathyLoadPromise = loadJSON('bathymetry.json').then(function(d) { obsBathyData = d; });
let salinityLoadPromise = loadJSON('salinity.json').then(function(d) { obsSalinityData = d; });
let windLoadPromise = loadJSON('wind_stress.json').then(function(d) { obsWindData = d; });
let seaIceLoadPromise = loadJSON('sea_ice.json').then(function(d) { obsSeaIceData = d; });
let mldLoadPromise = loadJSON('mixed_layer_depth.json').then(function(d) { obsMldData = d; });
let currentsLoadPromise = loadJSON('ocean_currents.json').then(function(d) { obsCurrentsData = d; });

// Optional dataset loaders — stub any that aren't loaded above
let albedoLoadPromise   = Promise.resolve();
let precipLoadPromise   = Promise.resolve();
let cloudLoadPromise    = Promise.resolve();
let airTempLoadPromise  = Promise.resolve();
let lstLoadPromise      = Promise.resolve();
let evapLoadPromise     = Promise.resolve();

// ============================================================
// MASK HELPERS
// ============================================================
function buildMask(nx, ny) {
  var m = new Uint8Array(nx * ny);
  if (!maskSrcBits) {
    // Fallback: simple rectangular ocean
    for (var j = 1; j < ny - 1; j++)
      for (var i = 0; i < nx; i++)
        m[j * nx + i] = 1;
    return m;
  }
  // Nearest-neighbor sample from mask source onto sim grid (both 1024x512 at -79.5..79.5)
  for (var j = 0; j < ny; j++) {
    var sj = Math.min(Math.floor(j * MASK_SRC_NY / ny), MASK_SRC_NY - 1);
    for (var i = 0; i < nx; i++) {
      var si = Math.min(Math.floor(i * MASK_SRC_NX / nx), MASK_SRC_NX - 1);
      m[j * nx + i] = maskSrcBits[sj * MASK_SRC_NX + si] || 0;
    }
  }
  // Ensure polar boundaries are land (j=0, j=ny-1)
  // Do NOT enforce walls at i=0 and i=nx-1 — periodic in longitude!
  for (var i = 0; i < nx; i++) { m[i] = 0; m[(ny - 1) * nx + i] = 0; }
  openStraits(m, nx, ny);
  return m;
}

// Open narrow straits that get closed at this resolution
function openStraits(m, nx, ny) {
  function lonToI(lon) { return Math.round((lon - LON0) / (LON1 - LON0) * (nx - 1)); }
  function latToJ(lat) { return Math.round((lat - LAT0) / (LAT1 - LAT0) * (ny - 1)); }
  function carve(lon0, lon1, lat0, lat1) {
    var i0 = lonToI(lon0), i1 = lonToI(lon1);
    var j0 = latToJ(lat0), j1 = latToJ(lat1);
    for (var j = Math.min(j0,j1); j <= Math.max(j0,j1); j++)
      for (var i = Math.min(i0,i1); i <= Math.max(i0,i1); i++)
        if (j > 0 && j < ny-1 && i >= 0 && i < nx) m[j * nx + i] = 1;
  }
  // Gibraltar Strait (~36N, -6 to -5)
  carve(-6.5, -4.5, 35.5, 36.5);
  // Bosphorus / Dardanelles (~40-41N, 26-30E) — connects Black Sea
  carve(25, 30, 40, 41.5);
  // Danish Straits (~55-56N, 10-13E) — connects Baltic
  carve(9, 13, 54.5, 56.5);
  // English Channel (~50-51N, -2 to 2E)
  carve(-2, 2, 49.5, 51);
  // Mozambique Channel (~15-25S, 40-45E)
  carve(39, 45, -25, -14);
  // Indonesian Throughflow (~5S-5N, 115-130E)
  carve(115, 130, -8, 0);
  console.log('Straits opened: Gibraltar, Bosphorus, Danish, English Channel, Mozambique, Indonesia');
}

function buildMaskU32(mask8, nx, ny) {
  var m = new Uint32Array(nx * ny);
  for (var k = 0; k < nx * ny; k++) m[k] = mask8[k];
  return m;
}

// ============================================================
// MAP UNDERLAY
// ============================================================
function lonToX(lon) { return ((lon - LON0) / (LON1 - LON0)) * W; }
function latToY(lat) { return (1 - (lat - LAT0) / (LAT1 - LAT0)) * H; }

function drawMapUnderlay() {
  mapCtx.clearRect(0, 0, W, H);
  mapCtx.fillStyle = '#060c16';
  mapCtx.fillRect(0, 0, W, H);

  // Draw land from the MASK with elevation-based coloring
  if (mask && NX && NY) {
    // Use same half-cell offset as field overlay
    var cw_ = W / (NX - 1), ch_ = H / (NY - 1);
    var ox_ = -cw_ / 2, oy_ = -ch_ / 2;
    var landElev = null;
    if (obsBathyData && obsBathyData.elevation) {
      landElev = obsBathyData.elevation;
    }
    for (var mj = 0; mj < NY; mj++) {
      for (var mi = 0; mi < NX; mi++) {
        var mk = mj * NX + mi;
        if (!mask[mk]) {
          if (landElev) {
            var mlat = LAT0 + (mj / (NY - 1)) * (LAT1 - LAT0);
            var mlon = LON0 + (mi / (NX - 1)) * (LON1 - LON0);
            var obsK = obsIndex(mlat, mlon, obsBathyData);
            var elev = 0;
            if (obsK >= 0) elev = landElev[obsK] || 0;
            var r, g, b;
            if (elev < 100) {
              var t = elev / 100;
              r = 26 + 16 * t; g = 62 + 32 * t; b = 32 + 16 * t;
            } else if (elev < 500) {
              var t = (elev - 100) / 400;
              r = 42 + 96 * t; g = 94 - 16 * t; b = 48 + 32 * t;
            } else if (elev < 2000) {
              var t = (elev - 500) / 1500;
              r = 138 - 32 * t; g = 122 - 42 * t; b = 80 - 16 * t;
            } else if (elev < 4000) {
              var t = (elev - 2000) / 2000;
              r = 106 + 48 * t; g = 80 + 64 * t; b = 64 + 80 * t;
            } else {
              var t = Math.min(1, (elev - 4000) / 4000);
              r = 154 + 54 * t; g = 144 + 64 * t; b = 144 + 64 * t;
            }
            mapCtx.fillStyle = 'rgb(' + Math.floor(r) + ',' + Math.floor(g) + ',' + Math.floor(b) + ')';
          } else {
            mapCtx.fillStyle = '#1a2e20';
          }
          mapCtx.fillRect(ox_ + mi * cw_, oy_ + (NY - 1 - mj) * ch_, cw_ + 0.5, ch_ + 0.5);
        }
      }
    }
  }
  mapCtx.strokeStyle = 'rgba(255,255,255,0.04)';
  mapCtx.lineWidth = 0.5;
  for (var lat = -60; lat <= 60; lat += 30) {
    var y = latToY(lat);
    mapCtx.beginPath(); mapCtx.moveTo(0, y); mapCtx.lineTo(W, y); mapCtx.stroke();
    mapCtx.fillStyle = 'rgba(255,255,255,0.1)'; mapCtx.font = '7px system-ui'; mapCtx.textAlign = 'right';
    mapCtx.fillText((lat >= 0 ? lat + '\u00b0N' : Math.abs(lat) + '\u00b0S'), W - 3, y - 2);
  }
  for (var lon = -120; lon <= 120; lon += 60) {
    var x = lonToX(lon);
    mapCtx.beginPath(); mapCtx.moveTo(x, 0); mapCtx.lineTo(x, H); mapCtx.stroke();
    mapCtx.fillStyle = 'rgba(255,255,255,0.1)'; mapCtx.font = '7px system-ui'; mapCtx.textAlign = 'center';
    mapCtx.fillText((lon <= 0 ? Math.abs(lon) + '\u00b0W' : lon + '\u00b0E'), x, H - 3);
  }
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
'  let invDx = 1.0 / params.dx;',
'  let invDy = 1.0 / params.dy;',
'  let invDx2 = invDx * invDx;',
'  let invDy2 = invDy * invDy;',
'',
'  // Arakawa Jacobian J(psi, zeta)',
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
'  let jac = (J1 + J2 + J3) / (12.0 * params.dx * params.dy);',
'',
'  // Latitude for this cell',
'  let lat = -79.5 + f32(j) / f32(ny - 1u) * 159.0;',
'  let latRad = lat * 3.14159265 / 180.0;',
'',
'  // Beta term: varies with latitude (beta ~ cos(lat) in real ocean)',
'  let betaLocal = params.beta * cos(latRad);',
'  let betaV = betaLocal * (psi[ke] - psi[kw]) * 0.5 * invDx;',
'',
'  // Wind curl: 3-belt pattern (trades + westerlies + polar easterlies)',
'  // cos(3φ) gives zero crossings at ±30° and ±90° — correct belt boundaries',
'  // SH westerlies 20% stronger than NH (observed asymmetry)',
'  // SH westerlies ~2x NH (observed); polar easterlies weakened by 0.7 factor',
'  let shBoost = select(1.0, 2.0, lat < 0.0);',
'  let polarDamp = select(1.0, 0.7, abs(lat) > 60.0);',
'  let F = params.windStrength * (-cos(3.0 * latRad) * shBoost * polarDamp) * 2.0;',
'',
'  // Friction',
'  let fric = -params.r * zeta[k];',
'',
'  // Biharmonic viscosity: A * laplacian(zeta)',
'  let lapZeta = invDx2 * (zeta[ke] + zeta[kw] - 2.0 * zeta[k])',
'             + invDy2 * (zeta[kn] + zeta[ks] - 2.0 * zeta[k]);',
'  let visc = params.A * lapZeta;',
'',
'  // Buoyancy coupling: density gradient from T AND S',
'  // rho = rho0 * (1 - alpha*T + beta*S), so drho/dx = -alpha*dT/dx + beta*dS/dx',
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
'  // Red cells: (i+j) % 2 == 0, Black cells: (i+j) % 2 == 1',
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
'  // SOR relaxation: omega ~ 1.98 for 360x180 grid (optimal for Laplacian)',
'  let omega = 1.85;',
'  psi[k] = psi[k] + omega * (psiNew - psi[k]);',
'}'
].join('\n');

// ============================================================
// COARSE-GRID POISSON: downsample zeta, SOR at 256x128, upsample psi
// ============================================================
var COARSE_NX = 512, COARSE_NY = 256;

var downsampleShaderCode = [
'struct DSParams { fineNx: u32, fineNy: u32, coarseNx: u32, coarseNy: u32 };',
'@group(0) @binding(0) var<storage, read> src: array<f32>;',
'@group(0) @binding(1) var<storage, read_write> dst: array<f32>;',
'@group(0) @binding(2) var<uniform> p: DSParams;',
'',
'@compute @workgroup_size(64)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let k = id.x;',
'  if (k >= p.coarseNx * p.coarseNy) { return; }',
'  let ci = k % p.coarseNx;',
'  let cj = k / p.coarseNx;',
'  let rx = p.fineNx / p.coarseNx;',
'  let ry = p.fineNy / p.coarseNy;',
'  let fi = ci * rx;',
'  let fj = cj * ry;',
'  var sum = 0.0;',
'  for (var dj = 0u; dj < ry; dj++) {',
'    for (var di = 0u; di < rx; di++) {',
'      sum += src[(fj + dj) * p.fineNx + fi + di];',
'    }',
'  }',
'  dst[k] = sum / f32(rx * ry);',
'}'
].join('\n');

var upsampleShaderCode = [
'struct DSParams { fineNx: u32, fineNy: u32, coarseNx: u32, coarseNy: u32 };',
'@group(0) @binding(0) var<storage, read> src: array<f32>;',
'@group(0) @binding(1) var<storage, read_write> dst: array<f32>;',
'@group(0) @binding(2) var<uniform> p: DSParams;',
'',
'@compute @workgroup_size(64)',
'fn main(@builtin(global_invocation_id) id: vec3u) {',
'  let k = id.x;',
'  if (k >= p.fineNx * p.fineNy) { return; }',
'  let fi = k % p.fineNx;',
'  let fj = k / p.fineNx;',
'  let cx = f32(fi) * f32(p.coarseNx - 1u) / f32(p.fineNx - 1u);',
'  let cy = f32(fj) * f32(p.coarseNy - 1u) / f32(p.fineNy - 1u);',
'  let ci = u32(cx);',
'  let cj = u32(cy);',
'  let fx = cx - f32(ci);',
'  let fy = cy - f32(cj);',
'  let ci1 = min(ci + 1u, p.coarseNx - 1u);',
'  let cj1 = min(cj + 1u, p.coarseNy - 1u);',
'  let cnx = p.coarseNx;',
'  dst[k] = src[cj * cnx + ci] * (1.0 - fx) * (1.0 - fy)',
'         + src[cj * cnx + ci1] * fx * (1.0 - fy)',
'         + src[cj1 * cnx + ci] * (1.0 - fx) * fy',
'         + src[cj1 * cnx + ci1] * fx * fy;',
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
'  // Zero on land cells and polar walls (j=0, j=ny-1)',
'  // Do NOT zero at i=0 or i=nx-1 — periodic in longitude',
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
'  let invDx = 1.0 / params.dx;',
'  let invDy = 1.0 / params.dy;',
'  let invDx2 = invDx * invDx;',
'  let invDy2 = invDy * invDy;',
'',
'  // Latitude',
'  let lat = -79.5 + f32(j) / f32(ny - 1u) * 159.0;',
'  let latRad = lat * 3.14159265 / 180.0;',
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
'  // Opposite sign to surface: cold deep water at high lat creates southward deep flow',
'  // Deep buoyancy: density gradient from deep T AND S (stacked at offset N)',
'  let N_doff = params.nx * params.ny;',
'  let dRhodxDeep = -params.alphaT * (deepTempIn[ke] - deepTempIn[kw]) + params.betaS * (deepTempIn[ke + N_doff] - deepTempIn[kw + N_doff]);',
'  let deepBuoyancy = dRhodxDeep * 0.5 * invDx;',
'',
'  deepZetaNew[k] = clamp(deepZeta[k] + params.dt * (-jac - betaV + fric + visc + coupling + deepBuoyancy), -500.0, 500.0);',
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
'  let invDx = 1.0 / params.dx;',
'  let invDy = 1.0 / params.dy;',
'  let invDx2 = invDx * invDx;',
'  let invDy2 = invDy * invDy;',
'',
'  // Advection: J(psi, T) — uses one-sided stencil near coasts',
'  let dPdx = (pE - pW) * 0.5 * invDx;',
'  let dPdy = (pN - pS) * 0.5 * invDy;',
'  let dTdx = (tE - tW) * 0.5 * invDx;',
'  let dTdy = (tN - tS) * 0.5 * invDy;',
'  let advec = dPdx * dTdy - dPdy * dTdx;',
'',
'  // Latitude and seasonal solar declination',
'  let lat = -79.5 + f32(j) / f32(ny - 1u) * 159.0;',
'  let latRad = lat * 3.14159265 / 180.0;',
'  let yearPhase = 2.0 * 3.14159265 * (params.simTime % 10.0) / 10.0;',
'  let declination = 23.44 * sin(yearPhase) * 3.14159265 / 180.0;',
'',
'  // Insolation with ice-albedo feedback',
'  let cosZenith = cos(latRad) * cos(declination) + sin(latRad) * sin(declination);',
'  var qSolar = params.sSolar * max(0.0, cosZenith);',
'  // Ice-albedo: gradual onset from 45° to full strength at 65°',
'  // Strength ramps with latitude; temp transition [-2, 8]°C',
'  if (abs(lat) > 45.0) {',
'    let iceT = clamp((tempIn[k] + 2.0) / 10.0, 0.0, 1.0);',
'    let iceFrac = 1.0 - iceT * iceT * (3.0 - 2.0 * iceT);',
'    // Latitude ramp: 0% effect at 45°, full effect at 65°',
'    let latRamp = clamp((abs(lat) - 45.0) / 20.0, 0.0, 1.0);',
'    // Max 50% solar reduction at full ice + full latitude',
'    qSolar *= 1.0 - 0.50 * iceFrac * latRamp;',
'  }',
'',
'  // Outgoing longwave: A + B*T (global heat balance)',
'  let olr = params.aOlr - params.bOlr * params.globalTempOffset + params.bOlr * tempIn[k];',
'',
'  // Net radiative heating',
'  let qNet = qSolar - olr;',
'',
'  // Freshwater forcing now affects SALINITY, not temperature (see salinity block below)',
'  let y = f32(j) / f32(ny - 1u);',
'',
'  // Diffusion with one-sided stencil',
'  let lapT = invDx2 * (tE + tW - 2.0 * tempIn[k])',
'           + invDy2 * (tN + tS - 2.0 * tempIn[k]);',
'  let diff = params.kappaDiff * lapT;',
'',
'  // Land-ocean heat exchange: coastal cells exchange heat with adjacent land',
'  var landFlux: f32 = 0.0;',
'  let nLand = f32(select(0u, 1u, mask[ke] == 0u)) + f32(select(0u, 1u, mask[kw] == 0u))',
'            + f32(select(0u, 1u, mask[kn] == 0u)) + f32(select(0u, 1u, mask[ks] == 0u));',
'  let nOcean = 4.0 - nLand;',
'  if (nLand > 0.0 && nOcean > 0.0) {',
'    let landT = 50.0 * max(0.0, cosZenith) - 20.0;',
'    // Scale by ocean fraction; clamp magnitude to prevent coastal hotspots',
'    let rawFlux = params.landHeatK * (landT - tempIn[k]) * (nOcean / 4.0);',
'    landFlux = clamp(rawFlux, -0.5, 0.5);',
'  }',
'',
'  // Temperature update: freshwater NO LONGER cools — it only affects salinity',
'  tempOut[k] = tempIn[k] + params.dt * (-advec + qNet + diff + landFlux);',
'',
'  // Two-layer vertical exchange — modulated by local depth',
'  let localDepth = depthField[k];',
'  let hSurf = min(params.hSurface, localDepth);',
'  let hDeep = max(1.0, localDepth - params.hSurface);',
'  let hasDeepLayer = select(0.0, 1.0, localDepth > params.hSurface);',
'',
'  // ── SALINITY (stacked at offset N in the same buffers) ──',
'  let N = params.nx * params.ny;',
'  let salK = k + N;',
'',
'  // Salinity neighbors (from stacked buffer)',
'  let sE = select(tempIn[salK], tempIn[idx(ip1,j) + N], mask[ke] != 0u);',
'  let sW = select(tempIn[salK], tempIn[idx(im1,j) + N], mask[kw] != 0u);',
'  let sN = select(tempIn[salK], tempIn[idx(i,j+1u) + N], mask[kn] != 0u);',
'  let sS = select(tempIn[salK], tempIn[idx(i,j-1u) + N], mask[ks] != 0u);',
'',
'  // Salinity advection: J(ψ, S) — reuses dPdx, dPdy already computed',
'  let dSdx = (sE - sW) * 0.5 * invDx;',
'  let dSdy = (sN - sS) * 0.5 * invDy;',
'  let salAdvec = dPdx * dSdy - dPdy * dSdx;',
'',
'  // Salinity diffusion',
'  let lapS = invDx2 * (sE + sW - 2.0 * tempIn[salK])',
'           + invDy2 * (sN + sS - 2.0 * tempIn[salK]);',
'  let salDiff = params.kappaSal * lapS;',
'',
'  // Salinity restoring toward climatological pattern',
'  let salClim = 34.0 + 2.0 * cos(2.0 * latRad) - 0.5 * cos(4.0 * latRad);',
'  let salRestore = params.salRestoring * (salClim - tempIn[salK]);',
'',
'  // Freshwater forcing: REDUCES salinity (makes water lighter — correct physics)',
'  var fwSal: f32 = 0.0;',
'  if (y > 0.75) {',
'    fwSal = -params.freshwater * 3.0 * (y - 0.75) * 4.0;',
'  }',
'',
'  tempOut[salK] = tempIn[salK] + params.dt * (-salAdvec + salDiff + salRestore + fwSal);',
'',
'  // ── DENSITY-BASED VERTICAL MIXING ──',
'  // No latitude threshold — convection happens wherever surface is denser than deep',
'  let rhoSurf = -params.alphaT * tempOut[k] + params.betaS * tempOut[salK];',
'  let rhoDeep = -params.alphaT * deepTempIn[k] + params.betaS * deepTempIn[k + N];',
'  let drho = rhoSurf - rhoDeep;',
'',
'  // Convection (drho > 0): dense surface sinks fast, rate ~ density difference',
'  // Stable (drho < 0): slow diffusive upwelling (Munk abyssal recipes)',
'  var gamma = select(params.gammaMix * 0.5, params.gammaMix + params.gammaDeepForm * min(1.0, drho * 10.0), drho > 0.0);',
'',
'  let vertExchangeT = gamma * (tempOut[k] - deepTempIn[k]) * hasDeepLayer;',
'  tempOut[k] = clamp(tempOut[k] - params.dt * vertExchangeT / hSurf, -10.0, 40.0);',
'',
'  let vertExchangeS = gamma * (tempOut[salK] - deepTempIn[k + N]) * hasDeepLayer;',
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
'  // Deep salinity: vertical exchange + diffusion',
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
// WebGPU RENDER SHADERS (fullscreen quad)
// ============================================================
var renderVertexShaderCode = [
'@vertex',
'fn main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {',
'  // Fullscreen triangle: 3 vertices, no buffer needed',
'  var pos = array<vec2f, 3>(',
'    vec2f(-1.0, -1.0),',
'    vec2f( 3.0, -1.0),',
'    vec2f(-1.0,  3.0)',
'  );',
'  return vec4f(pos[vi], 0.0, 1.0);',
'}'
].join('\n');

var renderFragmentShaderCode = [
'struct RenderParams {',
'  nx: u32, ny: u32,',
'  fieldMode: u32, _pad: u32,',
'  absMax: f32, maxSpd: f32,',
'  canvasW: f32, canvasH: f32,',
'  simTime: f32, _pad2: u32, _pad3: u32, _pad4: u32,',
'};',
'',
'@group(0) @binding(0) var<storage, read> psi: array<f32>;',
'@group(0) @binding(1) var<storage, read> zeta: array<f32>;',
'@group(0) @binding(2) var<storage, read> temp: array<f32>;',
'@group(0) @binding(3) var<storage, read> mask: array<u32>;',
'@group(0) @binding(4) var<uniform> rp: RenderParams;',
'',
'fn psiColor(val: f32, absMax: f32) -> vec3f {',
'  let t = clamp(val / (absMax + 1e-30), -1.0, 1.0);',
'  if (t < 0.0) {',
'    let s = -t;',
'    return vec3f((40.0 + 60.0 * s) / 255.0, (80.0 + 100.0 * s) / 255.0, (160.0 + 95.0 * s) / 255.0);',
'  }',
'  return vec3f((200.0 + 55.0 * t) / 255.0, (100.0 - 40.0 * t) / 255.0, (80.0 - 40.0 * t) / 255.0);',
'}',
'',
'fn vortColor(val: f32, absMax: f32) -> vec3f {',
'  let t = clamp(val / (absMax + 1e-30), -1.0, 1.0);',
'  if (t < 0.0) {',
'    let s = -t;',
'    return vec3f((30.0 + 20.0 * s) / 255.0, (60.0 + 140.0 * s) / 255.0, (40.0 + 60.0 * s) / 255.0);',
'  }',
'  return vec3f((180.0 + 75.0 * t) / 255.0, (60.0 + 40.0 * t) / 255.0, (120.0 + 60.0 * t) / 255.0);',
'}',
'',
'fn speedColor(spd: f32, maxSpd: f32) -> vec3f {',
'  let t = clamp(spd / (maxSpd + 1e-30), 0.0, 1.0);',
'  if (t < 0.25) { let s = t / 0.25; return vec3f((10.0 + 10.0 * s) / 255.0, (15.0 + 35.0 * s) / 255.0, (40.0 + 60.0 * s) / 255.0); }',
'  if (t < 0.5) { let s = (t - 0.25) / 0.25; return vec3f((20.0 + 30.0 * s) / 255.0, (50.0 + 80.0 * s) / 255.0, (100.0 + 80.0 * s) / 255.0); }',
'  if (t < 0.75) { let s = (t - 0.5) / 0.25; return vec3f((50.0 + 120.0 * s) / 255.0, (130.0 + 80.0 * s) / 255.0, (180.0 - 60.0 * s) / 255.0); }',
'  let s = (t - 0.75) / 0.25;',
'  return vec3f((170.0 + 80.0 * s) / 255.0, (210.0 + 30.0 * s) / 255.0, (120.0 - 80.0 * s) / 255.0);',
'}',
'',
'fn tempColor(tIn: f32) -> vec3f {',
'  let t = clamp(tIn, -5.0, 30.0);',
'  if (t < 5.0) {',
'    let s = (t - (-5.0)) / 10.0;',
'    return vec3f((20.0 + 10.0 * s) / 255.0, (30.0 + 150.0 * s) / 255.0, (140.0 + 80.0 * s) / 255.0);',
'  }',
'  if (t < 15.0) {',
'    let s = (t - 5.0) / 10.0;',
'    return vec3f((30.0 - 10.0 * s) / 255.0, (180.0 + 20.0 * s) / 255.0, (220.0 - 120.0 * s) / 255.0);',
'  }',
'  if (t < 22.0) {',
'    let s = (t - 15.0) / 7.0;',
'    return vec3f((20.0 + 210.0 * s) / 255.0, (200.0 + 30.0 * s) / 255.0, (100.0 - 60.0 * s) / 255.0);',
'  }',
'  if (t < 26.0) {',
'    let s = (t - 22.0) / 4.0;',
'    return vec3f((230.0 + 20.0 * s) / 255.0, (230.0 - 80.0 * s) / 255.0, (40.0 - 10.0 * s) / 255.0);',
'  }',
'  let s = (t - 26.0) / 4.0;',
'  return vec3f(250.0 / 255.0, (150.0 - 100.0 * s) / 255.0, (30.0 - 30.0 * s) / 255.0);',
'}',
'',
'@fragment',
'fn main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {',
'  let nx = rp.nx;',
'  let ny = rp.ny;',
'  // Map fragment position to grid cell',
'  let fi = fragCoord.x / rp.canvasW * f32(nx);',
'  // Flip Y: top of canvas = high j (north)',
'  let fj = (1.0 - fragCoord.y / rp.canvasH) * f32(ny);',
'  let i = u32(clamp(fi, 0.0, f32(nx) - 1.0));',
'  let j = u32(clamp(fj, 0.0, f32(ny) - 1.0));',
'  let k = j * nx + i;',
'',
'  // Land: seasonal temperature coloring',
'  if (mask[k] == 0u) {',
'    let lat = -80.0 + f32(j) / f32(ny - 1u) * 160.0;',
'    let latRad = lat * 3.14159265 / 180.0;',
'    let yearPhase = 2.0 * 3.14159265 * rp.simTime / 10.0;',
'    let decl = 23.44 * sin(yearPhase) * 3.14159265 / 180.0;',
'    let cosZ = cos(latRad) * cos(decl) + sin(latRad) * sin(decl);',
'    let landT = 35.0 * max(0.0, cosZ) - 15.0;',
'    // Dark muted earth tones',
'    let tN = clamp((landT + 20.0) / 50.0, 0.0, 1.0);',
'    let lr = (20.0 + 40.0 * tN) / 255.0;',
'    let lg = (30.0 + 30.0 * tN) / 255.0;',
'    let lb = (15.0 + 10.0 * tN) / 255.0;',
'    // Snow cover when land temp < -5',
'    if (landT < -5.0) {',
'      let snow = clamp((-5.0 - landT) / 15.0, 0.0, 1.0);',
'      return vec4f(mix(vec3f(lr, lg, lb), vec3f(0.75, 0.8, 0.85), snow), 0.85);',
'    }',
'    return vec4f(lr, lg, lb, 0.85);',
'  }',
'',
'  var col: vec3f;',
'  if (rp.fieldMode == 0u) {',
'    // Streamfunction',
'    col = psiColor(psi[k], rp.absMax);',
'  } else if (rp.fieldMode == 1u) {',
'    // Vorticity',
'    col = vortColor(zeta[k], rp.absMax);',
'  } else if (rp.fieldMode == 2u) {',
'    // Speed: compute from psi gradients with periodic x',
'    if (j < 1u || j >= ny - 1u) {',
'      col = vec3f(0.04, 0.06, 0.16);',
'    } else {',
'      let sip1 = select(i + 1u, 0u, i == nx - 1u);',
'      let sim1 = select(i - 1u, nx - 1u, i == 0u);',
'      let invDx = f32(nx - 1u);',
'      let invDy = f32(ny - 1u);',
'      let u = -(psi[(j + 1u) * nx + i] - psi[(j - 1u) * nx + i]) * 0.5 * invDy;',
'      let v = (psi[j * nx + sip1] - psi[j * nx + sim1]) * 0.5 * invDx;',
'      let spd = sqrt(u * u + v * v);',
'      col = speedColor(spd, rp.maxSpd);',
'    }',
'  } else {',
'    // Temperature',
'    col = tempColor(temp[k]);',
'  }',
'',
'  return vec4f(col, 190.0 / 255.0);',
'}'
].join('\n');

// ============================================================
// WebGPU SOLVER
// ============================================================
var gpuDevice = null;
var gpuPsiBuf, gpuZetaBuf, gpuZetaNewBuf, gpuMaskBuf, gpuParamsBuf;
var gpuReadbackBuf, gpuZetaReadbackBuf;
var gpuTempBuf, gpuTempNewBuf, gpuTempReadbackBuf;
var gpuDeepTempBuf, gpuDeepTempNewBuf, gpuDeepTempReadbackBuf;
var gpuDeepPsiBuf, gpuDeepZetaBuf, gpuDeepZetaNewBuf, gpuDeepPsiReadbackBuf;
var gpuDepthBuf;
var gpuTimestepPipeline, gpuPoissonPipeline, gpuEnforceBCPipeline, gpuTemperaturePipeline, gpuDeepTimestepPipeline;
var gpuTimestepBindGroup, gpuPoissonBindGroup, gpuEnforceBCBindGroup, gpuTemperatureBindGroup;
var gpuSwapTimestepBindGroup; // for after swap
var gpuSwapTemperatureBindGroup;

// Coarse-grid Poisson state
var gpuCoarsePsiBuf, gpuCoarseZetaBuf, gpuCoarseMaskBuf;
var gpuCoarseDeepPsiBuf, gpuCoarseDeepZetaBuf;
var gpuDSParamsBuf, gpuCoarseParamsRedBuf, gpuCoarseParamsBlackBuf;
var gpuDownsamplePipeline, gpuUpsamplePipeline;
var gpuDownsampleBindGroup, gpuDownsampleSwapBindGroup;
var gpuUpsampleBindGroup, gpuUpsampleDeepBindGroup;
var gpuDownsampleDeepBindGroup, gpuDownsampleDeepSwapBindGroup;
var gpuCoarsePoissonRedBG, gpuCoarsePoissonBlackBG;
var gpuCoarseDeepPoissonRedBG, gpuCoarseDeepPoissonBlackBG;
var COARSE_POISSON_ITERS = 20;  // 20 red-black iters at 512×256: removes ~25% error per step.
                                // With warm-starting from previous psi, this converges fine.
                                // Was 40 — halving saves 40 dispatches/step.

// GPU Render pipeline state
var gpuRenderPipeline = null;
var gpuRenderParamsBuf = null;
var gpuRenderBindGroup = null;
var gpuCanvasCtx = null;
var gpuRenderFormat = null;
var gpuRenderEnabled = false;
var readbackFrameCounter = 0;
var READBACK_INTERVAL = 5; // only read back every N frames

async function initWebGPU() {
  if (!navigator.gpu) return false;
  var adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return false;
  gpuDevice = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: GPU_NX * GPU_NY * 4 * 4,  // stacked T+S buffers
      maxBufferSize: GPU_NX * GPU_NY * 4 * 4
    }
  });
  if (!gpuDevice) return false;

  NX = GPU_NX; NY = GPU_NY;
  dx = 1.0 / (NX - 1); dy = 1.0 / (NY - 1);
  invDx = 1 / dx; invDy = 1 / dy;
  invDx2 = invDx * invDx; invDy2 = invDy * invDy;
  cellW = W / NX; cellH = H / NY;

  mask = buildMask(NX, NY);
  var maskU32 = buildMaskU32(mask, NX, NY);

  var bufSize = NX * NY * 4;

  // Create GPU buffers — COPY_DST added to psi/zeta/zetaNew so reset paths
  // (gpuReset, scenario load, paint apply) can writeBuffer without device loss.
  gpuPsiBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuZetaBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuZetaNewBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuMaskBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuParamsBuf = gpuDevice.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // 40 fields = 160 bytes (includes salinity params)
  gpuReadbackBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpuZetaReadbackBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  // Stacked layout: first NX*NY = temperature, second NX*NY = salinity (2x size)
  var tracerBufSize = bufSize * 2;
  gpuTempBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuTempNewBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  gpuTempReadbackBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpuDeepTempBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuDeepTempNewBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  gpuDeepTempReadbackBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpuDeepPsiBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuDeepZetaBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuDeepZetaNewBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  gpuDeepPsiReadbackBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpuDepthBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  // Upload mask
  gpuDevice.queue.writeBuffer(gpuMaskBuf, 0, maskU32);

  // Create pipelines
  gpuTimestepPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: timestepShaderCode }), entryPoint: 'main' }
  });

  gpuPoissonPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: poissonShaderCode }), entryPoint: 'main' }
  });

  gpuEnforceBCPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: enforceBCShaderCode }), entryPoint: 'main' }
  });

  gpuTemperaturePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: temperatureShaderCode }), entryPoint: 'main' }
  });

  gpuDeepTimestepPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: deepTimestepShaderCode }), entryPoint: 'main' }
  });

  // Build bind groups
  rebuildBindGroups();

  // CPU-side readback arrays — initialize with Earth-like conditions
  psi = new Float32Array(NX * NY);
  zeta = new Float32Array(NX * NY);
  temp = new Float32Array(NX * NY);
  deepTemp = new Float32Array(NX * NY);
  sal = new Float32Array(NX * NY);
  deepSal = new Float32Array(NX * NY);

  deepPsi = new Float32Array(NX * NY);
  deepZeta = new Float32Array(NX * NY);
  seaIce = new Float32Array(NX * NY);
  airTemp = new Float32Array(NX * NY);

  // Generate bathymetry from distance to coast
  generateDepthField();
  gpuDevice.queue.writeBuffer(gpuDepthBuf, 0, depth);

  // Build wind curl field from observed ERA5 data
  generateWindCurlField();

  // Initialize circulation from observed GODAS currents
  initStommelSolution();
  gpuDevice.queue.writeBuffer(gpuPsiBuf, 0, psi);
  gpuDevice.queue.writeBuffer(gpuZetaBuf, 0, zeta);
  gpuDevice.queue.writeBuffer(gpuDeepPsiBuf, 0, deepPsi);
  gpuDevice.queue.writeBuffer(gpuDeepZetaBuf, 0, deepZeta);

  // Realistic temperature + salinity from observations
  initTemperatureField();

  // Initialize atmosphere from surface temperatures
  initLandTemp(); // ensure landTempField exists
  for (var ak = 0; ak < NX * NY; ak++) {
    airTemp[ak] = mask[ak] ? temp[ak] : (landTempField ? landTempField[ak] : 15);
  }

  // Pack T+S into stacked buffers for GPU
  var surfTracer = new Float32Array(NX * NY * 2);
  var deepTracer = new Float32Array(NX * NY * 2);
  for (var tk = 0; tk < NX * NY; tk++) {
    surfTracer[tk] = temp[tk];
    surfTracer[tk + NX * NY] = sal[tk];
    deepTracer[tk] = deepTemp[tk];
    deepTracer[tk + NX * NY] = deepSal[tk];
  }
  gpuDevice.queue.writeBuffer(gpuTempBuf, 0, surfTracer);
  gpuDevice.queue.writeBuffer(gpuDeepTempBuf, 0, deepTracer);

  return true;
}

function generateDepthField() {
  depth = new Float32Array(NX * NY);

  // Use real ETOPO1 bathymetry when available
  if (obsBathyData && obsBathyData.depth) {
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        if (!mask[k]) { depth[k] = 0; continue; }
        var lon = LON0 + (i / (NX - 1)) * (LON1 - LON0);
        var obsK = obsIndex(lat, lon, obsBathyData);
        if (obsK < 0) { depth[k] = 200; continue; }
        var d = obsBathyData.depth[obsK];
        if (d != null && !isNaN(d) && d > 0) {
          depth[k] = Math.min(5500, Math.max(50, d)); // clamp to reasonable range
        } else {
          depth[k] = 200; // ocean cell with no bathymetry data → shallow default
        }
      }
    }
    console.log('Using real ETOPO1 bathymetry');
    return;
  }

  // Fallback: BFS from all land cells to compute distance-to-coast
  // Then map distance to depth: shelf (200m) near coast, abyssal (4000m) far from coast
  console.log('Using BFS distance-to-coast bathymetry (fallback)');
  var dist = new Float32Array(NX * NY);
  for (var k = 0; k < NX * NY; k++) dist[k] = 9999;

  // Initialize: land cells and ocean cells adjacent to land get distance 0 and 1
  var queue = [];
  for (var j = 0; j < NY; j++) for (var i = 0; i < NX; i++) {
    var k = j * NX + i;
    if (!mask[k]) { dist[k] = 0; continue; }
    // Check if adjacent to land
    var ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;
    var hasLand = false;
    if (j > 0 && !mask[(j-1)*NX+i]) hasLand = true;
    if (j < NY-1 && !mask[(j+1)*NX+i]) hasLand = true;
    if (!mask[j*NX+ip1]) hasLand = true;
    if (!mask[j*NX+im1]) hasLand = true;
    if (hasLand) { dist[k] = 1; queue.push(k); }
  }

  // BFS
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

  // Map distance to depth with smooth profile
  // 1 cell from coast: 200m (continental shelf)
  // 3 cells: ~1000m (continental slope)
  // 6+ cells: 4000m (abyssal plain)
  for (var k2 = 0; k2 < NX * NY; k2++) {
    if (!mask[k2]) { depth[k2] = 0; continue; }
    var d = dist[k2];
    // Smooth ramp: shelf -> slope -> abyss
    var t = Math.min(1, Math.max(0, (d - 1) / 5)); // 0 at coast, 1 at 6 cells out
    t = t * t * (3 - 2 * t); // smoothstep
    depth[k2] = 200 + 3800 * t; // 200m to 4000m
  }
}

// Map sim grid (i,j) to obs data index. Obs grids are 360x160 over -79.5..79.5 / -179.5..179.5
// Map lat/lon to data array index. Data files store rows north-first (row 0 = lat1).
function obsIndex(lat, lon, obsData) {
  var onx = obsData.nx || 1024, ony = obsData.ny || 512;
  var olat0 = obsData.lat0 || -79.5, olat1 = obsData.lat1 || 79.5;
  var olon0 = obsData.lon0 || -180, olon1 = obsData.lon1 || 180;
  // Row 0 = north (lat1), row ony-1 = south (lat0)
  var oj = Math.round((olat1 - lat) / (olat1 - olat0) * (ony - 1));
  var oi = Math.round((lon - olon0) / (olon1 - olon0) * (onx - 1));
  if (oj < 0 || oj >= ony || oi < 0 || oi >= onx) return -1;
  return oj * onx + oi;
}

function initTemperatureField() {
  var useObs = obsSSTData && obsSSTData.sst;
  var useDeepObs = obsDeepData && obsDeepData.temp;

  for (var j = 0; j < NY; j++) {
    var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    for (var i = 0; i < NX; i++) {
      var k = j * NX + i;
      if (!mask[k]) { temp[k] = 0; deepTemp[k] = 0; continue; }
      var lon = LON0 + (i / (NX - 1)) * (LON1 - LON0);

      // Surface temperature: from NOAA observations or latitude formula fallback
      var gotSST = false;
      if (useObs) {
        var obsK = obsIndex(lat, lon, obsSSTData);
        if (obsK >= 0) {
          var sv = obsSSTData.sst[obsK];
          if (sv != null && !isNaN(sv) && sv > -90) {
            temp[k] = sv;
            gotSST = true;
          }
        }
      }
      if (!gotSST) {
        var tBase = 28 - 0.55 * Math.abs(lat) - 0.0003 * Math.pow(Math.abs(lat), 2);
        temp[k] = Math.max(-2, Math.min(30, tBase));
      }

      // Deep temperature: from WOA23 or simple latitude formula
      var gotDeep = false;
      if (useDeepObs) {
        var obsKd = obsIndex(lat, lon, obsDeepData);
        if (obsKd >= 0) {
          var dv = obsDeepData.temp[obsKd];
          if (dv != null && !isNaN(dv) && dv > -90) {
            deepTemp[k] = dv;
            gotDeep = true;
          }
        }
      }
      if (!gotDeep) {
        var yFrac = j / (NY - 1);
        deepTemp[k] = 0.5 + 3.0 * yFrac;
      }

      // Surface salinity: from observations if available, else analytical
      var gotSal = false;
      if (obsSalinityData && obsSalinityData.salinity) {
        var salK = obsIndex(lat, lon, obsSalinityData);
        if (salK >= 0) {
          var sv2 = obsSalinityData.salinity[salK];
          if (sv2 != null && !isNaN(sv2) && sv2 > 0) { sal[k] = sv2; gotSal = true; }
        }
      }
      if (!gotSal) {
        var latRad = lat * Math.PI / 180;
        sal[k] = 34.0 + 2.0 * Math.cos(2 * latRad) - 0.5 * Math.cos(4 * latRad);
      }

      // Deep salinity: more uniform, slightly fresher than surface mean
      deepSal[k] = 34.7 + 0.2 * Math.cos(2 * latRad);

      // Sea ice: from observed ice fraction if available, else from SST
      if (seaIce) {
        var gotIce = false;
        if (obsSeaIceData && obsSeaIceData.ice_fraction) {
          var iceK = obsIndex(lat, lon, obsSeaIceData);
          if (iceK >= 0) {
            var frac = obsSeaIceData.ice_fraction[iceK];
            if (frac != null && !isNaN(frac) && frac > 0.05) {
              seaIce[k] = frac * 2.0; // fraction → thickness estimate (2m max for 100% cover)
              gotIce = true;
            }
          }
        }
        if (!gotIce) {
          seaIce[k] = temp[k] < T_FREEZE ? Math.min(ICE_MAX, (T_FREEZE - temp[k]) * 0.5) : 0;
        }
      }
    }
  }
}

// Build wind curl field from observed data or analytical fallback
function generateWindCurlField() {
  windCurlField = new Float32Array(NX * NY);
  if (obsWindData && obsWindData.wind_curl) {
    // Remap observed wind curl to sim grid, normalize to nondimensional units
    // Real curl is ~10⁻⁶ N/m³, need to scale to nondimensional forcing amplitude
    var maxCurl = 0;
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        if (!mask[k]) { windCurlField[k] = 0; continue; }
        var lon = LON0 + (i / (NX - 1)) * (LON1 - LON0);
        var obsK = obsIndex(lat, lon, obsWindData);
        if (obsK >= 0) {
          windCurlField[k] = obsWindData.wind_curl[obsK] || 0;
          var a = Math.abs(windCurlField[k]);
          if (a > maxCurl) maxCurl = a;
        }
      }
    }
    // Normalize: scale so max curl ≈ 2 (matching analytical amplitude)
    if (maxCurl > 0) {
      var scale = 2.0 / maxCurl;
      for (var k = 0; k < NX * NY; k++) windCurlField[k] *= scale;
    }
    console.log('Using observed ERA5 wind stress curl (max raw: ' + maxCurl.toExponential(2) + ')');
  } else {
    // Analytical 3-belt pattern: trades + westerlies + polar easterlies
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var latRad = lat * Math.PI / 180;
      var shBoost = lat < 0 ? 2.0 : 1.0;
      var polarDamp = Math.abs(lat) > 60 ? 0.7 : 1.0;
      var curl = (-Math.cos(3 * latRad) * shBoost * polarDamp) * 2;
      for (var i = 0; i < NX; i++) windCurlField[j * NX + i] = curl;
    }
    console.log('Using analytical wind curl (no observed data)');
  }
}

function initStommelSolution() {
  // Initialize circulation from observed GODAS currents if available,
  // otherwise start from rest. Observed currents give a huge head start
  // (skip years of spinup).
  if (obsCurrentsData && obsCurrentsData.u && obsCurrentsData.v) {
    // Compute vorticity ζ = ∂v/∂x - ∂u/∂y from observed currents
    // Scale to nondimensional units: u_obs is in m/s, model u ~ dpsi/dy ~ O(1)
    // Real velocity scale U ~ 0.1 m/s for gyres, so scale factor ≈ 1/0.1 = 10
    var U_SCALE = 10.0;
    for (var j = 1; j < NY - 1; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        if (!mask[k]) { zeta[k] = 0; continue; }
        var lon = LON0 + (i / (NX - 1)) * (LON1 - LON0);
        var ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;

        // Get u,v at neighbors
        var obsK = obsIndex(lat, lon, obsCurrentsData);
        var latN = LAT0 + ((j+1) / (NY - 1)) * (LAT1 - LAT0);
        var latS = LAT0 + ((j-1) / (NY - 1)) * (LAT1 - LAT0);
        var lonE = LON0 + (ip1 / (NX - 1)) * (LON1 - LON0);
        var lonW = LON0 + (im1 / (NX - 1)) * (LON1 - LON0);

        var obsE = obsIndex(lat, lonE, obsCurrentsData);
        var obsW = obsIndex(lat, lonW, obsCurrentsData);
        var obsN = obsIndex(latN, lon, obsCurrentsData);
        var obsS = obsIndex(latS, lon, obsCurrentsData);

        var vE = obsE >= 0 ? (obsCurrentsData.v[obsE] || 0) : 0;
        var vW = obsW >= 0 ? (obsCurrentsData.v[obsW] || 0) : 0;
        var uN = obsN >= 0 ? (obsCurrentsData.u[obsN] || 0) : 0;
        var uS = obsS >= 0 ? (obsCurrentsData.u[obsS] || 0) : 0;

        var dvdx = (vE - vW) * 0.5 * invDx * U_SCALE;
        var dudy = (uN - uS) * 0.5 * invDy * U_SCALE;
        zeta[k] = Math.max(-500, Math.min(500, dvdx - dudy));
      }
    }
    // Solve Poisson to get initial psi from zeta (CPU SOR, many iterations for cold start)
    initSOR();
    for (var iter = 0; iter < 200; iter++) cpuSolveSOR(1);
    console.log('Initialized circulation from GODAS observed currents');
  } else {
    for (var k = 0; k < NX * NY; k++) { psi[k] = 0; zeta[k] = 0; }
    console.log('No observed currents — starting from rest');
  }
  // Deep layer starts from rest (deep circulation is slow, will develop from coupling)
  for (var k = 0; k < NX * NY; k++) { deepPsi[k] = 0; deepZeta[k] = 0; }
}

// Paint tool: push CPU-side changes to GPU
function updateGPUBuffersAfterPaint() {
  if (!gpuDevice) return;
  // Re-upload mask
  var maskU32 = new Uint32Array(NX * NY);
  for (var k = 0; k < NX * NY; k++) maskU32[k] = mask[k];
  gpuDevice.queue.writeBuffer(gpuMaskBuf, 0, maskU32);
  // Re-upload zeta (for heat/cold/wind painting)
  var zetaF32 = new Float32Array(NX * NY);
  for (var k2 = 0; k2 < NX * NY; k2++) zetaF32[k2] = zeta[k2];
  gpuDevice.queue.writeBuffer(gpuZetaBuf, 0, zetaF32);
  // Re-upload psi (zeroed on new land)
  var psiF32 = new Float32Array(NX * NY);
  for (var k3 = 0; k3 < NX * NY; k3++) psiF32[k3] = psi[k3];
  gpuDevice.queue.writeBuffer(gpuPsiBuf, 0, psiF32);
  // Re-upload temperature
  var tempF32 = new Float32Array(NX * NY);
  for (var k4 = 0; k4 < NX * NY; k4++) tempF32[k4] = temp[k4];
  gpuDevice.queue.writeBuffer(gpuTempBuf, 0, tempF32);
  // Regenerate depth from mask and upload
  if (gpuDepthBuf) {
    generateDepthField();
    gpuDevice.queue.writeBuffer(gpuDepthBuf, 0, depth);
  }
  // Re-upload deep temperature
  if (deepTemp) {
    var deepF32 = new Float32Array(NX * NY);
    for (var k5 = 0; k5 < NX * NY; k5++) deepF32[k5] = deepTemp[k5];
    gpuDevice.queue.writeBuffer(gpuDeepTempBuf, 0, deepF32);
  }
}

function rebuildBindGroups() {
  // Timestep: reads psi, zeta, temp, deepPsi -> writes zetaNew
  gpuTimestepBindGroup = gpuDevice.createBindGroup({
    layout: gpuTimestepPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuZetaNewBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuTempBuf } },
      { binding: 6, resource: { buffer: gpuDeepPsiBuf } },
    ]
  });

  // Swapped version
  gpuSwapTimestepBindGroup = gpuDevice.createBindGroup({
    layout: gpuTimestepPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuZetaBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuTempBuf } },
      { binding: 6, resource: { buffer: gpuDeepPsiBuf } },
    ]
  });

  // Poisson: reads/writes psi, reads zeta
  // Red-Black SOR: two params buffers with different color flags
  var redBuf = new ArrayBuffer(160);
  var redU32 = new Uint32Array(redBuf);
  new Float32Array(redBuf).set(new Float32Array(gpuParamsBuf.size ? 40 : 40)); // will be overwritten by uploadParams
  redU32[32] = 0; // color = red
  gpuParamsRedBuf = gpuDevice.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  var blackBuf = new ArrayBuffer(160);
  var blackU32 = new Uint32Array(blackBuf);
  blackU32[32] = 1; // color = black
  gpuParamsBlackBuf = gpuDevice.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  gpuPoissonBindGroupRed = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsRedBuf } },
    ]
  });
  gpuPoissonBindGroupBlack = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBlackBuf } },
    ]
  });
  gpuPoissonBindGroupSwapRed = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsRedBuf } },
    ]
  });
  gpuPoissonBindGroupSwapBlack = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBlackBuf } },
    ]
  });

  // Deep Poisson reuses the same red/black approach
  gpuDeepPoissonBindGroupRed = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsRedBuf } },
    ]
  });
  gpuDeepPoissonBindGroupBlack = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBlackBuf } },
    ]
  });

  // EnforceBC: reads/writes psi and zeta, reads mask
  gpuEnforceBCBindGroup = gpuDevice.createBindGroup({
    layout: gpuEnforceBCPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });

  gpuEnforceBCBindGroupSwap = gpuDevice.createBindGroup({
    layout: gpuEnforceBCPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });

  // Temperature: reads psi, tempIn, deepTempIn, depth -> writes tempOut, deepTempOut
  gpuTemperatureBindGroup = gpuDevice.createBindGroup({
    layout: gpuTemperaturePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuTempBuf } },
      { binding: 2, resource: { buffer: gpuTempNewBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuDeepTempBuf } },
      { binding: 6, resource: { buffer: gpuDeepTempNewBuf } },
      { binding: 7, resource: { buffer: gpuDepthBuf } },
    ]
  });

  // Swapped: reads psi, tempNew, deepTempNew, depth -> writes temp, deepTemp
  gpuSwapTemperatureBindGroup = gpuDevice.createBindGroup({
    layout: gpuTemperaturePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuTempNewBuf } },
      { binding: 2, resource: { buffer: gpuTempBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuDeepTempNewBuf } },
      { binding: 6, resource: { buffer: gpuDeepTempBuf } },
      { binding: 7, resource: { buffer: gpuDepthBuf } },
    ]
  });
  // Deep timestep: reads deepPsi, deepZeta, surfacePsi -> writes deepZetaNew
  gpuDeepTimestepBindGroup = gpuDevice.createBindGroup({
    layout: gpuDeepTimestepPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuDeepZetaNewBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuPsiBuf } },
      { binding: 6, resource: { buffer: gpuDeepTempBuf } },
    ]
  });
  gpuSwapDeepTimestepBindGroup = gpuDevice.createBindGroup({
    layout: gpuDeepTimestepPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuPsiBuf } },
      { binding: 6, resource: { buffer: gpuDeepTempBuf } },
    ]
  });

  // Deep Poisson: reuses pipeline, different buffers
  gpuDeepPoissonBindGroup = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });
  gpuDeepPoissonBindGroupSwap = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });

  // Deep enforceBC: reuses pipeline, different buffers
  gpuDeepEnforceBCBindGroup = gpuDevice.createBindGroup({
    layout: gpuEnforceBCPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });
  gpuDeepEnforceBCBindGroupSwap = gpuDevice.createBindGroup({
    layout: gpuEnforceBCPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });

  // ── COARSE-GRID POISSON SETUP ──
  var coarseBufSize = COARSE_NX * COARSE_NY * 4;
  gpuCoarsePsiBuf = gpuDevice.createBuffer({ size: coarseBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuCoarseZetaBuf = gpuDevice.createBuffer({ size: coarseBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuCoarseMaskBuf = gpuDevice.createBuffer({ size: coarseBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuCoarseDeepPsiBuf = gpuDevice.createBuffer({ size: coarseBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuCoarseDeepZetaBuf = gpuDevice.createBuffer({ size: coarseBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  // Build coarse mask: ocean if any fine cell in block is ocean
  var coarseMask = new Uint32Array(COARSE_NX * COARSE_NY);
  var rxM = NX / COARSE_NX, ryM = NY / COARSE_NY;
  for (var cj = 0; cj < COARSE_NY; cj++) {
    for (var ci = 0; ci < COARSE_NX; ci++) {
      var ocean = 0;
      for (var dj = 0; dj < ryM && !ocean; dj++)
        for (var di = 0; di < rxM && !ocean; di++)
          if (mask[(cj * ryM + dj) * NX + ci * rxM + di]) ocean = 1;
      coarseMask[cj * COARSE_NX + ci] = ocean;
    }
  }
  gpuDevice.queue.writeBuffer(gpuCoarseMaskBuf, 0, coarseMask);

  // DS params buffer (shared by downsample + upsample)
  var dsBuf = new ArrayBuffer(16);
  new Uint32Array(dsBuf).set([NX, NY, COARSE_NX, COARSE_NY]);
  gpuDSParamsBuf = gpuDevice.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  gpuDevice.queue.writeBuffer(gpuDSParamsBuf, 0, dsBuf);

  // Coarse params (same Params struct, but nx/ny/dx/dy for coarse grid)
  var coarseParamsBuf = new ArrayBuffer(160);
  var cf32 = new Float32Array(coarseParamsBuf);
  var cu32 = new Uint32Array(coarseParamsBuf);
  cu32[0] = COARSE_NX; cu32[1] = COARSE_NY;
  cf32[2] = 1.0 / (COARSE_NX - 1); cf32[3] = 1.0 / (COARSE_NY - 1);
  // Rest of params don't matter for Poisson (only uses nx/ny/dx/dy/mask)
  cu32[32] = 0; // red
  gpuCoarseParamsRedBuf = gpuDevice.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  gpuDevice.queue.writeBuffer(gpuCoarseParamsRedBuf, 0, coarseParamsBuf);
  cu32[32] = 1; // black
  gpuCoarseParamsBlackBuf = gpuDevice.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  gpuDevice.queue.writeBuffer(gpuCoarseParamsBlackBuf, 0, coarseParamsBuf);

  // Downsample + upsample pipelines
  gpuDownsamplePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: downsampleShaderCode }), entryPoint: 'main' }
  });
  gpuUpsamplePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: upsampleShaderCode }), entryPoint: 'main' }
  });

  // Downsample bind groups (surface: zeta → coarse zeta)
  var dsLayout = gpuDownsamplePipeline.getBindGroupLayout(0);
  gpuDownsampleBindGroup = gpuDevice.createBindGroup({ layout: dsLayout, entries: [
    { binding: 0, resource: { buffer: gpuZetaBuf } },
    { binding: 1, resource: { buffer: gpuCoarseZetaBuf } },
    { binding: 2, resource: { buffer: gpuDSParamsBuf } },
  ]});
  gpuDownsampleSwapBindGroup = gpuDevice.createBindGroup({ layout: dsLayout, entries: [
    { binding: 0, resource: { buffer: gpuZetaNewBuf } },
    { binding: 1, resource: { buffer: gpuCoarseZetaBuf } },
    { binding: 2, resource: { buffer: gpuDSParamsBuf } },
  ]});
  // Deep downsample (always reads from gpuDeepZetaBuf after swap)
  gpuDownsampleDeepBindGroup = gpuDevice.createBindGroup({ layout: dsLayout, entries: [
    { binding: 0, resource: { buffer: gpuDeepZetaBuf } },
    { binding: 1, resource: { buffer: gpuCoarseDeepZetaBuf } },
    { binding: 2, resource: { buffer: gpuDSParamsBuf } },
  ]});
  gpuDownsampleDeepSwapBindGroup = gpuDevice.createBindGroup({ layout: dsLayout, entries: [
    { binding: 0, resource: { buffer: gpuDeepZetaNewBuf } },
    { binding: 1, resource: { buffer: gpuCoarseDeepZetaBuf } },
    { binding: 2, resource: { buffer: gpuDSParamsBuf } },
  ]});

  // Upsample bind groups (coarse psi → fine psi)
  var usLayout = gpuUpsamplePipeline.getBindGroupLayout(0);
  gpuUpsampleBindGroup = gpuDevice.createBindGroup({ layout: usLayout, entries: [
    { binding: 0, resource: { buffer: gpuCoarsePsiBuf } },
    { binding: 1, resource: { buffer: gpuPsiBuf } },
    { binding: 2, resource: { buffer: gpuDSParamsBuf } },
  ]});
  gpuUpsampleDeepBindGroup = gpuDevice.createBindGroup({ layout: usLayout, entries: [
    { binding: 0, resource: { buffer: gpuCoarseDeepPsiBuf } },
    { binding: 1, resource: { buffer: gpuDeepPsiBuf } },
    { binding: 2, resource: { buffer: gpuDSParamsBuf } },
  ]});

  // Coarse SOR bind groups (reuse existing Poisson pipeline, coarse buffers)
  var pLayout = gpuPoissonPipeline.getBindGroupLayout(0);
  gpuCoarsePoissonRedBG = gpuDevice.createBindGroup({ layout: pLayout, entries: [
    { binding: 0, resource: { buffer: gpuCoarsePsiBuf } },
    { binding: 1, resource: { buffer: gpuCoarseZetaBuf } },
    { binding: 2, resource: { buffer: gpuCoarseMaskBuf } },
    { binding: 3, resource: { buffer: gpuCoarseParamsRedBuf } },
  ]});
  gpuCoarsePoissonBlackBG = gpuDevice.createBindGroup({ layout: pLayout, entries: [
    { binding: 0, resource: { buffer: gpuCoarsePsiBuf } },
    { binding: 1, resource: { buffer: gpuCoarseZetaBuf } },
    { binding: 2, resource: { buffer: gpuCoarseMaskBuf } },
    { binding: 3, resource: { buffer: gpuCoarseParamsBlackBuf } },
  ]});
  gpuCoarseDeepPoissonRedBG = gpuDevice.createBindGroup({ layout: pLayout, entries: [
    { binding: 0, resource: { buffer: gpuCoarseDeepPsiBuf } },
    { binding: 1, resource: { buffer: gpuCoarseDeepZetaBuf } },
    { binding: 2, resource: { buffer: gpuCoarseMaskBuf } },
    { binding: 3, resource: { buffer: gpuCoarseParamsRedBuf } },
  ]});
  gpuCoarseDeepPoissonBlackBG = gpuDevice.createBindGroup({ layout: pLayout, entries: [
    { binding: 0, resource: { buffer: gpuCoarseDeepPsiBuf } },
    { binding: 1, resource: { buffer: gpuCoarseDeepZetaBuf } },
    { binding: 2, resource: { buffer: gpuCoarseMaskBuf } },
    { binding: 3, resource: { buffer: gpuCoarseParamsBlackBuf } },
  ]});

  console.log('Coarse-grid Poisson initialized: ' + COARSE_NX + 'x' + COARSE_NY);
}

var gpuPoissonBindGroupSwap, gpuEnforceBCBindGroupSwap;
var gpuDeepTimestepBindGroup, gpuSwapDeepTimestepBindGroup;
var gpuDeepPoissonBindGroup, gpuDeepPoissonBindGroupSwap;
var gpuDeepEnforceBCBindGroup, gpuDeepEnforceBCBindGroupSwap;
// Red-Black SOR Poisson bind groups + params buffers
var gpuParamsRedBuf, gpuParamsBlackBuf;
var gpuPoissonBindGroupRed, gpuPoissonBindGroupBlack;
var gpuPoissonBindGroupSwapRed, gpuPoissonBindGroupSwapBlack;
var gpuDeepPoissonBindGroupRed, gpuDeepPoissonBindGroupBlack;

function initGPURenderPipeline() {
  var renderCanvas = document.getElementById('gpu-render-canvas');
  gpuCanvasCtx = renderCanvas.getContext('webgpu');
  if (!gpuCanvasCtx) { console.warn('Could not get webgpu context for render canvas'); return false; }

  gpuRenderFormat = navigator.gpu.getPreferredCanvasFormat();
  gpuCanvasCtx.configure({
    device: gpuDevice,
    format: gpuRenderFormat,
    alphaMode: 'premultiplied'
  });

  // Render params uniform: nx, ny, fieldMode, pad, absMax, maxSpd, canvasW, canvasH
  gpuRenderParamsBuf = gpuDevice.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  var vertModule = gpuDevice.createShaderModule({ code: renderVertexShaderCode });
  var fragModule = gpuDevice.createShaderModule({ code: renderFragmentShaderCode });

  // Create bind group layout explicitly so we can share buffers
  var renderBGLayout = gpuDevice.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ]
  });

  var renderPipelineLayout = gpuDevice.createPipelineLayout({ bindGroupLayouts: [renderBGLayout] });

  gpuRenderPipeline = gpuDevice.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: { module: vertModule, entryPoint: 'main' },
    fragment: {
      module: fragModule,
      entryPoint: 'main',
      targets: [{
        format: gpuRenderFormat,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
        }
      }]
    },
    primitive: { topology: 'triangle-list' }
  });

  gpuRenderBindGroup = gpuDevice.createBindGroup({
    layout: renderBGLayout,
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuTempBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuRenderParamsBuf } },
    ]
  });

  gpuRenderEnabled = true;
  return true;
}

function gpuRenderField() {
  if (!gpuRenderEnabled) return;

  // Compute absMax/maxSpd on GPU-side via a quick scan of the last readback data
  var absMax = 1.0;
  var maxSpd = 1.0;
  var fieldMode = 3; // default temp
  if (showField === 'psi') {
    fieldMode = 0;
    absMax = 0;
    for (var k = 0; k < NX * NY; k++) { var a = Math.abs(psi[k]); if (a > absMax) absMax = a; }
    if (absMax < 1e-30) absMax = 1;
  } else if (showField === 'vort') {
    fieldMode = 1;
    absMax = 0;
    for (var k = 0; k < NX * NY; k++) { var a = Math.abs(zeta[k]); if (a > absMax) absMax = a; }
    if (absMax < 1e-30) absMax = 1;
  } else if (showField === 'speed') {
    fieldMode = 2;
    maxSpd = 0;
    for (var j = 1; j < NY - 1; j++) for (var i = 1; i < NX - 1; i++) {
      var vel = getVel(i, j);
      var s = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (s > maxSpd) maxSpd = s;
    }
    if (maxSpd < 1e-30) maxSpd = 1;
  }

  // Upload render params
  var buf = new ArrayBuffer(48);
  var u32v = new Uint32Array(buf);
  var f32v = new Float32Array(buf);
  u32v[0] = NX;
  u32v[1] = NY;
  u32v[2] = fieldMode;
  u32v[3] = 0; // pad
  f32v[4] = absMax;
  f32v[5] = maxSpd;
  f32v[6] = W; // canvas width
  f32v[7] = H; // canvas height
  f32v[8] = simTime;
  u32v[9] = 0; u32v[10] = 0; u32v[11] = 0; // pad
  gpuDevice.queue.writeBuffer(gpuRenderParamsBuf, 0, buf);

  var textureView = gpuCanvasCtx.getCurrentTexture().createView();
  var encoder = gpuDevice.createCommandEncoder();
  var pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: textureView,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: 'clear',
      storeOp: 'store'
    }]
  });
  pass.setPipeline(gpuRenderPipeline);
  pass.setBindGroup(0, gpuRenderBindGroup);
  pass.draw(3);
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
}

function uploadParams() {
  // Params struct: nx(u32), ny(u32), dx(f32), dy(f32), dt(f32), beta(f32), r(f32), A(f32),
  //   windStrength(f32), doubleGyre(u32), alphaT(f32), simTime(f32), yearSpeed(f32), freshwater(f32), pad, pad
  var buf = new ArrayBuffer(160);
  var u32 = new Uint32Array(buf);
  var f32 = new Float32Array(buf);
  u32[0] = NX;
  u32[1] = NY;
  f32[2] = dx;
  f32[3] = dy;
  f32[4] = dt;
  f32[5] = beta;
  f32[6] = r_friction;
  f32[7] = A_visc;
  f32[8] = windStrength;
  u32[9] = doubleGyre ? 1 : 0;
  f32[10] = alpha_T;
  f32[11] = simTime;
  f32[12] = yearSpeed;
  f32[13] = freshwaterForcing;
  f32[14] = globalTempOffset;
  f32[15] = gamma_mix;
  f32[16] = gamma_deep_form;
  f32[17] = kappa_deep;
  f32[18] = H_surface;
  f32[19] = H_deep;
  f32[20] = F_couple_s; f32[21] = F_couple_d;
  f32[22] = S_solar; f32[23] = A_olr;
  f32[24] = B_olr; f32[25] = kappa_diff;
  f32[26] = r_deep; f32[27] = 0.02; // landHeatK
  // Salinity parameters
  f32[28] = beta_S;             // haline contraction coefficient
  f32[29] = kappa_sal;          // salinity diffusion
  f32[30] = kappa_deep_sal;     // deep salinity diffusion
  f32[31] = salRestoringRate;   // surface salinity restoring rate
  u32[32] = 0; u32[33] = 0; u32[34] = 0; u32[35] = 0; // pad to 160
  // Note: in WGSL buf must be 16-byte aligned, 160 = 40 * 4 bytes, OK
  // But uniform buffer size must match struct size exactly
  // 36 used fields * 4 = 144 bytes, pad to 160 for alignment
  gpuDevice.queue.writeBuffer(gpuParamsBuf, 0, buf);
  // Red-black SOR params: copy main params, then set color flag
  if (gpuParamsRedBuf) {
    u32[32] = 0; // red
    gpuDevice.queue.writeBuffer(gpuParamsRedBuf, 0, buf);
    u32[32] = 1; // black
    gpuDevice.queue.writeBuffer(gpuParamsBlackBuf, 0, buf);
    u32[32] = 0; // restore
  }
}

var POISSON_ITERS = 80;        // 1024x512 needs more iterations for SOR convergence
var DEEP_POISSON_ITERS = 40;   // deep layer also needs more at higher resolution

function gpuRunSteps(nSteps) {
  uploadParams();

  var wgX = Math.ceil(NX / 8);
  var wgY = Math.ceil(NY / 8);
  var wgLinear = Math.ceil((NX * NY) / 64);

  var encoder = gpuDevice.createCommandEncoder();

  for (var s = 0; s < nSteps; s++) {
    var isEven = (s % 2 === 0);

    // Timestep: compute zetaNew from zeta and psi (reads temp for buoyancy)
    var tsPass = encoder.beginComputePass();
    tsPass.setPipeline(gpuTimestepPipeline);
    tsPass.setBindGroup(0, isEven ? gpuTimestepBindGroup : gpuSwapTimestepBindGroup);
    tsPass.dispatchWorkgroups(wgX, wgY);
    tsPass.end();

    // Temperature step: advect and force temperature
    var tempPass = encoder.beginComputePass();
    tempPass.setPipeline(gpuTemperaturePipeline);
    tempPass.setBindGroup(0, isEven ? gpuTemperatureBindGroup : gpuSwapTemperatureBindGroup);
    tempPass.dispatchWorkgroups(wgX, wgY);
    tempPass.end();

    // After timestep, the "new" zeta is in the opposite buffer.
    // Enforce BC on psi and the new zeta
    var bcPass = encoder.beginComputePass();
    bcPass.setPipeline(gpuEnforceBCPipeline);
    bcPass.setBindGroup(0, isEven ? gpuEnforceBCBindGroupSwap : gpuEnforceBCBindGroup);
    bcPass.dispatchWorkgroups(wgLinear);
    bcPass.end();

    // Surface Poisson: downsample zeta → coarse SOR → upsample psi
    var coarseWgLinear = Math.ceil((COARSE_NX * COARSE_NY) / 64);
    var coarseWgX = Math.ceil(COARSE_NX / 8), coarseWgY = Math.ceil(COARSE_NY / 8);
    var fineWgLinear = Math.ceil((NX * NY) / 64);

    // Downsample zeta to coarse grid
    var dsPass = encoder.beginComputePass();
    dsPass.setPipeline(gpuDownsamplePipeline);
    dsPass.setBindGroup(0, isEven ? gpuDownsampleSwapBindGroup : gpuDownsampleBindGroup);
    dsPass.dispatchWorkgroups(coarseWgLinear);
    dsPass.end();

    // SOR on coarse grid (25 iterations converges at 256x128)
    for (var pi = 0; pi < COARSE_POISSON_ITERS; pi++) {
      var cpR = encoder.beginComputePass();
      cpR.setPipeline(gpuPoissonPipeline);
      cpR.setBindGroup(0, gpuCoarsePoissonRedBG);
      cpR.dispatchWorkgroups(coarseWgX, coarseWgY);
      cpR.end();
      var cpB = encoder.beginComputePass();
      cpB.setPipeline(gpuPoissonPipeline);
      cpB.setBindGroup(0, gpuCoarsePoissonBlackBG);
      cpB.dispatchWorkgroups(coarseWgX, coarseWgY);
      cpB.end();
    }

    // Upsample psi back to fine grid
    var usPass = encoder.beginComputePass();
    usPass.setPipeline(gpuUpsamplePipeline);
    usPass.setBindGroup(0, gpuUpsampleBindGroup);
    usPass.dispatchWorkgroups(fineWgLinear);
    usPass.end();

    // Deep layer: only update every 4th step (deep circulation is 40x slower than surface)
    // Saves ~45 dispatches per skipped step
    if (s % 4 === 0) {
      var deepTsPass = encoder.beginComputePass();
      deepTsPass.setPipeline(gpuDeepTimestepPipeline);
      deepTsPass.setBindGroup(0, isEven ? gpuDeepTimestepBindGroup : gpuSwapDeepTimestepBindGroup);
      deepTsPass.dispatchWorkgroups(wgX, wgY);
      deepTsPass.end();

      var deepBcPass = encoder.beginComputePass();
      deepBcPass.setPipeline(gpuEnforceBCPipeline);
      deepBcPass.setBindGroup(0, isEven ? gpuDeepEnforceBCBindGroupSwap : gpuDeepEnforceBCBindGroup);
      deepBcPass.dispatchWorkgroups(wgLinear);
      deepBcPass.end();

      // Deep Poisson: downsample → coarse SOR → upsample
      var ddsPass = encoder.beginComputePass();
      ddsPass.setPipeline(gpuDownsamplePipeline);
      ddsPass.setBindGroup(0, isEven ? gpuDownsampleDeepSwapBindGroup : gpuDownsampleDeepBindGroup);
      ddsPass.dispatchWorkgroups(coarseWgLinear);
      ddsPass.end();

      for (var dpi = 0; dpi < COARSE_POISSON_ITERS; dpi++) {
        var dcpR = encoder.beginComputePass();
        dcpR.setPipeline(gpuPoissonPipeline);
        dcpR.setBindGroup(0, gpuCoarseDeepPoissonRedBG);
        dcpR.dispatchWorkgroups(coarseWgX, coarseWgY);
        dcpR.end();
        var dcpB = encoder.beginComputePass();
        dcpB.setPipeline(gpuPoissonPipeline);
        dcpB.setBindGroup(0, gpuCoarseDeepPoissonBlackBG);
        dcpB.dispatchWorkgroups(coarseWgX, coarseWgY);
        dcpB.end();
      }

      var dusPass = encoder.beginComputePass();
      dusPass.setPipeline(gpuUpsamplePipeline);
      dusPass.setBindGroup(0, gpuUpsampleDeepBindGroup);
      dusPass.dispatchWorkgroups(fineWgLinear);
      dusPass.end();
    } // end deep layer skip
  }

  // After all steps, ensure final results are in primary buffers
  if (nSteps % 2 !== 0) {
    encoder.copyBufferToBuffer(gpuZetaNewBuf, 0, gpuZetaBuf, 0, NX * NY * 4);
    encoder.copyBufferToBuffer(gpuTempNewBuf, 0, gpuTempBuf, 0, NX * NY * 4 * 2);   // T + S stacked
    encoder.copyBufferToBuffer(gpuDeepTempNewBuf, 0, gpuDeepTempBuf, 0, NX * NY * 4 * 2); // Td + Sd stacked
    encoder.copyBufferToBuffer(gpuDeepZetaNewBuf, 0, gpuDeepZetaBuf, 0, NX * NY * 4);
  }

  // Readback when needed
  var needReadback = gpuRenderEnabled ? (readbackFrameCounter % READBACK_INTERVAL === 0) : true;
  if (needReadback) {
    encoder.copyBufferToBuffer(gpuPsiBuf, 0, gpuReadbackBuf, 0, NX * NY * 4);
    encoder.copyBufferToBuffer(gpuZetaBuf, 0, gpuZetaReadbackBuf, 0, NX * NY * 4);
    encoder.copyBufferToBuffer(gpuTempBuf, 0, gpuTempReadbackBuf, 0, NX * NY * 4 * 2);   // T + S
    encoder.copyBufferToBuffer(gpuDeepTempBuf, 0, gpuDeepTempReadbackBuf, 0, NX * NY * 4 * 2); // Td + Sd
    encoder.copyBufferToBuffer(gpuDeepPsiBuf, 0, gpuDeepPsiReadbackBuf, 0, NX * NY * 4);
  }

  gpuDevice.queue.submit([encoder.finish()]);
  totalSteps += nSteps;
  simTime += nSteps * dt * yearSpeed;
}

var readbackPending = false;

async function gpuReadback() {
  if (readbackPending) return;
  readbackPending = true;
  try {
    await gpuReadbackBuf.mapAsync(GPUMapMode.READ);
    var data = new Float32Array(gpuReadbackBuf.getMappedRange().slice(0));
    gpuReadbackBuf.unmap();
    psi = data;

    await gpuZetaReadbackBuf.mapAsync(GPUMapMode.READ);
    var zData = new Float32Array(gpuZetaReadbackBuf.getMappedRange().slice(0));
    gpuZetaReadbackBuf.unmap();
    zeta = zData;

    // Stacked readback: first NX*NY = temperature, second NX*NY = salinity
    await gpuTempReadbackBuf.mapAsync(GPUMapMode.READ);
    var tSData = new Float32Array(gpuTempReadbackBuf.getMappedRange().slice(0));
    gpuTempReadbackBuf.unmap();
    temp = tSData.subarray(0, NX * NY);
    sal = tSData.subarray(NX * NY, NX * NY * 2);

    await gpuDeepTempReadbackBuf.mapAsync(GPUMapMode.READ);
    var dSData = new Float32Array(gpuDeepTempReadbackBuf.getMappedRange().slice(0));
    gpuDeepTempReadbackBuf.unmap();
    deepTemp = dSData.subarray(0, NX * NY);
    deepSal = dSData.subarray(NX * NY, NX * NY * 2);

    await gpuDeepPsiReadbackBuf.mapAsync(GPUMapMode.READ);
    var dpData = new Float32Array(gpuDeepPsiReadbackBuf.getMappedRange().slice(0));
    gpuDeepPsiReadbackBuf.unmap();
    deepPsi = dpData;

    // Update sea ice from readback temperature (thermodynamic phase change)
    if (seaIce && temp) {
      for (var ik = 0; ik < NX * NY; ik++) {
        if (!mask[ik]) continue;
        var hIce = seaIce[ik];
        var T = temp[ik];
        if (T < T_FREEZE && hIce < ICE_MAX) {
          seaIce[ik] = Math.min(ICE_MAX, hIce + (T_FREEZE - T) * ICE_GROW_RATE * dt * READBACK_INTERVAL);
        } else if (T > T_FREEZE && hIce > 0) {
          seaIce[ik] = Math.max(0, hIce - (T - T_FREEZE) * ICE_MELT_RATE * dt * READBACK_INTERVAL);
        }
      }
    }

    // ── ATMOSPHERE DIFFUSION STEP ──
    // Run on CPU during readback. Couples land and ocean via air temperature.
    // This is the ONLY mechanism that transports oceanic warmth inland.
    atmosphereStep();

  } catch (e) {
    // readback failed, skip this frame
  }
  readbackPending = false;
}

// 1-layer atmosphere on COARSE grid (128×64) for performance.
// Jacobi smoothing at coarse res is cheap and each pass covers ~8 fine cells.
var ATM_CNX = 128, ATM_CNY = 64;
var ATM_SMOOTH_PASSES = 8;    // 8 passes at coarse res ≈ 60+ fine cells range
var ATM_SURFACE_RELAX = 0.15;
var ATM_FEEDBACK_SST = 0.003;
var ATM_FEEDBACK_LAND = 0.05;
var atmCoarse = null, atmScratch = null; // pre-allocated

function atmosphereStep() {
  if (!airTemp || !temp || !mask) return;
  var N = NX * NY;
  var cnx = ATM_CNX, cny = ATM_CNY;
  var cn = cnx * cny;
  var rx = NX / cnx, ry = NY / cny; // 8, 8

  // Lazy init coarse buffers
  if (!atmCoarse) { atmCoarse = new Float32Array(cn); atmScratch = new Float32Array(cn); }

  // Step 1: Downsample surface temp to coarse grid + relax air toward it
  for (var cj = 0; cj < cny; cj++) {
    for (var ci = 0; ci < cnx; ci++) {
      var ck = cj * cnx + ci;
      // Sample center of coarse cell
      var fi = Math.floor(ci * rx + rx * 0.5);
      var fj = Math.floor(cj * ry + ry * 0.5);
      var fk = fj * NX + fi;
      var surfT = mask[fk] ? temp[fk] : (landTempField ? landTempField[fk] : 15);
      atmCoarse[ck] += ATM_SURFACE_RELAX * (surfT - atmCoarse[ck]);
    }
  }

  // Step 2: Jacobi smoothing on coarse grid (very fast: 128×64 = 8192 cells)
  for (var pass = 0; pass < ATM_SMOOTH_PASSES; pass++) {
    atmScratch.set(atmCoarse);
    for (var cj = 1; cj < cny - 1; cj++) {
      for (var ci = 0; ci < cnx; ci++) {
        var ck = cj * cnx + ci;
        var cip1 = (ci + 1) % cnx, cim1 = (ci - 1 + cnx) % cnx;
        var avg = 0.25 * (atmScratch[cj*cnx+cip1] + atmScratch[cj*cnx+cim1] +
                          atmScratch[(cj+1)*cnx+ci] + atmScratch[(cj-1)*cnx+ci]);
        atmCoarse[ck] = 0.5 * atmScratch[ck] + 0.5 * avg;
      }
    }
  }

  // Step 3: Upsample coarse air temp to fine grid + apply feedback
  var sstChanged = false;
  for (var j = 0; j < NY; j++) {
    // Bilinear from coarse grid
    var cy = (j / (NY - 1)) * (cny - 1);
    var cj0 = Math.min(Math.floor(cy), cny - 2);
    var fy = cy - cj0;
    for (var i = 0; i < NX; i++) {
      var k = j * NX + i;
      var cx = (i / (NX - 1)) * (cnx - 1);
      var ci0 = Math.min(Math.floor(cx), cnx - 2);
      var fx = cx - ci0;
      var a00 = atmCoarse[cj0 * cnx + ci0];
      var a10 = atmCoarse[cj0 * cnx + ci0 + 1];
      var a01 = atmCoarse[(cj0 + 1) * cnx + ci0];
      var a11 = atmCoarse[(cj0 + 1) * cnx + ci0 + 1];
      var airT = a00*(1-fx)*(1-fy) + a10*fx*(1-fy) + a01*(1-fx)*fy + a11*fx*fy;
      airTemp[k] = airT;

      if (mask[k]) {
        var correction = ATM_FEEDBACK_SST * (airT - temp[k]);
        if (Math.abs(correction) > 0.001) { temp[k] += correction; sstChanged = true; }
      } else if (landTempField) {
        landTempField[k] += ATM_FEEDBACK_LAND * (airT - landTempField[k]);
      }
    }
  }

  // Upload corrected SST back to GPU if changed
  if (sstChanged && gpuDevice && gpuTempBuf) {
    if (!atmosphereStep._uploadBuf) atmosphereStep._uploadBuf = new Float32Array(NX * NY * 2);
    var surfTracer = atmosphereStep._uploadBuf;
    for (var k = 0; k < N; k++) {
      surfTracer[k] = temp[k];
      surfTracer[k + N] = sal[k];
    }
    gpuDevice.queue.writeBuffer(gpuTempBuf, 0, surfTracer);
  }
}

function gpuReset() {
  psi = new Float32Array(NX * NY);
  zeta = new Float32Array(NX * NY);
  temp = new Float32Array(NX * NY);
  deepTemp = new Float32Array(NX * NY);
  sal = new Float32Array(NX * NY);
  deepSal = new Float32Array(NX * NY);
  seaIce = new Float32Array(NX * NY);
  deepPsi = new Float32Array(NX * NY);
  deepZeta = new Float32Array(NX * NY);
  airTemp = new Float32Array(NX * NY);
  generateWindCurlField();
  initStommelSolution();
  initTemperatureField();
  initLandTemp();
  for (var ak = 0; ak < NX * NY; ak++) {
    airTemp[ak] = mask[ak] ? temp[ak] : (landTempField ? landTempField[ak] : 15);
  }
  gpuDevice.queue.writeBuffer(gpuPsiBuf, 0, psi);
  gpuDevice.queue.writeBuffer(gpuZetaBuf, 0, zeta);
  gpuDevice.queue.writeBuffer(gpuZetaNewBuf, 0, new Float32Array(NX * NY));
  // Pack T+S stacked for GPU
  var surfTr = new Float32Array(NX * NY * 2);
  var deepTr = new Float32Array(NX * NY * 2);
  for (var rk = 0; rk < NX * NY; rk++) {
    surfTr[rk] = temp[rk]; surfTr[rk + NX * NY] = sal[rk];
    deepTr[rk] = deepTemp[rk]; deepTr[rk + NX * NY] = deepSal[rk];
  }
  gpuDevice.queue.writeBuffer(gpuTempBuf, 0, surfTr);
  gpuDevice.queue.writeBuffer(gpuTempNewBuf, 0, new Float32Array(NX * NY * 2));
  gpuDevice.queue.writeBuffer(gpuDeepTempBuf, 0, deepTr);
  gpuDevice.queue.writeBuffer(gpuDeepTempNewBuf, 0, new Float32Array(NX * NY * 2));
  deepPsi = new Float32Array(NX * NY);
  deepZeta = new Float32Array(NX * NY);
  gpuDevice.queue.writeBuffer(gpuDeepPsiBuf, 0, deepPsi);
  gpuDevice.queue.writeBuffer(gpuDeepZetaBuf, 0, deepZeta);
  gpuDevice.queue.writeBuffer(gpuDeepZetaNewBuf, 0, new Float32Array(NX * NY));
  totalSteps = 0;
  simTime = 0;
  readbackFrameCounter = 0;
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
  generateDepthField();
  initTemperatureField();
}

function cpuI(i, j) { return j * NX + i; }

function cpuWindCurl(i, j) {
  // Read from precomputed field (observed ERA5 or analytical fallback)
  if (windCurlField) return windStrength * windCurlField[j * NX + i];
  // Fallback if field not yet generated
  var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
  var latRad = lat * Math.PI / 180;
  var shBoost = lat < 0 ? 2.0 : 1.0;
  var polarDamp = Math.abs(lat) > 60 ? 0.7 : 1.0;
  return windStrength * (-Math.cos(3 * latRad) * shBoost * polarDamp) * 2;
}

function cpuCosLat(j) {
  var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
  return Math.max(Math.cos(lat * Math.PI / 180), 0.087); // clamp near poles
}

function cpuBeta(j) {
  var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
  var latRad = lat * Math.PI / 180;
  return beta * Math.cos(latRad);
}

var rhoGS, omegaSOR;

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
  if (NX & (NX - 1)) console.warn('FFT Poisson: NX=' + NX + ' is not power of 2!');
  console.log('FFT Poisson solver initialized: NX=' + NX + ' NY=' + NY);
}

function cpuSolveFFT(psiArr, zetaArr) {
  var tmpR = new Float64Array(NX), tmpI = new Float64Array(NX);
  var hatR = new Float64Array(NX * NY), hatI = new Float64Array(NX * NY);
  for (var j = 0; j < NY; j++) {
    for (var i = 0; i < NX; i++) { tmpR[i] = zetaArr[j*NX+i]; tmpI[i] = 0; }
    fftRadix2(tmpR, tmpI, NX, false);
    for (var m = 0; m < NX; m++) { hatR[m*NY+j] = tmpR[m]; hatI[m*NY+j] = tmpI[m]; }
  }
  var pHR = new Float64Array(NX * NY), pHI = new Float64Array(NX * NY);
  for (var m = 0; m < NX; m++) {
    var km2 = invDx2 * 2 * (Math.cos(2 * Math.PI * m / NX) - 1);
    var b = new Float64Array(NY), dR = new Float64Array(NY), dI = new Float64Array(NY);
    b[0] = 1; b[NY-1] = 1;
    for (var j = 1; j < NY-1; j++) {
      b[j] = km2 - 2 * invDy2;
      dR[j] = hatR[m*NY+j]; dI[j] = hatI[m*NY+j];
    }
    for (var j = 1; j < NY - 1; j++) {
      var cp = (j-1 > 0) ? invDy2 : 0;
      var w = invDy2 / b[j-1];
      b[j] -= w * cp;
      dR[j] -= w * dR[j-1]; dI[j] -= w * dI[j-1];
    }
    pHR[m*NY+(NY-1)] = 0; pHI[m*NY+(NY-1)] = 0;
    for (var j = NY-2; j >= 1; j--) {
      var c = invDy2;
      pHR[m*NY+j] = (dR[j] - c * pHR[m*NY+(j+1)]) / b[j];
      pHI[m*NY+j] = (dI[j] - c * pHI[m*NY+(j+1)]) / b[j];
    }
  }
  for (var j = 0; j < NY; j++) {
    for (var m = 0; m < NX; m++) { tmpR[m] = pHR[m*NY+j]; tmpI[m] = pHI[m*NY+j]; }
    fftRadix2(tmpR, tmpI, NX, true);
    for (var i = 0; i < NX; i++) psiArr[j*NX+i] = tmpR[i];
  }
}

function cpuSolveSOR(nIter) {
  var cx = invDx2, cy = invDy2, cc = -2 * (cx + cy), invCC = 1 / cc;
  for (var iter = 0; iter < nIter; iter++) {
    for (var j = 1; j < NY - 1; j++) for (var i = 0; i < NX; i++) {
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

function cpuSolveDeepSOR(nIter) {
  var cx = invDx2, cy = invDy2, cc = -2 * (cx + cy), invCC = 1 / cc;
  for (var iter = 0; iter < nIter; iter++) {
    for (var j = 1; j < NY - 1; j++) for (var i = 0; i < NX; i++) {
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
    // Vorticity update — damp near coasts, full physics in interior
    if (!mask[cpuI(ip1, j)] || !mask[cpuI(im1, j)] || !mask[cpuI(i, j + 1)] || !mask[cpuI(i, j - 1)]) {
      cpuZetaNew[k] = zeta[k] * 0.9;
    } else if (!mask[cpuI(ip1, j + 1)] || !mask[cpuI(im1, j + 1)] || !mask[cpuI(ip1, j - 1)] || !mask[cpuI(im1, j - 1)]) {
      cpuZetaNew[k] = zeta[k] * 0.95;
    } else {
      var jac = cpuJacobian(i, j);
      var betaV = cpuBeta(j) * (psi[cpuI(ip1, j)] - psi[cpuI(im1, j)]) * 0.5 * invDx;
      var F = cpuWindCurl(i, j);
      var fric = -r_friction * zeta[k];
      var visc = A_visc * cpuLaplacian(zeta, i, j);
      // Density-based buoyancy: -alpha*dT/dx + beta*dS/dx
      var dRhodx_cpu = -alpha_T * (temp[cpuI(ip1, j)] - temp[cpuI(im1, j)]) + beta_S * (sal[cpuI(ip1, j)] - sal[cpuI(im1, j)]);
      var buoyancy = -dRhodx_cpu * 0.5 * invDx;
      var coupling = F_couple_s * (deepPsi[k] - psi[k]);
      cpuZetaNew[k] = Math.max(-500, Math.min(500, zeta[k] + dt * (-jac - betaV + F + fric + visc + buoyancy + coupling)));
    }

    // Temperature equation — one-sided stencil near coasts (always runs)
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
    var dPdx = (pE - pW) * 0.5 * invDx;
    var dPdy = (pN - pS) * 0.5 * invDy;
    var dTdx = (tE - tW) * 0.5 * invDx;
    var dTdy = (tN - tS) * 0.5 * invDy;
    var advec = dPdx * dTdy - dPdy * dTdx;
    var latRad = lat * Math.PI / 180;
    var yearPhase = 2 * Math.PI * simTime / T_YEAR;
    var declination = 23.44 * Math.sin(yearPhase) * Math.PI / 180;
    var cosZenith = Math.cos(latRad) * Math.cos(declination) + Math.sin(latRad) * Math.sin(declination);
    var qSolar = S_solar * Math.max(0, cosZenith);
    // ── THERMODYNAMIC SEA ICE ──
    // Ice thickness in meters. Phase change at T_FREEZE with latent heat buffer.
    var hIce = seaIce ? seaIce[k] : 0;

    // Ice insulates: reduce atmospheric heat exchange when ice is thick
    var iceInsulFactor = hIce > 0 ? 1.0 / (1.0 + hIce / ICE_INSULATION) : 1.0;
    qSolar *= iceInsulFactor;

    // Ice-albedo: ice-covered cells reflect ~50% of incoming solar
    if (hIce > 0.1) {
      qSolar *= 0.5;
    }
    var olr = A_olr - B_olr * globalTempOffset + B_olr * temp[k];
    var qNet = qSolar - olr;
    var lapT = invDx2 * (tE + tW - 2 * temp[k]) + invDy2 * (tN + tS - 2 * temp[k]);
    var diff = kappa_diff * lapT;
    // Land-ocean heat exchange: use actual land temperature field (not fake solar formula)
    var nLand = (!mask[ke] ? 1 : 0) + (!mask[kw] ? 1 : 0) + (!mask[kn] ? 1 : 0) + (!mask[ks] ? 1 : 0);
    var nOcean = 4 - nLand;
    var landFlux = 0;
    if (nLand > 0 && nOcean > 0 && landTempField) {
      // Average temperature of adjacent land cells
      var landSum = 0, landCnt = 0;
      if (!mask[ke]) { landSum += landTempField[ke]; landCnt++; }
      if (!mask[kw]) { landSum += landTempField[kw]; landCnt++; }
      if (!mask[kn]) { landSum += landTempField[kn]; landCnt++; }
      if (!mask[ks]) { landSum += landTempField[ks]; landCnt++; }
      var landT = landCnt > 0 ? landSum / landCnt : temp[k];
      var rawFlux = 0.02 * (landT - temp[k]) * (nOcean / 4);
      landFlux = Math.max(-0.5, Math.min(0.5, rawFlux));
    }
    // Temperature: freshwater no longer cools — only affects salinity
    cpuTempNew[k] = Math.max(-10, Math.min(40, temp[k] + dt * (-advec + qNet + diff + landFlux)));

    // ── PHASE CHANGE: latent heat buffer at freezing point ──
    if (seaIce) {
      var T = cpuTempNew[k];
      if (T < T_FREEZE && hIce < ICE_MAX) {
        // Freezing: excess cold goes into ice growth, SST stays at T_FREEZE
        var dT = T_FREEZE - T;  // how far below freezing
        var dh = dT * ICE_GROW_RATE * dt;  // ice thickness gained
        seaIce[k] = Math.min(ICE_MAX, hIce + dh);
        cpuTempNew[k] = T_FREEZE;  // latent heat holds water at freezing
        // Brine rejection: freezing expels salt
        cpuSalNew ? 0 : 0; // handled below
      } else if (T > T_FREEZE && hIce > 0) {
        // Melting: excess heat goes into melting ice, SST stays at T_FREEZE
        var dT2 = T - T_FREEZE;
        var dh2 = dT2 * ICE_MELT_RATE * dt;
        var melt = Math.min(hIce, dh2);
        seaIce[k] = hIce - melt;
        cpuTempNew[k] = T_FREEZE + (dT2 - melt / (ICE_MELT_RATE * dt)) * (hIce > melt ? 0 : 1);
        // Simplified: if all ice melts, remaining heat warms water; otherwise stays at T_FREEZE
        if (seaIce[k] > 0.01) cpuTempNew[k] = T_FREEZE;
      } else {
        seaIce[k] = hIce; // no change
      }
    }

    // ── CPU SALINITY ──
    var sE = mask[ke] ? sal[ke] : sal[k];
    var sW = mask[kw] ? sal[kw] : sal[k];
    var sN2 = mask[kn] ? sal[kn] : sal[k];
    var sS2 = mask[ks] ? sal[ks] : sal[k];
    var dSdx = (sE - sW) * 0.5 * invDx;
    var dSdy = (sN2 - sS2) * 0.5 * invDy;
    var salAdvec = dPdx * dSdy - dPdy * dSdx;
    var lapS = invDx2 * (sE + sW - 2 * sal[k]) + invDy2 * (sN2 + sS2 - 2 * sal[k]);
    var salDiff = kappa_sal * lapS;
    var latRad = lat * Math.PI / 180;
    var salClim = 34.0 + 2.0 * Math.cos(2 * latRad) - 0.5 * Math.cos(4 * latRad);
    var salRestore = salRestoringRate * (salClim - sal[k]);
    var fwSal = 0;
    if (y > 0.75) fwSal = -freshwaterForcing * 3.0 * (y - 0.75) * 4.0;
    cpuSalNew[k] = Math.max(28, Math.min(40, sal[k] + dt * (-salAdvec + salDiff + salRestore + fwSal)));

    // Two-layer vertical exchange — use observed MLD where available
    var localDepth = depth ? depth[k] : 4000;
    var hSurf = H_surface;
    if (obsMldData && obsMldData.mld) {
      var mldK = obsIndex(lat, lon, obsMldData);
      if (mldK >= 0) {
        var mldVal = obsMldData.mld[mldK];
        if (mldVal > 10 && mldVal < 1000) hSurf = mldVal;
      }
    }
    hSurf = Math.min(hSurf, localDepth);
    var hDeep = Math.max(1, localDepth - hSurf);
    var hasDeep = localDepth > hSurf ? 1 : 0;

    // Density-based vertical mixing — no latitude threshold
    // Surface denser than deep → convective sinking (fast)
    // Surface lighter than deep → diffusive upwelling (slow, Munk's abyssal recipes)
    var rhoSurf = -alpha_T * cpuTempNew[k] + beta_S * cpuSalNew[k];
    var rhoDeep = -alpha_T * deepTemp[k] + beta_S * deepSal[k];
    var gamma;
    if (rhoSurf > rhoDeep) {
      // Convection: dense surface water sinks. Rate proportional to density difference.
      var drho = rhoSurf - rhoDeep;
      gamma = gamma_mix + gamma_deep_form * Math.min(1, drho * 10);
    } else {
      // Stable stratification: slow diffusive upwelling
      gamma = gamma_mix * 0.5;
    }

    var vertExchangeT = gamma * (cpuTempNew[k] - deepTemp[k]) * hasDeep;
    cpuTempNew[k] -= dt * vertExchangeT / hSurf;
    var vertExchangeS = gamma * (cpuSalNew[k] - deepSal[k]) * hasDeep;
    cpuSalNew[k] -= dt * vertExchangeS / hSurf;

    // Deep layer: temperature + salinity
    var dE = mask[ke] ? deepTemp[ke] : deepTemp[k];
    var dW = mask[kw] ? deepTemp[kw] : deepTemp[k];
    var dN = mask[kn] ? deepTemp[kn] : deepTemp[k];
    var dS = mask[ks] ? deepTemp[ks] : deepTemp[k];
    var lapDeep = invDx2 * (dE + dW - 2 * deepTemp[k]) + invDy2 * (dN + dS - 2 * deepTemp[k]);
    var deepDiff = kappa_deep * lapDeep;
    cpuDeepTempNew[k] = Math.max(-5, Math.min(30, deepTemp[k] + dt * (vertExchangeT / hDeep + deepDiff) * hasDeep));

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
  cpuSolveSOR(25);

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
    // Deep buoyancy forcing: density-driven overturning from deep temperature gradients
    // Deep density-based buoyancy
    var dRhodxDeep2 = -alpha_T * (deepTemp[ke2] - deepTemp[kw2]) + beta_S * (deepSal[ke2] - deepSal[kw2]);
    var deepBuoyancy2 = dRhodxDeep2 * 0.5 * invDx;
    cpuDeepZetaNew[k2] = deepZeta[k2] + dt * (-jac2 - betaV2 + fric2 + visc2 + coupling2 + deepBuoyancy2);
  }
  var tmpDZ = deepZeta; deepZeta = cpuDeepZetaNew; cpuDeepZetaNew = tmpDZ;
  for (var k3 = 0; k3 < NX * NY; k3++) { if (!mask[k3]) { deepPsi[k3] = 0; deepZeta[k3] = 0; } }
  cpuSolveDeepSOR(15);

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
  seaIce = new Float64Array(NX * NY);
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
  // Periodic wrapping in x
  i = ((i % NX) + NX) % NX;
  var ip1 = (i + 1) % NX;
  var im1 = (i - 1 + NX) % NX;
  return [
    -(psi[(j + 1) * NX + i] - psi[(j - 1) * NX + i]) * 0.5 * invDy,  // u = -dpsi/dy
    (psi[j * NX + ip1] - psi[j * NX + im1]) * 0.5 * invDx              // v = dpsi/dx
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
    // Periodic wrapping in x
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
// RENDERING
// ============================================================
function psiToRGB(val, absMax) {
  var t = Math.max(-1, Math.min(1, val / (absMax + 1e-30)));
  if (t < 0) {
    var s = -t;
    return [Math.floor(40 + 60 * s), Math.floor(80 + 100 * s), Math.floor(160 + 95 * s)];
  }
  return [Math.floor(200 + 55 * t), Math.floor(100 - 40 * t), Math.floor(80 - 40 * t)];
}

function vortToRGB(val, absMax) {
  var t = Math.max(-1, Math.min(1, val / (absMax + 1e-30)));
  if (t < 0) {
    var s = -t;
    return [Math.floor(30 + 20 * s), Math.floor(60 + 140 * s), Math.floor(40 + 60 * s)];
  }
  return [Math.floor(180 + 75 * t), Math.floor(60 + 40 * t), Math.floor(120 + 60 * t)];
}

function speedToRGB(spd, maxSpd) {
  var t = Math.min(1, spd / (maxSpd + 1e-30));
  if (t < 0.25) { var s = t / 0.25; return [Math.floor(10 + 10 * s), Math.floor(15 + 35 * s), Math.floor(40 + 60 * s)]; }
  if (t < 0.5) { var s = (t - 0.25) / 0.25; return [Math.floor(20 + 30 * s), Math.floor(50 + 80 * s), Math.floor(100 + 80 * s)]; }
  if (t < 0.75) { var s = (t - 0.5) / 0.25; return [Math.floor(50 + 120 * s), Math.floor(130 + 80 * s), Math.floor(180 - 60 * s)]; }
  var s = (t - 0.75) / 0.25;
  return [Math.floor(170 + 80 * s), Math.floor(210 + 30 * s), Math.floor(120 - 80 * s)];
}

function salToRGB(s) {
  // Salinity colormap: 30-38 PSU
  // Fresh (brown/tan) → reference 35 (white) → salty (deep purple)
  var t = Math.max(0, Math.min(1, (s - 30) / 8)); // 0 at 30 PSU, 1 at 38 PSU
  var mid = (35 - 30) / 8; // 0.625
  if (t < mid) {
    var f = t / mid;
    return [Math.floor(140 + 115 * f), Math.floor(100 + 120 * f), Math.floor(60 + 140 * f)]; // brown → white
  } else {
    var f = (t - mid) / (1 - mid);
    return [Math.floor(255 - 155 * f), Math.floor(220 - 140 * f), Math.floor(200 - 60 * f)]; // white → purple
  }
}

function densityToRGB(temp, sal) {
  // Density anomaly: ρ' = -α(T-15) + β(S-35)
  // Light (warm+fresh = buoyant) to dark (cold+salty = dense)
  var rho = -0.05 * (temp - 15) + 0.8 * ((sal || 35) - 35);
  // Normalize: light water (~-1.5) to dense water (~+1.5)
  var t = Math.max(0, Math.min(1, (rho + 1.5) / 3.0));
  // Light=warm yellow, heavy=deep blue
  if (t < 0.5) {
    var f = t / 0.5;
    return [Math.floor(255 - 55 * f), Math.floor(240 - 80 * f), Math.floor(120 + 40 * f)]; // yellow → teal
  } else {
    var f = (t - 0.5) / 0.5;
    return [Math.floor(200 - 170 * f), Math.floor(160 - 130 * f), Math.floor(160 - 40 * f)]; // teal → dark blue
  }
}

function depthToRGB(d) {
  // Light blue (shallow) to dark navy (deep)
  var t = Math.min(1, Math.max(0, d / 4000));
  t = Math.sqrt(t); // expand shallow range
  var r = Math.floor(120 - 100 * t);
  var g = Math.floor(200 - 160 * t);
  var b = Math.floor(220 - 140 * t);
  return [r, g, b];
}

function tempToRGB(t) {
  // Thermal colormap with ICE: white/ice < -1.8, deep blue -1.8 to 5, cyan to green to yellow to red
  if (t > 30) t = 30;
  // Ice: below freezing point of seawater (-1.8C) = white/light blue
  if (t < -1.8) {
    var s = Math.max(0, Math.min(1, (t - (-10)) / 8.2)); // -10 to -1.8
    return [Math.floor(180 + 60 * s), Math.floor(200 + 40 * s), Math.floor(220 + 30 * s)]; // white-ish to pale blue
  }
  if (t < 2) {
    var s = (t - (-1.8)) / 3.8; // -1.8 to 2
    return [Math.floor(240 - 210 * s), Math.floor(240 - 170 * s), Math.floor(250 - 100 * s)]; // pale blue to deep blue
  }
  if (t < 8) {
    var s = (t - 2) / 6;
    return [Math.floor(30 - 10 * s), Math.floor(70 + 130 * s), Math.floor(150 + 70 * s)]; // deep blue to cyan
  }
  if (t < 15) {
    var s = (t - 8) / 7;
    return [Math.floor(20 + 20 * s), Math.floor(200 + 10 * s), Math.floor(220 - 130 * s)]; // cyan to teal/green
  }
  if (t < 22) {
    var s = (t - 15) / 7;
    return [Math.floor(40 + 190 * s), Math.floor(210 + 20 * s), Math.floor(90 - 50 * s)]; // green to yellow
  }
  if (t < 27) {
    var s = (t - 22) / 5;
    return [Math.floor(230 + 20 * s), Math.floor(230 - 100 * s), Math.floor(40 - 10 * s)]; // yellow to orange
  }
  var s = (t - 27) / 3;
  return [Math.floor(250), Math.floor(130 - 80 * s), Math.floor(30 - 20 * s)]; // orange to red
}

// Use ImageData for fast field rendering
var fieldCanvas = document.createElement('canvas');
var fieldCtx;

function initFieldCanvas() {
  fieldCanvas.width = NX;
  fieldCanvas.height = NY;
  fieldCtx = fieldCanvas.getContext('2d');
}

// Land temperature with thermal inertia + altitude lapse rate
var landTempField = null;
var landCanvas = null, landCtx_ = null, landTmpCanvas = null;
var lastLandTime = -999;

var landElevCache = null; // cached elevation lapse rate per cell (avoids obsIndex per frame)

function initLandTemp() {
  if (landTempField && landTempField.length === NX * NY) return;
  landTempField = new Float32Array(NX * NY);
  landElevCache = new Float32Array(NX * NY); // cache elevation lapse
  var hasElev = obsBathyData && obsBathyData.elevation;
  for (var j = 0; j < NY; j++) {
    var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    var cosZ = Math.max(0, Math.cos(lat * Math.PI / 180));
    var baseT = 50 * cosZ - 20;
    for (var i = 0; i < NX; i++) {
      var k = j * NX + i;
      if (mask[k]) continue;
      var elev = 0;
      if (hasElev) {
        var lon = LON0 + (i / (NX - 1)) * (LON1 - LON0);
        var obsK = obsIndex(lat, lon, obsBathyData);
        if (obsK >= 0) elev = obsBathyData.elevation[obsK] || 0;
      }
      landElevCache[k] = 6.5 * elev / 1000; // lapse rate correction (°C)
      landTempField[k] = baseT - landElevCache[k];
    }
  }
}

function drawSeasonalLand() {
  if (!mask || !NX || !NY) return;
  if (!landTempField) initLandTemp();

  if (!landCanvas) {
    landCanvas = document.createElement('canvas');
    landCanvas.width = W; landCanvas.height = H;
    landCtx_ = landCanvas.getContext('2d');
    landTmpCanvas = document.createElement('canvas');
    landTmpCanvas.width = NX; landTmpCanvas.height = NY;
  }

  // Update smoothly — every 0.05 sim time units (~18 updates per year)
  var timeDelta = simTime - lastLandTime;
  if (Math.abs(timeDelta) > 0.05 || lastLandTime < 0) {
    lastLandTime = simTime;

    var yearPhase = 2 * Math.PI * (simTime % T_YEAR) / T_YEAR;
    var decl = 23.44 * Math.sin(yearPhase) * Math.PI / 180;
    var hasElev = obsBathyData && obsBathyData.elevation;

    // Land temp: 75% solar + 25% atmosphere. Uses cached elevation (no obsIndex per frame).
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var latRad = lat * Math.PI / 180;
      var cosZ = Math.cos(latRad) * Math.cos(decl) + Math.sin(latRad) * Math.sin(decl);
      var solarT = 50 * Math.max(0, cosZ) - 20;
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        if (mask[k]) continue;
        var solarTarget = solarT - (landElevCache ? landElevCache[k] : 0);
        var targetT = airTemp ? 0.75 * solarTarget + 0.25 * airTemp[k] : solarTarget;
        landTempField[k] += 0.08 * (targetT - landTempField[k]);
      }
    }

    // Render to pixel buffer
    var tmpCtx = landTmpCanvas.getContext('2d');
    var imgData = tmpCtx.createImageData(NX, NY);
    var px = imgData.data;
    for (var mj = 0; mj < NY; mj++) {
      var dstRow = NY - 1 - mj;
      for (var mi = 0; mi < NX; mi++) {
        var mk = mj * NX + mi;
        var idx = (dstRow * NX + mi) * 4;
        if (!mask[mk]) {
          var rgb = tempToRGB(landTempField[mk]);
          px[idx] = rgb[0]; px[idx+1] = rgb[1]; px[idx+2] = rgb[2]; px[idx+3] = 255;
        }
      }
    }
    tmpCtx.putImageData(imgData, 0, 0);
    landCtx_.clearRect(0, 0, W, H);
    landCtx_.imageSmoothingEnabled = false;
    // Use same half-cell offset as field overlay so land aligns with ocean data
    var ldx = -W / (2 * (NX - 1));
    var ldy = -H / (2 * (NY - 1));
    var ldw = W * NX / (NX - 1);
    var ldh = H * NY / (NY - 1);
    landCtx_.drawImage(landTmpCanvas, ldx, ldy, ldw, ldh);
  }

  ctx.drawImage(landCanvas, 0, 0);
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  drawSeasonalLand();

  // Compute field image
  var imgData = fieldCtx.createImageData(NX, NY);
  var data = imgData.data;

  var absMax = 0;
  var maxSpd = 0;
  var k;

  if (showField === 'psi') {
    for (k = 0; k < NX * NY; k++) { var a = Math.abs(psi[k]); if (a > absMax) absMax = a; }
  } else if (showField === 'deepflow') {
    if (deepPsi) for (k = 0; k < NX * NY; k++) { var a = Math.abs(deepPsi[k]); if (a > absMax) absMax = a; }
  } else if (showField === 'vort') {
    for (k = 0; k < NX * NY; k++) { var a = Math.abs(zeta[k]); if (a > absMax) absMax = a; }
  } else if (showField === 'temp' || showField === 'deeptemp') {
    // temp uses fixed colormap, no normalization needed
  } else {
    for (var j = 1; j < NY - 1; j++) for (var i = 1; i < NX - 1; i++) {
      var vel = getVel(i, j);
      var s = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (s > maxSpd) maxSpd = s;
    }
  }
  if (absMax < 1e-30) absMax = 1;
  if (maxSpd < 1e-30) maxSpd = 1;

  for (var j = 0; j < NY; j++) {
    for (var i = 0; i < NX; i++) {
      var srcK = j * NX + i;
      // Flip vertically: row 0 of image = top = j=NY-1
      var dstRow = NY - 1 - j;
      var dstIdx = (dstRow * NX + i) * 4;

      if (!mask[srcK] && showField !== 'airtemp') {
        // Land: show seasonal temperature or elevation
        var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
        var latRad = lat * Math.PI / 180;
        var yearPhase = 2 * Math.PI * simTime / T_YEAR;
        var decl = 23.44 * Math.sin(yearPhase) * Math.PI / 180;
        var cosZ = Math.cos(latRad) * Math.cos(decl) + Math.sin(latRad) * Math.sin(decl);
        var landT = 50 * Math.max(0, cosZ) - 20; // seasonal land temperature

        // Blend elevation (static) with seasonal temp (dynamic)
        var elev = 0;
        if (obsBathyData && obsBathyData.elevation) {
          var lon = LON0 + (i / (NX - 1)) * (LON1 - LON0);
          var obsK = obsIndex(lat, lon, obsBathyData);
          if (obsK >= 0) elev = obsBathyData.elevation[obsK] || 0;
        }

        // Color: warm land = green/brown, cold land = white/gray, high elevation = lighter
        var r, g, b;
        var elevFade = Math.min(1, elev / 3000); // 0 at sea level, 1 at 3000m+
        if (landT < -5) {
          // Frozen: white-blue, lighter at high elevation
          var f = Math.max(0, Math.min(1, (landT + 20) / 15));
          r = Math.floor(180 + 60 * f + 15 * elevFade);
          g = Math.floor(190 + 50 * f + 15 * elevFade);
          b = Math.floor(210 + 30 * f + 15 * elevFade);
        } else if (landT < 10) {
          // Cool: gray-green, browner at high elevation
          var f = (landT + 5) / 15;
          r = Math.floor(100 + 40 * f + 40 * elevFade);
          g = Math.floor(110 + 50 * f - 20 * elevFade);
          b = Math.floor(90 - 20 * f + 30 * elevFade);
        } else if (landT < 25) {
          // Warm: green, browner at high elevation
          var f = (landT - 10) / 15;
          r = Math.floor(140 + 50 * f + 30 * elevFade);
          g = Math.floor(160 - 30 * f - 40 * elevFade);
          b = Math.floor(70 - 20 * f + 20 * elevFade);
        } else {
          // Hot: tan/brown desert
          r = Math.floor(190 + 20 * elevFade);
          g = Math.floor(160 - 30 * elevFade);
          b = Math.floor(80 + 20 * elevFade);
        }
        data[dstIdx] = r; data[dstIdx + 1] = g; data[dstIdx + 2] = b; data[dstIdx + 3] = 220;
        continue;
      }

      var rgb;
      if (showField === 'psi') rgb = psiToRGB(psi[srcK], absMax);
      else if (showField === 'vort') rgb = vortToRGB(zeta[srcK], absMax);
      else if (showField === 'temp') rgb = tempToRGB(temp[srcK]);
      else if (showField === 'deeptemp') rgb = tempToRGB(deepTemp ? deepTemp[srcK] : 0);
      else if (showField === 'deepflow') rgb = psiToRGB(deepPsi ? deepPsi[srcK] : 0, absMax);
      else if (showField === 'sal') rgb = salToRGB(sal ? sal[srcK] : 35);
      else if (showField === 'density') rgb = densityToRGB(temp[srcK], sal ? sal[srcK] : 35);
      else if (showField === 'depth') rgb = depthToRGB(depth ? depth[srcK] : 0);
      else if (showField === 'airtemp') {
        // Show air temp everywhere (land + ocean)
        rgb = tempToRGB(airTemp ? airTemp[srcK] : 15);
      }
      else {
        var vel = getVel(i, j);
        rgb = speedToRGB(Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]), maxSpd);
      }
      data[dstIdx] = rgb[0]; data[dstIdx + 1] = rgb[1]; data[dstIdx + 2] = rgb[2]; data[dstIdx + 3] = 190;
    }
  }
  fieldCtx.putImageData(imgData, 0, 0);

  // Draw field — the mask IS the coastline. Editable like SimEarth.
  // Land cells have alpha=0 in ImageData so only ocean shows through.
  ctx.imageSmoothingEnabled = false; // crisp grid cells
  var fieldDx = -W / (2 * (NX - 1));
  var fieldDy = -H / (2 * (NY - 1));
  var fieldDw = W * NX / (NX - 1);
  var fieldDh = H * NY / (NY - 1);
  ctx.drawImage(fieldCanvas, fieldDx, fieldDy, fieldDw, fieldDh);

  // Contour lines for streamfunction
  if (showField === 'psi' && absMax > 1e-20) {
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 0.5;
    var nContours = 16;
    for (var c = 1; c < nContours; c++) {
      var level = -absMax + 2 * absMax * c / nContours;
      for (var j = 0; j < NY - 1; j++) for (var i = 0; i < NX - 1; i++) {
        var v00 = psi[j * NX + i] - level, v10 = psi[j * NX + i + 1] - level;
        var v01 = psi[(j + 1) * NX + i] - level, v11 = psi[(j + 1) * NX + i + 1] - level;
        var s00 = v00 > 0 ? 1 : 0, s10 = v10 > 0 ? 1 : 0, s01 = v01 > 0 ? 1 : 0, s11 = v11 > 0 ? 1 : 0;
        var sum = s00 + s10 + s01 + s11;
        if (sum > 0 && sum < 4) {
          var cx_ = i * cellW + cellW / 2, cy_ = (NY - 1 - j) * cellH - cellH / 2;
          ctx.beginPath(); ctx.arc(cx_, cy_, 0.5, 0, Math.PI * 2); ctx.stroke();
        }
      }
    }
  }

  // Particles
  if (showParticles) {
    for (var p = 0; p < NP; p++) {
      var x = px[p] * cellW, y = (NY - 1 - py[p]) * cellH;
      var vel = getVel(px[p], py[p]);
      var spd = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      var alpha = Math.min(1, page_[p] / 20) * Math.min(1, (MAX_AGE - page_[p]) / 20);
      var bright = Math.min(1, spd / (maxSpd || 1) * 3);
      ctx.fillStyle = 'rgba(' + Math.floor(200 + 55 * bright) + ',' + Math.floor(220 + 35 * bright) + ',' + Math.floor(240 + 15 * bright) + ',' + (alpha * 0.6) + ')';
      ctx.fillRect(x - 0.5, y - 0.5, 1.5, 1.5);
    }
  }

  // Basin boundary
  ctx.strokeStyle = '#2a4050';
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, W, H);

  // Geographic labels
  ctx.fillStyle = 'rgba(200,220,240,0.15)';
  ctx.font = '9px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('N. Atlantic', lonToX(-40), latToY(35));
  ctx.fillText('N. Pacific', lonToX(-160), latToY(35));
  ctx.fillText('Indian', lonToX(75), latToY(-15));
  ctx.fillText('S. Ocean', lonToX(0), latToY(-60));

  // Color legend
  if (showField === "temp" || showField === "deeptemp") {
    var lx = W - 30, ly = 20, lh = 140, lw = 12;
    for (var li = 0; li < lh; li++) {
      var t_ = -10 + 40 * (lh - li) / lh;
      var rgb_ = tempToRGB(t_);
      ctx.fillStyle = "rgb(" + rgb_[0] + "," + rgb_[1] + "," + rgb_[2] + ")";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    ctx.strokeStyle = "rgba(255,255,255,.15)";
    ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillStyle = "rgba(255,255,255,.5)";
    ctx.font = "8px system-ui"; ctx.textAlign = "left";
    ctx.fillText("30\u00b0C", lx + lw + 3, ly + 6);
    ctx.fillText("15\u00b0", lx + lw + 3, ly + lh * 0.375 + 3);
    ctx.fillText("0\u00b0", lx + lw + 3, ly + lh * 0.75 + 3);
    ctx.fillStyle = "rgba(200,230,255,.4)";
    ctx.fillText("ICE", lx - 1, ly + lh - 8);
    ctx.fillText("-10\u00b0", lx + lw + 3, ly + lh);
    ctx.fillStyle = "rgba(255,255,255,.5)";
    ctx.fillText(showField === "deeptemp" ? "Deep" : "SST", lx, ly - 4);
  }
  // Western boundary annotation
  if (totalSteps > 500) {
    var maxWestVel = 0;
    for (var j = Math.floor(NY / 4); j < Math.floor(NY * 3 / 4); j++) {
      var vel = getVel(3, j);
      var sv = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (sv > maxWestVel) maxWestVel = sv;
    }
    if (maxWestVel > 0.5) {
      ctx.fillStyle = 'rgba(200,160,100,0.5)';
      ctx.font = '9px system-ui';
      ctx.save(); ctx.translate(20, H / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText('Western Boundary Current', 0, 0);
      ctx.restore();
    }
  }
  drawMagnifier();
}

// Draw overlay: map underlay, particles, contours, labels (for GPU render mode)
// The 2D canvas is transparent, sitting on top of the WebGPU canvas
function drawOverlay() {
  ctx.clearRect(0, 0, W, H);

  drawSeasonalLand();

  // Sea ice overlay: white with opacity proportional to ice fraction
  if (seaIce && NX && NY) {
    // Update ice canvas every 50 steps
    if (!drawOverlay._iceCvs || totalSteps % 50 === 0) {
      if (!drawOverlay._iceCvs) {
        drawOverlay._iceCvs = document.createElement('canvas');
        drawOverlay._iceCvs.width = NX; drawOverlay._iceCvs.height = NY;
      }
      var iceCvs = drawOverlay._iceCvs;
      var iceCtx = iceCvs.getContext('2d');
      var iceImg = iceCtx.createImageData(NX, NY);
      var iceD = iceImg.data;
      for (var j = 0; j < NY; j++) {
        var dstR = NY - 1 - j;
        for (var i = 0; i < NX; i++) {
          var hIce = seaIce[j * NX + i];
          if (hIce > 0.05 && mask[j * NX + i]) {
            var idx = (dstR * NX + i) * 4;
            // Thin ice: translucent blue-white. Thick ice: opaque white.
            var thick = Math.min(1, hIce / 2.0); // fully opaque at 2m
            iceD[idx] = 200 + Math.floor(40 * thick);
            iceD[idx+1] = 210 + Math.floor(35 * thick);
            iceD[idx+2] = 230 + Math.floor(20 * thick);
            iceD[idx+3] = Math.floor(80 + 175 * thick); // min 80 alpha so thin ice is visible
          }
        }
      }
      iceCtx.putImageData(iceImg, 0, 0);
    }
    if (drawOverlay._iceCvs) {
      var ldx = -W / (2 * (NX - 1)), ldy = -H / (2 * (NY - 1));
      var ldw = W * NX / (NX - 1), ldh = H * NY / (NY - 1);
      ctx.drawImage(drawOverlay._iceCvs, ldx, ldy, ldw, ldh);
    }
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;
  for (var lat = -60; lat <= 60; lat += 30) {
    var gy = latToY(lat);
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.1)'; ctx.font = '7px system-ui'; ctx.textAlign = 'right';
    ctx.fillText((lat >= 0 ? lat + '\u00b0N' : Math.abs(lat) + '\u00b0S'), W - 3, gy - 2);
  }
  for (var lon = -120; lon <= 120; lon += 60) {
    var gx = lonToX(lon);
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.1)'; ctx.font = '7px system-ui'; ctx.textAlign = 'center';
    ctx.fillText((lon <= 0 ? Math.abs(lon) + '\u00b0W' : lon + '\u00b0E'), gx, H - 3);
  }

  // Contour lines for streamfunction
  if (showField === 'psi') {
    var absMax = 0;
    for (var k = 0; k < NX * NY; k++) { var a = Math.abs(psi[k]); if (a > absMax) absMax = a; }
    if (absMax > 1e-20) {
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 0.5;
      var nContours = 16;
      for (var c = 1; c < nContours; c++) {
        var level = -absMax + 2 * absMax * c / nContours;
        for (var j = 0; j < NY - 1; j++) for (var i = 0; i < NX - 1; i++) {
          var v00 = psi[j * NX + i] - level, v10 = psi[j * NX + i + 1] - level;
          var v01 = psi[(j + 1) * NX + i] - level, v11 = psi[(j + 1) * NX + i + 1] - level;
          var s00 = v00 > 0 ? 1 : 0, s10 = v10 > 0 ? 1 : 0, s01 = v01 > 0 ? 1 : 0, s11 = v11 > 0 ? 1 : 0;
          var sum = s00 + s10 + s01 + s11;
          if (sum > 0 && sum < 4) {
            var cx_ = i * cellW + cellW / 2, cy_ = (NY - 1 - j) * cellH - cellH / 2;
            ctx.beginPath(); ctx.arc(cx_, cy_, 0.5, 0, Math.PI * 2); ctx.stroke();
          }
        }
      }
    }
  }

  // Particles
  var maxSpd = 1;
  if (showParticles) {
    for (var p = 0; p < NP; p++) {
      var x = px[p] * cellW, y = (NY - 1 - py[p]) * cellH;
      var vel = getVel(px[p], py[p]);
      var spd = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (spd > maxSpd) maxSpd = spd;
    }
    for (var p = 0; p < NP; p++) {
      var x = px[p] * cellW, y = (NY - 1 - py[p]) * cellH;
      var vel = getVel(px[p], py[p]);
      var spd = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      var alpha = Math.min(1, page_[p] / 20) * Math.min(1, (MAX_AGE - page_[p]) / 20);
      var bright = Math.min(1, spd / (maxSpd || 1) * 3);
      ctx.fillStyle = 'rgba(' + Math.floor(200 + 55 * bright) + ',' + Math.floor(220 + 35 * bright) + ',' + Math.floor(240 + 15 * bright) + ',' + (alpha * 0.6) + ')';
      ctx.fillRect(x - 0.5, y - 0.5, 1.5, 1.5);
    }
  }

  // Basin boundary
  ctx.strokeStyle = '#2a4050';
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, W, H);

  // Geographic labels
  ctx.fillStyle = 'rgba(200,220,240,0.15)';
  ctx.font = '9px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('N. Atlantic', lonToX(-40), latToY(35));
  ctx.fillText('N. Pacific', lonToX(-160), latToY(35));
  ctx.fillText('Indian', lonToX(75), latToY(-15));
  ctx.fillText('S. Ocean', lonToX(0), latToY(-60));

  // Color legend
  if (showField === "temp" || showField === "deeptemp") {
    var lx = W - 30, ly = 20, lh = 140, lw = 12;
    for (var li = 0; li < lh; li++) {
      var t_ = -10 + 40 * (lh - li) / lh;
      var rgb_ = tempToRGB(t_);
      ctx.fillStyle = "rgb(" + rgb_[0] + "," + rgb_[1] + "," + rgb_[2] + ")";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    ctx.strokeStyle = "rgba(255,255,255,.15)";
    ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillStyle = "rgba(255,255,255,.5)";
    ctx.font = "8px system-ui"; ctx.textAlign = "left";
    ctx.fillText("30\u00b0C", lx + lw + 3, ly + 6);
    ctx.fillText("15\u00b0", lx + lw + 3, ly + lh * 0.375 + 3);
    ctx.fillText("0\u00b0", lx + lw + 3, ly + lh * 0.75 + 3);
    ctx.fillStyle = "rgba(200,230,255,.4)";
    ctx.fillText("ICE", lx - 1, ly + lh - 8);
    ctx.fillText("-10\u00b0", lx + lw + 3, ly + lh);
    ctx.fillStyle = "rgba(255,255,255,.5)";
    ctx.fillText(showField === "deeptemp" ? "Deep" : "SST", lx, ly - 4);
  }
  // Western boundary annotation
  if (totalSteps > 500) {
    var maxWestVel = 0;
    for (var j = Math.floor(NY / 4); j < Math.floor(NY * 3 / 4); j++) {
      var vel = getVel(3, j);
      var sv = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (sv > maxWestVel) maxWestVel = sv;
    }
    if (maxWestVel > 0.5) {
      ctx.fillStyle = 'rgba(200,160,100,0.5)';
      ctx.font = '9px system-ui';
      ctx.save(); ctx.translate(20, H / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText('Western Boundary Current', 0, 0);
      ctx.restore();
    }
  }

  // Magnifying glass (toggle with Z key)
  drawMagnifier();
}

// ============================================================
// MAGNIFYING GLASS — zoom lens following cursor over simulation
// ============================================================
var magActive = false;
var magX = 0, magY = 0;       // cursor position in sim-view coords
var MAG_RADIUS = 80;          // lens radius in pixels
var MAG_ZOOM = 4;             // zoom factor
var magLensCanvas = null;
var magLensCtx = null;

function initMagnifier() {
  magLensCanvas = document.createElement('canvas');
  magLensCanvas.width = MAG_RADIUS * 2;
  magLensCanvas.height = MAG_RADIUS * 2;
  magLensCtx = magLensCanvas.getContext('2d');

  var simView = document.querySelector('.sim-view');
  simView.addEventListener('mousemove', function(e) {
    if (!magActive) return;
    var rect = simCanvas.getBoundingClientRect();
    magX = e.clientX - rect.left;
    magY = e.clientY - rect.top;
  });
  simView.addEventListener('mouseleave', function() { magX = -999; });

  // Toggle with 'Z' key
  document.addEventListener('keydown', function(e) {
    if (e.key === 'z' || e.key === 'Z') {
      magActive = !magActive;
      simCanvas.style.cursor = magActive ? 'none' : '';
    }
    // 'D' key: capture diagnostic grid of all views
    if (e.key === 'd' || e.key === 'D') {
      captureDiagnosticGrid();
    }
  });
}

// Capture all view modes into a 3x3 grid image and display in a popup
function captureDiagnosticGrid() {
  var views = [
    { field: 'temp', label: 'SST' },
    { field: 'airtemp', label: 'Air Temp' },
    { field: 'sal', label: 'Salinity' },
    { field: 'psi', label: 'Streamfunction' },
    { field: 'vort', label: 'Vorticity' },
    { field: 'speed', label: 'Speed' },
    { field: 'deeptemp', label: 'Deep Temp' },
    { field: 'depth', label: 'Depth' },
    { field: 'density', label: 'Density' },
  ];
  var cols = 3, rows = 3;
  var cw = 512, ch = 256;
  var pad = 2;
  var gridW = cols * cw + (cols - 1) * pad;
  var gridH = rows * ch + (rows - 1) * pad;

  var gridCvs = document.createElement('canvas');
  gridCvs.width = gridW; gridCvs.height = gridH;
  var gctx = gridCvs.getContext('2d');
  gctx.fillStyle = '#080e18';
  gctx.fillRect(0, 0, gridW, gridH);

  var prevField = showField;
  var gpuCanvas = document.getElementById('gpu-render-canvas');

  // Temporarily disable GPU rendering so draw() uses CPU path for ALL fields
  var wasGpuRender = gpuRenderEnabled;
  gpuRenderEnabled = false;

  for (var vi = 0; vi < views.length; vi++) {
    var v = views[vi];
    var col = vi % cols, row = Math.floor(vi / cols);
    var x = col * (cw + pad), y = row * (ch + pad);

    showField = v.field;
    draw(); // CPU render — guaranteed to work for all fields
    gctx.drawImage(simCanvas, 0, 0, W, H, x, y, cw, ch);

    // Label
    gctx.fillStyle = 'rgba(0,0,0,0.5)';
    gctx.fillRect(x, y, 100, 18);
    gctx.fillStyle = '#a0d0f0';
    gctx.font = 'bold 12px system-ui';
    gctx.fillText(v.label, x + 4, y + 13);
  }

  showField = prevField;
  gpuRenderEnabled = wasGpuRender;

  // Show in popup overlay
  var overlay = document.createElement('div');
  overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:9999;display:flex;align-items:center;justify-content:center;cursor:pointer';
  overlay.onclick = function() { document.body.removeChild(overlay); };
  var img = new Image();
  img.src = gridCvs.toDataURL('image/png');
  img.style.cssText = 'max-width:95%;max-height:95%;border:1px solid #2a4058;border-radius:8px';
  overlay.appendChild(img);

  // Right-click to save hint
  var hint = document.createElement('div');
  hint.style.cssText = 'position:absolute;bottom:20px;color:#6a8aa4;font-size:12px';
  hint.textContent = 'Click to close · Right-click image to save';
  overlay.appendChild(hint);

  document.body.appendChild(overlay);
  console.log('Diagnostic grid captured — click to close, right-click to save');
}

function drawMagnifier() {
  if (!magActive || magX < 0 || !magLensCtx) return;

  var gpuCanvas = document.getElementById('gpu-render-canvas');
  var rect = simCanvas.getBoundingClientRect();
  var scaleX = W / rect.width;
  var scaleY = H / rect.height;

  // Source region in canvas coords
  var srcX = magX * scaleX;
  var srcY = magY * scaleY;
  var srcR = MAG_RADIUS / MAG_ZOOM * scaleX;

  var d = MAG_RADIUS * 2;
  magLensCtx.clearRect(0, 0, d, d);

  // Clip to circle
  magLensCtx.save();
  magLensCtx.beginPath();
  magLensCtx.arc(MAG_RADIUS, MAG_RADIUS, MAG_RADIUS - 2, 0, Math.PI * 2);
  magLensCtx.clip();

  // Draw GPU canvas (field rendering) zoomed
  magLensCtx.drawImage(gpuCanvas,
    srcX - srcR, srcY - srcR, srcR * 2, srcR * 2,
    0, 0, d, d);

  // Draw CPU canvas (land, ice, particles) zoomed on top
  magLensCtx.drawImage(simCanvas,
    srcX - srcR, srcY - srcR, srcR * 2, srcR * 2,
    0, 0, d, d);

  magLensCtx.restore();

  // Draw lens border
  magLensCtx.strokeStyle = 'rgba(255,255,255,0.6)';
  magLensCtx.lineWidth = 2;
  magLensCtx.beginPath();
  magLensCtx.arc(MAG_RADIUS, MAG_RADIUS, MAG_RADIUS - 2, 0, Math.PI * 2);
  magLensCtx.stroke();

  // Draw crosshair
  magLensCtx.strokeStyle = 'rgba(255,255,255,0.2)';
  magLensCtx.lineWidth = 0.5;
  magLensCtx.beginPath();
  magLensCtx.moveTo(MAG_RADIUS, 4); magLensCtx.lineTo(MAG_RADIUS, d - 4);
  magLensCtx.moveTo(4, MAG_RADIUS); magLensCtx.lineTo(d - 4, MAG_RADIUS);
  magLensCtx.stroke();

  // Coordinate label
  var lon = LON0 + (srcX / W) * (LON1 - LON0);
  var lat = LAT1 - (srcY / H) * (LAT1 - LAT0);
  var lonStr = (lon < 0 ? Math.abs(lon).toFixed(1) + '°W' : lon.toFixed(1) + '°E');
  var latStr = (lat < 0 ? Math.abs(lat).toFixed(1) + '°S' : lat.toFixed(1) + '°N');
  magLensCtx.fillStyle = 'rgba(0,0,0,0.6)';
  magLensCtx.fillRect(4, d - 20, d - 8, 16);
  magLensCtx.fillStyle = '#b0d0e8';
  magLensCtx.font = '11px system-ui';
  magLensCtx.textAlign = 'center';
  magLensCtx.fillText(latStr + ', ' + lonStr, MAG_RADIUS, d - 8);

  // Composite onto main canvas at cursor position
  ctx.drawImage(magLensCanvas, magX - MAG_RADIUS, magY - MAG_RADIUS);
}

// Init magnifier after DOM ready
setTimeout(initMagnifier, 100);

// Velocity profile
var profCanvas = document.getElementById('profile');
var profCtx = profCanvas.getContext('2d');

function drawProfile() {
  var dpr = devicePixelRatio || 1;
  var r = profCanvas.getBoundingClientRect();
  profCanvas.width = r.width * dpr; profCanvas.height = r.height * dpr;
  profCtx.scale(dpr, dpr);
  var w = r.width, h = r.height;
  profCtx.clearRect(0, 0, w, h);

  var jMid = Math.floor(NY / 2);
  var maxV = 0;
  var vals = [];
  for (var i = 0; i < NX; i++) {
    var vel = getVel(i, jMid);
    vals.push(vel[1]);
    if (Math.abs(vel[1]) > maxV) maxV = Math.abs(vel[1]);
  }
  if (maxV < 1e-30) maxV = 1;

  var m = { l: 30, r: 6, t: 8, b: 18 }, pw = w - m.l - m.r, ph = h - m.t - m.b;

  profCtx.strokeStyle = '#1a2838'; profCtx.lineWidth = 1;
  profCtx.beginPath(); profCtx.moveTo(m.l, m.t); profCtx.lineTo(m.l, h - m.b); profCtx.lineTo(w - m.r, h - m.b); profCtx.stroke();

  var zy = m.t + ph / 2;
  profCtx.strokeStyle = '#2a3848'; profCtx.setLineDash([3, 3]);
  profCtx.beginPath(); profCtx.moveTo(m.l, zy); profCtx.lineTo(w - m.r, zy); profCtx.stroke(); profCtx.setLineDash([]);

  profCtx.fillStyle = '#4a7090'; profCtx.font = '7px system-ui';
  profCtx.textAlign = 'center'; profCtx.fillText('x (west to east)', w / 2, h - 2);
  profCtx.textAlign = 'right'; profCtx.fillText('v', m.l - 4, m.t + ph / 2 + 3);

  profCtx.strokeStyle = '#50b0e0'; profCtx.lineWidth = 1.5;
  profCtx.beginPath();
  for (var i = 0; i < NX; i++) {
    var x = m.l + (i / (NX - 1)) * pw;
    var y = zy - (vals[i] / maxV) * (ph / 2) * 0.9;
    if (i === 0) profCtx.moveTo(x, y); else profCtx.lineTo(x, y);
  }
  profCtx.stroke();

  profCtx.fillStyle = 'rgba(200,140,60,0.15)';
  profCtx.fillRect(m.l, m.t, pw * 0.08, ph);
}

// Radiative balance profile
var radCanvas = document.getElementById('rad-profile');
var radCtx = radCanvas.getContext('2d');

function drawRadProfile() {
  var dpr = devicePixelRatio || 1;
  var r = radCanvas.getBoundingClientRect();
  radCanvas.width = r.width * dpr; radCanvas.height = r.height * dpr;
  radCtx.scale(dpr, dpr);
  var w = r.width, h = r.height;
  radCtx.clearRect(0, 0, w, h);

  var m = { l: 30, r: 6, t: 8, b: 18 }, pw = w - m.l - m.r, ph = h - m.t - m.b;

  // Compute solar, OLR, and net for each latitude row (zonally averaged)
  var yearPhase = 2 * Math.PI * simTime / T_YEAR;
  var decl = 23.44 * Math.sin(yearPhase) * Math.PI / 180;

  var solar = [], olrArr = [], netArr = [], meanT = [];
  var maxVal = 0.1;

  for (var j = 0; j < NY; j++) {
    var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    var latRad = lat * Math.PI / 180;
    var cosZ = Math.cos(latRad) * Math.cos(decl) + Math.sin(latRad) * Math.sin(decl);

    // Zonal mean temperature
    var tSum = 0, tCount = 0;
    for (var i = 0; i < NX; i++) {
      var k = j * NX + i;
      if (mask[k]) { tSum += temp[k]; tCount++; }
    }
    var avgT = tCount > 0 ? tSum / tCount : 0;
    meanT.push(avgT);

    // Ice fraction for this temperature
    var iceT = Math.max(0, Math.min(1, (avgT + 3) / 5));
    var iceFrac = 1 - iceT * iceT * (3 - 2 * iceT);
    var albedoFactor = 0.15 + 0.85 * (1 - iceFrac);

    var qs = S_solar * Math.max(0, cosZ) * albedoFactor;
    var olr = A_olr + B_olr * avgT;
    var net = qs - olr;
    solar.push(qs);
    olrArr.push(olr);
    netArr.push(net);

    var mv = Math.max(Math.abs(qs), Math.abs(olr), Math.abs(net));
    if (mv > maxVal) maxVal = mv;
  }

  // Axes
  radCtx.strokeStyle = '#1a2838'; radCtx.lineWidth = 1;
  radCtx.beginPath(); radCtx.moveTo(m.l, m.t); radCtx.lineTo(m.l, h - m.b); radCtx.lineTo(w - m.r, h - m.b); radCtx.stroke();

  // Zero line
  var zy = m.t + ph * (maxVal / (2 * maxVal));
  radCtx.strokeStyle = '#2a3848'; radCtx.setLineDash([3, 3]);
  radCtx.beginPath(); radCtx.moveTo(m.l, zy); radCtx.lineTo(w - m.r, zy); radCtx.stroke(); radCtx.setLineDash([]);

  // Labels
  radCtx.fillStyle = '#4a7090'; radCtx.font = '7px system-ui';
  radCtx.textAlign = 'center'; radCtx.fillText('80\u00b0S                    latitude                    80\u00b0N', w / 2, h - 2);
  radCtx.textAlign = 'right'; radCtx.fillText('0', m.l - 3, zy + 3);

  function plotLine(arr, color) {
    radCtx.strokeStyle = color; radCtx.lineWidth = 1.5;
    radCtx.beginPath();
    for (var j2 = 0; j2 < NY; j2++) {
      var x = m.l + (j2 / (NY - 1)) * pw;
      var y = zy - (arr[j2] / maxVal) * (ph / 2) * 0.9;
      if (j2 === 0) radCtx.moveTo(x, y); else radCtx.lineTo(x, y);
    }
    radCtx.stroke();
  }

  plotLine(solar, '#e8a040');    // Solar: warm orange
  plotLine(olrArr, '#e05050');   // OLR: red
  plotLine(netArr, '#40c080');   // Net: green

  // Legend
  radCtx.font = '7px system-ui'; radCtx.textAlign = 'left';
  var lx = m.l + 4, ly = m.t + 8;
  radCtx.fillStyle = '#e8a040'; radCtx.fillText('Solar', lx, ly);
  radCtx.fillStyle = '#e05050'; radCtx.fillText('OLR', lx + 30, ly);
  radCtx.fillStyle = '#40c080'; radCtx.fillText('Net', lx + 55, ly);
}

// ============================================================
// MAIN LOOP
// ============================================================
var frameCount = 0;

async function gpuTick() {
  if (!paused) {
    gpuRunSteps(stepsPerFrame);
    readbackFrameCounter++;
    // Only do expensive readback every Nth frame if GPU render is active
    var needReadback = gpuRenderEnabled ? ((readbackFrameCounter - 1) % READBACK_INTERVAL === 0) : true;
    if (needReadback) {
      await gpuReadback();
      var needReupload = stabilityCheck();
      if (needReupload) updateGPUBuffersAfterPaint();
    }
    advectParticles();
  }
  // GPU render: draw field directly from GPU buffers (no readback needed)
  // Deep temp view uses CPU canvas since deep temp isn't in the render pipeline
  // GPU render only supports: temp, psi, vort, speed. Others fall back to CPU draw().
  var gpuFields = { temp: 1, psi: 1, vort: 1, speed: 1 };
  if (gpuRenderEnabled && gpuFields[showField]) {
    gpuRenderField();
    drawOverlay(); // particles, contours, labels on 2D canvas
  } else {
    draw();
  }
  updateStats();
  frameCount++;
  if (frameCount % 10 === 0) { drawProfile(); drawRadProfile(); }
  requestAnimationFrame(gpuTick);
}

function stabilityCheck() {
  // Single pass: CFL check, clamp, and NaN detection
  var maxV = 0;
  var maxZeta = 0;
  var blownUp = false;
  var N = NX * NY;

  for (var k = 0; k < N; k++) {
    if (!mask[k]) continue;

    // Track max vorticity
    var az = Math.abs(zeta[k]);
    if (az > maxZeta) maxZeta = az;

    // Clamp zeta
    if (az > 500) { zeta[k] = zeta[k] > 0 ? 500 : -500; blownUp = true; }

    // Clamp temperatures
    if (temp[k] > 40) temp[k] = 40;
    else if (temp[k] < -10) temp[k] = -10;
    if (deepTemp[k] > 30) deepTemp[k] = 30;
    else if (deepTemp[k] < -2) deepTemp[k] = -2;

    // NaN check
    if (zeta[k] !== zeta[k] || psi[k] !== psi[k] || temp[k] !== temp[k]) {
      zeta[k] = 0; psi[k] = 0; temp[k] = 0; deepTemp[k] = 0;
      blownUp = true;
    }
  }

  // Sample velocity at every 4th point (good enough for CFL)
  for (var j = 1; j < NY - 1; j += 2) for (var ii = 1; ii < NX - 1; ii += 4) {
    if (!mask[j * NX + ii]) continue;
    var vel = getVel(ii, j);
    var s2 = vel[0] * vel[0] + vel[1] * vel[1];
    if (s2 > maxV) maxV = s2; // compare squared to avoid sqrt
  }
  maxV = Math.sqrt(maxV);

  // CFL condition with smoothing to prevent dt oscillation
  var dx_ = 1.0 / (NX - 1);
  var dtTarget = maxV > 0 ? Math.min(dtBase, 0.3 * dx_ / maxV) : dtBase;
  dt = 0.7 * dt + 0.3 * dtTarget; // low-pass filter

  // Emergency damping
  if (maxZeta > 200) {
    var damp = 200 / maxZeta;
    for (var k3 = 0; k3 < N; k3++) { if (mask[k3]) zeta[k3] *= damp; }
    blownUp = true;
  }

  return blownUp;
}

function cpuTick() {
  if (!paused) {
    stabilityCheck();
    for (var i = 0; i < stepsPerFrame; i++) cpuTimestep();
    advectParticles();
  }
  draw();
  updateStats();
  frameCount++;
  if (frameCount % 5 === 0) { drawProfile(); drawRadProfile(); }
  requestAnimationFrame(cpuTick);
}

var monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

function updateStats() {
  // Sample every 4th point for velocity stats (8x fewer cells)
  var maxV = 0, ke = 0, sampleCount = 0;
  for (var j = 1; j < NY - 1; j += 2) for (var i = 1; i < NX - 1; i += 4) {
    if (!mask[j * NX + i]) continue;
    var vel = getVel(i, j);
    var s2 = vel[0] * vel[0] + vel[1] * vel[1];
    if (s2 > maxV * maxV) maxV = Math.sqrt(s2);
    ke += s2;
    sampleCount++;
  }
  // Scale KE estimate to full grid
  var totalOcean = 0; for (var k = 0; k < NX * NY; k++) { if (mask[k]) totalOcean++; }
  if (sampleCount > 0) ke *= totalOcean / sampleCount;
  document.getElementById('stat-vel').textContent = maxV.toFixed(3);
  document.getElementById('stat-ke').textContent = ke.toExponential(2);
  document.getElementById('stat-step').textContent = totalSteps;

  // Season display
  var yearFrac = (simTime % T_YEAR) / T_YEAR;
  if (yearFrac < 0) yearFrac += 1;
  var monthIdx = Math.floor(yearFrac * 12) % 12;
  document.getElementById('stat-season').textContent = monthNames[monthIdx];

  // AMOC strength: difference between surface and deep northward transport
  // at ~45N (North Atlantic). Positive = surface north, deep south = healthy AMOC
  var amocSum = 0, amocCount = 0;
  var jAmoc = Math.floor(NY * 0.65); // ~45N
  // Restrict to Atlantic basin (roughly lon -80 to 0 = grid i ~56 to 180)
  var iAtlW = Math.floor(0.28 * NX), iAtlE = Math.floor(0.5 * NX);
  for (var i = iAtlW; i < iAtlE; i++) {
    var k = jAmoc * NX + i;
    if (mask[k]) {
      var vSurf = (psi[k + 1] - psi[k - 1]) * 0.5 * invDx;
      var vDeep = deepPsi ? (deepPsi[k + 1] - deepPsi[k - 1]) * 0.5 * invDx : 0;
      amocSum += (vSurf - vDeep);
      amocCount++;
    }
  }
  amocStrength = amocCount > 0 ? amocSum / amocCount : 0;
  var amocDisplay = Math.abs(amocStrength);
  var amocEl = document.getElementById('stat-amoc');
  // Always show numeric value with qualitative label
  var amocSign = amocStrength >= 0 ? '+' : '-';
  if (amocDisplay < 0.001) {
    amocEl.textContent = amocSign + amocDisplay.toExponential(1) + ' weak';
    amocEl.style.color = '#e06050';
  } else if (amocDisplay > 0.05) {
    amocEl.textContent = amocSign + amocDisplay.toFixed(3) + ' strong';
    amocEl.style.color = '#4aba70';
  } else {
    amocEl.textContent = amocSign + amocDisplay.toFixed(3);
    amocEl.style.color = '#4a9ec8';
  }
}

// ============================================================
// RESET
// ============================================================
function resetSim() {
  if (useGPU) gpuReset();
  else cpuReset();
  initParticles();
}

// ============================================================
// CONTROLS
// ============================================================
// Info button click handlers
document.querySelectorAll('.info-btn').forEach(function(btn) {
  btn.addEventListener('click', function(e) {
    e.stopPropagation();
    var pop = this.closest('.cg').querySelector('.info-pop');
    var wasOpen = pop.classList.contains('show');
    // Close all
    document.querySelectorAll('.info-pop.show').forEach(function(p) { p.classList.remove('show'); });
    if (!wasOpen) {
      pop.textContent = this.dataset.info;
      pop.classList.add('show');
    }
  });
});
document.addEventListener('click', function() {
  document.querySelectorAll('.info-pop.show').forEach(function(p) { p.classList.remove('show'); });
});

document.getElementById('wind-slider').oninput = function(e) { windStrength = +e.target.value; document.getElementById('wind-val').textContent = windStrength.toFixed(2); };
document.getElementById('r-slider').oninput = function(e) { r_friction = +e.target.value; document.getElementById('r-val').textContent = r_friction.toFixed(3); };
document.getElementById('a-slider').oninput = function(e) { A_visc = +e.target.value; document.getElementById('a-val').textContent = A_visc.toExponential(1); };
document.getElementById('speed-slider').oninput = function(e) { stepsPerFrame = +e.target.value; document.getElementById('speed-val').textContent = stepsPerFrame; };
document.getElementById('btn-reset').onclick = resetSim;
document.getElementById('btn-pause').onclick = function() { paused = !paused; this.textContent = paused ? 'Resume' : 'Pause'; this.classList.toggle('active', paused); };
document.getElementById('btn-doublegyre').onclick = function() { doubleGyre = true; this.classList.add('active'); document.getElementById('btn-singlegyre').classList.remove('active'); resetSim(); };
document.getElementById('btn-singlegyre').onclick = function() { doubleGyre = false; this.classList.add('active'); document.getElementById('btn-doublegyre').classList.remove('active'); resetSim(); };
var viewBtnSelector = '#btn-psi,#btn-vort,#btn-speed,#btn-temp,#btn-deeptemp,#btn-deepflow,#btn-sal,#btn-density,#btn-depth,#btn-airtemp';
document.getElementById('btn-psi').onclick = function() { showField = 'psi'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-vort').onclick = function() { showField = 'vort'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-speed').onclick = function() { showField = 'speed'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-temp').onclick = function() { showField = 'temp'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-deeptemp').onclick = function() { showField = 'deeptemp'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-deepflow').onclick = function() { showField = 'deepflow'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-sal').onclick = function() { showField = 'sal'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-density').onclick = function() { showField = 'density'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-depth').onclick = function() { showField = 'depth'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-airtemp').onclick = function() { showField = 'airtemp'; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
document.getElementById('btn-particles').onclick = function() { showParticles = !showParticles; this.classList.toggle('active', showParticles); };

// New sliders
document.getElementById('year-speed-slider').oninput = function(e) { yearSpeed = +e.target.value; document.getElementById('year-speed-val').textContent = yearSpeed.toFixed(2); };
document.getElementById('fw-slider').oninput = function(e) { freshwaterForcing = +e.target.value; document.getElementById('fw-val').textContent = freshwaterForcing.toFixed(2); };
document.getElementById('gt-slider').oninput = function(e) { globalTempOffset = +e.target.value; document.getElementById('gt-val').textContent = globalTempOffset.toFixed(1); };
document.getElementById('dwf-slider').oninput = function(e) { gamma_deep_form = +e.target.value; document.getElementById('dwf-val').textContent = gamma_deep_form.toFixed(2); };
document.getElementById('solar-slider').oninput = function(e) { S_solar = +e.target.value; document.getElementById('solar-val').textContent = S_solar.toFixed(0); };

// ============================================================
// PAINT TOOL
// ============================================================
var paintMode = 'none';
var brushSize = 3;

document.getElementById('brush-slider').oninput = function(e) {
  brushSize = +e.target.value;
  document.getElementById('brush-val').textContent = brushSize;
};

// Paint button selection
document.querySelectorAll('.ptile').forEach(function(btn) {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.ptile').forEach(function(b) { b.classList.remove('active'); });
    // Show brush size popup when a paint tool is selected
    document.getElementById('brush-popup').classList.toggle('show', this.dataset.mode !== 'none');
    this.classList.add('active');
    paintMode = this.dataset.mode;
    simCanvas.style.cursor = paintMode === 'none' ? 'default' : 'crosshair';
  });
});
// Activate "Select" by default
document.querySelector('.ptile[data-mode="none"]').classList.add('active');

// Canvas coordinates to grid coordinates
function canvasToGrid(cx, cy) {
  var rect = simCanvas.getBoundingClientRect();
  var rx = (cx - rect.left) / rect.width;
  var ry = (cy - rect.top) / rect.height;
  var gi = Math.floor(rx * NX);
  var gj = Math.floor((1 - ry) * NY); // flip Y
  return [gi, gj];
}

function applyBrush(cx, cy) {
  if (paintMode === 'none') return;
  var coords = canvasToGrid(cx, cy);
  var ci = coords[0], cj = coords[1];
  var r2 = brushSize * brushSize;

  for (var dj = -brushSize; dj <= brushSize; dj++) {
    for (var di = -brushSize; di <= brushSize; di++) {
      if (di * di + dj * dj > r2) continue;
      var i = ((ci + di) % NX + NX) % NX, j = cj + dj;
      if (j < 1 || j >= NY - 1) continue;
      var k = j * NX + i;

      if (paintMode === 'land') {
        mask[k] = 0;
        psi[k] = 0;
        zeta[k] = 0;
        temp[k] = 0;
      } else if (paintMode === 'ocean') {
        mask[k] = 1;
        if (temp[k] === 0) {
          var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
          temp[k] = 28.0 - 0.5 * Math.abs(lat);
        }
      } else if (paintMode === 'heat') {
        if (mask[k]) {
          zeta[k] -= 0.5; // anticyclonic (warm core = negative vorticity in NH)
          temp[k] += 3;   // also warm the water
        }
      } else if (paintMode === 'cold') {
        if (mask[k]) {
          zeta[k] += 0.5; // cyclonic (cold core = positive vorticity in NH)
          temp[k] -= 3;   // also cool the water
        }
      } else if (paintMode === 'ice') {
        // Ice: ocean that's frozen. Keeps mask=1 so physics runs, but freezes the water.
        mask[k] = 1;
        temp[k] = -5;
        if (deepTemp) deepTemp[k] = -1;
        zeta[k] = 0; // ice suppresses circulation
        psi[k] = 0;
      } else if (paintMode === 'wind-cw') {
        if (mask[k]) zeta[k] -= 0.3;
      } else if (paintMode === 'wind-ccw') {
        if (mask[k]) zeta[k] += 0.3;
      }
    }
  }
  // Update GPU buffers if using WebGPU
  if (typeof updateGPUBuffersAfterPaint === 'function') {
    updateGPUBuffersAfterPaint();
  }
  drawMapUnderlay(); // redraw map if land changed
  readbackFrameCounter = 0; // force readback next frame to sync CPU data
}

var painting = false;
simCanvas.addEventListener('mousedown', function(e) {
  if (paintMode === 'none') return;
  painting = true;
  applyBrush(e.clientX, e.clientY);
});
simCanvas.addEventListener('mousemove', function(e) {
  if (!painting) return;
  applyBrush(e.clientX, e.clientY);
});
window.addEventListener('mouseup', function() { painting = false; });

// Touch support
simCanvas.addEventListener('touchstart', function(e) {
  if (paintMode === 'none') return;
  e.preventDefault();
  painting = true;
  applyBrush(e.touches[0].clientX, e.touches[0].clientY);
}, { passive: false });
simCanvas.addEventListener('touchmove', function(e) {
  if (!painting) return;
  e.preventDefault();
  applyBrush(e.touches[0].clientX, e.touches[0].clientY);
}, { passive: false });
simCanvas.addEventListener('touchend', function() { painting = false; });

// ============================================================
// INIT
// ============================================================
async function init() {
  await Promise.all([maskLoadPromise, coastLoadPromise, sstLoadPromise, deepLoadPromise, bathyLoadPromise,
    salinityLoadPromise, windLoadPromise, albedoLoadPromise, precipLoadPromise, cloudLoadPromise,
    seaIceLoadPromise, airTempLoadPromise, lstLoadPromise, evapLoadPromise]);
  drawMapUnderlay();

  var gpuOk = false;
  try {
    gpuOk = await initWebGPU();
  } catch (e) {
    console.warn('WebGPU init failed:', e);
    gpuOk = false;
  }

  if (gpuOk) {
    useGPU = true;
    document.getElementById('backend-badge').textContent = 'GPU';
    document.getElementById('backend-badge').className = 'gpu-badge gpu';
    console.log('WebGPU active: ' + NX + 'x' + NY + ' grid');
    // Initialize GPU render pipeline for direct-to-screen rendering
    try {
      initGPURenderPipeline();
      if (gpuRenderEnabled) console.log('GPU render pipeline active — no per-frame readback for rendering');
    } catch (e) {
      console.warn('GPU render pipeline failed, falling back to readback:', e);
      gpuRenderEnabled = false;
    }
  } else {
    useGPU = false;
    document.getElementById('gpu-warning').style.display = 'block';
    document.getElementById('backend-badge').textContent = 'CPU';
    document.getElementById('backend-badge').className = 'gpu-badge cpu';
    initCPU();
    initSOR();
    console.log('CPU fallback: ' + NX + 'x' + NY + ' grid');
  }

  // Redraw map now that mask + bathymetry are loaded (first call was too early)
  drawMapUnderlay();

  initFieldCanvas();
  initParticles();

  if (useGPU) gpuTick();
  else cpuTick();
}

// ============================================================
// ONBOARDING
// ============================================================
(function() {
  var overlay = document.getElementById('onboarding-overlay');
  if (!localStorage.getItem('amoc-onboarded')) {
    overlay.classList.remove('hidden');
  }
  document.getElementById('btn-start-exploring').addEventListener('click', function() {
    overlay.classList.add('hidden');
    localStorage.setItem('amoc-onboarded', '1');
  });
})();

// ============================================================
// PALEOCLIMATE SCENARIOS
// ============================================================
var originalMask = null; // saved after mask loads

function showScenarioExplanation(text) {
  var el = document.getElementById('scenario-explanation');
  var textEl = document.getElementById('scenario-text');
  if (!text) {
    el.classList.remove('show');
    return;
  }
  textEl.innerHTML = text;
  el.classList.add('show');
}

function lonLatToGrid(lon, lat) {
  var gi = Math.round((lon - LON0) / (LON1 - LON0) * (NX - 1));
  var gj = Math.round((lat - LAT0) / (LAT1 - LAT0) * (NY - 1));
  return [gi, gj];
}

function setMaskRect(lon0, lon1, lat0, lat1, value) {
  var g0 = lonLatToGrid(lon0, lat0);
  var g1 = lonLatToGrid(lon1, lat1);
  var iMin = Math.min(g0[0], g1[0]);
  var iMax = Math.max(g0[0], g1[0]);
  var jMin = Math.max(1, Math.min(g0[1], g1[1]));
  var jMax = Math.min(NY - 2, Math.max(g0[1], g1[1]));
  for (var j = jMin; j <= jMax; j++) {
    for (var i = iMin; i <= iMax; i++) {
      var wi = ((i % NX) + NX) % NX;
      var k = j * NX + wi;
      mask[k] = value;
      if (value === 0) {
        psi[k] = 0; zeta[k] = 0; temp[k] = 0;
      } else if (value === 1 && temp[k] === 0) {
        var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
        temp[k] = 28.0 - 0.5 * Math.abs(lat);
      }
    }
  }
}

function applyMaskChange() {
  drawMapUnderlay();
  updateGPUBuffersAfterPaint();
  initParticles();
}

// Open Drake Passage (ensure it's open)
document.getElementById('sc-drake').addEventListener('click', function() {
  // Open Drake Passage: lon -70 to -55, lat -70 to -55
  setMaskRect(-70, -55, -70, -55, 1);
  applyMaskChange();
  showScenarioExplanation('<strong>Drake Passage Opened</strong> &mdash; 35 million years ago, South America separated from Antarctica, opening the Drake Passage. The Antarctic Circumpolar Current formed, thermally isolating Antarctica and triggering its glaciation.');
});

// Close Drake Passage
document.getElementById('sc-close-drake').addEventListener('click', function() {
  // Block Drake Passage: fill in land from lon -70 to -55, lat -70 to -55
  setMaskRect(-70, -55, -70, -55, 0);
  applyMaskChange();
  showScenarioExplanation('<strong>Drake Passage Closed</strong> &mdash; Before 35 million years ago, South America and Antarctica were connected. Without the Drake Passage, the Antarctic Circumpolar Current cannot flow, and heat can reach Antarctica from the tropics.');
});

// Panama Seaway (toggle)
var panamaOpen = false;
document.getElementById('sc-panama').addEventListener('click', function() {
  if (!panamaOpen) {
    // Open seaway: remove land between N and S America (lon -85 to -75, lat 5 to 18)
    setMaskRect(-85, -75, 5, 18, 1);
    applyMaskChange();
    showScenarioExplanation('<strong>Panama Seaway Opened</strong> &mdash; Before 3 million years ago, the Atlantic and Pacific were connected through Central America. Opening this seaway weakens the Gulf Stream by allowing water to leak westward.');
    panamaOpen = true;
    this.querySelector('.sc-title').textContent = 'Close Panama Seaway';
  } else {
    // Close seaway: restore land
    setMaskRect(-85, -75, 5, 18, 0);
    applyMaskChange();
    showScenarioExplanation('<strong>Isthmus of Panama Formed</strong> &mdash; The Isthmus of Panama rose 3 million years ago, blocking Atlantic-Pacific exchange. This redirected warm water northward via the Gulf Stream, intensifying heat transport to Europe.');
    panamaOpen = false;
    this.querySelector('.sc-title').textContent = 'Open Panama Seaway';
  }
});

// Melt Greenland
document.getElementById('sc-greenland').addEventListener('click', function() {
  freshwaterForcing = 2.0;
  document.getElementById('fw-slider').value = 2.0;
  document.getElementById('fw-val').textContent = '2.00';
  showScenarioExplanation('<strong>Greenland Melting</strong> &mdash; Greenland\'s ice sheet contains enough fresh water to raise sea levels 7 meters. As it melts, the fresh water caps the North Atlantic, preventing the dense salty water from sinking. This could collapse the AMOC &mdash; as early as the 2030s.');
});

// Ice Age
document.getElementById('sc-iceage').addEventListener('click', function() {
  globalTempOffset = -8;
  freshwaterForcing = 0;
  document.getElementById('gt-slider').value = -8;
  document.getElementById('gt-val').textContent = '-8.0';
  document.getElementById('fw-slider').value = 0;
  document.getElementById('fw-val').textContent = '0.00';
  showScenarioExplanation('<strong>Ice Age</strong> &mdash; 20,000 years ago: sea levels 120m lower, ice sheets covering Canada and Scandinavia. The AMOC was weaker and shallower, with deep water forming further south.');
});

// Reset to Present
document.getElementById('sc-reset').addEventListener('click', function() {
  // Restore parameters
  windStrength = 1.0; document.getElementById('wind-slider').value = 1; document.getElementById('wind-val').textContent = '1.00';
  r_friction = 0.04; document.getElementById('r-slider').value = 0.04; document.getElementById('r-val').textContent = '0.040';
  A_visc = 2e-4; document.getElementById('a-slider').value = 0.0002; document.getElementById('a-val').textContent = '2.0e-4';
  freshwaterForcing = 0; document.getElementById('fw-slider').value = 0; document.getElementById('fw-val').textContent = '0.00';
  globalTempOffset = 0; document.getElementById('gt-slider').value = 0; document.getElementById('gt-val').textContent = '0.0';
  yearSpeed = 1.0; document.getElementById('year-speed-slider').value = 1; document.getElementById('year-speed-val').textContent = '1.00';
  // Restore mask
  if (originalMask) {
    for (var k = 0; k < NX * NY; k++) mask[k] = originalMask[k];
  }
  panamaOpen = false;
  document.getElementById('sc-panama').querySelector('.sc-title').textContent = 'Open Panama Seaway';
  resetSim();
  drawMapUnderlay();
  updateGPUBuffersAfterPaint();
  showScenarioExplanation('');
});

// Save original mask after init
var _origInit = init;
init = async function() {
  await _origInit();
  originalMask = new Uint8Array(NX * NY);
  for (var k = 0; k < NX * NY; k++) originalMask[k] = mask[k];
};

init();

// ============================================================
// EXPERIMENTATION API (window.lab)
// ============================================================
// Closure access to all top-level let-vars in this script tag.
window.lab = (function() {
  async function ensureReady() {
    var waited = 0;
    while ((!useGPU || !gpuDevice) && waited < 20000) {
      await new Promise(function(r) { setTimeout(r, 100); });
      waited += 100;
    }
    if (!useGPU || !gpuDevice) throw new Error('lab: GPU not initialized (CPU fallback path not supported by lab API yet)');
  }

  function getParams() {
    return {
      // Dynamics
      beta: beta, r: r_friction, A: A_visc, windStrength: windStrength,
      dt: dt, doubleGyre: doubleGyre,
      // Thermodynamics
      S_solar: S_solar, A_olr: A_olr, B_olr: B_olr,
      kappa_diff: kappa_diff, alpha_T: alpha_T,
      // Two-layer / thermohaline
      H_surface: H_surface, H_deep: H_deep,
      gamma_mix: gamma_mix, gamma_deep_form: gamma_deep_form,
      kappa_deep: kappa_deep,
      F_couple_s: F_couple_s, F_couple_d: F_couple_d, r_deep: r_deep,
      // Forcings
      yearSpeed: yearSpeed, freshwaterForcing: freshwaterForcing,
      globalTempOffset: globalTempOffset, T_YEAR: T_YEAR,
      // Numerics
      stepsPerFrame: stepsPerFrame,
      POISSON_ITERS: POISSON_ITERS, DEEP_POISSON_ITERS: DEEP_POISSON_ITERS,
      // State
      totalSteps: totalSteps, simTime: simTime, paused: paused,
      showField: showField, NX: NX, NY: NY, useGPU: useGPU
    };
  }

  function setParams(p) {
    if ('beta' in p) beta = p.beta;
    if ('r' in p) r_friction = p.r;
    if ('A' in p) A_visc = p.A;
    if ('windStrength' in p) windStrength = p.windStrength;
    if ('dt' in p) dt = p.dt;
    if ('doubleGyre' in p) doubleGyre = p.doubleGyre;
    if ('S_solar' in p) S_solar = p.S_solar;
    if ('A_olr' in p) A_olr = p.A_olr;
    if ('B_olr' in p) B_olr = p.B_olr;
    if ('kappa_diff' in p) kappa_diff = p.kappa_diff;
    if ('alpha_T' in p) alpha_T = p.alpha_T;
    if ('H_surface' in p) H_surface = p.H_surface;
    if ('H_deep' in p) H_deep = p.H_deep;
    if ('gamma_mix' in p) gamma_mix = p.gamma_mix;
    if ('gamma_deep_form' in p) gamma_deep_form = p.gamma_deep_form;
    if ('kappa_deep' in p) kappa_deep = p.kappa_deep;
    if ('F_couple_s' in p) F_couple_s = p.F_couple_s;
    if ('F_couple_d' in p) F_couple_d = p.F_couple_d;
    if ('r_deep' in p) r_deep = p.r_deep;
    if ('yearSpeed' in p) yearSpeed = p.yearSpeed;
    if ('freshwaterForcing' in p) freshwaterForcing = p.freshwaterForcing;
    if ('globalTempOffset' in p) globalTempOffset = p.globalTempOffset;
    if ('T_YEAR' in p) T_YEAR = p.T_YEAR;
    if ('stepsPerFrame' in p) stepsPerFrame = p.stepsPerFrame;
    if ('POISSON_ITERS' in p) POISSON_ITERS = p.POISSON_ITERS;
    if ('DEEP_POISSON_ITERS' in p) DEEP_POISSON_ITERS = p.DEEP_POISSON_ITERS;
    // Mirror to sliders where possible so the UI doesn't lie
    var sliderMap = {
      windStrength: 'wind-slider', r: 'r-slider', A: 'a-slider',
      stepsPerFrame: 'speed-slider', yearSpeed: 'year-speed-slider',
      freshwaterForcing: 'fw-slider', globalTempOffset: 'gt-slider'
    };
    for (var k in sliderMap) {
      if (k in p) {
        var el = document.getElementById(sliderMap[k]);
        if (el) el.value = p[k];
      }
    }
    return getParams();
  }

  async function step(n) {
    await ensureReady();
    var wasPaused = paused;
    paused = true;
    // Wait for any in-flight readback from main tick
    while (readbackPending) await new Promise(function(r){ setTimeout(r, 5); });
    // Run in chunks so the GPU driver doesn't time out on huge n
    var CHUNK = 500;
    var done = 0;
    while (done < n) {
      var k = Math.min(CHUNK, n - done);
      gpuRunSteps(k);
      done += k;
      // Let the command queue drain periodically
      if (done % (CHUNK * 10) === 0) await gpuDevice.queue.onSubmittedWorkDone();
    }
    await gpuDevice.queue.onSubmittedWorkDone();
    await gpuReadback();
    paused = wasPaused;
    return { step: totalSteps, simTime: simTime, simYears: simTime / T_YEAR };
  }

  function fields() {
    return {
      psi: psi ? new Float32Array(psi) : null,
      zeta: zeta ? new Float32Array(zeta) : null,
      temp: temp ? new Float32Array(temp) : null,
      deepTemp: deepTemp ? new Float32Array(deepTemp) : null,
      deepPsi: deepPsi ? new Float32Array(deepPsi) : null,
      mask: mask ? new Uint8Array(mask) : null,
      NX: NX, NY: NY, LAT0: LAT0, LAT1: LAT1, LON0: LON0, LON1: LON1
    };
  }

  function _lat(j) { return LAT0 + (j / (NY - 1)) * (LAT1 - LAT0); }

  function diagnostics(opts) {
    opts = opts || {};
    var includeProfiles = !!opts.profiles;
    if (!temp || !psi) return { error: 'no fields yet — call step() first' };
    var maxVel = 0, KE = 0, oceanCells = 0;
    var zonalSumT = new Float64Array(NY), zonalSumPsi = new Float64Array(NY);
    var zonalSumU  = new Float64Array(NY), zonalSumV = new Float64Array(NY);
    var zonalN    = new Int32Array(NY);
    var tropSum = 0, tropN = 0, polarSum = 0, polarN = 0, globSum = 0, globN = 0;
    var nhPolarSum = 0, nhPolarN = 0, shPolarSum = 0, shPolarN = 0;
    var iceArea = 0;
    var iDx = invDx, iDy = invDy;
    for (var j = 1; j < NY - 1; j++) {
      var lat = _lat(j);
      var absLat = Math.abs(lat);
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        if (!mask[k]) continue;
        oceanCells++;
        var ip1 = (i + 1) % NX, im1 = (i - 1 + NX) % NX;
        var u = -(psi[(j + 1) * NX + i] - psi[(j - 1) * NX + i]) * 0.5 * iDy;
        var v = (psi[j * NX + ip1] - psi[j * NX + im1]) * 0.5 * iDx;
        var s2 = u * u + v * v;
        if (s2 > maxVel * maxVel) maxVel = Math.sqrt(s2);
        KE += s2;
        var T = temp[k];
        zonalSumT[j] += T; zonalSumPsi[j] += psi[k];
        zonalSumU[j]  += u; zonalSumV[j]   += v;
        zonalN[j]++;
        globSum += T; globN++;
        if (absLat < 20) { tropSum += T; tropN++; }
        if (absLat > 60) { polarSum += T; polarN++;
          if (lat > 0) { nhPolarSum += T; nhPolarN++; }
          else { shPolarSum += T; shPolarN++; }
        }
        if (T < -1.5) iceArea++;
      }
    }
    KE *= 0.5;

    // AMOC proxy — Atlantic-band zonal mean meridional transport at ~45N
    var amocSum = 0, amocN = 0;
    var jAmoc = Math.floor(NY * 0.65);
    var iAtlW = Math.floor(0.28 * NX), iAtlE = Math.floor(0.5 * NX);
    for (var ia = iAtlW; ia < iAtlE; ia++) {
      var ka = jAmoc * NX + ia;
      if (!mask[ka]) continue;
      var vSurf = (psi[ka + 1] - psi[ka - 1]) * 0.5 * iDx;
      var vDeep = deepPsi ? (deepPsi[ka + 1] - deepPsi[ka - 1]) * 0.5 * iDx : 0;
      amocSum += vSurf - vDeep;
      amocN++;
    }
    var amoc = amocN > 0 ? amocSum / amocN : 0;

    // ACC proxy — zonal mean u at ~60S
    var jACC = Math.round((-60 - LAT0) / (LAT1 - LAT0) * (NY - 1));
    var accSum = 0, accN = 0;
    for (var ic = 0; ic < NX; ic++) {
      var kc = jACC * NX + ic;
      if (!mask[kc]) continue;
      var uc = -(psi[(jACC + 1) * NX + ic] - psi[(jACC - 1) * NX + ic]) * 0.5 * iDy;
      accSum += uc;
      accN++;
    }
    var accU = accN > 0 ? accSum / accN : 0;

    // Atlantic subtropical gyre strength (range of psi in 0-40N Atlantic band)
    var maxPsiAtl = -Infinity, minPsiAtl = Infinity;
    var jAtl0 = Math.floor(0.5 * NY), jAtl1 = Math.floor(0.75 * NY);
    for (var jg = jAtl0; jg < jAtl1; jg++)
      for (var ig = iAtlW; ig < iAtlE; ig++) {
        var kg = jg * NX + ig;
        if (!mask[kg]) continue;
        if (psi[kg] > maxPsiAtl) maxPsiAtl = psi[kg];
        if (psi[kg] < minPsiAtl) minPsiAtl = psi[kg];
      }

    var out = {
      step: totalSteps,
      simTime: simTime,
      simYears: simTime / T_YEAR,
      seasonFrac: ((simTime % T_YEAR) / T_YEAR + 1) % 1,
      oceanCells: oceanCells,
      maxVel: maxVel,
      KE: KE,
      globalSST: globN ? globSum / globN : NaN,
      tropicalSST: tropN ? tropSum / tropN : NaN,
      polarSST: polarN ? polarSum / polarN : NaN,
      nhPolarSST: nhPolarN ? nhPolarSum / nhPolarN : NaN,
      shPolarSST: shPolarN ? shPolarSum / shPolarN : NaN,
      amoc: amoc,
      accU: accU,
      iceArea: iceArea,
      gyreMaxPsi: maxPsiAtl,
      gyreMinPsi: minPsiAtl,
      gyreRangePsi: maxPsiAtl - minPsiAtl
    };

    if (includeProfiles) {
      var zT = new Float32Array(NY), zP = new Float32Array(NY), zU = new Float32Array(NY);
      for (var jz = 0; jz < NY; jz++) {
        zT[jz] = zonalN[jz] > 0 ? zonalSumT[jz] / zonalN[jz] : NaN;
        zP[jz] = zonalN[jz] > 0 ? zonalSumPsi[jz] / zonalN[jz] : NaN;
        zU[jz] = zonalN[jz] > 0 ? zonalSumU[jz] / zonalN[jz] : NaN;
      }
      var lats = new Float32Array(NY);
      for (var jl = 0; jl < NY; jl++) lats[jl] = _lat(jl);
      out.zonalMeanT = Array.from(zT);
      out.zonalMeanPsi = Array.from(zP);
      out.zonalMeanU = Array.from(zU);
      out.latitudes = Array.from(lats);
    }
    return out;
  }

  async function reset() {
    await ensureReady();
    var wasPaused = paused;
    paused = true;
    while (readbackPending) await new Promise(function(r){ setTimeout(r, 5); });
    if (typeof gpuReset === 'function') gpuReset();
    await gpuReadback();
    paused = wasPaused;
    return { step: totalSteps };
  }

  function view(name) {
    var valid = ['psi','vort','speed','temp','deeptemp','deepflow','depth'];
    if (valid.indexOf(name) < 0) throw new Error('view must be one of ' + valid.join(','));
    showField = name;
    return showField;
  }

  function pause() { paused = true; return paused; }
  function resume() { paused = false; return paused; }
  function isPaused() { return paused; }

  async function sweep(knob, values, opts) {
    opts = opts || {};
    var stepsPerPoint = opts.stepsPerPoint || 50000;
    var settleSteps = opts.settleSteps || 0;
    var resetBetween = !!opts.resetBetween;
    var results = [];
    for (var i = 0; i < values.length; i++) {
      var v = values[i];
      if (resetBetween) await reset();
      setParams({ [knob]: v });
      if (settleSteps) await step(settleSteps);
      await step(stepsPerPoint);
      var d = diagnostics();
      d._sweep_knob = knob; d._sweep_value = v;
      results.push(d);
    }
    return results;
  }

  async function timeSeries(totalStepsWanted, opts) {
    opts = opts || {};
    var interval = opts.interval || 10000;
    var series = [];
    var remaining = totalStepsWanted;
    while (remaining > 0) {
      var k = Math.min(interval, remaining);
      await step(k);
      var d = diagnostics();
      series.push(d);
      remaining -= k;
    }
    return series;
  }

  // Scenario triggers (click through the existing UI buttons so originalMask handling stays correct)
  function scenario(name) {
    var map = {
      'drake-open': 'sc-drake', 'drake-close': 'sc-close-drake',
      'panama-open': 'sc-panama', 'greenland': 'sc-greenland',
      'iceage': 'sc-iceage', 'present': 'sc-reset'
    };
    if (!(name in map)) throw new Error('scenario must be one of ' + Object.keys(map).join(','));
    document.getElementById(map[name]).click();
    return name;
  }

  return {
    getParams: getParams, setParams: setParams,
    step: step, reset: reset, view: view,
    pause: pause, resume: resume, isPaused: isPaused,
    fields: fields, diagnostics: diagnostics,
    sweep: sweep, timeSeries: timeSeries,
    scenario: scenario,
    _version: '0.1'
  };
})();
console.log('[lab] experimentation API ready at window.lab — try lab.getParams(), lab.step(1000), lab.diagnostics()');
