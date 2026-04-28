// {{PARAMS}} — replaced at load time with shared Params struct

@group(0) @binding(0) var<storage, read> psi: array<f32>;
@group(0) @binding(1) var<storage, read> tempIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> tempOut: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> deepTempIn: array<f32>;
@group(0) @binding(6) var<storage, read_write> deepTempOut: array<f32>;
@group(0) @binding(7) var<storage, read> depthField: array<f32>;
@group(0) @binding(8) var<storage, read> salClimatology: array<f32>;
@group(0) @binding(9) var<storage, read> ekmanVel: array<f32>;

fn idx(i: u32, j: u32) -> u32 { return j * params.nx + i; }

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let j = id.y;
  let nx = params.nx;
  let ny = params.ny;
  if (i >= nx) { return; }
  let k = idx(i, j);
  if (j < 1u || j >= ny - 1u || mask[k] == 0u) { tempOut[k] = 0.0; deepTempOut[k] = 0.0; return; }

  // Periodic wrapping in x
  let ip1 = select(i + 1u, 0u, i == nx - 1u);
  let im1 = select(i - 1u, nx - 1u, i == 0u);

  let ke = idx(ip1, j); let kw = idx(im1, j);
  let kn = idx(i, j + 1u); let ks = idx(i, j - 1u);
  // One-sided stencil: use self temp/psi for land neighbors (zero-gradient BC)
  let tE = select(tempIn[k], tempIn[ke], mask[ke] != 0u);
  let tW = select(tempIn[k], tempIn[kw], mask[kw] != 0u);
  let tN = select(tempIn[k], tempIn[kn], mask[kn] != 0u);
  let tS = select(tempIn[k], tempIn[ks], mask[ks] != 0u);
  let pE = select(psi[k], psi[ke], mask[ke] != 0u);
  let pW = select(psi[k], psi[kw], mask[kw] != 0u);
  let pN = select(psi[k], psi[kn], mask[kn] != 0u);
  let pS = select(psi[k], psi[ks], mask[ks] != 0u);

  // Metric correction for spherical geometry
  let lat = -79.5 + f32(j) / f32(ny - 1u) * 159.0;
  let latRad = lat * 3.14159265 / 180.0;
  let cosLat = max(cos(latRad), 0.087);
  let invDx = 1.0 / (params.dx * cosLat);
  let invDy = 1.0 / params.dy;
  let invDx2 = invDx * invDx;
  let invDy2 = invDy * invDy;

  // Advection: J(psi, T) + Ekman transport
  let dPdx = (pE - pW) * 0.5 * invDx;
  let dPdy = (pN - pS) * 0.5 * invDy;
  let dTdx = (tE - tW) * 0.5 * invDx;
  let dTdy = (tN - tS) * 0.5 * invDy;
  let geoAdvec = dPdx * dTdy - dPdy * dTdx;

  // Ekman heat transport: u_ek * dT/dx + v_ek * dT/dy
  let N_ek = params.nx * params.ny;
  let u_ek = ekmanVel[k] * params.windStrength;
  let v_ek = ekmanVel[k + N_ek] * params.windStrength;
  let ekmanAdvec = u_ek * dTdx + v_ek * dTdy;
  let advec = geoAdvec + ekmanAdvec;

  // Seasonal solar declination (lat, latRad already computed above)
  let yearPhase = 2.0 * 3.14159265 * (params.simTime % 10.0) / 10.0;
  let declination = 23.44 * sin(yearPhase) * 3.14159265 / 180.0;

  // Insolation with ice-albedo feedback
  let cosZenith = cos(latRad) * cos(declination) + sin(latRad) * sin(declination);
  var qSolar = params.sSolar * max(0.0, cosZenith);
  if (abs(lat) > 45.0) {
    let iceT = clamp((tempIn[k] + 2.0) / 10.0, 0.0, 1.0);
    let iceFrac = 1.0 - iceT * iceT * (3.0 - 2.0 * iceT);
    let latRamp = clamp((abs(lat) - 45.0) / 20.0, 0.0, 1.0);
    qSolar *= 1.0 - 0.50 * iceFrac * latRamp;
  }

  // -- CLOUD PARAMETERIZATION --
  // Physical regime-based clouds: ITCZ convection, subtropical subsidence,
  // marine stratocumulus, mid-latitude storm tracks, polar stratus
  let absLat = abs(lat);

  // Humidity proxy: warm SST = more evaporation
  let humidity = clamp((tempIn[k] - 5.0) / 25.0, 0.0, 1.0);

  // Lower tropospheric stability: estimated air temp vs SST
  // Warm air over cold water = inversion = stratocumulus
  let airTempEst = 28.0 - 0.55 * absLat;
  let lts = clamp((airTempEst - tempIn[k]) / 15.0, 0.0, 1.0);

  // Seasonal ITCZ position (migrates ~5 deg with seasons)
  let itczLat = 5.0 * sin(yearPhase);

  // 1. ITCZ deep convection
  let itczDist = (lat - itczLat) / 10.0;
  let convCloud = 0.30 * exp(-itczDist * itczDist) * humidity;

  // 2. Warm-pool convection (SST > 26C threshold)
  let warmPool = 0.20 * clamp((tempIn[k] - 26.0) / 4.0, 0.0, 1.0);

  // 3. Subtropical subsidence (Hadley descent ~25 deg, suppresses clouds)
  let subDist = (absLat - 25.0) / 10.0;
  let subsidence = 0.25 * exp(-subDist * subDist);

  // 4. Marine stratocumulus (cold SST + stable air, subtropics)
  let stratocu = 0.30 * lts * clamp((35.0 - absLat) / 20.0, 0.0, 1.0);

  // 5. Mid-latitude storm track (40-75 deg)
  // NH gets an extra boost at 35-55N (observed 0.65-0.75 in N Atlantic/Pacific storm tracks)
  let nhStormBoost = select(0.0, 0.15 * clamp((absLat - 35.0) / 10.0, 0.0, 1.0)
                                     * clamp((58.0 - absLat) / 12.0, 0.0, 1.0), lat > 0.0);
  let stormTrack = 0.25 * clamp((absLat - 35.0) / 10.0, 0.0, 1.0)
                       * clamp((80.0 - absLat) / 15.0, 0.0, 1.0) + nhStormBoost;

  // 6. Southern Ocean boundary layer clouds (observed ~0.85 at 55-65S)
  // Gaussian peak centered at 60S — ACC forcing drives persistent low cloud deck
  let soDist = (absLat - 60.0) / 8.0;
  let soCloud = select(0.0,
    0.55 * exp(-soDist * soDist) + 0.20 * clamp((absLat - 48.0) / 10.0, 0.0, 1.0),
    lat < 0.0);

  // 7. Polar stratus (both hemispheres)
  let polarCloud = 0.10 * clamp((absLat - 60.0) / 10.0, 0.0, 1.0);

  // Combine: high clouds (convective) + low clouds (stratiform) - subsidence
  // Stratocu is capped to prevent cold-SST runaway in SH subtropics (30-40S):
  // excessive LTS-driven clouds over cold model SST cause a cold-amplifying feedback.
  let stratocuCapped = clamp(stratocu, 0.0, 0.20);
  let highCloud = convCloud + warmPool;
  let lowCloud = stratocuCapped + stormTrack + soCloud + polarCloud;
  let cloudFrac = clamp(highCloud + lowCloud - subsidence * (1.0 - humidity), 0.05, 0.90);

  // Convective fraction determines radiative properties
  let convFrac = select(0.0, clamp(highCloud / (highCloud + lowCloud + 0.01), 0.0, 1.0), cloudFrac > 0.05);

  // SW albedo: low clouds reflect more (0.35) than high clouds (0.20)
  let cloudAlbedo = cloudFrac * (0.35 * (1.0 - convFrac) + 0.20 * convFrac);
  qSolar *= 1.0 - cloudAlbedo;

  // Outgoing longwave: A + B*T (global heat balance)
  let olr = params.aOlr - params.bOlr * params.globalTempOffset + params.bOlr * tempIn[k];
  // Southern Ocean OLR enhancement: extra cooling south of 55S (dry polar air, less greenhouse)
  // Only apply at high southern latitudes — 30-40S is already too cold, don't add cooling there.
  let soOlrMult = select(1.0, 1.0 + 0.35 * clamp((absLat - 55.0) / 10.0, 0.0, 1.0), lat < -50.0);
  var effectiveOlr = olr * soOlrMult;
  // LW greenhouse: high clouds trap more (0.12) than low clouds (0.03)
  let cloudGreenhouse = cloudFrac * (0.03 * (1.0 - convFrac) + 0.12 * convFrac);
  // Water vapor greenhouse: Clausius-Clapeyron moisture at 80% RH
  let qSat = 3.75e-3 * exp(0.067 * tempIn[k]);
  let vaporGH = 0.4 * clamp(0.8 * qSat / 0.015, 0.0, 1.0);
  effectiveOlr = effectiveOlr * (1.0 - cloudGreenhouse) * (1.0 - vaporGH);

  // Net radiative heating
  let qNet = qSolar - effectiveOlr;

  let y = f32(j) / f32(ny - 1u);

  // Diffusion with one-sided stencil
  let lapT = invDx2 * (tE + tW - 2.0 * tempIn[k])
           + invDy2 * (tN + tS - 2.0 * tempIn[k]);
  let diff = params.kappaDiff * lapT;

  // Land-ocean heat exchange
  var landFlux: f32 = 0.0;
  let nLand = f32(select(0u, 1u, mask[ke] == 0u)) + f32(select(0u, 1u, mask[kw] == 0u))
            + f32(select(0u, 1u, mask[kn] == 0u)) + f32(select(0u, 1u, mask[ks] == 0u));
  let nOcean = 4.0 - nLand;
  if (nLand > 0.0 && nOcean > 0.0) {
    let landT = 50.0 * max(0.0, cosZenith) - 20.0;
    let rawFlux = params.landHeatK * (landT - tempIn[k]) * (nOcean / 4.0);
    landFlux = clamp(rawFlux, -0.5, 0.5);
  }

  tempOut[k] = tempIn[k] + params.dt * (-advec + qNet + diff + landFlux);

  // Variable mixed layer depth: deep in Southern Ocean + subpolar NH, shallow in tropics
  let mldBase = 30.0 + 70.0 * pow(absLat / 80.0, 1.5);
  let accDist = (lat + 50.0) / 12.0;
  let mldACC = select(0.0, 250.0 * exp(-accDist * accDist), lat < -35.0 && lat > -65.0);
  let subpDist = (lat - 62.0) / 8.0;
  let mldSubpolar = select(0.0, 150.0 * exp(-subpDist * subpDist), lat > 50.0 && lat < 75.0);
  let mixedLayerDepth = mldBase + mldACC + mldSubpolar;

  // Two-layer vertical exchange
  let localDepth = depthField[k];
  let hSurf = min(mixedLayerDepth, localDepth);
  let hDeep = max(1.0, localDepth - mixedLayerDepth);
  let hasDeepLayer = select(0.0, 1.0, localDepth > mixedLayerDepth);

  // -- SALINITY (stacked at offset N in the same buffers) --
  let N = params.nx * params.ny;
  let salK = k + N;

  let sE = select(tempIn[salK], tempIn[idx(ip1,j) + N], mask[ke] != 0u);
  let sW = select(tempIn[salK], tempIn[idx(im1,j) + N], mask[kw] != 0u);
  let sN = select(tempIn[salK], tempIn[idx(i,j+1u) + N], mask[kn] != 0u);
  let sS = select(tempIn[salK], tempIn[idx(i,j-1u) + N], mask[ks] != 0u);

  let dSdx = (sE - sW) * 0.5 * invDx;
  let dSdy = (sN - sS) * 0.5 * invDy;
  let salAdvec = dPdx * dSdy - dPdy * dSdx + u_ek * dSdx + v_ek * dSdy;

  let lapS = invDx2 * (sE + sW - 2.0 * tempIn[salK])
           + invDy2 * (sN + sS - 2.0 * tempIn[salK]);
  let salDiff = params.kappaSal * lapS;

  // Salinity restoring: use observed WOA23 climatology if available, else zonal formula
  let salClimObs = salClimatology[k];
  let salClim = select(34.0 + 2.0 * cos(2.0 * latRad) - 0.5 * cos(4.0 * latRad), salClimObs, salClimObs > 1.0);
  let salRestore = params.salRestoring * (salClim - tempIn[salK]);

  var fwSal: f32 = 0.0;
  if (y > 0.75) {
    fwSal = -params.freshwater * 3.0 * (y - 0.75) * 4.0;
  }

  tempOut[salK] = tempIn[salK] + params.dt * (-salAdvec + salDiff + salRestore + fwSal);

  // -- DENSITY-BASED DEEP WATER FORMATION --
  let rhoSurf = -params.alphaT * tempIn[k] + params.betaS * tempIn[salK];
  let rhoDeep = -params.alphaT * deepTempIn[k] + params.betaS * deepTempIn[k + N];

  var gamma = params.gammaMix;
  // Deep water formation:
  //   NH: NADW forms at >40N (N Atlantic deep water, physically correct)
  //   SH: AABW forms only in Weddell/Ross seas (>62S). Do NOT trigger at 40-62S
  //   which is SH mid-lats already too cold in model.
  let isDeepFormRegion = (lat > 40.0 && rhoSurf > rhoDeep)
                       || (lat < -62.0 && rhoSurf > rhoDeep);
  if (isDeepFormRegion) { gamma = params.gammaDeepForm; }

  let vertExchangeT = gamma * (tempIn[k] - deepTempIn[k]) * hasDeepLayer;
  tempOut[k] = clamp(tempOut[k] - params.dt * vertExchangeT / hSurf, -10.0, 40.0);

  let vertExchangeS = gamma * (tempIn[salK] - deepTempIn[k + N]) * hasDeepLayer;
  tempOut[salK] = clamp(tempOut[salK] - params.dt * vertExchangeS / hSurf, 28.0, 40.0);

  // -- DEEP LAYER: temperature + salinity --
  let dE = select(deepTempIn[k], deepTempIn[ke], mask[ke] != 0u);
  let dW = select(deepTempIn[k], deepTempIn[kw], mask[kw] != 0u);
  let dN = select(deepTempIn[k], deepTempIn[kn], mask[kn] != 0u);
  let dS = select(deepTempIn[k], deepTempIn[ks], mask[ks] != 0u);
  let lapDeep = invDx2 * (dE + dW - 2.0 * deepTempIn[k])
             + invDy2 * (dN + dS - 2.0 * deepTempIn[k]);
  let deepDiff = params.kappaDeep * lapDeep;
  deepTempOut[k] = clamp(deepTempIn[k] + params.dt * (vertExchangeT / hDeep + deepDiff) * hasDeepLayer, -5.0, 30.0);

  // Deep salinity
  let dsE = select(deepTempIn[k+N], deepTempIn[ke+N], mask[ke] != 0u);
  let dsW = select(deepTempIn[k+N], deepTempIn[kw+N], mask[kw] != 0u);
  let dsN = select(deepTempIn[k+N], deepTempIn[kn+N], mask[kn] != 0u);
  let dsS = select(deepTempIn[k+N], deepTempIn[ks+N], mask[ks] != 0u);
  let lapDeepSal = invDx2 * (dsE + dsW - 2.0 * deepTempIn[k+N])
                 + invDy2 * (dsN + dsS - 2.0 * deepTempIn[k+N]);
  let deepSalDiff = params.kappaDeepSal * lapDeepSal;
  deepTempOut[k+N] = clamp(deepTempIn[k+N] + params.dt * (vertExchangeS / hDeep + deepSalDiff) * hasDeepLayer, 33.0, 37.0);
}
