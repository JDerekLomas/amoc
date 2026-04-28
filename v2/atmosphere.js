/**
 * SimAMOC v2 — Atmosphere Component
 *
 * Energy balance atmosphere with moisture transport.
 * This is the biggest physics gap in v1 — adding it gives us:
 *   - Latent heat (dominant tropical energy transport)
 *   - Precipitation and evaporation
 *   - P-E freshwater flux → salinity → AMOC
 *
 * Interface (fluxes to/from coupler):
 *   IN:  SST, iceFrac
 *   OUT: heatFlux, freshwaterFlux (P-E), windCurl
 *
 * Physics:
 *   1. Solar heating: S * cos(zenith) * (1 - albedo)
 *   2. OLR cooling: A + B*T (linearized Stefan-Boltzmann)
 *   3. Evaporation: E = ρ_a * C_E * U * (q_sat(SST) - q_a)
 *   4. Moisture transport: dq/dt = E - P + diffusion
 *   5. Precipitation: when q > q_sat * RH_crit
 *   6. Cloud fraction: f(RH, convection, latitude)
 */

/**
 * Compute atmospheric fluxes for one timestep.
 * Returns { heatFlux, freshwaterFlux } arrays for the coupler.
 */
export function stepAtmosphere(state, grid, params) {
  const { nx, ny, n } = grid;
  const { dt, T_year } = params;
  const { S_solar, A_olr, B_olr } = params;
  const { moisture_e0, moisture_scale, precip_threshold, precip_rate } = params;
  const { ice_albedo, ocean_albedo } = params;

  const { temp, sal, mask, iceFrac, airTemp, humidity, precip, evap, cloudFrac } = state;

  const heatFlux = new Float32Array(n);
  const freshwaterFlux = new Float32Array(n);

  // Seasonal cycle
  const yearPhase = 2 * Math.PI * state.simTime / T_year;
  const decl = 23.44 * Math.sin(yearPhase) * Math.PI / 180;

  for (let j = 1; j < ny - 1; j++) {
    const lat = grid.latAt(j);
    const latRad = lat * Math.PI / 180;
    const cosZ = Math.cos(latRad) * Math.cos(decl) + Math.sin(latRad) * Math.sin(decl);

    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (!mask[k]) continue;

      const sst = temp[k];
      const ice = iceFrac[k];

      // --- 1. Solar heating ---
      const albedo = ice * ice_albedo + (1 - ice) * ocean_albedo;
      const cloud = cloudFrac[k];
      // Clouds reflect shortwave (albedo effect) and trap longwave (greenhouse)
      const cloudAlbedo = cloud * 0.3;  // average cloud shortwave albedo
      const qSolar = S_solar * Math.max(0, cosZ) * (1 - albedo) * (1 - cloudAlbedo);

      // --- 2. OLR cooling ---
      const olr = A_olr + B_olr * sst;
      // Cloud greenhouse effect: high clouds trap more than low clouds
      const cloudGH = cloud * 0.07;
      const effectiveOlr = olr * (1 - cloudGH);

      // --- 3. Evaporation ---
      // Clausius-Clapeyron: saturation specific humidity
      const qSat = moisture_e0 * Math.exp(moisture_scale * sst);
      const qa = humidity[k];
      // Evaporation rate (bulk formula, simplified)
      const windSpeed = 7.0; // mean surface wind ~7 m/s (could come from dynamics)
      const Ce = 1.2e-3;     // transfer coefficient
      const rhoA = 1.2;      // air density kg/m³
      const E = rhoA * Ce * windSpeed * Math.max(0, qSat - qa) * (1 - ice);
      evap[k] = E;

      // --- 4. Moisture budget ---
      // dq/dt = E - P + horizontal diffusion (done below)
      // For now, simple column moisture budget
      const RH = qa / Math.max(qSat, 1e-6);

      // --- 5. Precipitation ---
      let P = 0;
      if (RH > precip_threshold) {
        P = precip_rate * (RH - precip_threshold) * qSat;
      }
      // Convective precipitation (ITCZ-like): strong when SST > 26°C and high humidity
      if (sst > 26 && RH > 0.6) {
        P += precip_rate * 2.0 * (sst - 26) / 10 * qa;
      }
      precip[k] = P;

      // Update humidity: E adds moisture, P removes it
      humidity[k] = Math.max(0, qa + dt * (E - P));

      // --- 6. Cloud fraction ---
      // Regime-based: RH-driven + convective + latitude
      const absLat = Math.abs(lat);
      let cf = 0.15 + 0.6 * Math.max(0, RH - 0.3);  // base: RH-dependent
      // ITCZ deep convection
      if (absLat < 15 && sst > 26) cf = Math.max(cf, 0.5 + 0.2 * (sst - 26) / 5);
      // Subtropical subsidence (clear skies)
      if (absLat > 15 && absLat < 35) cf *= 0.6;
      // Storm tracks
      if (absLat > 40 && absLat < 60) cf = Math.max(cf, 0.5);
      // Southern Ocean
      if (lat < -50) cf = Math.max(cf, 0.6);
      cloudFrac[k] = Math.min(1, Math.max(0, cf));

      // --- Net heat flux to ocean ---
      // Latent heat: evaporation cools the ocean
      const Lv = 2.5e6;  // J/kg
      const latentHeat = Lv * E / (1025 * 4000 * 200);  // normalized by ρ*cp*H
      // Scale to model units (S_solar ~7 ≈ 240 W/m²)
      const latentScaled = latentHeat * 7 / 240;

      heatFlux[k] = qSolar - effectiveOlr - latentScaled;

      // --- Freshwater flux to ocean ---
      // P-E: positive means net freshening (precipitation > evaporation)
      // This drives salinity: more freshwater → lower salinity → lighter water → weaker AMOC
      freshwaterFlux[k] = -(P - E);  // negative sign: E-P > 0 means saltier

      // --- Update air temperature (simple relaxation to SST) ---
      airTemp[k] += dt * 0.1 * (sst - airTemp[k]);
    }
  }

  // Moisture diffusion (horizontal transport)
  diffuseField(humidity, mask, grid, 5e-4 * dt);

  return { heatFlux, freshwaterFlux };
}

/**
 * Simple horizontal diffusion for a scalar field.
 */
function diffuseField(field, mask, grid, kappa) {
  const { nx, ny } = grid;
  const invDx2 = 1 / (grid.dx * grid.dx);
  const invDy2 = 1 / (grid.dy * grid.dy);
  const tmp = new Float32Array(field);

  for (let j = 1; j < ny - 1; j++) {
    const cl = grid.cosLat[j];
    const invCl2 = 1 / (cl * cl + 0.001);
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (!mask[k]) continue;
      const ip = grid.iWrap(i + 1), im = grid.iWrap(i - 1);
      const ke = j * nx + ip, kw = j * nx + im;
      const kn = (j + 1) * nx + i, ks = (j - 1) * nx + i;
      const fE = mask[ke] ? tmp[ke] : tmp[k];
      const fW = mask[kw] ? tmp[kw] : tmp[k];
      const fN = mask[kn] ? tmp[kn] : tmp[k];
      const fS = mask[ks] ? tmp[ks] : tmp[k];
      field[k] = tmp[k] + kappa * (
        invDx2 * invCl2 * (fE + fW - 2 * tmp[k]) +
        invDy2 * (fN + fS - 2 * tmp[k])
      );
    }
  }
}
