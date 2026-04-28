// Atmosphere component (diagnostic / prescribed)
// Takes: SST, mask
// Produces: net heat flux (Q_net), wind stress (τx, τy), P-E freshwater flux
// Wind stress is prescribed from ERA5; radiation + P-E computed from SST

import { ATMOSPHERE } from './params.js';

export class Atmosphere {
  constructor(grid) {
    this.grid = grid;

    // Forcing data (loaded from observations)
    this.tauX_obs = grid.createField();
    this.tauY_obs = grid.createField();
    this.cloudFraction = grid.createField(0.5);
    this.precip_obs = grid.createField();     // observed precipitation (m/s)

    // Output fluxes
    this.Qnet = grid.createField();
    this.tauX = grid.createField();
    this.tauY = grid.createField();
    this.PmE = grid.createField();            // P - E (m/s, positive = freshening)

    // Diagnostic fields
    this.Qsw = grid.createField();
    this.Qolr = grid.createField();
  }

  computeFluxes(sst, mask) {
    const { nx, ny, cosLat, lat } = this.grid;
    const {
      solarConstant: S0, albedoOcean, albedoLand,
      olrA, olrB, cloudAlbedoEffect, cloudGreenhouseEffect
    } = ATMOSPHERE;
    const g = this.grid;

    for (let j = 0; j < ny; j++) {
      const cos = cosLat[j];
      const insolation = S0 * cos * 0.5;

      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        const isOcean = mask[k] > 0.5;

        const surfAlbedo = isOcean ? albedoOcean : albedoLand;
        const cf = this.cloudFraction[k];
        const effectiveAlbedo = surfAlbedo + cf * cloudAlbedoEffect;

        const sw = insolation * (1 - effectiveAlbedo);
        this.Qsw[k] = sw;

        const T = sst[k];
        const olr = olrA + olrB * T - cf * cloudGreenhouseEffect;
        this.Qolr[k] = olr;

        this.Qnet[k] = sw - olr;

        this.tauX[k] = this.tauX_obs[k];
        this.tauY[k] = this.tauY_obs[k];

        // P - E freshwater flux (m/s)
        if (isOcean) {
          const P = this.precip_obs[k];

          // Evaporation parameterized from SST (Clausius-Clapeyron)
          // E increases ~7%/°C. Baseline: ~1200 mm/yr at 20°C
          // E(T) = E₀ * exp(0.067 * (T - 20))
          const E0 = 1200 / (365.25 * 86400);  // mm/yr → m/s (3.8e-8 m/s)
          const E = E0 * Math.exp(0.067 * (T - 20));

          this.PmE[k] = P - E;
        } else {
          this.PmE[k] = 0;
        }
      }
    }
  }
}
