// Atmosphere component (diagnostic / prescribed)
// Takes: SST, mask
// Produces: net heat flux (Q_net), wind stress (τx, τy)
// Wind stress is prescribed from ERA5; radiation is computed from SST

import { ATMOSPHERE } from './params.js';

export class Atmosphere {
  constructor(grid) {
    this.grid = grid;

    // Forcing data (loaded from observations)
    this.tauX_obs = grid.createField();    // ERA5 wind stress x
    this.tauY_obs = grid.createField();    // ERA5 wind stress y
    this.cloudFraction = grid.createField(0.5);

    // Output fluxes (computed each step)
    this.Qnet = grid.createField();
    this.tauX = grid.createField();
    this.tauY = grid.createField();

    // Diagnostic fields
    this.Qsw = grid.createField();        // shortwave absorbed
    this.Qolr = grid.createField();       // outgoing longwave
  }

  // Compute fluxes from current SST
  computeFluxes(sst, mask) {
    const { nx, ny, cosLat, lat } = this.grid;
    const {
      solarConstant: S0, albedoOcean, albedoLand,
      olrA, olrB, cloudAlbedoEffect, cloudGreenhouseEffect
    } = ATMOSPHERE;
    const g = this.grid;

    for (let j = 0; j < ny; j++) {
      const cos = cosLat[j];
      const latj = lat[j];

      // Insolation: S0/4 * zenith-angle distribution
      // Simple annual-mean approximation
      const insolation = S0 * cos * 0.5;  // rough daily-mean

      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        const isOcean = mask[k] > 0.5;

        // Surface albedo
        const surfAlbedo = isOcean ? albedoOcean : albedoLand;
        const cf = this.cloudFraction[k];
        const effectiveAlbedo = surfAlbedo + cf * cloudAlbedoEffect;

        // Shortwave absorbed
        const sw = insolation * (1 - effectiveAlbedo);
        this.Qsw[k] = sw;

        // OLR (linearized, reduced by cloud greenhouse)
        const T = isOcean ? sst[k] : sst[k];  // use SST everywhere for now
        const olr = olrA + olrB * T - cf * cloudGreenhouseEffect;
        this.Qolr[k] = olr;

        // Net heat flux into surface (positive = warming)
        this.Qnet[k] = sw - olr;

        // Wind stress: pass through observed
        this.tauX[k] = this.tauX_obs[k];
        this.tauY[k] = this.tauY_obs[k];
      }
    }
  }
}
