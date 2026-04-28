// Ocean component
// Barotropic vorticity + SST + salinity + two-layer + density-driven overturning
// Takes: wind stress (τx, τy), net heat flux (Q_net), P-E freshwater flux, mask
// Produces: SST, salinity, streamfunction, velocities, deep T/S, MOC

import { EARTH, OCEAN, SIMULATION } from './params.js';

export class Ocean {
  constructor(grid) {
    this.grid = grid;
    const g = grid;

    // Surface layer state
    this.T = g.createField(15);       // SST (°C)
    this.S = g.createField(35);       // surface salinity (PSU)
    this.psi = g.createField();       // streamfunction (m²/s)
    this.zeta = g.createField();      // relative vorticity (1/s)
    this.u = g.createField();         // zonal velocity (m/s)
    this.v = g.createField();         // meridional velocity (m/s)
    this.rho_surf = g.createField();  // surface density anomaly (kg/m³)

    // Deep layer state
    this.Tdeep = g.createField(4);    // deep temperature (°C)
    this.Sdeep = g.createField(34.8); // deep salinity (PSU)
    this.vDeep = g.createField();     // deep meridional velocity (m/s)
    this.rho_deep = g.createField();  // deep density anomaly (kg/m³)

    // Vertical exchange
    this.wUp = g.createField();       // upwelling velocity (m/s, positive = upward)

    // Overturning diagnostic (ny array — Sv at each latitude)
    this.moc = new Float32Array(grid.ny);

    // Forcing fields (set by coupler)
    this.tauX = g.createField();
    this.tauY = g.createField();
    this.Qnet = g.createField();
    this.PmE = g.createField();       // precipitation minus evaporation (m/s, positive = freshening)
    this.mask = g.createField();

    // Observed salinity for restoring
    this.S_obs = g.createField(35);

    // Work arrays
    this._zetaNew = g.createField();
    this._Tnew = g.createField();
    this._Snew = g.createField();
    this._TdeepNew = g.createField();
    this._SdeepNew = g.createField();
  }

  initSST(sstData) {
    for (let k = 0; k < this.grid.size; k++) {
      if (this.mask[k] > 0.5) {
        this.T[k] = sstData[k];
        this.Tdeep[k] = Math.max(1, sstData[k] * 0.3 - 2);
      }
    }
  }

  initSalinity(salData) {
    for (let k = 0; k < this.grid.size; k++) {
      if (this.mask[k] > 0.5 && salData[k] > 0) {
        this.S[k] = salData[k];
        this.S_obs[k] = salData[k];
        // Deep salinity: slightly lower, more uniform
        this.Sdeep[k] = 34.7 + (salData[k] - 35) * 0.2;
      }
    }
  }

  step(dt) {
    this._stepVorticity(dt);
    this._solvePoissonSOR();
    this._computeVelocities();
    this._computeDensity();
    this._stepTemperature(dt);
    this._stepSalinity(dt);
    this._stepDeepOcean(dt);
    this._computeDeepFlow(dt);
    this._computeOverturning();
  }

  // Linear equation of state: ρ = ρ₀ (1 - αT(T - T₀) + βS(S - S₀))
  _computeDensity() {
    const { T, S, Tdeep, Sdeep, rho_surf, rho_deep, mask } = this;
    const alphaT = 2e-4;  // thermal expansion (1/°C)
    const betaS = 7.5e-4; // haline contraction (1/PSU)
    const T0 = 10, S0 = 35;

    for (let k = 0; k < this.grid.size; k++) {
      if (mask[k] > 0.5) {
        rho_surf[k] = -alphaT * (T[k] - T0) + betaS * (S[k] - S0);
        rho_deep[k] = -alphaT * (Tdeep[k] - T0) + betaS * (Sdeep[k] - S0);
      }
    }
  }

  _stepVorticity(dt) {
    const { nx, ny, cosLat, beta, dx, dy } = this.grid;
    const { rho, mixedLayerDepth: H, viscosity: AH, dragCoeff: r } = OCEAN;
    const { tauX, tauY, zeta, mask, _zetaNew } = this;
    const g = this.grid;

    for (let j = 1; j < ny - 1; j++) {
      const dxj = dx[j];
      const dyj = dy[j];

      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        if (mask[k] < 0.5) { _zetaNew[k] = 0; continue; }

        const ip = g.wrap(i + 1, j);
        const im = g.wrap(i - 1, j);
        const jp = g.idx(i, j + 1);
        const jm = g.idx(i, j - 1);

        const dtauY_dx = (tauY[ip] - tauY[im]) / (2 * dxj);
        const dtauX_dy = (tauX[jp] - tauX[jm]) / (2 * dyj);
        const curlTau = (dtauY_dx - dtauX_dy) / (rho * H);

        const betaV = beta[j] * this.v[k];

        const lap = (
          (zeta[ip] - 2 * zeta[k] + zeta[im]) / (dxj * dxj) +
          (zeta[jp] - 2 * zeta[k] + zeta[jm]) / (dyj * dyj)
        );

        _zetaNew[k] = zeta[k] + dt * (curlTau - betaV - r * zeta[k] + AH * lap);
      }
    }

    for (let i = 0; i < nx; i++) {
      _zetaNew[g.idx(i, 0)] = 0;
      _zetaNew[g.idx(i, ny - 1)] = 0;
    }
    this.zeta.set(_zetaNew);
  }

  _solvePoissonSOR() {
    const { nx, ny, dx, dy } = this.grid;
    const { psi, zeta, mask } = this;
    const g = this.grid;
    const omega = SIMULATION.sorOmega;
    const nIter = SIMULATION.sorIterations;

    for (let iter = 0; iter < nIter; iter++) {
      for (let j = 1; j < ny - 1; j++) {
        const dxj = dx[j];
        const dyj = dy[j];
        const ax = 1 / (dxj * dxj);
        const ay = 1 / (dyj * dyj);
        const denom = 2 * (ax + ay);

        for (let i = 0; i < nx; i++) {
          const k = g.idx(i, j);
          if (mask[k] < 0.5) { psi[k] = 0; continue; }

          const ip = g.wrap(i + 1, j);
          const im = g.wrap(i - 1, j);
          const jp = g.idx(i, j + 1);
          const jm = g.idx(i, j - 1);

          const pE = mask[ip] > 0.5 ? psi[ip] : 0;
          const pW = mask[im] > 0.5 ? psi[im] : 0;
          const pN = mask[jp] > 0.5 ? psi[jp] : 0;
          const pS = mask[jm] > 0.5 ? psi[jm] : 0;

          const psiNew = (ax * (pE + pW) + ay * (pN + pS) - zeta[k]) / denom;
          psi[k] += omega * (psiNew - psi[k]);
        }
      }
    }
  }

  _computeVelocities() {
    const { nx, ny, dx, dy } = this.grid;
    const { psi, u, v, mask } = this;
    const g = this.grid;

    for (let j = 1; j < ny - 1; j++) {
      const dxj = dx[j];
      const dyj = dy[j];
      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        if (mask[k] < 0.5) { u[k] = 0; v[k] = 0; continue; }
        const ip = g.wrap(i + 1, j);
        const im = g.wrap(i - 1, j);
        const jp = g.idx(i, j + 1);
        const jm = g.idx(i, j - 1);
        u[k] = -(psi[jp] - psi[jm]) / (2 * dyj);
        v[k] = (psi[ip] - psi[im]) / (2 * dxj);
      }
    }
  }

  _stepTemperature(dt) {
    const { nx, ny, dx, dy } = this.grid;
    const { rho, cp, mixedLayerDepth: H, diffusivity: kappa } = OCEAN;
    const { T, u, v, Qnet, mask, _Tnew, Tdeep, rho_surf, rho_deep } = this;
    const g = this.grid;
    const rhoCpH = rho * cp * H;
    const gammaMix = 3e-8;       // very slow background mixing (real ocean is weakly mixed)
    const gammaConvect = 5e-6;   // strong when surface denser than deep

    for (let j = 1; j < ny - 1; j++) {
      const dxj = dx[j];
      const dyj = dy[j];
      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        if (mask[k] < 0.5) { _Tnew[k] = T[k]; continue; }

        const ip = g.wrap(i + 1, j);
        const im = g.wrap(i - 1, j);
        const jp = g.idx(i, j + 1);
        const jm = g.idx(i, j - 1);

        const uj = u[k], vj = v[k];
        const dTdx = uj > 0 ? (T[k] - T[im]) / dxj : (T[ip] - T[k]) / dxj;
        const dTdy = vj > 0 ? (T[k] - T[jm]) / dyj : (T[jp] - T[k]) / dyj;
        const advection = -(uj * dTdx + vj * dTdy);

        const heating = Qnet[k] / rhoCpH;

        const lap = (
          (T[ip] - 2 * T[k] + T[im]) / (dxj * dxj) +
          (T[jp] - 2 * T[k] + T[jm]) / (dyj * dyj)
        );

        // Vertical mixing: density-driven convection
        const convecting = rho_surf[k] > rho_deep[k];
        const gamma = convecting ? gammaConvect : gammaMix;
        const vertMix = -gamma * (T[k] - Tdeep[k]);

        _Tnew[k] = T[k] + dt * (advection + heating + kappa * lap + vertMix);
        if (_Tnew[k] < -2) _Tnew[k] = -2;
        if (_Tnew[k] > 35) _Tnew[k] = 35;
      }
    }
    T.set(_Tnew);
  }

  _stepSalinity(dt) {
    const { nx, ny, dx, dy } = this.grid;
    const { diffusivity: kappa } = OCEAN;
    const { S, u, v, PmE, mask, _Snew, Sdeep, S_obs, rho_surf, rho_deep } = this;
    const g = this.grid;
    const H = OCEAN.mixedLayerDepth;
    const kappaSal = kappa * 0.5;
    const salRestoringRate = 5e-8;
    const gammaMix = 3e-8;
    const gammaConvect = 5e-6;

    for (let j = 1; j < ny - 1; j++) {
      const dxj = dx[j];
      const dyj = dy[j];
      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        if (mask[k] < 0.5) { _Snew[k] = S[k]; continue; }

        const ip = g.wrap(i + 1, j);
        const im = g.wrap(i - 1, j);
        const jp = g.idx(i, j + 1);
        const jm = g.idx(i, j - 1);

        // Upwind advection
        const uj = u[k], vj = v[k];
        const dSdx = uj > 0 ? (S[k] - S[im]) / dxj : (S[ip] - S[k]) / dxj;
        const dSdy = vj > 0 ? (S[k] - S[jm]) / dyj : (S[jp] - S[k]) / dyj;
        const advection = -(uj * dSdx + vj * dSdy);

        // P-E freshwater flux: dS/dt = -S₀ * (P-E) / H
        // P-E > 0 means net precipitation → freshening → S decreases
        const freshwaterFlux = -S[k] * PmE[k] / H;

        // Diffusion
        const lap = (
          (S[ip] - 2 * S[k] + S[im]) / (dxj * dxj) +
          (S[jp] - 2 * S[k] + S[jm]) / (dyj * dyj)
        );

        // Restoring toward observed climatology
        const restoring = salRestoringRate * (S_obs[k] - S[k]);

        // Vertical mixing with deep salinity (density-based)
        const convecting = rho_surf[k] > rho_deep[k];
        const gamma = convecting ? gammaConvect : gammaMix;
        const vertMix = -gamma * (S[k] - Sdeep[k]);

        _Snew[k] = S[k] + dt * (advection + freshwaterFlux + kappaSal * lap + restoring + vertMix);
        if (_Snew[k] < 20) _Snew[k] = 20;
        if (_Snew[k] > 42) _Snew[k] = 42;
      }
    }
    S.set(_Snew);
  }

  _stepDeepOcean(dt) {
    const { nx, ny, dx, dy, lat } = this.grid;
    const { diffusivity: kappa } = OCEAN;
    const { T, S, Tdeep, Sdeep, mask, _TdeepNew, _SdeepNew, rho_surf, rho_deep } = this;
    const g = this.grid;
    const kappaDeep = kappa * 0.05;
    const H_ratio = OCEAN.mixedLayerDepth / 4000;
    const gammaMix = 3e-8;
    const gammaConvect = 5e-6;

    for (let j = 1; j < ny - 1; j++) {
      const dxj = dx[j];
      const dyj = dy[j];
      const latj = lat[j];
      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        if (mask[k] < 0.5) {
          _TdeepNew[k] = Tdeep[k];
          _SdeepNew[k] = Sdeep[k];
          continue;
        }

        const ip = g.wrap(i + 1, j);
        const im = g.wrap(i - 1, j);
        const jp = g.idx(i, j + 1);
        const jm = g.idx(i, j - 1);

        // Deep diffusion
        const lapT = (
          (Tdeep[ip] - 2 * Tdeep[k] + Tdeep[im]) / (dxj * dxj) +
          (Tdeep[jp] - 2 * Tdeep[k] + Tdeep[jm]) / (dyj * dyj)
        );
        const lapS = (
          (Sdeep[ip] - 2 * Sdeep[k] + Sdeep[im]) / (dxj * dxj) +
          (Sdeep[jp] - 2 * Sdeep[k] + Sdeep[jm]) / (dyj * dyj)
        );

        // Vertical exchange — density-driven
        const convecting = rho_surf[k] > rho_deep[k];
        const gamma = convecting ? gammaConvect : gammaMix;
        const vertMixT = gamma * (T[k] - Tdeep[k]) * H_ratio;
        const vertMixS = gamma * (S[k] - Sdeep[k]) * H_ratio;

        // Deep advection by deep meridional flow
        const vd = this.vDeep[k];
        let advT = 0, advS = 0;
        if (Math.abs(vd) > 1e-8) {
          const dTdy = vd > 0
            ? (Tdeep[k] - Tdeep[jm]) / dyj
            : (Tdeep[jp] - Tdeep[k]) / dyj;
          advT = -vd * dTdy;
          const dSdy = vd > 0
            ? (Sdeep[k] - Sdeep[jm]) / dyj
            : (Sdeep[jp] - Sdeep[k]) / dyj;
          advS = -vd * dSdy;
        }

        _TdeepNew[k] = Tdeep[k] + dt * (kappaDeep * lapT + vertMixT + advT);
        _SdeepNew[k] = Sdeep[k] + dt * (kappaDeep * lapS + vertMixS + advS);
        if (_TdeepNew[k] < -2) _TdeepNew[k] = -2;
        if (_TdeepNew[k] > 20) _TdeepNew[k] = 20;
        if (_SdeepNew[k] < 30) _SdeepNew[k] = 30;
        if (_SdeepNew[k] > 38) _SdeepNew[k] = 38;
      }
    }
    Tdeep.set(_TdeepNew);
    Sdeep.set(_SdeepNew);
  }

  // Density-driven deep meridional flow
  // Where surface water is dense (cold + salty at high latitudes), it sinks
  // and flows equatorward at depth. Upwelling occurs in the tropics/south.
  // This is a simplified parameterization of AMOC:
  //   vDeep ∝ -∂ρ_deep/∂y (dense water flows toward less dense)
  //   wUp = vertical mass balance
  _computeDeepFlow(dt) {
    const { nx, ny, dx, dy, lat } = this.grid;
    const { mask, rho_surf, rho_deep, vDeep, wUp } = this;
    const g = this.grid;
    const H_surface = OCEAN.mixedLayerDepth;
    const H_deep = 4000;

    // Deep flow from meridional density gradient
    // vDeep = -g'/(f) * dρ/dy (thermal wind, simplified)
    const gPrime = 0.02;  // reduced gravity for density-driven flow (m/s²)

    for (let j = 1; j < ny - 1; j++) {
      const dyj = dy[j];
      const f = this.grid.f[j];
      const absF = Math.max(Math.abs(f), 1e-5); // avoid singularity at equator

      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        if (mask[k] < 0.5) { vDeep[k] = 0; wUp[k] = 0; continue; }

        const jp = g.idx(i, j + 1);
        const jm = g.idx(i, j - 1);

        // Deep density gradient (south-first: j+1 = north)
        const drho_dy = (rho_deep[jp] - rho_deep[jm]) / (2 * dyj);

        // Density-driven meridional flow
        // Negative gradient (denser to south) → flow toward south → vDeep < 0
        // But AMOC: dense water forms in north, flows south at depth
        vDeep[k] = -gPrime / absF * drho_dy * H_deep;

        // Clamp deep velocity
        if (vDeep[k] > 0.05) vDeep[k] = 0.05;
        if (vDeep[k] < -0.05) vDeep[k] = -0.05;

        // Upwelling from deep flow convergence/divergence
        // ∂w/∂z ≈ -∂vDeep/∂y → w ≈ -H_deep * ∂vDeep/∂y
        const dvdy = (vDeep[jp] - vDeep[jm]) / (2 * dyj);
        wUp[k] = -dvdy * H_deep;
      }
    }
  }

  // MOC: zonally-integrated deep southward transport gives AMOC
  _computeOverturning() {
    const { nx, ny, dx } = this.grid;
    const { vDeep, mask, moc } = this;
    const g = this.grid;
    const H_deep = 4000;

    for (let j = 0; j < ny; j++) {
      let transport = 0;
      const dxj = dx[j];
      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        if (mask[k] > 0.5) {
          // Deep northward transport (m³/s per cell)
          // AMOC convention: positive = northward deep flow (but real AMOC has
          // northward surface, southward deep — we show the overturning magnitude)
          transport += vDeep[k] * H_deep * dxj;
        }
      }
      moc[j] = -transport / 1e6;  // negative because AMOC = southward deep = positive overturning
    }
  }
}
