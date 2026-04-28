// Ocean component
// Barotropic vorticity equation + SST evolution + two-layer vertical structure
// Takes: wind stress (τx, τy), net heat flux (Q_net), mask
// Produces: SST, streamfunction (ψ), velocities (u, v), deep temperature, overturning

import { EARTH, OCEAN, SIMULATION } from './params.js';

export class Ocean {
  constructor(grid) {
    this.grid = grid;
    const g = grid;

    // Surface layer state
    this.T = g.createField(15);     // SST (°C)
    this.psi = g.createField();     // streamfunction (m²/s)
    this.zeta = g.createField();    // relative vorticity (1/s)
    this.u = g.createField();       // zonal velocity (m/s)
    this.v = g.createField();       // meridional velocity (m/s)

    // Deep layer state
    this.Tdeep = g.createField(4);  // deep temperature (°C), init ~4°C
    this.uDeep = g.createField();   // deep zonal velocity (weak)
    this.vDeep = g.createField();   // deep meridional velocity

    // Vertical exchange
    this.wUp = g.createField();     // upwelling velocity (m/s, positive = upward)

    // Overturning diagnostic: meridional overturning streamfunction
    // Stored as (ny) array — zonally integrated at each latitude
    this.moc = new Float32Array(grid.ny);

    // Forcing fields (set by coupler)
    this.tauX = g.createField();
    this.tauY = g.createField();
    this.Qnet = g.createField();
    this.mask = g.createField();

    // Work arrays
    this._zetaNew = g.createField();
    this._Tnew = g.createField();
    this._TdeepNew = g.createField();
  }

  initSST(sstData) {
    for (let k = 0; k < this.grid.size; k++) {
      if (this.mask[k] > 0.5) {
        this.T[k] = sstData[k];
        // Deep ocean: cooler version of surface, clamped
        this.Tdeep[k] = Math.max(1, sstData[k] * 0.3 - 2);
      }
    }
  }

  step(dt) {
    this._stepVorticity(dt);
    this._solvePoissonSOR();
    this._computeVelocities();
    this._stepTemperature(dt);
    this._stepDeepOcean(dt);
    this._computeOverturning();
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

        // Wind stress curl
        const dtauY_dx = (tauY[ip] - tauY[im]) / (2 * dxj);
        const dtauX_dy = (tauX[jp] - tauX[jm]) / (2 * dyj);
        const curlTau = (dtauY_dx - dtauX_dy) / (rho * H);

        // Beta effect
        const betaV = beta[j] * this.v[k];

        // Laplacian of vorticity
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
    const { T, u, v, Qnet, mask, _Tnew, Tdeep } = this;
    const g = this.grid;
    const rhoCpH = rho * cp * H;

    // Vertical mixing parameters
    const gammaMix = 2e-7;        // base vertical exchange rate (1/s)
    const gammaConvect = 5e-6;    // enhanced mixing when surface is denser (1/s)

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

        // Upwind advection
        const uj = u[k];
        const vj = v[k];
        const dTdx = uj > 0
          ? (T[k] - T[im]) / dxj
          : (T[ip] - T[k]) / dxj;
        const dTdy = vj > 0
          ? (T[k] - T[jm]) / dyj
          : (T[jp] - T[k]) / dyj;

        const advection = -(uj * dTdx + vj * dTdy);

        // Surface heat flux
        const heating = Qnet[k] / rhoCpH;

        // Diffusion
        const lap = (
          (T[ip] - 2 * T[k] + T[im]) / (dxj * dxj) +
          (T[jp] - 2 * T[k] + T[jm]) / (dyj * dyj)
        );
        const diffusion = kappa * lap;

        // Vertical mixing with deep layer
        // Enhanced when surface is colder than deep (convective instability)
        const dT = T[k] - Tdeep[k];
        const gamma = dT < 0 ? gammaConvect : gammaMix;
        const vertMix = -gamma * dT;

        _Tnew[k] = T[k] + dt * (advection + heating + diffusion + vertMix);
        if (_Tnew[k] < -2) _Tnew[k] = -2;
        if (_Tnew[k] > 35) _Tnew[k] = 35;
      }
    }
    T.set(_Tnew);
  }

  _stepDeepOcean(dt) {
    const { nx, ny, dx, dy, lat } = this.grid;
    const { diffusivity: kappa } = OCEAN;
    const { T, Tdeep, mask, _TdeepNew, wUp } = this;
    const g = this.grid;

    const gammaMix = 2e-7;
    const gammaConvect = 5e-6;
    const kappaDeep = kappa * 0.1;  // deep diffusion much weaker

    for (let j = 1; j < ny - 1; j++) {
      const dxj = dx[j];
      const dyj = dy[j];
      const latj = lat[j];

      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        if (mask[k] < 0.5) { _TdeepNew[k] = Tdeep[k]; continue; }

        const ip = g.wrap(i + 1, j);
        const im = g.wrap(i - 1, j);
        const jp = g.idx(i, j + 1);
        const jm = g.idx(i, j - 1);

        // Deep layer diffusion
        const lap = (
          (Tdeep[ip] - 2 * Tdeep[k] + Tdeep[im]) / (dxj * dxj) +
          (Tdeep[jp] - 2 * Tdeep[k] + Tdeep[jm]) / (dyj * dyj)
        );

        // Vertical exchange (opposite sign from surface)
        const dT = T[k] - Tdeep[k];
        const gamma = dT < 0 ? gammaConvect : gammaMix;
        const vertMix = gamma * dT * (OCEAN.mixedLayerDepth / 4000); // ratio of layer depths

        // Deep water formation: enhanced cooling at high northern latitudes
        // Represents NADW formation — cold dense surface water sinks
        let deepFormation = 0;
        if (latj > 55 && T[k] < 5) {
          deepFormation = 1e-6 * (5 - T[k]);
        }

        // Upwelling estimate for diagnostics
        wUp[k] = gamma * dT > 0 ? gamma * 0.001 : -gammaConvect * 0.001;

        _TdeepNew[k] = Tdeep[k] + dt * (kappaDeep * lap + vertMix + deepFormation);
        if (_TdeepNew[k] < -2) _TdeepNew[k] = -2;
        if (_TdeepNew[k] > 20) _TdeepNew[k] = 20;
      }
    }
    Tdeep.set(_TdeepNew);
  }

  // Compute meridional overturning circulation diagnostic
  // Approximation: MOC ≈ zonally-integrated meridional velocity × depth
  // For a two-layer model, the overturning is the difference between
  // surface and deep meridional transports
  _computeOverturning() {
    const { nx, ny, dx, lat } = this.grid;
    const { v, mask, moc, T, Tdeep } = this;
    const g = this.grid;
    const H_surface = OCEAN.mixedLayerDepth;

    for (let j = 0; j < ny; j++) {
      let transport = 0;
      const dxj = dx[j];
      for (let i = 0; i < nx; i++) {
        const k = g.idx(i, j);
        if (mask[k] > 0.5) {
          // Surface northward transport (m³/s per grid cell)
          transport += v[k] * H_surface * dxj;
        }
      }
      // Convert to Sverdrups
      moc[j] = transport / 1e6;
    }
  }
}
