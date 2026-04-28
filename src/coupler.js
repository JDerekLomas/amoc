// Flux coupler
// Orchestrates ocean + atmosphere exchange each timestep
// Owns the master simulation loop and model clock

import { SIMULATION } from './params.js';

export class Coupler {
  constructor(ocean, atmosphere, grid) {
    this.ocean = ocean;
    this.atmosphere = atmosphere;
    this.grid = grid;

    // Model clock
    this.modelTime = 0;           // seconds since start
    this.modelYear = 2025;
    this.modelMonth = 1;
    this.modelDay = 1;
    this.stepCount = 0;

    // Perturbation state
    this.freshwaterHosing = 0;    // Sv
    this.co2Multiplier = 1.0;
    this.globalTempOffset = 0;

    // Diagnostics (updated each coupling step)
    this.diagnostics = {
      globalMeanSST: 0,
      globalMeanSSS: 0,
      amocStrength: 0,
      energyBalance: 0,
      maxSpeed: 0,
    };
  }

  // Set by UI
  setFreshwaterHosing(sv) { this.freshwaterHosing = sv; }
  setCO2Multiplier(m) { this.co2Multiplier = m; }
  setTempOffset(offset) { this.globalTempOffset = offset; }

  step() {
    const { ocean, atmosphere, grid } = this;
    const dt = SIMULATION.dt;

    // 1. Atmosphere computes fluxes
    atmosphere.computeFluxes(ocean.T, ocean.mask);

    // 2. Transfer fluxes to ocean
    ocean.tauX.set(atmosphere.tauX);
    ocean.tauY.set(atmosphere.tauY);
    ocean.Qnet.set(atmosphere.Qnet);
    ocean.PmE.set(atmosphere.PmE);

    // 3. Apply perturbations
    this._applyPerturbations();

    // 4. Ocean steps forward
    ocean.step(dt);

    // 4. Advance clock
    this.modelTime += dt;
    this.stepCount++;


    this._advanceClock(dt);

    // 5. Update diagnostics
    this._computeDiagnostics();
  }

  _applyPerturbations() {
    const { ocean, grid } = this;
    const { nx, ny, lat } = grid;

    // Freshwater hosing: add fresh water to North Atlantic (50-70°N, 300-350°E)
    if (this.freshwaterHosing > 0) {
      const sv = this.freshwaterHosing;  // Sverdrups of freshwater
      // Convert to salinity tendency: dS/dt = -S * Fw / (H * A)
      // Fw = sv * 1e6 m³/s, spread over North Atlantic area
      let hosingCells = 0;
      for (let j = 0; j < ny; j++) {
        if (lat[j] < 50 || lat[j] > 70) continue;
        for (let i = 0; i < nx; i++) {
          const lonIdx = i * grid.dlon;
          if (lonIdx < 300 || lonIdx > 350) continue;
          const k = grid.idx(i, j);
          if (ocean.mask[k] > 0.5) hosingCells++;
        }
      }
      if (hosingCells > 0) {
        const cellArea = grid.dx[Math.round(ny * 0.75)] * grid.dy[Math.round(ny * 0.75)];
        const totalArea = hosingCells * cellArea;
        const freshwaterFlux = sv * 1e6 / totalArea;  // m/s of freshwater per unit area
        for (let j = 0; j < ny; j++) {
          if (lat[j] < 50 || lat[j] > 70) continue;
          for (let i = 0; i < nx; i++) {
            const lonIdx = i * grid.dlon;
            if (lonIdx < 300 || lonIdx > 350) continue;
            const k = grid.idx(i, j);
            if (ocean.mask[k] > 0.5) {
              ocean.PmE[k] += freshwaterFlux;
            }
          }
        }
      }
    }

    // CO2 effect: reduce OLR (greenhouse warming)
    // Each doubling of CO2 reduces OLR by ~3.7 W/m²
    if (this.co2Multiplier !== 1.0) {
      const dOLR = -3.7 * Math.log2(this.co2Multiplier);
      for (let k = 0; k < grid.size; k++) {
        ocean.Qnet[k] -= dOLR;  // reducing OLR = more net heating
      }
    }

    // Global temperature offset: direct heat flux perturbation
    if (this.globalTempOffset !== 0) {
      // Apply as restoring toward offset equilibrium
      // dQ = lambda * offset where lambda ≈ 1 W/m²/°C
      const dQ = 1.0 * this.globalTempOffset;
      for (let k = 0; k < grid.size; k++) {
        if (ocean.mask[k] > 0.5) ocean.Qnet[k] += dQ;
      }
    }
  }

  _advanceClock(dt) {
    const totalDays = this.modelTime / 86400;
    // Approximate: 30-day months, 360-day years
    this.modelYear = 2025 + Math.floor(totalDays / 360);
    this.modelMonth = 1 + Math.floor((totalDays % 360) / 30);
    this.modelDay = 1 + Math.floor(totalDays % 30);
  }

  _computeDiagnostics() {
    const { ocean, grid } = this;
    const { nx, ny } = grid;
    const { T, v, mask, psi } = ocean;

    // Global mean SST (ocean only)
    let sumT = 0, countT = 0;
    for (let k = 0; k < grid.size; k++) {
      if (mask[k] > 0.5) {
        sumT += T[k];
        countT++;
      }
    }
    this.diagnostics.globalMeanSST = countT > 0 ? sumT / countT : 0;

    // Global mean salinity
    let sumS = 0;
    for (let k = 0; k < grid.size; k++) {
      if (mask[k] > 0.5) sumS += ocean.S[k];
    }
    this.diagnostics.globalMeanSSS = countT > 0 ? sumS / countT : 0;

    // Max current speed
    let maxSpd = 0;
    for (let k = 0; k < grid.size; k++) {
      const spd = Math.sqrt(ocean.u[k] ** 2 + ocean.v[k] ** 2);
      if (spd > maxSpd) maxSpd = spd;
    }
    this.diagnostics.maxSpeed = maxSpd;

    // AMOC strength from MOC diagnostic at ~26.5°N
    let targetJ = 0;
    let minDist = Infinity;
    for (let j = 0; j < ny; j++) {
      const d = Math.abs(grid.lat[j] - 26.5);
      if (d < minDist) { minDist = d; targetJ = j; }
    }
    this.diagnostics.amocStrength = ocean.moc[targetJ];

    // Energy balance: mean Q_net over ocean
    let sumQ = 0;
    for (let k = 0; k < grid.size; k++) {
      if (mask[k] > 0.5) sumQ += ocean.Qnet[k];
    }
    this.diagnostics.energyBalance = countT > 0 ? sumQ / countT : 0;
  }

  getTimeString() {
    const m = String(this.modelMonth).padStart(2, '0');
    return `${this.modelYear}-${m}`;
  }
}
