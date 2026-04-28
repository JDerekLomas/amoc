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

    // Diagnostics (updated each coupling step)
    this.diagnostics = {
      globalMeanSST: 0,
      globalMeanSSS: 0,
      amocStrength: 0,
      energyBalance: 0,
      maxSpeed: 0,
    };
  }

  // One coupling step: atmosphere reads SST, computes fluxes; ocean reads fluxes, steps forward
  step() {
    const { ocean, atmosphere, grid } = this;
    const dt = SIMULATION.dt;

    // 1. Atmosphere computes fluxes from current ocean SST
    atmosphere.computeFluxes(ocean.T, ocean.mask);

    // 2. Transfer fluxes to ocean
    ocean.tauX.set(atmosphere.tauX);
    ocean.tauY.set(atmosphere.tauY);
    ocean.Qnet.set(atmosphere.Qnet);
    ocean.PmE.set(atmosphere.PmE);

    // 3. Ocean steps forward
    ocean.step(dt);

    // 4. Advance clock
    this.modelTime += dt;
    this.stepCount++;


    this._advanceClock(dt);

    // 5. Update diagnostics
    this._computeDiagnostics();
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
