/**
 * SimAMOC v2 — Flux Coupler
 *
 * Orchestrates the exchange between atmosphere and ocean.
 * The only place components interact. Enforces conservation.
 *
 * Timestep order:
 *   1. Atmosphere reads SST → computes heatFlux, freshwaterFlux
 *   2. Ocean reads fluxes → steps vorticity, tracers
 *   3. Poisson solve for streamfunction
 *   4. Enforce boundary conditions
 *   5. Ice update
 */

import { stepVorticity, solvePoissonSOR, stepTracers, enforceBC } from './ocean.js';
import { stepAtmosphere } from './atmosphere.js';

/**
 * Run one coupled timestep.
 */
export function coupledStep(state, grid, params) {
  // 1. Atmosphere: compute fluxes from current SST
  const { heatFlux, freshwaterFlux } = stepAtmosphere(state, grid, params);

  // 2. Ocean vorticity
  stepVorticity(state, grid, params);

  // 3. Poisson solve: ∇²ψ = ζ
  enforceBC(state, grid);
  solvePoissonSOR(state.psi, state.zeta, state.mask, grid, params.omega_sor, params.sor_iters);
  enforceBC(state, grid);

  // 4. Ocean tracers (temperature + salinity) with atmospheric fluxes
  stepTracers(state, grid, params, heatFlux, freshwaterFlux);

  // 5. Ice
  updateIce(state, grid, params);

  // 6. Bookkeeping
  state.step++;
  state.simTime += params.dt;
}

/**
 * Run N coupled timesteps.
 */
export function step(state, grid, params, n = 1) {
  for (let i = 0; i < n; i++) {
    coupledStep(state, grid, params);
  }
}

/**
 * Simple thermodynamic sea ice.
 * Ice forms when SST < freeze point, melts when > melt point.
 */
function updateIce(state, grid, params) {
  const { nx, ny } = grid;
  const { mask, temp, iceFrac } = state;
  const { ice_freeze, ice_melt } = params;

  for (let k = 0; k < grid.n; k++) {
    if (!mask[k]) continue;
    if (temp[k] < ice_freeze) {
      iceFrac[k] = Math.min(1, iceFrac[k] + 0.01);
      temp[k] = ice_freeze;  // clamp at freezing
    } else if (temp[k] > ice_melt) {
      iceFrac[k] = Math.max(0, iceFrac[k] - 0.01);
    }
  }
}

/**
 * Compute diagnostics from current state.
 */
export function diagnostics(state, grid) {
  const { nx, ny } = grid;
  const { temp, sal, psi, mask, iceFrac } = state;
  const invDx = 1 / grid.dx, invDy = 1 / grid.dy;

  let globT = 0, globN = 0, tropT = 0, tropN = 0, polarT = 0, polarN = 0;
  let maxVel = 0, KE = 0, iceArea = 0;
  const zonalT = new Float32Array(ny);
  const zonalN = new Int32Array(ny);

  for (let j = 1; j < ny - 1; j++) {
    const lat = grid.latAt(j);
    const absLat = Math.abs(lat);
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (!mask[k]) continue;

      const ip = grid.iWrap(i + 1), im = grid.iWrap(i - 1);
      const u = -(psi[(j + 1) * nx + i] - psi[(j - 1) * nx + i]) * 0.5 * invDy;
      const v = (psi[j * nx + ip] - psi[j * nx + im]) * 0.5 * invDx;
      const s2 = u * u + v * v;
      if (s2 > maxVel * maxVel) maxVel = Math.sqrt(s2);
      KE += s2;

      const T = temp[k];
      zonalT[j] += T; zonalN[j]++;
      globT += T; globN++;
      if (absLat < 20) { tropT += T; tropN++; }
      if (absLat > 60) { polarT += T; polarN++; }
      if (iceFrac[k] > 0.5) iceArea++;
    }
  }

  // Zonal means
  const zonalMeanT = new Float32Array(ny);
  for (let j = 0; j < ny; j++) {
    zonalMeanT[j] = zonalN[j] > 0 ? zonalT[j] / zonalN[j] : NaN;
  }

  return {
    step: state.step,
    simTime: state.simTime,
    simYears: state.simTime / (2 * Math.PI),
    globalSST: globN > 0 ? globT / globN : NaN,
    tropicalSST: tropN > 0 ? tropT / tropN : NaN,
    polarSST: polarN > 0 ? polarT / polarN : NaN,
    maxVel,
    KE: KE * 0.5,
    iceArea,
    zonalMeanT,
    latitudes: grid.lat,
  };
}
