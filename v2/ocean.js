/**
 * SimAMOC v2 — Ocean Component
 *
 * Barotropic vorticity equation + tracer advection for T and S.
 * Two layers: surface mixed layer + deep abyssal layer.
 *
 * Interface (fluxes from coupler):
 *   IN:  windCurl, heatFlux, freshwaterFlux
 *   OUT: SST, surfaceCurrents
 */

/**
 * Step the ocean vorticity equation.
 *
 * dζ/dt = -J(ψ,ζ) - β*v + curl(τ)/ρH + r*∇²ψ + A*∇⁴ψ
 *
 * J = Arakawa Jacobian (conserves energy + enstrophy)
 * β = planetary vorticity gradient
 * curl(τ) = wind stress curl
 * r = bottom friction
 * A = lateral viscosity
 */
export function stepVorticity(state, grid, params) {
  const { nx, ny, n } = grid;
  const { dt, beta, r_drag, A_visc, wind_strength } = params;
  const invDx = 1 / grid.dx;
  const invDy = 1 / grid.dy;
  const invDx2 = invDx * invDx;
  const invDy2 = invDy * invDy;

  const { psi, zeta, mask, windCurlTau } = state;
  const zetaNew = new Float32Array(n);

  for (let j = 1; j < ny - 1; j++) {
    const cl = grid.cosLat[j];
    const invCl = 1 / Math.max(cl, 0.01);

    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (!mask[k]) { zetaNew[k] = 0; continue; }

      const ip = grid.iWrap(i + 1), im = grid.iWrap(i - 1);
      const e = j * nx + ip, w = j * nx + im;
      const n_ = (j + 1) * nx + i, s = (j - 1) * nx + i;
      const ne = (j + 1) * nx + ip, nw = (j + 1) * nx + im;
      const se = (j - 1) * nx + ip, sw = (j - 1) * nx + im;

      // Arakawa Jacobian J(ψ, ζ) — conserves energy + enstrophy
      const J1 = (psi[e] - psi[w]) * (zeta[n_] - zeta[s])
               - (psi[n_] - psi[s]) * (zeta[e] - zeta[w]);
      const J2 = psi[e] * (zeta[ne] - zeta[se]) - psi[w] * (zeta[nw] - zeta[sw])
               - psi[n_] * (zeta[ne] - zeta[nw]) + psi[s] * (zeta[se] - zeta[sw]);
      const J3 = zeta[e] * (psi[ne] - psi[se]) - zeta[w] * (psi[nw] - psi[sw])
               - zeta[n_] * (psi[ne] - psi[nw]) + zeta[s] * (psi[se] - psi[sw]);
      const J = (J1 + J2 + J3) / (12 * grid.dx * cl * grid.dy);

      // Beta effect: β * cos(lat) * ∂ψ/∂x
      const betaV = beta * cl * (psi[e] - psi[w]) * 0.5 * invDx;

      // Rayleigh friction: -r * ζ (damps vorticity directly)
      const friction = -r_drag * zeta[k];

      // Viscosity: A * ∇²ζ (biharmonic would be ∇⁴ψ but Laplacian of ζ suffices)
      const lapZeta = invDx2 * invCl * invCl * (zeta[e] + zeta[w] - 2 * zeta[k])
                    + invDy2 * (zeta[n_] + zeta[s] - 2 * zeta[k]);
      const viscosity = A_visc * lapZeta;

      // Wind forcing
      const wind = windCurlTau[k] * wind_strength;

      const dz = dt * (-J - betaV + wind + friction + viscosity);
      zetaNew[k] = Math.max(-500, Math.min(500, zeta[k] + dz));
    }
  }

  state.zeta.set(zetaNew);
}

/**
 * Poisson solver: ∇²ψ = ζ using SOR (Successive Over-Relaxation).
 */
export function solvePoissonSOR(psiArr, zetaArr, mask, grid, omega, nIter) {
  const { nx, ny } = grid;
  const invDx2 = 1 / (grid.dx * grid.dx);
  const invDy2 = 1 / (grid.dy * grid.dy);
  const cx = invDx2, cy = invDy2;
  const cc = -2 * (cx + cy);
  const invCC = 1 / cc;

  for (let iter = 0; iter < nIter; iter++) {
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        if (!mask[k]) continue;

        const ip = grid.iWrap(i + 1), im = grid.iWrap(i - 1);
        const res = cx * (psiArr[j * nx + ip] + psiArr[j * nx + im])
                  + cy * (psiArr[(j + 1) * nx + i] + psiArr[(j - 1) * nx + i])
                  + cc * psiArr[k] - zetaArr[k];
        psiArr[k] -= omega * res * invCC;
      }
    }
  }
}

/**
 * Step temperature and salinity via advection + diffusion.
 *
 * dT/dt = -u·∇T + κ∇²T + Q_net / (ρ c_p H)
 * dS/dt = -u·∇S + κ_s∇²S + S₀(E-P)/H + restoring
 *
 * heatFlux and freshwaterFlux come from the coupler (atmosphere component).
 */
export function stepTracers(state, grid, params, heatFlux, freshwaterFlux) {
  const { nx, ny } = grid;
  const { dt, kappa_diff, kappa_sal, sal_restoring, pe_sal_flux } = params;
  const invDx = 1 / grid.dx;
  const invDy = 1 / grid.dy;
  const invDx2 = invDx * invDx;
  const invDy2 = invDy * invDy;
  const { psi, temp, sal, mask, obsSalTarget } = state;

  const newTemp = new Float32Array(temp);
  const newSal = new Float32Array(sal);

  for (let j = 1; j < ny - 1; j++) {
    const cl = grid.cosLat[j];
    const invCl = 1 / Math.max(cl, 0.01);

    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      if (!mask[k]) continue;

      const ip = grid.iWrap(i + 1), im = grid.iWrap(i - 1);
      const ke = j * nx + ip, kw = j * nx + im;
      const kn = (j + 1) * nx + i, ks = (j - 1) * nx + i;

      // Velocities from streamfunction
      const u = -(psi[kn] - psi[ks]) * 0.5 * invDy;
      const v = (psi[ke] - psi[kw]) * 0.5 * invDx * invCl;

      // Zero-gradient BC at land boundaries
      const tE = mask[ke] ? temp[ke] : temp[k];
      const tW = mask[kw] ? temp[kw] : temp[k];
      const tN = mask[kn] ? temp[kn] : temp[k];
      const tS = mask[ks] ? temp[ks] : temp[k];

      // Advection (upwind-biased central)
      const advT = u * (tN - tS) * 0.5 * invDy + v * (tE - tW) * 0.5 * invDx * invCl;

      // Diffusion
      const diffT = kappa_diff * (
        invDx2 * invCl * invCl * (tE + tW - 2 * temp[k]) +
        invDy2 * (tN + tS - 2 * temp[k])
      );

      // Heat flux from atmosphere (via coupler)
      const qNet = heatFlux ? heatFlux[k] : 0;

      newTemp[k] = temp[k] + dt * (-advT + diffT + qNet);

      // --- Salinity ---
      const sE = mask[ke] ? sal[ke] : sal[k];
      const sW = mask[kw] ? sal[kw] : sal[k];
      const sN = mask[kn] ? sal[kn] : sal[k];
      const sS = mask[ks] ? sal[ks] : sal[k];

      const advS = u * (sN - sS) * 0.5 * invDy + v * (sE - sW) * 0.5 * invDx * invCl;
      const diffS = kappa_sal * (
        invDx2 * invCl * invCl * (sE + sW - 2 * sal[k]) +
        invDy2 * (sN + sS - 2 * sal[k])
      );

      // Freshwater flux: P-E drives salinity
      // E-P > 0 (evaporation dominates) → salinity increases
      // P-E > 0 (precipitation dominates) → salinity decreases
      const peFlux = freshwaterFlux ? freshwaterFlux[k] * pe_sal_flux : 0;

      // Restoring toward observed salinity
      const restore = sal_restoring * (obsSalTarget[k] - sal[k]);

      newSal[k] = sal[k] + dt * (-advS + diffS + peFlux + restore);
    }
  }

  state.temp.set(newTemp);
  state.sal.set(newSal);
}

/**
 * Enforce boundary conditions: ψ=0 and ζ=0 on land and polar walls.
 */
export function enforceBC(state, grid) {
  const { nx, ny, n } = grid;
  const { psi, zeta, mask } = state;
  for (let k = 0; k < n; k++) {
    const j = (k / nx) | 0;
    if (!mask[k] || j === 0 || j === ny - 1) {
      psi[k] = 0;
      zeta[k] = 0;
    }
  }
}
