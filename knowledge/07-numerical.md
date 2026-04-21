# Numerical Methods

## Grid

- 360 x 180 (1 degree resolution)
- LON: -180 to +180 (i=0 → dateline, i=180 → prime meridian)
- LAT: -80 to +80 (j=0 → Antarctic, j=179 → Arctic)
- Conversion: `i = (lon+180)/360 * 359`, `j = (lat+80)/160 * 179`

## Time Stepping

- dt = 5e-5 (non-dimensional)
- Steps per frame: 50 (tunable 10-200)
- Forward Euler (explicit) for all terms

### CFL Constraint
```
max_vel * dt * nx < 1
With dt=5e-5, nx=360: max_vel < 55.6
```

### Diffusive Stability
```
kappa * dt / dx^2 < 0.5
dx = 1/360 ≈ 2.78e-3
kappa_diff * dt / dx^2 = 2.5e-4 * 5e-5 / (2.78e-3)^2 ≈ 0.0016 (safe)
```

## Poisson Solver

Inverts laplacian(psi) = zeta to get streamfunction from vorticity.
- Method: Iterative Jacobi
- Iterations: 60 (surface), 20 (deep)
- Under-convergence → noisy velocity → affects all diagnostics
- Consider: more iterations, SOR, or multigrid for improvement

## Arakawa Jacobian

Conserves both energy AND enstrophy (unlike centered differences).
Three-point stencil in both directions. Critical for long-term stability.
Prevents spectral energy cascade to grid scale.

## Clamping (NaN Prevention)

| Field | Range | Notes |
|-------|-------|-------|
| Surface temp | [-10, 40] C | |
| Deep temp | [-5, 30] C | |
| Vorticity | [-500, 500] | |
| Velocities | CFL-limited | |

## WebGPU Compute Pipeline

```
timestep()     → update vorticity (Jacobian + forcing)
poisson()      → solve laplacian(psi) = zeta (60 Jacobi iters)
enforceBC()    → boundary conditions on psi, zeta
temperature()  → advect + force temperature (both layers)
deepTimestep() → deep layer vorticity
deepPoisson()  → deep layer streamfunction (20 iters)
```

GPU readback every 5 frames for stability monitoring (not every frame).

## Known Numerical Issues

1. **Checkerboard patterns**: odd-even decoupling from centered differences → needs viscosity
2. **Coastal ringing**: sharp land/ocean transitions → needs smoothing or higher-order stencils
3. **Poisson under-convergence**: 60 Jacobi iterations may not be enough for strong forcing
4. **Forward Euler instability**: explicit diffusion requires kappa*dt/dx^2 < 0.5

## How Real Models Handle This

| Model | Time stepping | Barotropic | Poisson/Elliptic |
|-------|-------------|-----------|-----------------|
| MOM6 | Split-explicit | Sub-stepped ~5-15s | Implicit free surface |
| NEMO | Leapfrog + Asselin | Sub-stepped | Preconditioned CG |
| MITgcm | Adams-Bashforth | Implicit | Preconditioned CG |
| ROMS | Split-explicit | Sub-stepped | — |

Our sim avoids the barotropic CFL by solving the vorticity equation (not primitive equations with a free surface). The Poisson solve replaces the need for explicit pressure gradient calculation.
