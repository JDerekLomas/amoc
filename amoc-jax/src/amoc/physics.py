"""Barotropic vorticity dynamics on a spherical lat-lon grid.

The prognostic equation, in dimensionless grid units (dx=dy=1) with cos(lat)
metric corrections applied to differential operators:

    ∂ζ/∂t = -J(ψ, ζ) - β cos(φ) ∂ψ/∂x + curl(τ) - r ζ + A ∇²_grid ζ

This is the v1a formulation: barotropic, single layer, no thermodynamics.
v1b will add a deep layer with buoyancy coupling.

Conventions
-----------
- Periodic in x (longitude). Dirichlet (ψ=0, ζ=0) just outside in y (poles).
- Arakawa (1966) Jacobian: 9-point stencil that conserves energy, enstrophy,
  and total vorticity in the interior. Anti-symmetric: J(A,B) = -J(B,A).
- Grid Laplacian for viscosity matches the Poisson solver's stencil so the
  vorticity definition (ζ = ∇²_grid ψ) and the viscosity term are consistent.
- "Grid units" means dx=dy=1; physical scalings (Earth radius, time) are
  absorbed into the parameter values for now and will be calibrated in v1b/c.
"""
from __future__ import annotations

import jax.numpy as jnp

from .grid import Grid
from .poisson import grid_laplacian
from .state import Forcing, Params, State


def _pad_y_dirichlet(a: jnp.ndarray) -> jnp.ndarray:
    """Pad with one zero row on top and bottom for Dirichlet y-BC."""
    z = jnp.zeros_like(a[:1, :])
    return jnp.concatenate([z, a, z], axis=0)


def arakawa_jacobian(a: jnp.ndarray, b: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Arakawa (1966) energy-and-enstrophy-conserving Jacobian.

    Returns J(a, b) = (∂a/∂x ∂b/∂y - ∂a/∂y ∂b/∂x) / cos(lat),
    on the grid with periodic-x and Dirichlet-y boundaries (a, b zero outside).

    The 9-point form averages three discretizations (J++, J+x, Jx+) so that
    discrete ∫ a J(a,b) = 0 (energy) and ∫ J(a,b)² etc. are conserved.
    """
    # Periodic in x via roll; Dirichlet in y via pad-with-zeros.
    a_p = _pad_y_dirichlet(a)
    b_p = _pad_y_dirichlet(b)

    a_e = jnp.roll(a_p, -1, axis=1)        # i+1
    a_w = jnp.roll(a_p, 1, axis=1)         # i-1
    a_n = a_p[2:, :]                        # j+1 (after pad, equals original j+1)
    a_s = a_p[:-2, :]                       # j-1
    a_ne = jnp.roll(a_p, -1, axis=1)[2:, :]
    a_nw = jnp.roll(a_p, 1, axis=1)[2:, :]
    a_se = jnp.roll(a_p, -1, axis=1)[:-2, :]
    a_sw = jnp.roll(a_p, 1, axis=1)[:-2, :]
    a_e = a_e[1:-1, :]
    a_w = a_w[1:-1, :]

    b_e = jnp.roll(b_p, -1, axis=1)
    b_w = jnp.roll(b_p, 1, axis=1)
    b_n = b_p[2:, :]
    b_s = b_p[:-2, :]
    b_ne = jnp.roll(b_p, -1, axis=1)[2:, :]
    b_nw = jnp.roll(b_p, 1, axis=1)[2:, :]
    b_se = jnp.roll(b_p, -1, axis=1)[:-2, :]
    b_sw = jnp.roll(b_p, 1, axis=1)[:-2, :]
    b_e = b_e[1:-1, :]
    b_w = b_w[1:-1, :]

    # J++ : (a_e - a_w)(b_n - b_s) - (a_n - a_s)(b_e - b_w)
    Jpp = (a_e - a_w) * (b_n - b_s) - (a_n - a_s) * (b_e - b_w)
    # J+x : a_e (b_ne - b_se) - a_w (b_nw - b_sw) - a_n (b_ne - b_nw) + a_s (b_se - b_sw)
    Jpx = (
        a_e * (b_ne - b_se)
        - a_w * (b_nw - b_sw)
        - a_n * (b_ne - b_nw)
        + a_s * (b_se - b_sw)
    )
    # Jx+ : b_n (a_ne - a_nw) - b_s (a_se - a_sw) - b_e (a_ne - a_se) + b_w (a_nw - a_sw)
    Jxp = (
        b_n * (a_ne - a_nw)
        - b_s * (a_se - a_sw)
        - b_e * (a_ne - a_se)
        + b_w * (a_nw - a_sw)
    )

    # Each form has a 2dx*2dy = 4 in the denominator; average of three forms.
    J = (Jpp + Jpx + Jxp) / 12.0

    # Spherical metric: divide by cos(lat).
    # Avoid division by zero exactly at the poles (cos_lat=0); the Dirichlet
    # BC zeros ψ there anyway, so use a clipped denominator.
    cos_lat_safe = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    return J / cos_lat_safe


def vorticity_rhs(
    state: State, forcing: Forcing, params: Params, grid: Grid
) -> jnp.ndarray:
    """Return dζ/dt for the barotropic equation."""
    psi, zeta = state.psi, state.zeta

    advect = arakawa_jacobian(psi, zeta, grid)

    # β·v — planetary vorticity advection. On a sphere, v = (1/(R cos φ)) ∂_λψ
    # and β = 2Ω cos φ / R, so β·v = (2Ω/R²) ∂_λψ — the cos(φ) factors cancel.
    # In dimensionless grid units we just use β · ∂_iψ (no cos).
    dpsi_dx = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) * 0.5
    beta_v = params.beta * dpsi_dx

    visc = params.A_visc * grid_laplacian(zeta)
    fric = -params.r_friction * zeta
    wind = params.wind_strength * forcing.wind_curl

    rhs = -advect - beta_v + wind + fric + visc
    # Mask: land cells get zero tendency (and zero state, since rk2_step
    # also multiplies the predicted ζ by the mask). The Poisson solver
    # continues to solve on the full rectangle — ψ over land is
    # non-physical but never enters the RHS via masked cells.
    return rhs * forcing.ocean_mask
