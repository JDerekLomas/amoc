"""Two-layer baroclinic vorticity dynamics on a spherical lat-lon grid.

For v1b, two stacked layers (surface s, deep d) each obey a vorticity
equation. Coupling:

    dζ_s/dt = -(1/cosφ)J(ψ_s,ζ_s) - β∂_λψ_s
              + curl(τ)/H_s - r_s ζ_s + A∇²ζ_s
              - α ∂_λ b
              + F_cs (ψ_d - ψ_s)

    dζ_d/dt = -(1/cosφ)J(ψ_d,ζ_d) - β∂_λψ_d
              + 0           - r_d ζ_d + A∇²ζ_d
              + α ∂_λ b
              + F_cd (ψ_s - ψ_d)

The buoyancy term -α∂_λ b on the surface layer (and +α∂_λ b on the deep)
is the simplest representation of a baroclinic pressure-gradient force
that drives opposing flows in the two layers — i.e. the thermal-wind
shear. b is prescribed in v1b (proxy from observed SST climatology) and
will become prognostic in v1c.

The frictional coupling F_cs(ψ_d-ψ_s) drags layers together; the
asymmetric (F_cs > F_cd) reflects momentum transfer being faster from
ocean below into the surface than the reverse, given thickness ratio.

v1a is recovered by setting alpha_buoy = 0, F_couple_* = 0, and
ignoring the deep layer — that's a single-layer model with a stationary
zero deep field.
"""
from __future__ import annotations

import jax.numpy as jnp

from .grid import Grid
from .poisson import grid_laplacian
from .state import Forcing, Params, State


def _pad_y_dirichlet(a: jnp.ndarray) -> jnp.ndarray:
    z = jnp.zeros_like(a[:1, :])
    return jnp.concatenate([z, a, z], axis=0)


def arakawa_jacobian(a: jnp.ndarray, b: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Arakawa (1966) energy-and-enstrophy-conserving Jacobian.

    Returns J(a, b) = (∂_iψ ∂_jζ - ∂_jψ ∂_iζ) / cos(lat),
    on the grid with periodic-x and Dirichlet-y boundaries.
    """
    a_p = _pad_y_dirichlet(a)
    b_p = _pad_y_dirichlet(b)

    a_e = jnp.roll(a_p, -1, axis=1)[1:-1, :]
    a_w = jnp.roll(a_p, 1, axis=1)[1:-1, :]
    a_n = a_p[2:, :]
    a_s = a_p[:-2, :]
    a_ne = jnp.roll(a_p, -1, axis=1)[2:, :]
    a_nw = jnp.roll(a_p, 1, axis=1)[2:, :]
    a_se = jnp.roll(a_p, -1, axis=1)[:-2, :]
    a_sw = jnp.roll(a_p, 1, axis=1)[:-2, :]

    b_e = jnp.roll(b_p, -1, axis=1)[1:-1, :]
    b_w = jnp.roll(b_p, 1, axis=1)[1:-1, :]
    b_n = b_p[2:, :]
    b_s = b_p[:-2, :]
    b_ne = jnp.roll(b_p, -1, axis=1)[2:, :]
    b_nw = jnp.roll(b_p, 1, axis=1)[2:, :]
    b_se = jnp.roll(b_p, -1, axis=1)[:-2, :]
    b_sw = jnp.roll(b_p, 1, axis=1)[:-2, :]

    Jpp = (a_e - a_w) * (b_n - b_s) - (a_n - a_s) * (b_e - b_w)
    Jpx = (
        a_e * (b_ne - b_se)
        - a_w * (b_nw - b_sw)
        - a_n * (b_ne - b_nw)
        + a_s * (b_se - b_sw)
    )
    Jxp = (
        b_n * (a_ne - a_nw)
        - b_s * (a_se - a_sw)
        - b_e * (a_ne - a_se)
        + b_w * (a_nw - a_sw)
    )
    J = (Jpp + Jpx + Jxp) / 12.0
    cos_lat_safe = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    return J / cos_lat_safe


def _layer_rhs_common(
    psi: jnp.ndarray, zeta: jnp.ndarray,
    r: float, A: float, beta: float, grid: Grid,
) -> jnp.ndarray:
    """Terms common to both layers: advection, beta, viscosity, friction."""
    advect = arakawa_jacobian(psi, zeta, grid)
    dpsi_dx = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) * 0.5
    beta_v = beta * dpsi_dx
    visc = A * grid_laplacian(zeta)
    fric = -r * zeta
    return -advect - beta_v + fric + visc


def vorticity_rhs(
    state: State, forcing: Forcing, params: Params, grid: Grid
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (dζ_s/dt, dζ_d/dt) for the two-layer model.

    Both tendencies are masked to ocean cells.
    """
    # Buoyancy zonal gradient (centered, periodic in x). Mask off coastlines
    # where buoyancy jumps to zero (over land) — taking ∂_x across a coast
    # would produce an order-1 gradient from the discontinuity, multiplied by
    # alpha into the tendency, which destabilizes the stepper. We only keep
    # the gradient where both neighbors are ocean.
    b_e = jnp.roll(forcing.buoyancy, -1, axis=1)
    b_w = jnp.roll(forcing.buoyancy, 1, axis=1)
    m_e = jnp.roll(forcing.ocean_mask, -1, axis=1)
    m_w = jnp.roll(forcing.ocean_mask, 1, axis=1)
    db_dx = (b_e - b_w) * 0.5 * m_e * m_w * forcing.ocean_mask

    base_s = _layer_rhs_common(
        state.psi_s, state.zeta_s,
        params.r_friction_s, params.A_visc_s, params.beta, grid,
    )
    base_d = _layer_rhs_common(
        state.psi_d, state.zeta_d,
        params.r_friction_d, params.A_visc_d, params.beta, grid,
    )

    # Wind only on surface; scale by 1/H_s for layer-thickness consistency.
    # In dimensionless units we just keep wind_strength as the knob.
    wind = params.wind_strength * forcing.wind_curl

    # Buoyancy: -alpha * d_x b on surface, +alpha * d_x b on deep
    buoy_s = -params.alpha_buoy * db_dx
    buoy_d = +params.alpha_buoy * db_dx

    # Interfacial friction: drag proportional to vertical shear of horizontal
    # velocity, which translates to (ζ_s - ζ_d) in the vorticity formulation.
    # Asymmetric strength reflects momentum sink vs source: surface relaxes
    # toward deep at rate F_cs, deep relaxes toward surface at the slower
    # rate F_cd (heavier layer, more inertia).
    shear = state.zeta_s - state.zeta_d
    couple_s = -params.F_couple_s * shear
    couple_d = +params.F_couple_d * shear

    rhs_s = (base_s + wind + buoy_s + couple_s) * forcing.ocean_mask
    rhs_d = (base_d + buoy_d + couple_d) * forcing.ocean_mask
    return rhs_s, rhs_d
