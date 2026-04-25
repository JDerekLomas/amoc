"""Two-layer baroclinic ocean dynamics on a spherical lat-lon grid.

v1c: prognostic T, S in the surface layer plus T in the deep layer; the
buoyancy term in the vorticity equation is now *derived* from those fields
via a linear equation of state, rather than prescribed (as in v1b).

Equations -- in dimensionless grid units, ocean-cell only (RHS multiplied
by mask):

    ζ-tendencies (vorticity)
    dζ_s/dt = -(1/cosφ)J(ψ_s,ζ_s) - β∂_λψ_s
              + curl(τ)/H_s - r_s ζ_s + A∇²ζ_s
              - α_BC ∂_λ b           [thermal-wind shear forcing]
              - F_cs (ζ_s - ζ_d)     [interfacial friction]

    dζ_d/dt = -(1/cosφ)J(ψ_d,ζ_d) - β∂_λψ_d
              + 0           - r_d ζ_d + A∇²ζ_d
              + α_BC ∂_λ b
              + F_cd (ζ_s - ζ_d)

    Tracer tendencies (advection-diffusion + restoring + vertical mixing)
    dT_s/dt = -(1/cosφ)J(ψ_s, T_s) + κ_T ∇²T_s
              + (T*  - T_s)/τ_T              [Haney restoring at surface]
              - γ_eff(b) (T_s - T_d)         [vertical mixing]
    dS_s/dt = -(1/cosφ)J(ψ_s, S_s) + κ_S ∇²S_s
              + (S*  - S_s)/τ_S
              - γ_eff(b) (S_s - S_d_fixed)
              + F_fresh                      [freshwater flux, v1d]
    dT_d/dt = -(1/cosφ)J(ψ_d, T_d) + κ_T ∇²T_d
              + γ_eff(b) (H_s/H_d) (T_s - T_d)

where γ_eff(b) is enhanced when the surface is denser than the deep
(convective adjustment): γ_eff = γ_TS  if  b_s ≥ b_d  (surface lighter)
                          γ_eff = γ_conv  if  b_s < b_d  (surface denser)
b is computed from the linear EOS:  b = -α_T(T - T_0) + α_S(S - S_0)
with T_0, S_0 reference values that drop out of the *gradient* used in the
vorticity equation. We use b_s - b_d (the surface-deep contrast) as the
quantity that drives shear and convection.

v1a is recovered by setting alpha_BC = 0 and zeroing the deep layer.
v1b's prescribed buoyancy still works via Forcing.buoyancy if alpha_BC=0
and alpha_buoy>0 (legacy code path).
"""
from __future__ import annotations

import jax.numpy as jnp

from .grid import Grid
from .poisson import grid_laplacian
from .state import Forcing, Params, State


# Reference T, S for the EOS (drop out of gradients, kept as named values).
T0 = 15.0   # °C
S0 = 35.0   # psu


def _pad_y_dirichlet(a: jnp.ndarray) -> jnp.ndarray:
    z = jnp.zeros_like(a[:1, :])
    return jnp.concatenate([z, a, z], axis=0)


def arakawa_jacobian(a: jnp.ndarray, b: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Arakawa (1966) energy-and-enstrophy-conserving Jacobian.

    J(a, b) = (∂_iψ ∂_jζ - ∂_jψ ∂_iζ) / cos(lat),
    periodic-x and Dirichlet-y boundaries.
    """
    a_p = _pad_y_dirichlet(a)
    b_p = _pad_y_dirichlet(b)

    a_e  = jnp.roll(a_p, -1, axis=1)[1:-1, :]
    a_w  = jnp.roll(a_p,  1, axis=1)[1:-1, :]
    a_n  = a_p[2:, :]
    a_s_ = a_p[:-2, :]
    a_ne = jnp.roll(a_p, -1, axis=1)[2:,  :]
    a_nw = jnp.roll(a_p,  1, axis=1)[2:,  :]
    a_se = jnp.roll(a_p, -1, axis=1)[:-2, :]
    a_sw = jnp.roll(a_p,  1, axis=1)[:-2, :]

    b_e  = jnp.roll(b_p, -1, axis=1)[1:-1, :]
    b_w  = jnp.roll(b_p,  1, axis=1)[1:-1, :]
    b_n  = b_p[2:, :]
    b_s_ = b_p[:-2, :]
    b_ne = jnp.roll(b_p, -1, axis=1)[2:,  :]
    b_nw = jnp.roll(b_p,  1, axis=1)[2:,  :]
    b_se = jnp.roll(b_p, -1, axis=1)[:-2, :]
    b_sw = jnp.roll(b_p,  1, axis=1)[:-2, :]

    Jpp = (a_e - a_w) * (b_n - b_s_) - (a_n - a_s_) * (b_e - b_w)
    Jpx = (
        a_e * (b_ne - b_se)
        - a_w * (b_nw - b_sw)
        - a_n * (b_ne - b_nw)
        + a_s_ * (b_se - b_sw)
    )
    Jxp = (
        b_n * (a_ne - a_nw)
        - b_s_ * (a_se - a_sw)
        - b_e * (a_ne - a_se)
        + b_w * (a_nw - a_sw)
    )
    J = (Jpp + Jpx + Jxp) / 12.0
    cos_lat_safe = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    return J / cos_lat_safe


def buoyancy_from_TS(T: jnp.ndarray, S: jnp.ndarray, params: Params) -> jnp.ndarray:
    """Linear equation of state. Returns buoyancy anomaly (positive = light)."""
    return params.alpha_T * (T - T0) - params.alpha_S * (S - S0)


def _layer_dyn_common(
    psi: jnp.ndarray, zeta: jnp.ndarray,
    r: float, A: float, beta: float, grid: Grid,
) -> jnp.ndarray:
    """ζ tendencies common to both layers: advection, β, viscosity, friction."""
    advect = arakawa_jacobian(psi, zeta, grid)
    dpsi_dx = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) * 0.5
    visc = A * grid_laplacian(zeta)
    fric = -r * zeta
    return -advect - beta * dpsi_dx + fric + visc


def tracer_laplacian(field: jnp.ndarray) -> jnp.ndarray:
    """Grid Laplacian with periodic x and *no-flux* (Neumann) y boundaries.

    Tracers (T, S) don't have a Dirichlet=0 boundary at the poles — the
    pole is closed (no transport through it). So we replicate the
    boundary row instead of padding with zero. With this, the Laplacian
    of a uniform field is exactly zero everywhere, including the
    boundary rows.

    The streamfunction's grid Laplacian (in `poisson.py`) keeps the
    Dirichlet form because ψ=0 on the boundary is the correct BC for
    closed-basin streamfunction.
    """
    f_xp = jnp.roll(field, -1, axis=1)
    f_xm = jnp.roll(field,  1, axis=1)
    # No-flux y BC: replicate first/last row when "going outside".
    f_yp = jnp.concatenate([field[1:, :],  field[-1:, :]], axis=0)
    f_ym = jnp.concatenate([field[:1, :],  field[:-1, :]], axis=0)
    return (f_xp - 2 * field + f_xm) + (f_yp - 2 * field + f_ym)


def _masked_dx(field: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Zonal centered difference, masked across coastlines so a sharp jump
    at land doesn't produce a spurious O(1) gradient."""
    f_e = jnp.roll(field, -1, axis=1)
    f_w = jnp.roll(field,  1, axis=1)
    m_e = jnp.roll(mask,  -1, axis=1)
    m_w = jnp.roll(mask,   1, axis=1)
    return (f_e - f_w) * 0.5 * m_e * m_w * mask


def vorticity_rhs(
    state: State, forcing: Forcing, params: Params, grid: Grid
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (dζ_s/dt, dζ_d/dt). Used for v1a/b/c."""
    base_s = _layer_dyn_common(
        state.psi_s, state.zeta_s,
        params.r_friction_s, params.A_visc_s, params.beta, grid,
    )
    base_d = _layer_dyn_common(
        state.psi_d, state.zeta_d,
        params.r_friction_d, params.A_visc_d, params.beta, grid,
    )

    # Wind on surface only.
    wind = params.wind_strength * forcing.wind_curl

    # Buoyancy term. v1c: derived from T,S contrast between layers.
    # v1b legacy: prescribed Forcing.buoyancy field via alpha_buoy.
    b_s = buoyancy_from_TS(state.T_s, state.S_s, params)
    b_d = buoyancy_from_TS(state.T_d, forcing.S_d_const, params)
    b = b_s - b_d   # surface lighter than deep => positive
    db_dx_v1c = _masked_dx(b, forcing.ocean_mask) * params.alpha_BC

    db_dx_v1b = _masked_dx(forcing.buoyancy, forcing.ocean_mask) * params.alpha_buoy

    buoy_s = -(db_dx_v1c + db_dx_v1b)
    buoy_d = +(db_dx_v1c + db_dx_v1b)

    # Interfacial friction on the vorticity shear.
    shear = state.zeta_s - state.zeta_d
    couple_s = -params.F_couple_s * shear
    couple_d = +params.F_couple_d * shear

    rhs_s = (base_s + wind + buoy_s + couple_s) * forcing.ocean_mask
    rhs_d = (base_d        + buoy_d + couple_d) * forcing.ocean_mask
    return rhs_s, rhs_d


def _convective_gamma(b_s: jnp.ndarray, b_d: jnp.ndarray, params: Params) -> jnp.ndarray:
    """Enhanced vertical exchange where the surface is denser than the deep.

    b_s, b_d are buoyancy anomalies (positive = light). Surface dense means
    b_s < b_d. We smoothly interpolate between gamma_TS (background) and
    gamma_conv (convective) using a sigmoid in (b_d - b_s).
    """
    drho = b_d - b_s   # positive when surface is denser
    # Smooth indicator (sigmoid). Sharpness chosen so a buoyancy difference
    # of ~1e-3 (a few °C of temperature contrast at α_T=2e-4) is enough to
    # saturate the convective rate.
    indicator = 0.5 * (1.0 + jnp.tanh(2000.0 * drho))
    return params.gamma_TS + (params.gamma_conv - params.gamma_TS) * indicator


def tracer_rhs(
    state: State, forcing: Forcing, params: Params, grid: Grid,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (dT_s/dt, dS_s/dt, dT_d/dt). v1c thermodynamics."""
    # Advection of each tracer by its layer's streamfunction.
    advect_T_s = arakawa_jacobian(state.psi_s, state.T_s, grid)
    advect_S_s = arakawa_jacobian(state.psi_s, state.S_s, grid)
    advect_T_d = arakawa_jacobian(state.psi_d, state.T_d, grid)

    # Diffusion (grid Laplacian; cos(φ) consistency is the same as for ζ).
    diff_T_s = params.kappa_T * tracer_laplacian(state.T_s)
    diff_S_s = params.kappa_S * tracer_laplacian(state.S_s)
    diff_T_d = params.kappa_T * tracer_laplacian(state.T_d)

    # Surface restoring (Haney). Stronger on T than S, like real CMIP setups.
    restore_T = (forcing.T_target - state.T_s) / params.tau_T
    restore_S = (forcing.S_target - state.S_s) / params.tau_S

    # Vertical exchange between layers, enhanced where convectively unstable.
    b_s = buoyancy_from_TS(state.T_s, state.S_s, params)
    b_d = buoyancy_from_TS(state.T_d, forcing.S_d_const, params)
    gamma = _convective_gamma(b_s, b_d, params)
    dT_exchange = gamma * (state.T_s - state.T_d)
    dS_exchange = gamma * (state.S_s - forcing.S_d_const)

    # Mass conservation: heat removed from surface gets put into deep,
    # weighted by thickness ratio.
    H_ratio = params.H_s / params.H_d

    rhs_T_s = -advect_T_s + diff_T_s + restore_T - dT_exchange
    rhs_S_s = -advect_S_s + diff_S_s + restore_S - dS_exchange + forcing.F_fresh
    rhs_T_d = -advect_T_d + diff_T_d + H_ratio * dT_exchange

    m = forcing.ocean_mask
    return rhs_T_s * m, rhs_S_s * m, rhs_T_d * m
