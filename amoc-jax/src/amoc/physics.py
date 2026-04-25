"""Two-layer baroclinic ocean dynamics on a spherical lat-lon grid.

Full physics: vorticity dynamics (Arakawa Jacobian, beta, wind, buoyancy,
interfacial coupling, coastal damping), tracer transport (advection by
geostrophic + Ekman flow, diffusion, radiative forcing, vertical exchange
with variable MLD and convective adjustment), and deep layer dynamics
including prognostic salinity and meridional overturning tendency.
"""
from __future__ import annotations

import jax.numpy as jnp

from .grid import Grid
from .poisson import grid_laplacian
from .state import Forcing, Params, State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad_y_dirichlet(a: jnp.ndarray) -> jnp.ndarray:
    z = jnp.zeros_like(a[:1, :])
    return jnp.concatenate([z, a, z], axis=0)


def arakawa_jacobian(a: jnp.ndarray, b: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Arakawa (1966) J(a,b) with metric correction.
    Uses dx*cos(lat) for zonal and dy for meridional, matching JS convention."""
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
    cos_lat_safe = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    mDx = grid.dx_nd * cos_lat_safe  # physical dx at each latitude
    mDy = grid.dy_nd
    J = (Jpp + Jpx + Jxp) / (12.0 * mDx * mDy)
    return J


def tracer_laplacian(field: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Grid Laplacian with periodic x and no-flux (Neumann) y, with cos(lat) metric."""
    cos_lat = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    invDx2 = 1.0 / (grid.dx_nd * cos_lat) ** 2
    invDy2 = 1.0 / (grid.dy_nd ** 2)
    f_xp = jnp.roll(field, -1, axis=1)
    f_xm = jnp.roll(field,  1, axis=1)
    f_yp = jnp.concatenate([field[1:, :],  field[-1:, :]], axis=0)
    f_ym = jnp.concatenate([field[:1, :],  field[:-1, :]], axis=0)
    return invDx2 * (f_xp - 2 * field + f_xm) + invDy2 * (f_yp - 2 * field + f_ym)


def masked_laplacian(field: jnp.ndarray, mask: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Grid Laplacian with one-sided stencil at coastlines, with cos(lat) metric."""
    cos_lat = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    invDx2 = 1.0 / (grid.dx_nd * cos_lat) ** 2
    invDy2 = 1.0 / (grid.dy_nd ** 2)

    m_e = jnp.roll(mask, -1, axis=1)
    m_w = jnp.roll(mask,  1, axis=1)
    m_n = jnp.concatenate([mask[1:, :], mask[-1:, :]], axis=0)
    m_s = jnp.concatenate([mask[:1, :], mask[:-1, :]], axis=0)

    f_e = jnp.where(m_e, jnp.roll(field, -1, axis=1), field)
    f_w = jnp.where(m_w, jnp.roll(field,  1, axis=1), field)
    f_n = jnp.where(m_n, jnp.concatenate([field[1:, :], field[-1:, :]], axis=0), field)
    f_s = jnp.where(m_s, jnp.concatenate([field[:1, :], field[:-1, :]], axis=0), field)
    return invDx2 * (f_e - 2 * field + f_w) + invDy2 * (f_n - 2 * field + f_s)


def _masked_dx(field: jnp.ndarray, mask: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Zonal centered difference with metric, masked across coastlines."""
    cos_lat = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    invDx = 1.0 / (grid.dx_nd * cos_lat)
    f_e = jnp.roll(field, -1, axis=1)
    f_w = jnp.roll(field,  1, axis=1)
    m_e = jnp.roll(mask,  -1, axis=1)
    m_w = jnp.roll(mask,   1, axis=1)
    return (f_e - f_w) * 0.5 * invDx * m_e * m_w * mask


def _coastal_damping(zeta: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Compute damping factor for vorticity near coastlines.
    Returns multiplier: 1.0 for interior ocean, 0.9-0.95 near coast, 0 on land."""
    m_e = jnp.roll(mask, -1, axis=1)
    m_w = jnp.roll(mask,  1, axis=1)
    m_n = jnp.concatenate([mask[1:, :], mask[-1:, :]], axis=0)
    m_s = jnp.concatenate([mask[:1, :], mask[:-1, :]], axis=0)
    m_ne = jnp.roll(m_n, -1, axis=1)
    m_nw = jnp.roll(m_n,  1, axis=1)
    m_se = jnp.roll(m_s, -1, axis=1)
    m_sw = jnp.roll(m_s,  1, axis=1)

    cardinal_ok = m_e * m_w * m_n * m_s
    diagonal_ok = m_ne * m_nw * m_se * m_sw

    # Interior: all 8 neighbors ocean -> factor 1.0
    # Cardinal land neighbor -> 0.9
    # Only diagonal land neighbor -> 0.95
    # On land -> 0.0
    factor = jnp.where(
        mask < 0.5, 0.0,
        jnp.where(cardinal_ok < 0.5, 0.9,
                   jnp.where(diagonal_ok < 0.5, 0.95, 1.0))
    )
    return factor


# ---------------------------------------------------------------------------
# Vorticity
# ---------------------------------------------------------------------------

def _layer_dyn_common(
    psi: jnp.ndarray, zeta: jnp.ndarray,
    r: float, A: float, beta: float, grid: Grid,
) -> jnp.ndarray:
    """Vorticity tendencies common to both layers.
    Matches JS convention: derivatives use dx=1/(nx-1), cos(lat) metric."""
    advect = arakawa_jacobian(psi, zeta, grid)
    cos_lat = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    invDx = 1.0 / (grid.dx_nd * cos_lat)
    dpsi_dx = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) * 0.5 * invDx
    # beta varies with latitude: beta * cos(lat)
    lat_rad = jnp.deg2rad(grid.lat)[:, None]
    beta_local = beta * jnp.cos(lat_rad)
    beta_term = beta_local * dpsi_dx
    visc = A * grid_laplacian(zeta, grid.dx_nd, grid.dy_nd)
    fric = -r * zeta
    return -advect - beta_term + fric + visc


def vorticity_rhs(
    state: State, forcing: Forcing, params: Params, grid: Grid
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (dzeta_s/dt, dzeta_d/dt) with full physics."""
    mask = forcing.ocean_mask

    # Base dynamics
    base_s = _layer_dyn_common(
        state.psi_s, state.zeta_s,
        params.r_friction_s, params.A_visc_s, params.beta, grid,
    )
    base_d = _layer_dyn_common(
        state.psi_d, state.zeta_d,
        params.r_friction_d, params.A_visc_d, params.beta, grid,
    )

    # Wind forcing (surface only)
    wind = params.wind_strength * forcing.wind_curl

    # Buoyancy: density gradient from T and S (already has invDx from _masked_dx)
    # drho/dx = -alpha_T * dT/dx + beta_S * dS/dx
    dTdx_s = _masked_dx(state.T_s, mask, grid)
    dSdx_s = _masked_dx(state.S_s, mask, grid)
    dRhodx_s = -params.alpha_T * dTdx_s + params.beta_S * dSdx_s
    buoy_s = -dRhodx_s

    # Deep buoyancy from deep T and S
    dTdx_d = _masked_dx(state.T_d, mask, grid)
    dSdx_d = _masked_dx(state.S_d, mask, grid)
    dRhodx_d = -params.alpha_T * dTdx_d + params.beta_S * dSdx_d
    buoy_d = dRhodx_d

    # Interfacial coupling (on streamfunction, not vorticity shear)
    couple_s = params.F_couple_s * (state.psi_d - state.psi_s)
    couple_d = params.F_couple_d * (state.psi_s - state.psi_d)

    # Deep meridional overturning tendency: dTdy drives equatorward deep flow
    invDy = 1.0 / grid.dy_nd
    T_d_n = jnp.concatenate([state.T_d[1:, :], state.T_d[-1:, :]], axis=0)
    T_d_s = jnp.concatenate([state.T_d[:1, :], state.T_d[:-1, :]], axis=0)
    m_n = jnp.concatenate([mask[1:, :], mask[-1:, :]], axis=0)
    m_s = jnp.concatenate([mask[:1, :], mask[:-1, :]], axis=0)
    dTdy_d = (jnp.where(m_n, T_d_n, state.T_d) - jnp.where(m_s, T_d_s, state.T_d)) * 0.5 * invDy
    mot = params.mot_strength * dTdy_d

    # Coastal damping
    damp_s = _coastal_damping(state.zeta_s, mask)
    damp_d = _coastal_damping(state.zeta_d, mask)

    # Apply: where damping < 1, replace RHS with damping toward zero
    rhs_s_full = (base_s + wind + buoy_s + couple_s) * mask
    rhs_d_full = (base_d + buoy_d + couple_d + mot) * mask

    # For coastal cells (damp < 1), the effective update is:
    # zeta_new = zeta * damp_factor (instantaneous), which means
    # dzeta/dt = zeta * (damp_factor - 1) / dt
    # But simpler: we flag coastal cells and blend
    is_interior_s = (damp_s > 0.99)
    is_interior_d = (damp_d > 0.99)
    rhs_s = jnp.where(is_interior_s, rhs_s_full,
                       state.zeta_s * (damp_s - 1.0) / jnp.maximum(params.dt, 1e-10))
    rhs_d = jnp.where(is_interior_d, rhs_d_full,
                       state.zeta_d * (damp_d - 1.0) / jnp.maximum(params.dt, 1e-10))

    return rhs_s * mask, rhs_d * mask


# ---------------------------------------------------------------------------
# Tracers
# ---------------------------------------------------------------------------

def mixed_layer_depth(lat: jnp.ndarray) -> jnp.ndarray:
    """Variable mixed layer depth by latitude (meters). Shape (ny,)."""
    abs_lat = jnp.abs(lat)
    mld_base = 30.0 + 70.0 * jnp.power(abs_lat / 80.0, 1.5)
    # ACC enhancement
    d_acc = (lat + 50.0) / 12.0
    mld_acc = jnp.where(
        (lat < -35.0) & (lat > -65.0),
        250.0 * jnp.exp(-d_acc ** 2), 0.0
    )
    # Subpolar NH enhancement
    d_sub = (lat - 62.0) / 8.0
    mld_sub = jnp.where(
        (lat > 50.0) & (lat < 75.0),
        150.0 * jnp.exp(-d_sub ** 2), 0.0
    )
    return mld_base + mld_acc + mld_sub


def tracer_rhs(
    state: State, forcing: Forcing, params: Params, grid: Grid,
    q_net: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (dT_s/dt, dS_s/dt, dT_d/dt, dS_d/dt).

    q_net: (ny, nx) net radiative heating from atmosphere module.
    """
    mask = forcing.ocean_mask
    cos_lat = jnp.clip(grid.cos_lat, 1e-6, None)[:, None]
    invDx = 1.0 / (grid.dx_nd * cos_lat)
    invDy = 1.0 / grid.dy_nd

    # --- Geostrophic advection (Jacobian) ---
    advect_T_s = arakawa_jacobian(state.psi_s, state.T_s, grid)
    advect_S_s = arakawa_jacobian(state.psi_s, state.S_s, grid)
    advect_T_d = arakawa_jacobian(state.psi_d, state.T_d, grid)

    # --- Ekman advection ---
    # Use geostrophic velocity (from psi) gradients for Ekman transport
    # dPsi/dx and dPsi/dy give velocities, combined with dT/dx, dT/dy
    # In JS: ekmanAdvec = u_ek * dTdx + v_ek * dTdy (same convention)
    # Temperature gradients with one-sided stencil at coastlines
    m_e = jnp.roll(mask, -1, axis=1)
    m_w = jnp.roll(mask,  1, axis=1)
    m_n = jnp.concatenate([mask[1:, :], mask[-1:, :]], axis=0)
    m_s_nb = jnp.concatenate([mask[:1, :], mask[:-1, :]], axis=0)

    T_e = jnp.where(m_e, jnp.roll(state.T_s, -1, axis=1), state.T_s)
    T_w = jnp.where(m_w, jnp.roll(state.T_s,  1, axis=1), state.T_s)
    T_n = jnp.where(m_n,
        jnp.concatenate([state.T_s[1:, :], state.T_s[-1:, :]], axis=0), state.T_s)
    T_s_nb = jnp.where(m_s_nb,
        jnp.concatenate([state.T_s[:1, :], state.T_s[:-1, :]], axis=0), state.T_s)

    # Ekman velocities are nondimensional (grid-unit RMS~0.3), so
    # temperature gradients should also be in grid units (no invDx scaling)
    dTdx = (T_e - T_w) * 0.5
    dTdy = (T_n - T_s_nb) * 0.5

    u_ek = forcing.ekman_u * params.wind_strength
    v_ek = forcing.ekman_v * params.wind_strength
    ekman_T = u_ek * dTdx + v_ek * dTdy

    # Ekman salinity advection
    S_e = jnp.where(m_e, jnp.roll(state.S_s, -1, axis=1), state.S_s)
    S_w = jnp.where(m_w, jnp.roll(state.S_s,  1, axis=1), state.S_s)
    S_n = jnp.where(m_n,
        jnp.concatenate([state.S_s[1:, :], state.S_s[-1:, :]], axis=0), state.S_s)
    S_s_nb = jnp.where(m_s_nb,
        jnp.concatenate([state.S_s[:1, :], state.S_s[:-1, :]], axis=0), state.S_s)
    dSdx = (S_e - S_w) * 0.5
    dSdy = (S_n - S_s_nb) * 0.5
    ekman_S = u_ek * dSdx + v_ek * dSdy

    # --- Diffusion ---
    diff_T_s = params.kappa_T * masked_laplacian(state.T_s, mask, grid)
    diff_S_s = params.kappa_S * masked_laplacian(state.S_s, mask, grid)
    diff_T_d = params.kappa_deep_T * masked_laplacian(state.T_d, mask, grid)
    diff_S_d = params.kappa_deep_S * masked_laplacian(state.S_d, mask, grid)

    # --- Salinity restoring ---
    sal_restore = params.sal_restoring_rate * (forcing.sal_climatology - state.S_s)

    # --- Freshwater forcing (high-latitude hosing) ---
    # Use latitude as proxy for y_frac: lat ranges from ~-80 to ~80
    # y_frac > 0.75 corresponds to lat > ~40N in the original grid
    lat_b = grid.lat[:, None]
    # Map lat to [0,1] range: y_frac = (lat - lat_min) / (lat_max - lat_min)
    y_frac = (lat_b - grid.lat0) / (grid.lat1 - grid.lat0)
    fw_sal = jnp.where(y_frac > 0.75,
                        -params.freshwater_forcing * 3.0 * (y_frac - 0.75) * 4.0,
                        0.0)

    # --- Variable mixed layer depth ---
    mld = mixed_layer_depth(grid.lat)[:, None]  # (ny, 1)
    local_depth = forcing.depth_field
    h_surf = jnp.minimum(mld, local_depth)
    h_deep = jnp.maximum(1.0, local_depth - mld)
    has_deep = (local_depth > mld).astype(jnp.float32)

    # --- Density-based vertical exchange ---
    rho_surf = -params.alpha_T * state.T_s + params.beta_S * state.S_s
    rho_deep = -params.alpha_T * state.T_d + params.beta_S * state.S_d
    abs_lat = jnp.abs(grid.lat)[:, None]
    gamma = jnp.where(
        (abs_lat > 40.0) & (rho_surf > rho_deep),
        params.gamma_deep_form,
        params.gamma_mix,
    )

    vert_T = gamma * (state.T_s - state.T_d) * has_deep
    vert_S = gamma * (state.S_s - state.S_d) * has_deep

    # --- Land-ocean heat exchange ---
    # Count land neighbors
    m_e = jnp.roll(mask, -1, axis=1)
    m_w = jnp.roll(mask,  1, axis=1)
    m_n = jnp.concatenate([mask[1:, :], mask[-1:, :]], axis=0)
    m_s_mask = jnp.concatenate([mask[:1, :], mask[:-1, :]], axis=0)
    n_land = (1 - m_e) + (1 - m_w) + (1 - m_n) + (1 - m_s_mask)
    n_ocean = 4.0 - n_land
    land_flux = jnp.where(
        (n_land > 0) & (n_ocean > 0),
        jnp.clip(0.002 * (forcing.land_temp - state.T_s) * (n_ocean / 4.0), -0.1, 0.1),
        0.0,
    )

    # --- Haney restoring toward observed climatology ---
    restore_T_s = (forcing.T_target - state.T_s) / params.tau_T
    restore_T_d = (forcing.T_deep_target - state.T_d) / params.tau_deep_T

    # --- Assemble ---
    rhs_T_s = (-advect_T_s - ekman_T + q_net + diff_T_s + land_flux
               + restore_T_s
               - vert_T / jnp.maximum(h_surf, 1.0))
    rhs_S_s = (-advect_S_s - ekman_S + diff_S_s + sal_restore + fw_sal
               - vert_S / jnp.maximum(h_surf, 1.0))
    rhs_T_d = (diff_T_d + restore_T_d
               + vert_T / jnp.maximum(h_deep, 1.0)) * has_deep
    rhs_S_d = (diff_S_d + vert_S / jnp.maximum(h_deep, 1.0)) * has_deep

    return rhs_T_s * mask, rhs_S_s * mask, rhs_T_d * mask, rhs_S_d * mask
