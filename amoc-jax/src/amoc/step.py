"""Time stepping for the full coupled ocean-atmosphere model.

Forward Euler for simplicity (RK2 available but atmosphere coupling
makes it tricky). One step = one dt of ocean + atmosphere.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .atmosphere import atmosphere_step, cloud_fraction, radiation
from .grid import Grid
from .physics import tracer_rhs, vorticity_rhs
from .poisson import poisson_solve
from .state import Forcing, Params, SeasonalForcing, State, interpolate_month


def _resolve(zeta: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    return poisson_solve(zeta, dx, dy)


@jax.jit
def step(
    state: State, forcing: Forcing, params: Params, grid: Grid
) -> State:
    """One forward-Euler timestep of the full coupled model."""
    dt = params.dt
    mask = forcing.ocean_mask
    lat = grid.lat  # (ny,)

    # --- 1. Radiation (needs clouds which need SST) ---
    model_cloud, conv_frac = cloud_fraction(state.T_s, lat, state.sim_time)
    # Blend model clouds with observed (50/50) — nudge toward reality
    obs_cloud = forcing.obs_cloud
    cloud_frac = 0.5 * model_cloud + 0.5 * obs_cloud

    q_net = radiation(
        state.T_s, lat, state.sim_time, cloud_frac, conv_frac,
        state.moisture,
        S_solar=params.S_solar,
        A_olr=params.A_olr,
        B_olr=params.B_olr,
        global_temp_offset=params.global_temp_offset,
        greenhouse_q=params.greenhouse_q,
        q_ref=params.q_ref,
        obs_albedo=forcing.obs_albedo,
        obs_sea_ice=state.ice_frac,  # use prognostic ice, not static obs
    )

    # --- 2. Ocean vorticity ---
    dzeta_s, dzeta_d = vorticity_rhs(state, forcing, params, grid)

    # --- 3. Ocean tracers ---
    dT_s, dS_s, dT_d, dS_d = tracer_rhs(state, forcing, params, grid, q_net)

    # --- 4. Apply ocean updates (with stability clamping) ---
    new_zeta_s = jnp.clip(state.zeta_s + dt * dzeta_s, -500.0, 500.0)
    new_zeta_d = jnp.clip(state.zeta_d + dt * dzeta_d, -500.0, 500.0)
    new_T_s = jnp.clip(state.T_s + dt * dT_s, -10.0, 40.0)
    new_S_s = jnp.clip(state.S_s + dt * dS_s, 28.0, 40.0)
    new_T_d = jnp.clip(state.T_d + dt * dT_d, -5.0, 30.0)
    new_S_d = jnp.clip(state.S_d + dt * dS_d, 33.0, 37.0)

    # --- 5. Atmosphere step ---
    (new_air_temp, new_moisture, precip,
     ocean_T_feedback, sal_feedback) = atmosphere_step(
        state.air_temp, state.moisture, new_T_s, mask,
        forcing.land_temp, lat, state.sim_time,
        dt=dt,
        kappa_atm=params.kappa_atm,
        gamma_oa=params.gamma_oa,
        gamma_la=params.gamma_la,
        gamma_ao=params.gamma_ao,
        E0=params.E0,
        greenhouse_q=params.greenhouse_q,
        q_ref=params.q_ref,
        freshwater_scale_pe=params.freshwater_scale_pe,
        latent_heat_coeff=params.latent_heat_coeff,
    )

    # Apply atmosphere feedback on ocean
    new_T_s = jnp.clip(new_T_s + ocean_T_feedback * mask, -10.0, 40.0)
    new_S_s = jnp.clip(new_S_s + sal_feedback * mask, 28.0, 40.0)

    # --- 5b. Sea ice thermodynamics ---
    old_ice = state.ice_frac
    T_freeze = params.ice_freeze_T

    # Ice grows where SST < freezing and there's open water
    freeze_tendency = params.ice_grow_rate * jnp.maximum(0.0, T_freeze - new_T_s) * (1.0 - old_ice)
    # Ice melts where SST > freezing and there's ice
    melt_tendency = params.ice_melt_rate * jnp.maximum(0.0, new_T_s - T_freeze) * old_ice

    new_ice = jnp.clip(old_ice + dt * (freeze_tendency - melt_tendency), 0.0, 1.0)
    new_ice = new_ice * mask  # no ice on land

    # Ice formation clamps SST to freezing point (latent heat)
    new_T_s = jnp.where(
        (new_ice > old_ice) & (mask > 0.5),
        jnp.maximum(new_T_s, T_freeze),
        new_T_s,
    )

    # Brine rejection: ice formation expels salt, ice melt freshens
    ice_change = new_ice - old_ice
    sal_ice_flux = dt * params.ice_sal_flux * ice_change
    new_S_s = jnp.clip(new_S_s + sal_ice_flux * mask, 28.0, 40.0)

    # --- 6. Poisson solve + clamp ---
    new_psi_s = jnp.clip(_resolve(new_zeta_s, grid.dx_nd, grid.dy_nd), -50.0, 50.0)
    new_psi_d = jnp.clip(_resolve(new_zeta_d, grid.dx_nd, grid.dy_nd), -50.0, 50.0)

    # --- 7. Zero land ---
    new_zeta_s = new_zeta_s * mask
    new_zeta_d = new_zeta_d * mask
    new_T_s = new_T_s * mask
    new_S_s = jnp.where(mask > 0.5, new_S_s, 0.0)
    new_T_d = new_T_d * mask
    new_S_d = jnp.where(mask > 0.5, new_S_d, 0.0)

    # --- 8. Advance time ---
    new_sim_time = state.sim_time + dt * params.year_speed

    return State(
        psi_s=new_psi_s, zeta_s=new_zeta_s,
        psi_d=new_psi_d, zeta_d=new_zeta_d,
        T_s=new_T_s, S_s=new_S_s,
        T_d=new_T_d, S_d=new_S_d,
        air_temp=new_air_temp, moisture=new_moisture,
        ice_frac=new_ice,
        sim_time=new_sim_time,
    )


@partial(jax.jit, static_argnames=("n_steps",))
def run(
    state: State, forcing: Forcing, params: Params, grid: Grid, n_steps: int,
) -> State:
    def body(s, _):
        return step(s, forcing, params, grid), None
    final, _ = jax.lax.scan(body, state, xs=None, length=n_steps)
    return final


@partial(jax.jit, static_argnames=("n_steps", "save_every"))
def run_with_history(
    state: State, forcing: Forcing, params: Params, grid: Grid,
    n_steps: int, save_every: int,
) -> tuple[State, State]:
    n_saves = n_steps // save_every

    def body(carry, _):
        def inner(ss, __):
            return step(ss, forcing, params, grid), None
        s_out, _ = jax.lax.scan(inner, carry, xs=None, length=save_every)
        return s_out, s_out

    final, snaps = jax.lax.scan(body, state, xs=None, length=n_saves)
    return final, snaps


# ---------------------------------------------------------------------------
# Seasonal variants: time-varying SST target and albedo from monthly data
# ---------------------------------------------------------------------------

@jax.jit
def seasonal_step(
    state: State, forcing: Forcing, seasonal: SeasonalForcing,
    params: Params, grid: Grid,
) -> State:
    """One step with seasonally-varying SST target and albedo."""
    T_YEAR = params.T_YEAR
    t = state.sim_time

    # Interpolate monthly fields to current time
    sst_target_now = interpolate_month(seasonal.sst_monthly, t, T_YEAR)
    albedo_now = interpolate_month(seasonal.albedo_monthly, t, T_YEAR)

    # Override forcing fields with seasonal values (where monthly data exists)
    mask = forcing.ocean_mask
    # Use seasonal SST target where the monthly data is nonzero
    has_seasonal_sst = jnp.any(seasonal.sst_monthly[0] != 0)
    T_target = jnp.where(has_seasonal_sst, sst_target_now, forcing.T_target)
    T_target = jnp.where(mask > 0.5, T_target, 0.0)

    has_seasonal_albedo = jnp.any(seasonal.albedo_monthly[0] != 0)
    obs_albedo = jnp.where(has_seasonal_albedo, jnp.clip(albedo_now, 0.0, 1.0), forcing.obs_albedo)

    # Build a modified forcing with seasonal overrides
    forcing_now = forcing._replace(
        T_target=T_target,
        obs_albedo=obs_albedo,
    )

    return step(state, forcing_now, params, grid)


@partial(jax.jit, static_argnames=("n_steps",))
def seasonal_run(
    state: State, forcing: Forcing, seasonal: SeasonalForcing,
    params: Params, grid: Grid, n_steps: int,
) -> State:
    """Run n_steps with seasonal forcing."""
    def body(s, _):
        return seasonal_step(s, forcing, seasonal, params, grid), None
    final, _ = jax.lax.scan(body, state, xs=None, length=n_steps)
    return final
