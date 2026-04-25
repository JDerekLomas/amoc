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
from .state import Forcing, Params, State


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
    cloud_frac, conv_frac = cloud_fraction(state.T_s, lat, state.sim_time)
    q_net = radiation(
        state.T_s, lat, state.sim_time, cloud_frac, conv_frac,
        state.moisture,
        S_solar=params.S_solar,
        A_olr=params.A_olr,
        B_olr=params.B_olr,
        global_temp_offset=params.global_temp_offset,
        greenhouse_q=params.greenhouse_q,
        q_ref=params.q_ref,
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
