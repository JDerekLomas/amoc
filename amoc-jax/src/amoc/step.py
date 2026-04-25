"""RK2 over the full v1c state: 4 dynamical fields + 3 tracer fields."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .grid import Grid
from .physics import tracer_rhs, vorticity_rhs
from .poisson import poisson_solve
from .state import Forcing, Params, State


def _resolve(zeta: jnp.ndarray) -> jnp.ndarray:
    return poisson_solve(zeta)


def _compute_rhs(state: State, forcing: Forcing, params: Params, grid: Grid):
    """Return dζ_s, dζ_d, dT_s, dS_s, dT_d for the full state."""
    dzs, dzd = vorticity_rhs(state, forcing, params, grid)
    dts, dss, dtd = tracer_rhs(state, forcing, params, grid)
    return dzs, dzd, dts, dss, dtd


def _apply(state: State, dt: float,
           dzs, dzd, dts, dss, dtd) -> State:
    """Update all prognostic fields by dt and re-solve psi from new zeta."""
    new_zeta_s = state.zeta_s + dt * dzs
    new_zeta_d = state.zeta_d + dt * dzd
    return State(
        psi_s=_resolve(new_zeta_s), zeta_s=new_zeta_s,
        psi_d=_resolve(new_zeta_d), zeta_d=new_zeta_d,
        T_s=state.T_s + dt * dts,
        S_s=state.S_s + dt * dss,
        T_d=state.T_d + dt * dtd,
    )


@jax.jit
def rk2_step(
    state: State, forcing: Forcing, params: Params, grid: Grid
) -> State:
    """Heun's method (RK2) over the full state."""
    dt = params.dt
    k1 = _compute_rhs(state, forcing, params, grid)
    state_predict = _apply(state, dt, *k1)
    k2 = _compute_rhs(state_predict, forcing, params, grid)
    avg = tuple(0.5 * (a + b) for a, b in zip(k1, k2))
    return _apply(state, dt, *avg)


@partial(jax.jit, static_argnames=("n_steps",))
def run(
    state: State, forcing: Forcing, params: Params, grid: Grid, n_steps: int,
) -> State:
    def body(s, _):
        return rk2_step(s, forcing, params, grid), None
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
            return rk2_step(ss, forcing, params, grid), None
        s_out, _ = jax.lax.scan(inner, carry, xs=None, length=save_every)
        return s_out, s_out

    final, snaps = jax.lax.scan(body, state, xs=None, length=n_saves)
    return final, snaps
