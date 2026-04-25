"""Time integration: RK2 over both layers, fully JIT'd via lax.scan."""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .grid import Grid
from .physics import vorticity_rhs
from .poisson import poisson_solve
from .state import Forcing, Params, State


def _resolve(zeta: jnp.ndarray) -> jnp.ndarray:
    return poisson_solve(zeta)


def _state_step(state: State, dzs: jnp.ndarray, dzd: jnp.ndarray) -> State:
    """Apply tendencies to vorticities; resolve psi from each updated zeta."""
    new_zeta_s = state.zeta_s + dzs
    new_zeta_d = state.zeta_d + dzd
    return State(
        psi_s=_resolve(new_zeta_s),
        zeta_s=new_zeta_s,
        psi_d=_resolve(new_zeta_d),
        zeta_d=new_zeta_d,
    )


@jax.jit
def rk2_step(
    state: State, forcing: Forcing, params: Params, grid: Grid
) -> State:
    """Heun's method (RK2) over both layers."""
    dt = params.dt
    rhs1_s, rhs1_d = vorticity_rhs(state, forcing, params, grid)
    state_predict = _state_step(state, dt * rhs1_s, dt * rhs1_d)
    rhs2_s, rhs2_d = vorticity_rhs(state_predict, forcing, params, grid)
    return _state_step(state, 0.5 * dt * (rhs1_s + rhs2_s), 0.5 * dt * (rhs1_d + rhs2_d))


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
