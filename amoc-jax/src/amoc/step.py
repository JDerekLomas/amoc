"""Time integration: RK2 step + jax.lax.scan over many steps, fully JIT'd.

The streamfunction ψ is diagnosed from ζ via the Poisson solver at each
sub-stage; we never carry stale ψ through a half-step. This makes the step
consistent regardless of how many vorticity terms we add later (T, S
buoyancy, second layer).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .grid import Grid
from .physics import vorticity_rhs
from .poisson import poisson_solve
from .state import Forcing, Params, State


def _resolve_psi(zeta: jnp.ndarray) -> jnp.ndarray:
    return poisson_solve(zeta)


@jax.jit
def rk2_step(
    state: State, forcing: Forcing, params: Params, grid: Grid
) -> State:
    """Heun's method (RK2): predictor + corrector. Re-solves ψ at the half step."""
    dt = params.dt

    rhs1 = vorticity_rhs(state, forcing, params, grid)

    zeta_predict = state.zeta + dt * rhs1
    psi_predict = _resolve_psi(zeta_predict)
    state_predict = State(psi=psi_predict, zeta=zeta_predict)

    rhs2 = vorticity_rhs(state_predict, forcing, params, grid)
    zeta_new = state.zeta + 0.5 * dt * (rhs1 + rhs2)
    psi_new = _resolve_psi(zeta_new)
    return State(psi=psi_new, zeta=zeta_new)


@partial(jax.jit, static_argnames=("n_steps",))
def run(
    state: State,
    forcing: Forcing,
    params: Params,
    grid: Grid,
    n_steps: int,
) -> State:
    """Run n_steps forward, returning the final state. Forcing is constant."""
    def body(s, _):
        return rk2_step(s, forcing, params, grid), None

    final, _ = jax.lax.scan(body, state, xs=None, length=n_steps)
    return final


@partial(jax.jit, static_argnames=("n_steps", "save_every"))
def run_with_history(
    state: State,
    forcing: Forcing,
    params: Params,
    grid: Grid,
    n_steps: int,
    save_every: int,
) -> tuple[State, State]:
    """Run n_steps and also return ψ snapshots every `save_every` steps.

    Returns (final_state, history) where history.psi has shape
    (n_steps // save_every, ny, nx).
    """
    n_saves = n_steps // save_every

    def body(carry, _):
        s = carry
        # Run save_every inner steps in a tight scan.
        def inner(ss, __):
            return rk2_step(ss, forcing, params, grid), None
        s_out, _ = jax.lax.scan(inner, s, xs=None, length=save_every)
        return s_out, s_out

    final, snaps = jax.lax.scan(body, state, xs=None, length=n_saves)
    return final, snaps
