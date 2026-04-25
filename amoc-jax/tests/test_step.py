"""Time stepper: a single RK2 step is consistent; many steps are bounded."""
import jax.numpy as jnp
import numpy as np
import pytest

from amoc.grid import Grid
from amoc.state import Forcing, Params, State
from amoc.step import rk2_step, run


def test_one_rk2_step_no_forcing_no_change_on_zero_state():
    g = Grid.create(nx=32, ny=16)
    state = State(psi=jnp.zeros(g.shape), zeta=jnp.zeros(g.shape))
    forcing = Forcing(wind_curl=jnp.zeros(g.shape), ocean_mask=jnp.ones(g.shape))
    params = Params()
    new = rk2_step(state, forcing, params, g)
    assert jnp.all(new.zeta == 0)
    assert jnp.all(new.psi == 0)


def test_friction_only_decays_to_zero():
    """Pure linear friction: ζ(t) = ζ(0) e^{-rt}. After many steps ζ → 0."""
    g = Grid.create(nx=32, ny=16)
    rng = np.random.default_rng(123)
    zeta0 = jnp.asarray(rng.standard_normal(g.shape).astype(np.float32))
    # Make zeta consistent with Dirichlet ψ via Poisson; but for friction-only
    # test we don't need that — psi will be zeroed out anyway.
    state = State(psi=jnp.zeros(g.shape), zeta=zeta0)
    forcing = Forcing(wind_curl=jnp.zeros(g.shape), ocean_mask=jnp.ones(g.shape))
    # No advection (psi stays small), no beta, no viscosity, only friction.
    params = Params(r_friction=0.1, A_visc=0.0, beta=0.0, wind_strength=0.0, dt=0.5)
    out = run(state, forcing, params, g, n_steps=200)
    assert float(jnp.max(jnp.abs(out.zeta))) < float(jnp.max(jnp.abs(zeta0))) * 1e-3


def test_run_jit_compiles_and_returns_state():
    g = Grid.create(nx=64, ny=32)
    state = State(psi=jnp.zeros(g.shape), zeta=jnp.zeros(g.shape))
    forcing = Forcing(wind_curl=jnp.ones(g.shape) * 0.05, ocean_mask=jnp.ones(g.shape))
    params = Params(dt=0.1)
    out = run(state, forcing, params, g, n_steps=50)
    assert out.zeta.shape == g.shape
    assert jnp.all(jnp.isfinite(out.zeta))
    # Forcing was nonzero for 50 steps → zeta should have grown above zero.
    assert float(jnp.max(jnp.abs(out.zeta))) > 0.0
