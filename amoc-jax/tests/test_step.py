"""Two-layer time stepper: zero state stays zero; friction-only decays both layers."""
import jax.numpy as jnp
import numpy as np

from amoc.grid import Grid
from amoc.state import Forcing, Params, State
from amoc.step import rk2_step, run


def _zeros(g):
    return State(
        psi_s=jnp.zeros(g.shape), zeta_s=jnp.zeros(g.shape),
        psi_d=jnp.zeros(g.shape), zeta_d=jnp.zeros(g.shape),
    )


def _zero_forcing(g):
    return Forcing(
        wind_curl=jnp.zeros(g.shape),
        ocean_mask=jnp.ones(g.shape),
        buoyancy=jnp.zeros(g.shape),
    )


def test_zero_in_zero_out():
    g = Grid.create(nx=32, ny=16)
    new = rk2_step(_zeros(g), _zero_forcing(g), Params(), g)
    for f in (new.psi_s, new.zeta_s, new.psi_d, new.zeta_d):
        assert jnp.all(f == 0)


def test_friction_only_both_layers_decay():
    """Pure friction: ζ → ζ·(1-r·dt)^N. With small smooth ICs the advective
    correction stays tiny so decay still tracks 1/r time-scale."""
    g = Grid.create(nx=32, ny=16)
    # Use a smooth, low-amplitude IC so that advection (which scales like
    # |ψ|·|ζ|) is small even after the Poisson solve fills in psi.
    j = jnp.arange(16)[:, None]
    i = jnp.arange(32)[None, :]
    z_s = 0.1 * jnp.cos(jnp.pi * (j - 8) / 16) * jnp.cos(2 * jnp.pi * i / 32)
    z_d = 0.1 * jnp.cos(jnp.pi * (j - 8) / 16) * jnp.cos(4 * jnp.pi * i / 32)
    state = State(
        psi_s=jnp.zeros(g.shape), zeta_s=z_s,
        psi_d=jnp.zeros(g.shape), zeta_d=z_d,
    )
    params = Params(
        r_friction_s=0.1, A_visc_s=0.0,
        r_friction_d=0.1, A_visc_d=0.0,
        beta=0.0, wind_strength=0.0, dt=0.05,
        F_couple_s=0.0, F_couple_d=0.0, alpha_buoy=0.0,
    )
    out = run(state, _zero_forcing(g), params, g, n_steps=2000)
    # 2000 steps × dt 0.05 × r 0.1 = 10 friction times → ratio ≈ e^-10 ≈ 5e-5
    decay_s = float(jnp.max(jnp.abs(out.zeta_s))) / float(jnp.max(jnp.abs(z_s)))
    decay_d = float(jnp.max(jnp.abs(out.zeta_d))) / float(jnp.max(jnp.abs(z_d)))
    assert decay_s < 1e-2
    assert decay_d < 1e-2


def test_run_jit_compiles_and_returns_state():
    g = Grid.create(nx=64, ny=32)
    forcing = Forcing(
        wind_curl=jnp.ones(g.shape) * 0.02,
        ocean_mask=jnp.ones(g.shape),
        buoyancy=jnp.zeros(g.shape),
    )
    # Mild defaults that we know are stable from v1a tuning.
    params = Params(dt=0.01, beta=2.0, wind_strength=1.0, alpha_buoy=0.0)
    out = run(_zeros(g), forcing, params, g, n_steps=50)
    assert out.zeta_s.shape == g.shape
    assert jnp.all(jnp.isfinite(out.zeta_s))
    assert float(jnp.max(jnp.abs(out.zeta_s))) > 0.0
