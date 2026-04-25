"""Two-layer time stepper: zero state stays zero; friction-only decays both layers (v1c)."""
import jax.numpy as jnp
import numpy as np

from amoc.grid import Grid
from amoc.state import Params, State, trivial_forcing, zero_state
from amoc.step import rk2_step, run


def test_zero_in_zero_out():
    g = Grid.create(nx=32, ny=16)
    state = zero_state(g.shape)
    new = rk2_step(state, trivial_forcing(g.shape), Params(), g)
    for f in (new.psi_s, new.zeta_s, new.psi_d, new.zeta_d):
        assert jnp.all(f == 0)
    # Tracers untouched: they were initialized at the EOS reference and there
    # is no forcing or restoring perturbation.
    np.testing.assert_array_equal(np.asarray(new.T_s), np.asarray(state.T_s))


def test_friction_only_both_layers_decay():
    g = Grid.create(nx=32, ny=16)
    j = jnp.arange(16)[:, None]
    i = jnp.arange(32)[None, :]
    z_s = 0.1 * jnp.cos(jnp.pi * (j - 8) / 16) * jnp.cos(2 * jnp.pi * i / 32)
    z_d = 0.1 * jnp.cos(jnp.pi * (j - 8) / 16) * jnp.cos(4 * jnp.pi * i / 32)
    state = zero_state(g.shape)._replace(zeta_s=z_s, zeta_d=z_d)
    params = Params(
        r_friction_s=0.1, A_visc_s=0.0,
        r_friction_d=0.1, A_visc_d=0.0,
        beta=0.0, wind_strength=0.0, dt=0.05,
        F_couple_s=0.0, F_couple_d=0.0,
        alpha_buoy=0.0, alpha_BC=0.0,
        kappa_T=0.0, kappa_S=0.0, tau_T=1e9, tau_S=1e9,
        gamma_TS=0.0, gamma_conv=0.0,
    )
    out = run(state, trivial_forcing(g.shape), params, g, n_steps=2000)
    decay_s = float(jnp.max(jnp.abs(out.zeta_s))) / float(jnp.max(jnp.abs(z_s)))
    decay_d = float(jnp.max(jnp.abs(out.zeta_d))) / float(jnp.max(jnp.abs(z_d)))
    assert decay_s < 1e-2
    assert decay_d < 1e-2


def test_run_jit_compiles_and_returns_state():
    g = Grid.create(nx=64, ny=32)
    forcing = trivial_forcing(g.shape, wind=0.02)
    params = Params(dt=0.01, beta=2.0, wind_strength=1.0,
                    alpha_buoy=0.0, alpha_BC=0.0,
                    F_couple_s=0.0, F_couple_d=0.0,
                    kappa_T=0.0, kappa_S=0.0, tau_T=1e9, tau_S=1e9,
                    gamma_TS=0.0, gamma_conv=0.0)
    out = run(zero_state(g.shape), forcing, params, g, n_steps=50)
    assert out.zeta_s.shape == g.shape
    assert jnp.all(jnp.isfinite(out.zeta_s))
    assert float(jnp.max(jnp.abs(out.zeta_s))) > 0.0
