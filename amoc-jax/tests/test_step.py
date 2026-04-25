"""Time stepper: zero state stays zero; friction-only decays both layers."""
import jax.numpy as jnp
import numpy as np

from amoc.grid import Grid
from amoc.state import Params, trivial_forcing, zero_state
from amoc.step import step, run


def _quiet_params(**kw) -> Params:
    """Params with thermodynamics/atmosphere disabled for dynamics tests."""
    base = dict(
        alpha_T=0.0, beta_S=0.0,
        kappa_T=0.0, kappa_S=0.0,
        kappa_deep_T=0.0, kappa_deep_S=0.0,
        tau_T=1e9, tau_deep_T=1e9,
        gamma_mix=0.0, gamma_deep_form=0.0,
        sal_restoring_rate=0.0, mot_strength=0.0,
        gamma_oa=0.0, gamma_la=0.0, gamma_ao=0.0,
        kappa_atm=0.0, E0=0.0, freshwater_scale_pe=0.0,
    )
    base.update(kw)
    return Params(**base)


def test_zero_in_zero_out():
    g = Grid.create(nx=32, ny=16)
    state = zero_state(g.shape)
    new = step(state, trivial_forcing(g.shape), Params(), g)
    for f in (new.psi_s, new.zeta_s, new.psi_d, new.zeta_d):
        assert jnp.all(f == 0)
    # T_s shifts very slightly from radiation/atmosphere even at rest
    np.testing.assert_allclose(np.asarray(new.T_s), np.asarray(state.T_s), atol=1e-3)


def test_friction_only_both_layers_decay():
    g = Grid.create(nx=32, ny=16)
    j = jnp.arange(16)[:, None]
    i = jnp.arange(32)[None, :]
    z_s = 0.1 * jnp.cos(jnp.pi * (j - 8) / 16) * jnp.cos(2 * jnp.pi * i / 32)
    z_d = 0.1 * jnp.cos(jnp.pi * (j - 8) / 16) * jnp.cos(4 * jnp.pi * i / 32)
    state = zero_state(g.shape)._replace(zeta_s=z_s, zeta_d=z_d)
    params = _quiet_params(
        r_friction_s=0.1, A_visc_s=0.0,
        r_friction_d=0.1, A_visc_d=0.0,
        beta=0.0, wind_strength=0.0, dt=0.05,
        F_couple_s=0.0, F_couple_d=0.0,
    )
    out = run(state, trivial_forcing(g.shape), params, g, n_steps=2000)
    decay_s = float(jnp.max(jnp.abs(out.zeta_s))) / float(jnp.max(jnp.abs(z_s)))
    decay_d = float(jnp.max(jnp.abs(out.zeta_d))) / float(jnp.max(jnp.abs(z_d)))
    assert decay_s < 1e-2
    assert decay_d < 1e-2


def test_run_jit_compiles_and_returns_state():
    g = Grid.create(nx=64, ny=32)
    forcing = trivial_forcing(g.shape, wind=0.02)
    params = _quiet_params(
        dt=0.01, beta=2.0, wind_strength=1.0,
        F_couple_s=0.0, F_couple_d=0.0,
    )
    out = run(zero_state(g.shape), forcing, params, g, n_steps=50)
    assert out.zeta_s.shape == g.shape
    assert jnp.all(jnp.isfinite(out.zeta_s))
    assert float(jnp.max(jnp.abs(out.zeta_s))) > 0.0
