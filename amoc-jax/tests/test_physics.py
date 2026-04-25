"""Two-layer physics: Arakawa Jacobian properties + RHS sanity checks."""
import jax.numpy as jnp
import numpy as np
import pytest

from amoc.grid import Grid
from amoc.physics import arakawa_jacobian, vorticity_rhs, tracer_rhs
from amoc.state import Params, trivial_forcing, zero_state


def _rand(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def _quiet_params(**kw) -> Params:
    """Params with thermodynamics disabled for isolated dynamics tests."""
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


def test_jacobian_self_is_zero():
    g = Grid.create(nx=32, ny=16)
    a = _rand(g.shape, 0)
    j = arakawa_jacobian(a, a, g)
    assert float(jnp.max(jnp.abs(j))) < 1e-3  # cos(lat) metric introduces small residual


def test_jacobian_antisymmetric():
    g = Grid.create(nx=32, ny=16)
    a = _rand(g.shape, 1)
    b = _rand(g.shape, 2)
    np.testing.assert_allclose(
        np.asarray(arakawa_jacobian(a, b, g)),
        -np.asarray(arakawa_jacobian(b, a, g)),
        atol=2e-4,  # cos(lat) metric introduces float32 rounding
    )


def test_jacobian_zero_on_zeros():
    g = Grid.create(nx=16, ny=8)
    j = arakawa_jacobian(jnp.zeros(g.shape), _rand(g.shape, 5), g)
    assert float(jnp.max(jnp.abs(j))) < 1e-6


def test_rhs_zero_state_uniform_wind():
    g = Grid.create(nx=32, ny=16)
    state = zero_state(g.shape)
    forcing = trivial_forcing(g.shape, wind=0.1)
    params = _quiet_params(F_couple_s=0.0, F_couple_d=0.0, beta=0.0)
    rhs_s, rhs_d = vorticity_rhs(state, forcing, params, g)
    # With zero state and uniform wind, rhs_s should equal wind_strength * wind_curl
    # Default wind_strength = 1.0, so rhs_s ≈ 0.1
    np.testing.assert_allclose(np.asarray(rhs_s), 0.1, atol=1e-5)
    np.testing.assert_allclose(np.asarray(rhs_d), 0.0, atol=1e-6)


def test_tracer_rhs_restoring_drives_to_target():
    """Restoring should drive T_s toward T_target on timescale tau_T."""
    g = Grid.create(nx=16, ny=8)
    state = zero_state(g.shape)._replace(T_s=jnp.full(g.shape, 5.0))
    forcing = trivial_forcing(g.shape)._replace(T_target=jnp.full(g.shape, 25.0))
    params = _quiet_params(
        tau_T=1.0, wind_strength=0.0, beta=0.0,
        F_couple_s=0.0, F_couple_d=0.0,
    )
    # tracer_rhs returns (dT_s, dS_s, dT_d, dS_d) and needs q_net
    q_net = jnp.zeros(g.shape)
    dT_s, _, _, _ = tracer_rhs(state, forcing, params, g, q_net)
    # The restoring term = (T_target - T_s) / tau_T = (25 - 5) / 1 = 20
    # Other terms should be ~0 with everything disabled
    mean_dT = float(jnp.mean(dT_s))
    assert mean_dT > 15.0  # restoring dominates


def test_rhs_mask_zeros_land():
    g = Grid.create(nx=16, ny=8)
    state = zero_state(g.shape)
    f = trivial_forcing(g.shape, wind=1.0)._replace(
        ocean_mask=jnp.zeros(g.shape).at[3:5, :].set(1.0),
    )
    params = _quiet_params(F_couple_s=0.0, F_couple_d=0.0, beta=0.0)
    rhs_s, _ = vorticity_rhs(state, f, params, g)
    np.testing.assert_array_equal(np.asarray(rhs_s)[f.ocean_mask == 0], 0.0)
