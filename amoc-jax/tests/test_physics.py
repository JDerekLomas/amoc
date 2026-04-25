"""Two-layer physics: Arakawa Jacobian properties + RHS sanity checks."""
import jax.numpy as jnp
import numpy as np
import pytest

from amoc.grid import Grid
from amoc.physics import arakawa_jacobian, vorticity_rhs
from amoc.state import Forcing, Params, State


def _rand(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def _zero_state(shape):
    return State(
        psi_s=jnp.zeros(shape), zeta_s=jnp.zeros(shape),
        psi_d=jnp.zeros(shape), zeta_d=jnp.zeros(shape),
    )


def _trivial_forcing(shape, wind=0.0, buoy=0.0):
    arr = lambda v: jnp.full(shape, v)
    return Forcing(
        wind_curl=arr(wind),
        ocean_mask=jnp.ones(shape),
        buoyancy=arr(buoy),
    )


def test_jacobian_self_is_zero():
    g = Grid.create(nx=32, ny=16)
    a = _rand(g.shape, 0)
    j = arakawa_jacobian(a, a, g)
    assert float(jnp.max(jnp.abs(j))) < 1e-5


def test_jacobian_antisymmetric():
    g = Grid.create(nx=32, ny=16)
    a = _rand(g.shape, 1)
    b = _rand(g.shape, 2)
    jab = arakawa_jacobian(a, b, g)
    jba = arakawa_jacobian(b, a, g)
    np.testing.assert_allclose(np.asarray(jab), -np.asarray(jba), atol=1e-6)


def test_jacobian_zero_on_zeros():
    g = Grid.create(nx=16, ny=8)
    z = jnp.zeros(g.shape)
    b = _rand(g.shape, 5)
    j = arakawa_jacobian(z, b, g)
    assert float(jnp.max(jnp.abs(j))) < 1e-6


def test_jacobian_constant_interior():
    g = Grid.create(nx=16, ny=16)
    c = jnp.ones(g.shape) * 2.0
    b = _rand(g.shape, 5)
    j = arakawa_jacobian(c, b, g)
    interior = j[2:-2, :]
    assert float(jnp.max(jnp.abs(interior))) < 1e-6


def test_rhs_zero_state_uniform_wind():
    """With ψ=ζ=0 and uniform wind only on surface, both RHS reduce to forcing."""
    g = Grid.create(nx=32, ny=16)
    state = _zero_state(g.shape)
    forcing = _trivial_forcing(g.shape, wind=0.1)
    params = Params(F_couple_s=0.0, F_couple_d=0.0, alpha_buoy=0.0, beta=0.0)
    rhs_s, rhs_d = vorticity_rhs(state, forcing, params, g)
    np.testing.assert_allclose(np.asarray(rhs_s), 0.1, atol=1e-6)
    np.testing.assert_allclose(np.asarray(rhs_d), 0.0, atol=1e-6)


def test_rhs_friction_only():
    """Friction-only: each layer's RHS = -r ζ on its own ζ."""
    g = Grid.create(nx=16, ny=8)
    zeta_s = _rand(g.shape, 7)
    zeta_d = _rand(g.shape, 8)
    state = State(
        psi_s=jnp.zeros(g.shape), zeta_s=zeta_s,
        psi_d=jnp.zeros(g.shape), zeta_d=zeta_d,
    )
    forcing = _trivial_forcing(g.shape)
    params = Params(
        r_friction_s=0.04, A_visc_s=0.0,
        r_friction_d=0.10, A_visc_d=0.0,
        beta=0.0, wind_strength=0.0,
        F_couple_s=0.0, F_couple_d=0.0, alpha_buoy=0.0,
    )
    rhs_s, rhs_d = vorticity_rhs(state, forcing, params, g)
    np.testing.assert_allclose(np.asarray(rhs_s), -0.04 * np.asarray(zeta_s), atol=1e-5)
    np.testing.assert_allclose(np.asarray(rhs_d), -0.10 * np.asarray(zeta_d), atol=1e-5)


def test_rhs_buoyancy_opposes_layers():
    """Surface and deep buoyancy tendencies must be exactly opposite-signed
    when only the buoyancy term is active."""
    g = Grid.create(nx=32, ny=16)
    i = jnp.arange(32)[None, :]
    buoy = jnp.sin(2 * jnp.pi * i / 32) * jnp.ones((16, 32))
    state = _zero_state(g.shape)
    forcing = Forcing(wind_curl=jnp.zeros(g.shape), ocean_mask=jnp.ones(g.shape), buoyancy=buoy)
    params = Params(
        r_friction_s=0.0, A_visc_s=0.0, r_friction_d=0.0, A_visc_d=0.0,
        beta=0.0, wind_strength=0.0,
        F_couple_s=0.0, F_couple_d=0.0, alpha_buoy=0.1,
    )
    rhs_s, rhs_d = vorticity_rhs(state, forcing, params, g)
    np.testing.assert_allclose(np.asarray(rhs_s), -np.asarray(rhs_d), atol=1e-6)
    assert float(jnp.max(jnp.abs(rhs_s))) > 0.0


def test_rhs_mask_zeros_land():
    g = Grid.create(nx=16, ny=8)
    mask = jnp.zeros(g.shape).at[3:5, :].set(1.0)
    forcing = Forcing(wind_curl=jnp.ones(g.shape), ocean_mask=mask, buoyancy=jnp.zeros(g.shape))
    state = _zero_state(g.shape)
    params = Params(F_couple_s=0.0, F_couple_d=0.0, alpha_buoy=0.0, beta=0.0)
    rhs_s, _ = vorticity_rhs(state, forcing, params, g)
    np.testing.assert_array_equal(np.asarray(rhs_s)[mask == 0], 0.0)
