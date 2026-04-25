"""Physics: Arakawa Jacobian + barotropic vorticity RHS.

Property tests for the Jacobian (anti-symmetry, J(A,A)=0) plus a tendency-shape
sanity check for the RHS.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from amoc.grid import Grid
from amoc.physics import arakawa_jacobian, vorticity_rhs
from amoc.state import Forcing, Params, State


def _rand(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def test_jacobian_self_is_zero():
    """J(A, A) = 0 by definition (a function does not move itself)."""
    g = Grid.create(nx=32, ny=16)
    a = _rand(g.shape, 0)
    j = arakawa_jacobian(a, a, g)
    # Allow for floating-point round-off but very tight.
    assert float(jnp.max(jnp.abs(j))) < 1e-5


def test_jacobian_antisymmetric():
    """J(A, B) = -J(B, A)."""
    g = Grid.create(nx=32, ny=16)
    a = _rand(g.shape, 1)
    b = _rand(g.shape, 2)
    jab = arakawa_jacobian(a, b, g)
    jba = arakawa_jacobian(b, a, g)
    np.testing.assert_allclose(np.asarray(jab), -np.asarray(jba), atol=1e-6)


def test_jacobian_zero_on_zeros():
    """J(0, b) = 0 — no flow, no transport."""
    g = Grid.create(nx=16, ny=8)
    z = jnp.zeros(g.shape)
    b = _rand(g.shape, 5)
    j = arakawa_jacobian(z, b, g)
    assert float(jnp.max(jnp.abs(j))) < 1e-6


def test_jacobian_constant_interior():
    """For a constant ψ, J=0 at *interior* points (rows away from y-boundaries)."""
    g = Grid.create(nx=16, ny=16)
    c = jnp.ones(g.shape) * 2.0
    b = _rand(g.shape, 5)
    j = arakawa_jacobian(c, b, g)
    # Skip 1 row near each y-boundary where the implicit Dirichlet pad bites.
    interior = j[2:-2, :]
    assert float(jnp.max(jnp.abs(interior))) < 1e-6


def test_rhs_shape():
    g = Grid.create(nx=32, ny=16)
    state = State(psi=jnp.zeros(g.shape), zeta=jnp.zeros(g.shape))
    forcing = Forcing(wind_curl=jnp.ones(g.shape) * 0.1, ocean_mask=jnp.ones(g.shape))
    params = Params()
    rhs = vorticity_rhs(state, forcing, params, g)
    assert rhs.shape == g.shape
    # With zeta=psi=0 and uniform forcing, RHS should be ≈ forcing (after
    # multiplication by wind_strength) since friction/viscosity/J/beta all
    # vanish on zero state.
    np.testing.assert_allclose(
        np.asarray(rhs), np.asarray(forcing.wind_curl * params.wind_strength), atol=1e-6
    )


def test_rhs_friction_damps():
    """With only friction active, dζ/dt = -r ζ — sign opposes ζ."""
    g = Grid.create(nx=16, ny=8)
    zeta = _rand(g.shape, 7)
    psi = jnp.zeros(g.shape)
    state = State(psi=psi, zeta=zeta)
    forcing = Forcing(wind_curl=jnp.zeros(g.shape), ocean_mask=jnp.ones(g.shape))
    params = Params(r_friction=0.04, A_visc=0.0, beta=0.0, wind_strength=0.0)
    rhs = vorticity_rhs(state, forcing, params, g)
    # rhs = -r * zeta exactly (no other terms active because psi=0 → J=0, beta=0)
    np.testing.assert_allclose(
        np.asarray(rhs), np.asarray(-params.r_friction * zeta), atol=1e-5
    )
