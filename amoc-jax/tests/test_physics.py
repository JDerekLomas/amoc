"""Two-layer physics: Arakawa Jacobian properties + RHS sanity checks (v1c)."""
import jax.numpy as jnp
import numpy as np
import pytest

from amoc.grid import Grid
from amoc.physics import arakawa_jacobian, vorticity_rhs, tracer_rhs, T0, S0, buoyancy_from_TS
from amoc.state import Forcing, Params, State, trivial_forcing, zero_state


def _rand(shape, seed):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def test_jacobian_self_is_zero():
    g = Grid.create(nx=32, ny=16)
    a = _rand(g.shape, 0)
    j = arakawa_jacobian(a, a, g)
    assert float(jnp.max(jnp.abs(j))) < 1e-5


def test_jacobian_antisymmetric():
    g = Grid.create(nx=32, ny=16)
    a = _rand(g.shape, 1)
    b = _rand(g.shape, 2)
    np.testing.assert_allclose(
        np.asarray(arakawa_jacobian(a, b, g)),
        -np.asarray(arakawa_jacobian(b, a, g)),
        atol=1e-6,
    )


def test_jacobian_zero_on_zeros():
    g = Grid.create(nx=16, ny=8)
    j = arakawa_jacobian(jnp.zeros(g.shape), _rand(g.shape, 5), g)
    assert float(jnp.max(jnp.abs(j))) < 1e-6


def test_rhs_zero_state_uniform_wind():
    g = Grid.create(nx=32, ny=16)
    state = zero_state(g.shape)
    forcing = trivial_forcing(g.shape, wind=0.1)
    params = Params(F_couple_s=0.0, F_couple_d=0.0,
                    alpha_buoy=0.0, alpha_BC=0.0, beta=0.0)
    rhs_s, rhs_d = vorticity_rhs(state, forcing, params, g)
    np.testing.assert_allclose(np.asarray(rhs_s), 0.1, atol=1e-6)
    np.testing.assert_allclose(np.asarray(rhs_d), 0.0, atol=1e-6)


def test_buoyancy_from_TS_sign():
    """Warmer water is more buoyant (positive b); saltier water is less buoyant."""
    params = Params()
    # Warmer than reference, same salinity
    b1 = float(buoyancy_from_TS(jnp.array(20.0), jnp.array(S0), params))
    # Cooler
    b2 = float(buoyancy_from_TS(jnp.array(10.0), jnp.array(S0), params))
    assert b1 > 0 and b2 < 0
    # Saltier than reference, same temperature
    b3 = float(buoyancy_from_TS(jnp.array(T0), jnp.array(36.0), params))
    assert b3 < 0


def test_tracer_rhs_restoring_drives_to_target():
    """In isolation, restoring should drive T toward T_target on time-scale tau_T."""
    g = Grid.create(nx=16, ny=8)
    state = zero_state(g.shape)._replace(T_s=jnp.full(g.shape, 5.0))  # below ref
    forcing = trivial_forcing(g.shape)._replace(T_target=jnp.full(g.shape, 25.0))
    params = Params(
        kappa_T=0.0, kappa_S=0.0, tau_T=1.0, tau_S=1.0,
        gamma_TS=0.0, gamma_conv=0.0,
        F_couple_s=0.0, F_couple_d=0.0,
        alpha_buoy=0.0, alpha_BC=0.0, beta=0.0, wind_strength=0.0,
    )
    dT_s, _, _ = tracer_rhs(state, forcing, params, g)
    expected = (25.0 - 5.0) / 1.0
    np.testing.assert_allclose(np.asarray(dT_s), expected, atol=1e-6)


def test_convective_adjustment_activates_when_surface_denser():
    """When surface T,S make ρ_s > ρ_d, the exchange rate jumps to gamma_conv."""
    g = Grid.create(nx=8, ny=8)
    # Surface very salty + cold => denser than deep at reference T,S.
    state = zero_state(g.shape)._replace(
        T_s=jnp.full(g.shape, 0.0), S_s=jnp.full(g.shape, 38.0),
        T_d=jnp.full(g.shape, T0),
    )
    forcing = trivial_forcing(g.shape)
    params = Params(
        kappa_T=0.0, kappa_S=0.0, tau_T=1e9, tau_S=1e9,
        gamma_TS=0.001, gamma_conv=0.05,
        F_couple_s=0.0, F_couple_d=0.0,
        alpha_buoy=0.0, alpha_BC=0.0, beta=0.0, wind_strength=0.0,
    )
    dT_s, _, _ = tracer_rhs(state, forcing, params, g)
    # Exchange brings T_s up toward T_d. Magnitude should reflect gamma_conv (~0.05),
    # not gamma_TS (~0.001). T_s - T_d = -15. Expected: -gamma_conv * (T_s - T_d) = +0.75.
    np.testing.assert_allclose(np.asarray(dT_s), 0.05 * 15.0, atol=1e-3)


def test_buoyancy_ts_drives_baroclinic_shear():
    """A nonzero T_s gradient (with T_d uniform) should produce equal-and-opposite
    buoyancy tendencies on the two layers, the v1c thermal-wind shear forcing."""
    g = Grid.create(nx=32, ny=16)
    i = jnp.arange(32)[None, :]
    state = zero_state(g.shape)._replace(
        T_s=jnp.broadcast_to(T0 + 5.0 * jnp.sin(2 * jnp.pi * i / 32), g.shape),
    )
    forcing = trivial_forcing(g.shape)
    params = Params(
        F_couple_s=0.0, F_couple_d=0.0,
        beta=0.0, wind_strength=0.0,
        alpha_buoy=0.0, alpha_BC=1.0,
        r_friction_s=0.0, A_visc_s=0.0,
        r_friction_d=0.0, A_visc_d=0.0,
    )
    rhs_s, rhs_d = vorticity_rhs(state, forcing, params, g)
    np.testing.assert_allclose(np.asarray(rhs_s), -np.asarray(rhs_d), atol=1e-6)
    assert float(jnp.max(jnp.abs(rhs_s))) > 0.0


def test_rhs_mask_zeros_land():
    g = Grid.create(nx=16, ny=8)
    state = zero_state(g.shape)
    f = trivial_forcing(g.shape, wind=1.0)._replace(
        ocean_mask=jnp.zeros(g.shape).at[3:5, :].set(1.0),
    )
    params = Params(F_couple_s=0.0, F_couple_d=0.0,
                    alpha_buoy=0.0, alpha_BC=0.0, beta=0.0)
    rhs_s, _ = vorticity_rhs(state, f, params, g)
    np.testing.assert_array_equal(np.asarray(rhs_s)[f.ocean_mask == 0], 0.0)
