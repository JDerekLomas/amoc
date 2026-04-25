"""Grid: spherical lat-lon mesh with metric factors, Coriolis, beta."""
import jax.numpy as jnp
import numpy as np
import pytest

from amoc.grid import Grid


def test_grid_shape_and_extent():
    g = Grid(nx=360, ny=160, lat0=-79.5, lat1=79.5)
    assert g.nx == 360 and g.ny == 160
    assert g.lon.shape == (360,)
    assert g.lat.shape == (160,)
    # Cell-centered: lat values lie strictly inside (lat0, lat1).
    assert g.lat[0] > -79.5 and g.lat[-1] < 79.5
    # Periodic longitude spans 360 degrees.
    np.testing.assert_allclose(g.lon[-1] - g.lon[0], 360 - g.dlon, rtol=1e-6)


def test_metric_cos_lat():
    g = Grid(nx=4, ny=180, lat0=-89.5, lat1=89.5)
    # cos(lat) is positive everywhere off the poles, max at equator.
    assert jnp.all(g.cos_lat > 0)
    eq_idx = int(jnp.argmin(jnp.abs(g.lat)))
    assert g.cos_lat[eq_idx] == pytest.approx(1.0, abs=1e-3)


def test_coriolis_and_beta():
    g = Grid(nx=8, ny=180, lat0=-89.5, lat1=89.5)
    # f changes sign across the equator.
    assert g.f[0] < 0 and g.f[-1] > 0
    # f at +45° is ~ 2*Omega*sin(45°)
    Omega = 7.2921159e-5
    idx = int(jnp.argmin(jnp.abs(g.lat - 45.0)))
    assert float(g.f[idx]) == pytest.approx(2 * Omega * np.sin(np.deg2rad(45)), rel=1e-2)
    # beta = df/dy is positive everywhere on a sphere (df/dy = 2 Omega cos(lat) / R)
    assert jnp.all(g.beta > 0)


def test_periodic_x_step():
    g = Grid(nx=360, ny=160, lat0=-79.5, lat1=79.5)
    # dx in radians per cell on the unit sphere; convert to meters via R.
    R = 6.371e6
    expected = 2 * np.pi * R / 360
    assert float(g.dx_eq) == pytest.approx(expected, rel=1e-6)
