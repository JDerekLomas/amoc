"""Data loader: read existing JSON fields, bilinear-resample to model grid."""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from amoc.data import load_field, resample_to_grid
from amoc.grid import Grid

REPO = Path(__file__).resolve().parents[2]


def test_load_returns_2d_array_at_source_resolution():
    f = load_field(REPO / "wind_stress_1deg.json", "tau_x")
    assert f.shape == (160, 360)
    # tau_x in Pa is order 0.01 to 0.2 in absolute value at most points.
    assert float(jnp.max(jnp.abs(f))) < 1.0


def test_resample_to_same_grid_is_identity():
    """1deg source onto 1deg model grid must round-trip via interp."""
    src = jnp.asarray(np.linspace(-1, 1, 160 * 360).reshape(160, 360))
    g = Grid(nx=360, ny=160, lat0=-80.0, lat1=80.0)  # cell-edge bounds
    src_lat = np.linspace(-79.5, 79.5, 160)  # cell centers (1deg convention)
    src_lon = np.linspace(-179.5, 179.5, 360)
    out = resample_to_grid(src, src_lat=src_lat, src_lon=src_lon, grid=g)
    # Centers of g.lat are at -79.5, -78.5, ... matching src_lat exactly.
    np.testing.assert_allclose(np.asarray(g.lat), src_lat, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out), np.asarray(src), atol=1e-6)


def test_resample_periodic_x_wraps():
    """A field that is 0 except at lon=180 must interpolate correctly past the seam."""
    nx_src, ny_src = 8, 4
    src = np.zeros((ny_src, nx_src))
    src[:, 0] = 1.0  # spike at lon=-157.5 (cell 0 center)
    src[:, -1] = 1.0  # spike at lon=+157.5 (cell -1)
    src_lon = np.linspace(-157.5, 157.5, nx_src)
    src_lat = np.linspace(-30.0, 30.0, ny_src)
    g = Grid(nx=16, ny=4, lat0=-40.0, lat1=40.0)  # dlat=20, centers at -30,-10,10,30
    out = resample_to_grid(jnp.asarray(src), src_lat=src_lat, src_lon=src_lon, grid=g)
    # No NaNs, output finite.
    assert jnp.all(jnp.isfinite(out))
    # The wrap-around region (around lon=±180) should not be all zero —
    # it must blend between the two spikes.
    assert float(out.max()) > 0.4


def test_resample_upscale_smooth():
    """Upsampling 1deg -> 0.5deg should produce a finite, smooth field."""
    src = np.cos(np.deg2rad(np.linspace(-79.5, 79.5, 160)))[:, None] * \
          np.ones(360)[None, :]
    src_lat = np.linspace(-79.5, 79.5, 160)
    src_lon = np.linspace(-179.5, 179.5, 360)
    g = Grid(nx=720, ny=320, lat0=-80.0, lat1=80.0)
    out = resample_to_grid(jnp.asarray(src), src_lat=src_lat, src_lon=src_lon, grid=g)
    assert out.shape == g.shape
    assert jnp.all(jnp.isfinite(out))
    # Result should still resemble cos(lat).
    expected = jnp.cos(jnp.deg2rad(g.lat))[:, None] * jnp.ones(g.nx)[None, :]
    # Skip rows that fall outside source range (target extends past source centers).
    interior = slice(2, -2)
    np.testing.assert_allclose(
        np.asarray(out[interior]), np.asarray(expected[interior]), atol=2e-3
    )
