"""Data loader: resample to model grid, load binary fields."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from amoc.data import resample_to_grid, load_bin_field
from amoc.grid import Grid

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"


def test_resample_to_same_grid_is_identity():
    """1deg source onto 1deg model grid must round-trip via interp."""
    src = jnp.asarray(np.linspace(-1, 1, 160 * 360).reshape(160, 360))
    g = Grid.create(nx=360, ny=160, lat0=-80.0, lat1=80.0)
    src_lat = np.linspace(-79.5, 79.5, 160)
    src_lon = np.linspace(-179.5, 179.5, 360)
    out = resample_to_grid(src, src_lat=src_lat, src_lon=src_lon, grid=g)
    np.testing.assert_allclose(np.asarray(g.lat), src_lat, atol=1e-6)
    np.testing.assert_allclose(np.asarray(out), np.asarray(src), atol=1e-6)


def test_resample_periodic_x_wraps():
    """A field that is 0 except at lon extremes must interpolate correctly past the seam."""
    nx_src, ny_src = 8, 4
    src = np.zeros((ny_src, nx_src))
    src[:, 0] = 1.0
    src[:, -1] = 1.0
    src_lon = np.linspace(-157.5, 157.5, nx_src)
    src_lat = np.linspace(-30.0, 30.0, ny_src)
    g = Grid.create(nx=16, ny=4, lat0=-40.0, lat1=40.0)
    out = resample_to_grid(jnp.asarray(src), src_lat=src_lat, src_lon=src_lon, grid=g)
    assert jnp.all(jnp.isfinite(out))
    assert float(out.max()) > 0.4


def test_resample_upscale_smooth():
    """Upsampling 1deg -> 0.5deg should produce a finite, smooth field."""
    src = np.cos(np.deg2rad(np.linspace(-79.5, 79.5, 160)))[:, None] * \
          np.ones(360)[None, :]
    src_lat = np.linspace(-79.5, 79.5, 160)
    src_lon = np.linspace(-179.5, 179.5, 360)
    g = Grid.create(nx=720, ny=320, lat0=-80.0, lat1=80.0)
    out = resample_to_grid(jnp.asarray(src), src_lat=src_lat, src_lon=src_lon, grid=g)
    assert out.shape == g.shape
    assert jnp.all(jnp.isfinite(out))
    expected = jnp.cos(jnp.deg2rad(g.lat))[:, None] * jnp.ones(g.nx)[None, :]
    interior = slice(2, -2)
    np.testing.assert_allclose(
        np.asarray(out[interior]), np.asarray(expected[interior]), atol=2e-3
    )


@pytest.mark.skipif(not (DATA_DIR / "bin" / "sst.json").exists(),
                     reason="Data directory not available")
def test_load_bin_field_sst():
    """Load SST from binary format and check shape/range."""
    arr = load_bin_field(DATA_DIR / "bin", "sst", "sst")
    assert arr is not None
    assert arr.shape == (512, 1024)
    assert arr.dtype == np.float32
    assert float(arr[arr != 0].min()) > -5.0
    assert float(arr.max()) < 35.0
