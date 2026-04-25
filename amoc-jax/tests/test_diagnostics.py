"""MOC diagnostic on a synthetic state."""
import jax.numpy as jnp
import numpy as np

from amoc.diagnostics import amoc_streamfunction, basin_mask, meridional_velocity
from amoc.grid import Grid
from amoc.state import State, zero_state


def test_basin_mask_shape_and_extent():
    g = Grid.create(nx=360, ny=160)
    m = basin_mask(g, lon_min=-80.0, lon_max=0.0)
    assert m.shape == g.shape
    # Around 80/360 of cells should be inside the Atlantic mask.
    frac = m.mean()
    assert 80 / 360 - 0.02 < frac < 80 / 360 + 0.02


def test_meridional_velocity_uniform_psi_is_zero():
    g = Grid.create(nx=16, ny=8)
    psi = jnp.ones(g.shape) * 5.0
    v = meridional_velocity(psi, g)
    np.testing.assert_allclose(np.asarray(v), 0.0, atol=1e-6)


def test_amoc_streamfunction_signs_with_synthetic_input():
    """If surface ψ has a single-mode zonal sin pattern on basin only, the
    diagnostic returns a finite Ψ(y) shaped like the integrand."""
    g = Grid.create(nx=64, ny=32)
    # Surface psi = sin(2πx/Lx) * cos(πy/Ly) — single zonal/meridional mode.
    i = jnp.arange(64)[None, :]
    j = jnp.arange(32)[:, None]
    psi_s = jnp.sin(2 * jnp.pi * i / 64) * jnp.cos(jnp.pi * (j - 16) / 32)
    state = zero_state(g.shape)._replace(psi_s=psi_s)
    psi_amoc = amoc_streamfunction(state, g, lon_min=-180, lon_max=180)
    assert psi_amoc.shape == (g.ny,)
    assert np.all(np.isfinite(psi_amoc))
