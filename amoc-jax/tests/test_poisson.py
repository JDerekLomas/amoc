"""FFT/DST Poisson solver: round-trip nabla^2 -> solve must reproduce the source."""
import jax.numpy as jnp
import numpy as np
import pytest

from amoc.poisson import grid_laplacian, poisson_solve


def _grid_laplacian_ref(psi: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Reference 5-point grid Laplacian: periodic in x, Dirichlet in y (psi=0 outside)."""
    invDx2 = 1.0 / (dx * dx)
    invDy2 = 1.0 / (dy * dy)
    psi_xp = jnp.roll(psi, -1, axis=1)
    psi_xm = jnp.roll(psi, 1, axis=1)
    z = jnp.zeros_like(psi[:1, :])
    psi_yp = jnp.concatenate([psi[1:, :], z], axis=0)
    psi_ym = jnp.concatenate([z, psi[:-1, :]], axis=0)
    return invDx2 * (psi_xp - 2 * psi + psi_xm) + invDy2 * (psi_yp - 2 * psi + psi_ym)


def test_grid_laplacian_matches_ref():
    rng = np.random.default_rng(0)
    psi = jnp.asarray(rng.standard_normal((32, 64)).astype(np.float32))
    dx = 1.0 / 63
    dy = 1.0 / 31
    np.testing.assert_allclose(
        np.asarray(grid_laplacian(psi, dx, dy)),
        np.asarray(_grid_laplacian_ref(psi, dx, dy)),
        atol=1e-5,
    )


def test_poisson_solve_round_trip_random():
    """For random Dirichlet-zero psi, solve(laplacian(psi)) must equal psi."""
    rng = np.random.default_rng(42)
    ny, nx = 64, 128
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    psi = rng.standard_normal((ny, nx)).astype(np.float32)
    psi[0, :] = 0
    psi[-1, :] = 0
    psi = jnp.asarray(psi)

    zeta = grid_laplacian(psi, dx, dy)
    psi_back = poisson_solve(zeta, dx, dy)
    np.testing.assert_allclose(
        np.asarray(psi_back[1:-1]), np.asarray(psi[1:-1]), atol=2e-4
    )


def test_poisson_solve_zero_input():
    dx = 1.0 / 31
    dy = 1.0 / 15
    zeta = jnp.zeros((16, 32))
    psi = poisson_solve(zeta, dx, dy)
    assert jnp.all(psi == 0)


def test_poisson_solve_eigenfunction():
    """sin(pi j / (ny+1)) cos(2pi i k / nx) is an eigenfunction with known eigenvalue."""
    ny, nx = 64, 128
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    invDx2 = 1.0 / (dx * dx)
    invDy2 = 1.0 / (dy * dy)
    j = jnp.arange(ny)[:, None]
    i = jnp.arange(nx)[None, :]
    kx_n = 3
    psi_true = jnp.sin(jnp.pi * (j + 1) / (ny + 1)) * jnp.cos(2 * jnp.pi * kx_n * i / nx)
    eig = invDy2 * 2 * (jnp.cos(jnp.pi / (ny + 1)) - 1) + invDx2 * 2 * (jnp.cos(2 * jnp.pi * kx_n / nx) - 1)
    zeta = eig * psi_true
    psi = poisson_solve(zeta, dx, dy)
    np.testing.assert_allclose(np.asarray(psi), np.asarray(psi_true), atol=1e-4)
