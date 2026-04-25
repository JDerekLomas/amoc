"""FFT/DST Poisson solver: round-trip ∇² → solve must reproduce the source."""
import jax.numpy as jnp
import numpy as np
import pytest

from amoc.poisson import grid_laplacian, poisson_solve


def _grid_laplacian_ref(psi: jnp.ndarray) -> jnp.ndarray:
    """Reference 5-point grid Laplacian: periodic in x, Dirichlet in y (psi=0 outside)."""
    psi_xp = jnp.roll(psi, -1, axis=1)
    psi_xm = jnp.roll(psi, 1, axis=1)
    # Pad y with zeros (Dirichlet BC).
    z = jnp.zeros_like(psi[:1, :])
    psi_yp = jnp.concatenate([psi[1:, :], z], axis=0)
    psi_ym = jnp.concatenate([z, psi[:-1, :]], axis=0)
    return (psi_xp - 2 * psi + psi_xm) + (psi_yp - 2 * psi + psi_ym)


def test_grid_laplacian_matches_ref():
    rng = np.random.default_rng(0)
    psi = jnp.asarray(rng.standard_normal((32, 64)).astype(np.float32))
    np.testing.assert_allclose(
        np.asarray(grid_laplacian(psi)),
        np.asarray(_grid_laplacian_ref(psi)),
        atol=1e-5,
    )


def test_poisson_solve_round_trip_random():
    """For random Dirichlet-zero ψ, solve(∇²ψ) must equal ψ."""
    rng = np.random.default_rng(42)
    psi = rng.standard_normal((64, 128)).astype(np.float32)
    # Force Dirichlet BC at top/bottom.
    psi[0, :] = 0
    psi[-1, :] = 0
    psi = jnp.asarray(psi)

    zeta = grid_laplacian(psi)
    psi_back = poisson_solve(zeta)
    # Compare interior; first/last rows are part of the Dirichlet BC.
    np.testing.assert_allclose(
        np.asarray(psi_back[1:-1]), np.asarray(psi[1:-1]), atol=2e-4
    )


def test_poisson_solve_zero_input():
    zeta = jnp.zeros((16, 32))
    psi = poisson_solve(zeta)
    assert jnp.all(psi == 0)


def test_poisson_solve_eigenfunction():
    """sin(π j / (ny+1)) cos(2π i k / nx) is an eigenfunction with known eigenvalue."""
    ny, nx = 64, 128
    j = jnp.arange(ny)[:, None]
    i = jnp.arange(nx)[None, :]
    kx_n = 3
    psi_true = jnp.sin(jnp.pi * (j + 1) / (ny + 1)) * jnp.cos(2 * jnp.pi * kx_n * i / nx)
    eig = 2 * (jnp.cos(jnp.pi / (ny + 1)) - 1) + 2 * (jnp.cos(2 * jnp.pi * kx_n / nx) - 1)
    zeta = eig * psi_true
    psi = poisson_solve(zeta)
    np.testing.assert_allclose(np.asarray(psi), np.asarray(psi_true), atol=1e-4)
