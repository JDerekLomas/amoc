"""FFT/DST Poisson solver for the streamfunction.

Solves ∇²_grid ψ = ζ on a grid with:
  - periodic x (longitude)
  - Dirichlet y (poles): ψ = 0 just outside the first/last rows.

Method: FFT in x, DST-I in y, diagonal divide, inverse transforms. The "grid
Laplacian" here is the 5-point stencil with unit spacing — physical metric
factors (cos(lat), Earth radius) live in the dynamics operators, not the
Poisson inversion. This matches the convention used by the existing model.

The k=0, j=0 mode (constant) has eigenvalue 0 along x but is rescued by the
Dirichlet condition in y, so the operator is invertible without any nullspace
projection.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def grid_laplacian(psi: jnp.ndarray) -> jnp.ndarray:
    """5-point grid Laplacian. Periodic in x, Dirichlet (zero) in y."""
    psi_xp = jnp.roll(psi, -1, axis=1)
    psi_xm = jnp.roll(psi, 1, axis=1)
    z = jnp.zeros_like(psi[:1, :])
    psi_yp = jnp.concatenate([psi[1:, :], z], axis=0)
    psi_ym = jnp.concatenate([z, psi[:-1, :]], axis=0)
    return (psi_xp - 2 * psi + psi_xm) + (psi_yp - 2 * psi + psi_ym)


def _dst_I_real(y: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Type-I DST along `axis` via length-2(N+1) FFT. Input must be real.

    Y_k = sum_{j=0}^{N-1} y_j sin(π (j+1)(k+1) / (N+1)),  k = 0..N-1
    """
    N = y.shape[axis]
    zeros_shape = list(y.shape)
    zeros_shape[axis] = 1
    z = jnp.zeros(zeros_shape, dtype=y.dtype)
    minus = -jnp.flip(y, axis=axis)
    ext = jnp.concatenate([z, y, z, minus], axis=axis)
    F = jnp.fft.fft(ext, axis=axis)
    sl = [slice(None)] * y.ndim
    sl[axis] = slice(1, N + 1)
    return -jnp.imag(F[tuple(sl)]) / 2.0


def _dst_I(y: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Type-I DST along `axis`. Handles complex input by linearity."""
    if jnp.iscomplexobj(y):
        return _dst_I_real(y.real, axis=axis) + 1j * _dst_I_real(y.imag, axis=axis)
    return _dst_I_real(y, axis=axis)


def _idst_I(Y: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Inverse DST-I. DST-I is involutive up to factor 2/(N+1)."""
    N = Y.shape[axis]
    return _dst_I(Y, axis=axis) * (2.0 / (N + 1))


@jax.jit
def poisson_solve(zeta: jnp.ndarray) -> jnp.ndarray:
    """Return ψ such that grid_laplacian(ψ) = ζ, with periodic-x / Dirichlet-y."""
    ny, nx = zeta.shape
    # x-eigenvalues for the 5-point stencil along periodic x:
    #   λ_x[i] = 2(cos(2π i / nx) - 1)
    kx = jnp.arange(nx)
    lam_x = 2.0 * (jnp.cos(2.0 * jnp.pi * kx / nx) - 1.0)
    # y-eigenvalues for Dirichlet BC (sine basis), j = 0..ny-1:
    #   λ_y[j] = 2(cos(π (j+1) / (ny+1)) - 1)
    ky = jnp.arange(ny)
    lam_y = 2.0 * (jnp.cos(jnp.pi * (ky + 1) / (ny + 1)) - 1.0)
    eig = lam_y[:, None] + lam_x[None, :]  # (ny, nx)

    # FFT in x then DST-I in y (or other order — both are linear).
    Z_xhat = jnp.fft.fft(zeta, axis=1)
    Z_hat = _dst_I(Z_xhat, axis=0)
    # Avoid divide-by-zero at the mathematical zero mode (none for this BC, but
    # be defensive in case of numerical underflow).
    safe = jnp.where(jnp.abs(eig) < 1e-30, 1.0, eig)
    Psi_hat = jnp.where(jnp.abs(eig) < 1e-30, 0.0, Z_hat / safe)
    Psi_xhat = _idst_I(Psi_hat, axis=0)
    psi = jnp.fft.ifft(Psi_xhat, axis=1).real
    return psi
