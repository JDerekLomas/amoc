"""FFT/DST Poisson solver for the streamfunction.

Solves (invDx2 * Δx + invDy2 * Δy) ψ = ζ on a grid with:
  - periodic x (longitude)
  - Dirichlet y (poles): ψ = 0 just outside the first/last rows.

Uses dx = 1/(nx-1), dy = 1/(ny-1) convention matching the JS model.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def grid_laplacian(psi: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """5-point grid Laplacian with dx/dy scaling.
    Periodic in x, Dirichlet (zero) in y."""
    invDx2 = 1.0 / (dx * dx)
    invDy2 = 1.0 / (dy * dy)
    psi_xp = jnp.roll(psi, -1, axis=1)
    psi_xm = jnp.roll(psi, 1, axis=1)
    z = jnp.zeros_like(psi[:1, :])
    psi_yp = jnp.concatenate([psi[1:, :], z], axis=0)
    psi_ym = jnp.concatenate([z, psi[:-1, :]], axis=0)
    return invDx2 * (psi_xp - 2 * psi + psi_xm) + invDy2 * (psi_yp - 2 * psi + psi_ym)


def _dst_I_real(y: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
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
    if jnp.iscomplexobj(y):
        return _dst_I_real(y.real, axis=axis) + 1j * _dst_I_real(y.imag, axis=axis)
    return _dst_I_real(y, axis=axis)


def _idst_I(Y: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    N = Y.shape[axis]
    return _dst_I(Y, axis=axis) * (2.0 / (N + 1))


@jax.jit
def poisson_solve(zeta: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """Return ψ such that grid_laplacian(ψ, dx, dy) = ζ."""
    ny, nx = zeta.shape
    invDx2 = 1.0 / (dx * dx)
    invDy2 = 1.0 / (dy * dy)

    kx = jnp.arange(nx)
    lam_x = invDx2 * 2.0 * (jnp.cos(2.0 * jnp.pi * kx / nx) - 1.0)
    ky = jnp.arange(ny)
    lam_y = invDy2 * 2.0 * (jnp.cos(jnp.pi * (ky + 1) / (ny + 1)) - 1.0)
    eig = lam_y[:, None] + lam_x[None, :]

    Z_xhat = jnp.fft.fft(zeta, axis=1)
    Z_hat = _dst_I(Z_xhat, axis=0)
    safe = jnp.where(jnp.abs(eig) < 1e-30, 1.0, eig)
    Psi_hat = jnp.where(jnp.abs(eig) < 1e-30, 0.0, Z_hat / safe)
    Psi_xhat = _idst_I(Psi_hat, axis=0)
    psi = jnp.fft.ifft(Psi_xhat, axis=1).real
    return psi
