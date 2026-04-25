"""Physical-correctness regression tests.

These run a small idealized configuration to steady state and check
that the simulator reproduces known analytical results.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from amoc.diagnostics import meridional_velocity
from amoc.grid import Grid
from amoc.state import Params, trivial_forcing, zero_state
from amoc.step import run


def _dynamics_only_params(**kw) -> Params:
    """Params that disable thermodynamics so we test vorticity dynamics in isolation."""
    base = dict(
        r_friction_s=0.2, A_visc_s=0.005,
        r_friction_d=0.2, A_visc_d=0.005,
        beta=2.0, wind_strength=1.0, dt=0.005,
        F_couple_s=0.0, F_couple_d=0.0,
        alpha_T=0.0, beta_S=0.0,
        kappa_T=0.0, kappa_S=0.0,
        kappa_deep_T=0.0, kappa_deep_S=0.0,
        tau_T=1e9, tau_deep_T=1e9,
        gamma_mix=0.0, gamma_deep_form=0.0,
        sal_restoring_rate=0.0,
        mot_strength=0.0,
        gamma_oa=0.0, gamma_la=0.0, gamma_ao=0.0,
        kappa_atm=0.0, E0=0.0,
        freshwater_scale_pe=0.0,
    )
    base.update(kw)
    return Params(**base)


def test_friction_balances_forcing_at_steady_state():
    nx, ny = 32, 16
    g = Grid.create(nx=nx, ny=ny)
    curl_const = 0.5
    forcing = trivial_forcing(g.shape)._replace(
        wind_curl=jnp.full(g.shape, curl_const, dtype=jnp.float32),
    )
    params = _dynamics_only_params(
        r_friction_s=0.5, A_visc_s=0.0,
        r_friction_d=0.5, A_visc_d=0.0,
        beta=0.0, dt=0.05,
    )
    out = run(zero_state(g.shape), forcing, params, g, n_steps=1000)
    zeta = np.asarray(out.zeta_s)
    expected = curl_const / params.r_friction_s
    interior = zeta[2:-2, :]
    np.testing.assert_allclose(interior, expected, rtol=0.05)


def test_no_forcing_no_growth():
    g = Grid.create(nx=32, ny=16)
    out = run(zero_state(g.shape), trivial_forcing(g.shape),
              _dynamics_only_params(dt=0.05), g, n_steps=2000)
    assert float(jnp.max(jnp.abs(out.zeta_s))) == 0.0
    assert float(jnp.max(jnp.abs(out.zeta_d))) == 0.0


def test_sverdrup_response_localized_curl():
    """Localized positive curl in NH -> psi minimum (cyclonic gyre) at that latitude."""
    nx, ny = 32, 32
    g = Grid.create(nx=nx, ny=ny, lat0=-30.0, lat1=30.0)
    j = jnp.arange(ny)[:, None]
    bump = jnp.exp(-((j - 24) ** 2) / 8.0)
    forcing = trivial_forcing(g.shape)._replace(
        wind_curl=jnp.broadcast_to(bump, (ny, nx)).astype(jnp.float32),
    )
    out = run(zero_state(g.shape), forcing, _dynamics_only_params(), g, n_steps=4000)
    zeta = np.asarray(out.zeta_s)
    psi = np.asarray(out.psi_s)
    j_zeta = int(np.argmax(zeta.mean(axis=1)))
    assert abs(j_zeta - 24) <= 2
    j_psi = int(np.argmin(psi.mean(axis=1)))
    assert abs(j_psi - 24) <= 4
    assert psi[j_psi].mean() < 0


def test_curl_sign_drives_correct_gyre_sense():
    """Negative curl in NH -> psi maximum (anticyclonic = subtropical)."""
    nx, ny = 32, 32
    g = Grid.create(nx=nx, ny=ny, lat0=-30.0, lat1=30.0)
    j = jnp.arange(ny)[:, None]
    bump = -jnp.exp(-((j - 24) ** 2) / 8.0)
    forcing = trivial_forcing(g.shape)._replace(
        wind_curl=jnp.broadcast_to(bump, (ny, nx)).astype(jnp.float32),
    )
    out = run(zero_state(g.shape), forcing, _dynamics_only_params(), g, n_steps=4000)
    psi = np.asarray(out.psi_s)
    j_max = int(np.argmax(psi.mean(axis=1)))
    assert abs(j_max - 24) <= 4
    assert psi[j_max].mean() > 0
