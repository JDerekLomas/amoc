"""Physical-correctness regression tests.

These run a small idealized configuration to steady state and check
that the simulator reproduces *known analytical results*. They take
seconds (not milliseconds like the unit tests) but they would have
caught both the cos(φ) bug and the data-orientation bug in CI without
any manual inspection — the analytical answer simply wouldn't match.

These complement the unit tests in test_physics.py / test_step.py:

- Unit tests check operator properties (J(A,A)=0, antisymmetry, friction
  damping). They're cheap and catch algorithmic typos.
- Correctness tests check whole-simulator behavior against analytical
  ocean physics. They're more expensive and catch *physical* bugs that
  pass the unit tests.

Test configurations are idealized — small box, smooth wind, no
continents, no buoyancy. v1a barotropic regime only. v1b/c/d will add
their own correctness tests.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from amoc.diagnostics import meridional_velocity
from amoc.grid import Grid
from amoc.state import Forcing, Params, State
from amoc.step import run


def _spinup_params(beta: float = 4.0) -> Params:
    """Pure-linear setup: friction balances wind curl. No advection
    contributes much because we'll start from rest with weak wind."""
    return Params(
        r_friction_s=0.1, A_visc_s=0.005,
        r_friction_d=0.1, A_visc_d=0.005,
        beta=beta, wind_strength=1.0, dt=0.005,
        F_couple_s=0.0, F_couple_d=0.0, alpha_buoy=0.0,
    )


def test_sverdrup_response_localized_curl():
    """A localized curl bump should produce a localized ψ response with
    sign and location consistent with the Stommel/Munk closed-basin
    solution -- here in periodic-x, the ψ has a minimum (cyclonic) at
    the latitude of positive curl forcing.

    Sign convention here: u = -∂_yψ, v = ∂_xψ, so ζ = ∇²ψ. Positive
    curl(τ) → friction balance gives ζ > 0 → Poisson(ζ>0) → ψ has a
    *minimum* (negative). The flow circulating around that minimum is
    anti-clockwise looking down → cyclonic in NH = subpolar gyre.
    See docs/physics.md §4 for the full sign-convention derivation.
    """
    nx, ny = 32, 32
    g = Grid.create(nx=nx, ny=ny, lat0=-30.0, lat1=30.0)
    # Bump localized in northern half (around lat ≈ +14°, j=24).
    j = jnp.arange(ny)[:, None]
    bump = jnp.exp(-((j - 24) ** 2) / 8.0)
    curl_pos = jnp.broadcast_to(bump, (ny, nx))
    forcing = Forcing(
        wind_curl=jnp.asarray(curl_pos, dtype=jnp.float32),
        ocean_mask=jnp.ones((ny, nx)),
        buoyancy=jnp.zeros((ny, nx)),
    )
    state = State(
        psi_s=jnp.zeros((ny, nx)), zeta_s=jnp.zeros((ny, nx)),
        psi_d=jnp.zeros((ny, nx)), zeta_d=jnp.zeros((ny, nx)),
    )
    params = Params(
        r_friction_s=0.2, A_visc_s=0.005,
        r_friction_d=0.2, A_visc_d=0.005,
        beta=2.0, wind_strength=1.0, dt=0.005,
        F_couple_s=0.0, F_couple_d=0.0, alpha_buoy=0.0,
    )
    out = run(state, forcing, params, g, n_steps=4000)
    zeta = np.asarray(out.zeta_s)
    psi = np.asarray(out.psi_s)

    # ζ should peak (positive max) where curl is concentrated.
    j_zeta = int(np.argmax(zeta.mean(axis=1)))
    assert abs(j_zeta - 24) <= 2, f"ζ peak should be at j≈24, got j={j_zeta}"

    # ψ should reach its *minimum* (most negative) near the same row.
    j_psi = int(np.argmin(psi.mean(axis=1)))
    assert abs(j_psi - 24) <= 4, f"ψ minimum should be at j≈24 (cyclonic gyre), got j={j_psi}"
    assert psi[j_psi].mean() < 0, "ψ should be negative under positive curl (cyclonic)"


def test_friction_balances_forcing_at_steady_state():
    """With pure linear friction and uniform wind curl over a periodic
    domain, the steady state has ζ = curl(τ) / r exactly (no Sverdrup
    transport because curl is constant in y → no β·v contribution).

    This isolates the friction balance from the β-term.
    """
    nx, ny = 32, 16
    g = Grid.create(nx=nx, ny=ny)
    curl_const = 0.5
    forcing = Forcing(
        wind_curl=jnp.full((ny, nx), curl_const, dtype=jnp.float32),
        ocean_mask=jnp.ones((ny, nx)),
        buoyancy=jnp.zeros((ny, nx)),
    )
    state = State(
        psi_s=jnp.zeros((ny, nx)), zeta_s=jnp.zeros((ny, nx)),
        psi_d=jnp.zeros((ny, nx)), zeta_d=jnp.zeros((ny, nx)),
    )
    params = Params(
        r_friction_s=0.5, A_visc_s=0.0,
        r_friction_d=0.5, A_visc_d=0.0,
        beta=0.0, wind_strength=1.0, dt=0.05,
        F_couple_s=0.0, F_couple_d=0.0, alpha_buoy=0.0,
    )
    # 1000 steps × dt 0.05 × r 0.5 = 25 friction times (saturated).
    out = run(state, forcing, params, g, n_steps=1000)
    zeta = np.asarray(out.zeta_s)
    expected = curl_const / params.r_friction_s
    # In the interior, zeta should be exactly forcing/r. Boundary rows
    # are affected by the Dirichlet ψ=0 condition.
    interior = zeta[2:-2, :]
    np.testing.assert_allclose(interior, expected, rtol=0.05)


def test_no_forcing_no_growth():
    """With zero wind and zero buoyancy, ζ from rest must remain at zero
    indefinitely. No spurious growth from numerical noise."""
    nx, ny = 32, 16
    g = Grid.create(nx=nx, ny=ny)
    forcing = Forcing(
        wind_curl=jnp.zeros((ny, nx)),
        ocean_mask=jnp.ones((ny, nx)),
        buoyancy=jnp.zeros((ny, nx)),
    )
    state = State(
        psi_s=jnp.zeros((ny, nx)), zeta_s=jnp.zeros((ny, nx)),
        psi_d=jnp.zeros((ny, nx)), zeta_d=jnp.zeros((ny, nx)),
    )
    params = Params(dt=0.05, beta=2.0, wind_strength=1.0, alpha_buoy=0.0,
                    F_couple_s=0.0, F_couple_d=0.0)
    out = run(state, forcing, params, g, n_steps=2000)
    assert float(jnp.max(jnp.abs(out.zeta_s))) == 0.0
    assert float(jnp.max(jnp.abs(out.zeta_d))) == 0.0


def test_curl_sign_drives_correct_gyre_sense():
    """Negative curl in NH (anticyclonic forcing) → ψ has a maximum
    (anticyclonic = NH subtropical gyre). Positive curl → ψ minimum
    (cyclonic = NH subpolar gyre). Tests the sign coupling between
    forcing and gyre sense.
    """
    nx, ny = 32, 32
    g = Grid.create(nx=nx, ny=ny, lat0=-30.0, lat1=30.0)
    j = jnp.arange(ny)[:, None]
    # Negative bump in NH (anticyclonic forcing).
    bump = -jnp.exp(-((j - 24) ** 2) / 8.0)
    curl = jnp.broadcast_to(bump, (ny, nx))
    forcing = Forcing(
        wind_curl=jnp.asarray(curl, dtype=jnp.float32),
        ocean_mask=jnp.ones((ny, nx)),
        buoyancy=jnp.zeros((ny, nx)),
    )
    state0 = State(
        psi_s=jnp.zeros((ny, nx)), zeta_s=jnp.zeros((ny, nx)),
        psi_d=jnp.zeros((ny, nx)), zeta_d=jnp.zeros((ny, nx)),
    )
    params = Params(
        r_friction_s=0.2, A_visc_s=0.005,
        r_friction_d=0.2, A_visc_d=0.005,
        beta=2.0, wind_strength=1.0, dt=0.005,
        F_couple_s=0.0, F_couple_d=0.0, alpha_buoy=0.0,
    )
    out = run(state0, forcing, params, g, n_steps=4000)
    psi = np.asarray(out.psi_s)
    j_max = int(np.argmax(psi.mean(axis=1)))
    # Negative curl → ζ negative → ψ has a max (anticyclonic gyre).
    assert abs(j_max - 24) <= 4, f"ψ max should be at j≈24, got {j_max}"
    assert psi[j_max].mean() > 0, "ψ should be positive (max) under negative curl (anticyclonic)"
