"""Simulation state and parameters as JIT-friendly pytrees.

NamedTuples register as JAX pytrees automatically, so state flows through
jax.jit and jax.lax.scan with no extra ceremony.

State has two layers from v1b onwards (surface s, deep d). v1a runs with
this same shape and just leaves the deep layer at zero by setting deep
forcing to zero. v1c will add T and S as additional fields here; v1d
adds a freshwater-flux scalar to Forcing.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class State(NamedTuple):
    """Prognostic ocean state.

    v1a: only psi_s, zeta_s used (rest set to zero).
    v1b: + psi_d, zeta_d (deep layer).
    v1c: + T_s, S_s, T_d (prognostic temperature & salinity). The buoyancy
         that drives the baroclinic flow is then derived as
         b = -alpha_T(T_s - T_d) + alpha_S(S_s - S_d_implicit) — see
         physics.py.
    """
    psi_s:  jnp.ndarray   # (ny, nx) surface-layer streamfunction
    zeta_s: jnp.ndarray   # (ny, nx) surface-layer relative vorticity
    psi_d:  jnp.ndarray   # (ny, nx) deep-layer streamfunction
    zeta_d: jnp.ndarray   # (ny, nx) deep-layer relative vorticity
    T_s:    jnp.ndarray   # (ny, nx) surface temperature, °C  (v1c)
    S_s:    jnp.ndarray   # (ny, nx) surface salinity, psu     (v1c)
    T_d:    jnp.ndarray   # (ny, nx) deep temperature, °C     (v1c)


class Params(NamedTuple):
    """Tunable physics parameters. Static across a simulation."""
    # ------- v1a/b dynamics -------
    r_friction_s: float = 0.04
    A_visc_s: float = 2.0e-4
    H_s: float = 100.0              # surface-layer thickness (m)
    r_friction_d: float = 0.10
    A_visc_d: float = 2.0e-4
    H_d: float = 3900.0             # deep-layer thickness (m)
    beta: float = 2.0
    wind_strength: float = 1.0
    dt: float = 0.01
    F_couple_s: float = 0.5         # surface->deep coupling on shear ζ_s-ζ_d
    F_couple_d: float = 0.0125
    alpha_buoy: float = 0.05        # legacy v1b prescribed-buoyancy strength
    # ------- v1c thermodynamics -------
    # Linear EOS coefficients (seawater approximations):
    alpha_T: float = 2.0e-4         # thermal expansion, 1/°C
    alpha_S: float = 8.0e-4         # haline contraction, 1/psu
    # Diffusion of T & S (set in dimensionless model units; calibration in v1d):
    kappa_T: float = 5.0e-4
    kappa_S: float = 5.0e-4
    # Surface restoring (Haney-style) toward observed climatology:
    tau_T: float = 60.0             # restoring time-scale (model units)
    tau_S: float = 180.0            # salinity is restored more weakly
    # Vertical mixing (interface between surface & deep):
    gamma_TS: float = 0.001         # background T,S exchange
    gamma_conv: float = 0.05        # enhanced when surface denser than deep
    # Internal pressure-gradient strength derived from layer T,S contrast.
    # alpha_BC ≈ g'/H_s * f^-2 in proper scaling; for v1c we keep it as a
    # tuning knob until v1d adds physical-unit calibration.
    alpha_BC: float = 1.0


class Forcing(NamedTuple):
    """Time-independent external forcing fields and geometry."""
    wind_curl:  jnp.ndarray  # (ny, nx) RMS-normalized curl(tau)
    ocean_mask: jnp.ndarray  # (ny, nx) 1.0 over ocean, 0.0 over land
    buoyancy:   jnp.ndarray  # (ny, nx) [legacy v1b] prescribed buoyancy
    # v1c: surface restoring targets + freshwater flux pattern
    T_target:   jnp.ndarray  # (ny, nx) SST climatology, °C
    S_target:   jnp.ndarray  # (ny, nx) SSS climatology, psu
    S_d_const:  jnp.ndarray  # (ny, nx) deep salinity (held fixed in v1c)
    F_fresh:    jnp.ndarray  # (ny, nx) freshwater flux pattern (set in v1d)


def zero_state(grid_shape: tuple[int, int]) -> State:
    """Convenience: rest state for tests and spin-up. Tracer fields at the
    EOS reference T0=15°C, S0=35 psu so b_s = b_d = 0 initially."""
    z = jnp.zeros(grid_shape)
    from .physics import T0 as _T0, S0 as _S0  # local import to avoid cycle
    return State(
        psi_s=z, zeta_s=z, psi_d=z, zeta_d=z,
        T_s=jnp.full(grid_shape, _T0), S_s=jnp.full(grid_shape, _S0),
        T_d=jnp.full(grid_shape, _T0),
    )


def trivial_forcing(grid_shape: tuple[int, int],
                    *, wind: float = 0.0, mask: float = 1.0) -> Forcing:
    """Convenience: zero forcing apart from optional uniform wind."""
    z = jnp.zeros(grid_shape)
    from .physics import T0 as _T0, S0 as _S0
    return Forcing(
        wind_curl=jnp.full(grid_shape, wind),
        ocean_mask=jnp.full(grid_shape, mask),
        buoyancy=z,
        T_target=jnp.full(grid_shape, _T0),
        S_target=jnp.full(grid_shape, _S0),
        S_d_const=jnp.full(grid_shape, _S0),
        F_fresh=z,
    )
