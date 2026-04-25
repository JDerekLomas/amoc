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
    """Prognostic ocean state. Two layers from v1b onwards."""
    psi_s: jnp.ndarray    # (ny, nx) surface-layer streamfunction
    zeta_s: jnp.ndarray   # (ny, nx) surface-layer relative vorticity
    psi_d: jnp.ndarray    # (ny, nx) deep-layer streamfunction
    zeta_d: jnp.ndarray   # (ny, nx) deep-layer relative vorticity


class Params(NamedTuple):
    """Tunable physics parameters. Static across a simulation."""
    # Surface layer
    r_friction_s: float = 0.04      # linear drag, surface
    A_visc_s: float = 2.0e-4        # Laplacian viscosity, surface
    H_s: float = 100.0              # surface-layer thickness (m)
    # Deep layer
    r_friction_d: float = 0.10      # linear drag, deep (stronger)
    A_visc_d: float = 2.0e-4
    H_d: float = 3900.0             # deep-layer thickness (m)
    # Shared
    beta: float = 2.0
    wind_strength: float = 1.0
    dt: float = 0.01
    # 2-layer coupling
    F_couple_s: float = 0.5         # surface relax toward deep psi
    F_couple_d: float = 0.0125      # deep relax toward surface psi
    # Buoyancy: rho derived from prescribed buoyancy field b(λ,φ).
    # The d_x b term enters as an "internal pressure-gradient" body force
    # acting opposite-signed on the two layers; alpha controls strength.
    alpha_buoy: float = 0.05        # buoyancy coupling strength


class Forcing(NamedTuple):
    """Time-independent external forcing fields and geometry."""
    wind_curl: jnp.ndarray   # (ny, nx) RMS-normalized curl(tau)
    ocean_mask: jnp.ndarray  # (ny, nx) 1.0 over ocean, 0.0 over land
    buoyancy: jnp.ndarray    # (ny, nx) prescribed buoyancy field (e.g. -alpha_T * SST)
