"""Simulation state and parameters as JIT-friendly pytrees.

Using flax-style frozen dataclasses (registered as JAX pytrees) keeps the
step function pure: state and params flow in, new state flows out, no global
mutation. Adding fields later (T, S, deep psi) is a one-line change.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class State(NamedTuple):
    """Prognostic ocean state at a single time."""
    psi: jnp.ndarray   # (ny, nx) streamfunction (nondim)
    zeta: jnp.ndarray  # (ny, nx) relative vorticity (nondim)


class Params(NamedTuple):
    """Tunable physics parameters. Static across a simulation."""
    r_friction: float = 0.04        # linear bottom drag
    A_visc: float = 2.0e-4          # Laplacian viscosity (grid units)
    beta: float = 0.5               # planetary vorticity gradient coefficient
    wind_strength: float = 1.0      # multiplier on RMS-normalized wind curl
    dt: float = 0.05                # time step (nondimensional)


class Forcing(NamedTuple):
    """Time-independent external forcing fields."""
    wind_curl: jnp.ndarray  # (ny, nx) RMS-normalized curl(tau)
