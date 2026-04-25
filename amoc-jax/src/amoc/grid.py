"""Spherical lat-lon grid with metric factors and Coriolis terms.

Conventions:
- Cell-centered grid. lat[j], lon[i] are cell centers.
- Longitude is periodic over 360 degrees.
- Latitude bounds [lat0, lat1] are *cell-edge* extremes; cell centers sit
  half a cell inside.
- Distances are in meters; times in seconds.

Grid is a NamedTuple so JAX treats it as a pytree (can pass through jit).
Construct via `Grid.create(nx, ny, ...)` to compute derived fields.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

R_EARTH = 6.371e6           # m
OMEGA = 7.2921159e-5        # rad/s


class Grid(NamedTuple):
    nx: int
    ny: int
    lat0: float
    lat1: float
    lon0: float
    lon1: float
    dlat: float
    dlon: float
    dx_eq: float
    dy: float
    lat: jnp.ndarray       # (ny,) cell-center latitudes (degrees)
    lon: jnp.ndarray       # (nx,)
    cos_lat: jnp.ndarray   # (ny,)
    f: jnp.ndarray         # (ny,) Coriolis (1/s)
    beta: jnp.ndarray      # (ny,) df/dy on sphere (1/(m*s))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def dx(self) -> jnp.ndarray:
        """Zonal cell width by latitude (ny,), meters."""
        return self.dx_eq * self.cos_lat

    @classmethod
    def create(
        cls,
        nx: int,
        ny: int,
        lat0: float = -79.5,
        lat1: float = 79.5,
        lon0: float = -180.0,
        lon1: float = 180.0,
    ) -> "Grid":
        dlon = (lon1 - lon0) / nx
        dlat = (lat1 - lat0) / ny
        lon = lon0 + (jnp.arange(nx) + 0.5) * dlon
        lat = lat0 + (jnp.arange(ny) + 0.5) * dlat
        lat_rad = jnp.deg2rad(lat)
        cos_lat = jnp.cos(lat_rad)
        f = 2 * OMEGA * jnp.sin(lat_rad)
        beta = 2 * OMEGA * cos_lat / R_EARTH
        dx_eq = float(R_EARTH * jnp.deg2rad(dlon))
        dy = float(R_EARTH * jnp.deg2rad(dlat))
        return cls(
            nx=nx, ny=ny,
            lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1,
            dlat=dlat, dlon=dlon,
            dx_eq=dx_eq, dy=dy,
            lat=lat, lon=lon, cos_lat=cos_lat, f=f, beta=beta,
        )


def make_grid(nx: int, ny: int, **kw) -> Grid:
    """Backwards-compatible factory."""
    return Grid.create(nx, ny, **kw)
