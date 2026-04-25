"""Spherical lat-lon grid with metric factors and Coriolis terms.

Conventions:
- Cell-centered grid. lat[j], lon[i] are cell centers.
- Longitude is periodic over 360 degrees.
- Latitude bounds [lat0, lat1] are the *cell-edge* extremes; cell centers
  sit half a cell inside.
- Distances are in meters; times in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

R_EARTH = 6.371e6           # m
OMEGA = 7.2921159e-5        # rad/s


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    lat0: float = -79.5
    lat1: float = 79.5
    lon0: float = -180.0
    lon1: float = 180.0

    # Computed in __post_init__ via object.__setattr__ (frozen dataclass).
    lat: jnp.ndarray = None       # (ny,) degrees
    lon: jnp.ndarray = None       # (nx,) degrees
    dlat: float = 0.0             # degrees per cell
    dlon: float = 0.0             # degrees per cell
    cos_lat: jnp.ndarray = None   # (ny,)
    f: jnp.ndarray = None         # (ny,) Coriolis parameter, 1/s
    beta: jnp.ndarray = None      # (ny,) df/dy on sphere, 1/(m*s)
    dx_eq: float = 0.0            # meters per cell at equator
    dy: float = 0.0               # meters per cell (constant)

    def __post_init__(self):
        dlon = (self.lon1 - self.lon0) / self.nx
        dlat = (self.lat1 - self.lat0) / self.ny
        lon = self.lon0 + (jnp.arange(self.nx) + 0.5) * dlon
        lat = self.lat0 + (jnp.arange(self.ny) + 0.5) * dlat
        lat_rad = jnp.deg2rad(lat)
        cos_lat = jnp.cos(lat_rad)
        f = 2 * OMEGA * jnp.sin(lat_rad)
        beta = 2 * OMEGA * cos_lat / R_EARTH
        dx_eq = R_EARTH * jnp.deg2rad(dlon)
        dy = R_EARTH * jnp.deg2rad(dlat)
        # Bypass frozen dataclass to set computed fields.
        for k, v in dict(
            lat=lat, lon=lon, dlat=dlat, dlon=dlon,
            cos_lat=cos_lat, f=f, beta=beta,
            dx_eq=float(dx_eq), dy=float(dy),
        ).items():
            object.__setattr__(self, k, v)

    @property
    def shape(self) -> tuple[int, int]:
        """(ny, nx) — row-major, latitude is the slow axis."""
        return (self.ny, self.nx)

    def dx(self) -> jnp.ndarray:
        """Zonal cell width as a function of latitude, shape (ny,), meters."""
        return self.dx_eq * self.cos_lat
