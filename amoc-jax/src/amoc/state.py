"""Simulation state and parameters as JIT-friendly pytrees.

NamedTuples register as JAX pytrees automatically, so state flows through
jax.jit and jax.lax.scan with no extra ceremony.

Full coupled model: two-layer ocean (vorticity + T/S) + 1-layer atmosphere
(air temp + moisture).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class State(NamedTuple):
    """Prognostic ocean + atmosphere state."""
    # --- Ocean dynamics (two layers) ---
    psi_s:    jnp.ndarray   # (ny, nx) surface streamfunction
    zeta_s:   jnp.ndarray   # (ny, nx) surface vorticity
    psi_d:    jnp.ndarray   # (ny, nx) deep streamfunction
    zeta_d:   jnp.ndarray   # (ny, nx) deep vorticity
    # --- Ocean tracers ---
    T_s:      jnp.ndarray   # (ny, nx) surface temperature, C
    S_s:      jnp.ndarray   # (ny, nx) surface salinity, psu
    T_d:      jnp.ndarray   # (ny, nx) deep temperature, C
    S_d:      jnp.ndarray   # (ny, nx) deep salinity, psu
    # --- Atmosphere ---
    air_temp: jnp.ndarray   # (ny, nx) 1-layer atmospheric temperature, C
    moisture: jnp.ndarray   # (ny, nx) specific humidity, kg/kg
    # --- Time ---
    sim_time: float         # continuous time for seasonal cycle


class Params(NamedTuple):
    """Tunable physics parameters. Static across a simulation."""
    # --- Time ---
    dt: float = 5e-5
    year_speed: float = 1.0
    T_YEAR: float = 10.0          # sim time units per year
    # --- Surface dynamics ---
    r_friction_s: float = 0.04
    A_visc_s: float = 2e-4
    H_s: float = 100.0
    r_friction_d: float = 0.10
    A_visc_d: float = 2e-4
    H_d: float = 3900.0
    beta: float = 2.0
    wind_strength: float = 1.0
    F_couple_s: float = 0.5
    F_couple_d: float = 0.0125
    # --- Buoyancy (linear EOS) ---
    alpha_T: float = 0.05          # thermal expansion (model units)
    beta_S: float = 0.8            # haline contraction (model units)
    # --- Tracer diffusion ---
    kappa_T: float = 2.5e-4
    kappa_S: float = 2.5e-4
    kappa_deep_T: float = 2e-5
    kappa_deep_S: float = 2e-5
    # --- Salinity restoring ---
    sal_restoring_rate: float = 0.005
    # --- Vertical mixing ---
    gamma_mix: float = 0.001
    gamma_deep_form: float = 0.05
    # --- Radiation ---
    S_solar: float = 6.2
    A_olr: float = 1.8
    B_olr: float = 0.13
    global_temp_offset: float = 0.0
    # --- Atmosphere ---
    kappa_atm: float = 3e-3
    gamma_oa: float = 0.005       # ocean -> atmosphere exchange
    gamma_la: float = 0.01        # land -> atmosphere exchange
    gamma_ao: float = 0.001       # atmosphere -> ocean feedback
    E0: float = 0.003             # evaporation coefficient
    greenhouse_q: float = 0.4
    q_ref: float = 0.015
    freshwater_scale_pe: float = 0.5
    latent_heat_coeff: float = 800.0
    # --- SST restoring (Haney) ---
    tau_T: float = 0.01           # SST restoring timescale (model time units)
    tau_deep_T: float = 0.1       # deep T restoring timescale (weaker)
    # --- Freshwater ---
    freshwater_forcing: float = 0.0
    # --- Deep overturning ---
    mot_strength: float = 0.05    # meridional overturning tendency


class Forcing(NamedTuple):
    """Time-independent external forcing fields and geometry."""
    wind_curl:       jnp.ndarray  # (ny, nx) pre-scaled curl(tau)
    ocean_mask:      jnp.ndarray  # (ny, nx) 1=ocean, 0=land
    # --- Tracer targets (Haney restoring) ---
    T_target:        jnp.ndarray  # (ny, nx) observed SST climatology, C
    T_deep_target:   jnp.ndarray  # (ny, nx) observed deep T, C
    sal_climatology: jnp.ndarray  # (ny, nx) WOA23 salinity, psu
    # --- Ekman transport ---
    ekman_u:         jnp.ndarray  # (ny, nx) Ekman zonal velocity
    ekman_v:         jnp.ndarray  # (ny, nx) Ekman meridional velocity
    # --- Bathymetry ---
    depth_field:     jnp.ndarray  # (ny, nx) ocean depth in meters
    # --- Land ---
    land_temp:       jnp.ndarray  # (ny, nx) land surface temperature, C
    # --- Observed fields for physics alignment ---
    obs_mld:         jnp.ndarray  # (ny, nx) observed mixed layer depth, m
    obs_cloud:       jnp.ndarray  # (ny, nx) MODIS cloud fraction, 0-1
    obs_albedo:      jnp.ndarray  # (ny, nx) observed surface albedo, 0-1
    obs_sea_ice:     jnp.ndarray  # (ny, nx) observed sea ice fraction, 0-1
    obs_precip:      jnp.ndarray  # (ny, nx) observed precipitation, mm/yr
    obs_evap:        jnp.ndarray  # (ny, nx) observed evaporation
    obs_water_vapor: jnp.ndarray  # (ny, nx) column water vapor (normalized)
    # --- Observed ocean currents (for init/validation) ---
    obs_u:           jnp.ndarray  # (ny, nx) observed surface current u, m/s
    obs_v:           jnp.ndarray  # (ny, nx) observed surface current v, m/s
    # --- Additional land/vegetation ---
    obs_ndvi:        jnp.ndarray  # (ny, nx) NDVI vegetation index
    obs_snow:        jnp.ndarray  # (ny, nx) snow cover fraction
    obs_chlorophyll: jnp.ndarray  # (ny, nx) ocean chlorophyll, mg/m3
    obs_pressure:    jnp.ndarray  # (ny, nx) surface pressure, hPa


def zero_state(grid_shape: tuple[int, int]) -> State:
    """Rest state for initialization. T=15C, S=35psu, air=15C, q=q_sat(15)."""
    z = jnp.zeros(grid_shape)
    T0, S0 = 15.0, 35.0
    q0 = 3.75e-3 * jnp.exp(0.067 * T0)  # q_sat(15C)
    return State(
        psi_s=z, zeta_s=z, psi_d=z, zeta_d=z,
        T_s=jnp.full(grid_shape, T0),
        S_s=jnp.full(grid_shape, S0),
        T_d=jnp.full(grid_shape, T0),
        S_d=jnp.full(grid_shape, S0 + 0.7),  # deep slightly saltier
        air_temp=jnp.full(grid_shape, T0),
        moisture=jnp.full(grid_shape, q0 * 0.8),
        sim_time=0.0,
    )


def trivial_forcing(grid_shape: tuple[int, int],
                    *, wind: float = 0.0, mask: float = 1.0) -> Forcing:
    """Zero forcing for tests."""
    z = jnp.zeros(grid_shape)
    return Forcing(
        wind_curl=jnp.full(grid_shape, wind),
        ocean_mask=jnp.full(grid_shape, mask),
        T_target=jnp.full(grid_shape, 15.0),
        T_deep_target=jnp.full(grid_shape, 3.0),
        sal_climatology=jnp.full(grid_shape, 35.0),
        ekman_u=z,
        ekman_v=z,
        depth_field=jnp.full(grid_shape, 4000.0),
        land_temp=jnp.full(grid_shape, 15.0),
        obs_mld=jnp.full(grid_shape, 50.0),
        obs_cloud=jnp.full(grid_shape, 0.5),
        obs_albedo=jnp.full(grid_shape, 0.06),
        obs_sea_ice=z,
        obs_precip=z,
        obs_evap=z,
        obs_water_vapor=jnp.full(grid_shape, 0.5),
        obs_u=z,
        obs_v=z,
        obs_ndvi=z,
        obs_snow=z,
        obs_chlorophyll=z,
        obs_pressure=jnp.full(grid_shape, 1013.0),
    )
