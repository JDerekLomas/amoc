"""Atmosphere physics for the coupled ocean-atmosphere climate model.

Pure JAX functions operating on (ny, nx) arrays. Implements:
- Clausius-Clapeyron saturation humidity
- 7-regime cloud fraction model
- Shortwave/longwave radiation with ice-albedo and cloud feedbacks
- Atmosphere time-stepping (diffusion, surface exchange, moisture cycle)
- Latitude-dependent mixed layer depth

All latitude arrays are in degrees, south-to-north (row 0 = southern boundary).
Periodic in x (longitude), Neumann boundaries in y (latitude).
"""
from __future__ import annotations

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# 1. Saturation specific humidity (Clausius-Clapeyron approximation)
# ---------------------------------------------------------------------------

def q_sat(T: jnp.ndarray) -> jnp.ndarray:
    """Saturation specific humidity as a function of temperature (deg C)."""
    return 3.75e-3 * jnp.exp(0.067 * T)


# ---------------------------------------------------------------------------
# 2. Seven-regime cloud fraction model
# ---------------------------------------------------------------------------

def cloud_fraction(
    T_s: jnp.ndarray,
    lat: jnp.ndarray,
    sim_time: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute cloud fraction and convective fraction.

    Parameters
    ----------
    T_s : (ny, nx) surface temperature in deg C
    lat : (ny,) latitude in degrees
    sim_time : scalar, simulation time (period = 10)

    Returns
    -------
    cloud_frac, conv_frac : both (ny, nx)
    """
    year_phase = 2.0 * jnp.pi * (sim_time % 10.0) / 10.0
    abs_lat = jnp.abs(lat[:, jnp.newaxis])                 # (ny, 1)

    humidity = jnp.clip((T_s - 5.0) / 25.0, 0.0, 1.0)

    air_temp_est = 28.0 - 0.55 * abs_lat
    lts = jnp.clip((air_temp_est - T_s) / 15.0, 0.0, 1.0)

    itcz_lat = 5.0 * jnp.sin(year_phase)
    itcz_dist = (lat[:, jnp.newaxis] - itcz_lat) / 10.0

    conv_cloud = 0.30 * jnp.exp(-itcz_dist ** 2) * humidity
    warm_pool = 0.20 * jnp.clip((T_s - 26.0) / 4.0, 0.0, 1.0)

    sub_dist = (abs_lat - 25.0) / 10.0
    subsidence = 0.25 * jnp.exp(-sub_dist ** 2)

    stratocu = 0.30 * lts * jnp.clip((35.0 - abs_lat) / 20.0, 0.0, 1.0)

    storm_track = (
        0.25
        * jnp.clip((abs_lat - 35.0) / 10.0, 0.0, 1.0)
        * jnp.clip((80.0 - abs_lat) / 15.0, 0.0, 1.0)
    )

    so_cloud = jnp.where(
        lat[:, jnp.newaxis] < 0.0,
        0.35 * jnp.clip((abs_lat - 45.0) / 10.0, 0.0, 1.0),
        0.0,
    )

    polar_cloud = 0.15 * jnp.clip((abs_lat - 55.0) / 15.0, 0.0, 1.0)

    high_cloud = conv_cloud + warm_pool
    low_cloud = stratocu + storm_track + so_cloud + polar_cloud

    cloud_frac = jnp.clip(
        high_cloud + low_cloud - subsidence * (1.0 - humidity),
        0.05,
        0.85,
    )

    conv_frac = jnp.where(
        cloud_frac > 0.05,
        jnp.clip(high_cloud / (high_cloud + low_cloud + 0.01), 0.0, 1.0),
        0.0,
    )

    return cloud_frac, conv_frac


# ---------------------------------------------------------------------------
# 3. Net radiative heating
# ---------------------------------------------------------------------------

def radiation(
    T_s: jnp.ndarray,
    lat: jnp.ndarray,
    sim_time: float,
    cloud_frac: jnp.ndarray,
    conv_frac: jnp.ndarray,
    moisture: jnp.ndarray,
    *,
    S_solar: float,
    A_olr: float,
    B_olr: float,
    global_temp_offset: float,
    greenhouse_q: float,
    q_ref: float,
) -> jnp.ndarray:
    """Compute net radiative heating q_net = SW_down - OLR.

    Parameters
    ----------
    T_s : (ny, nx) surface temperature (deg C)
    lat : (ny,) latitude in degrees
    sim_time : scalar simulation time (period = 10)
    cloud_frac, conv_frac : (ny, nx) from cloud_fraction()
    moisture : (ny, nx) specific humidity, or None to use q_sat(T_s)
    S_solar, A_olr, B_olr, global_temp_offset, greenhouse_q, q_ref : params

    Returns
    -------
    q_net : (ny, nx) net radiative heating (W m-2 equivalent)
    """
    year_phase = 2.0 * jnp.pi * (sim_time % 10.0) / 10.0
    lat_rad = jnp.deg2rad(lat[:, jnp.newaxis])             # (ny, 1)
    abs_lat = jnp.abs(lat[:, jnp.newaxis])

    declination = jnp.deg2rad(23.44 * jnp.sin(year_phase))
    cos_zenith = (
        jnp.cos(lat_rad) * jnp.cos(declination)
        + jnp.sin(lat_rad) * jnp.sin(declination)
    )
    q_solar = S_solar * jnp.maximum(0.0, cos_zenith)

    # Ice-albedo feedback (poleward of 45 deg)
    ice_t = jnp.clip((T_s + 2.0) / 10.0, 0.0, 1.0)
    ice_frac = 1.0 - ice_t ** 2 * (3.0 - 2.0 * ice_t)     # smoothstep
    lat_ramp = jnp.clip((abs_lat - 45.0) / 20.0, 0.0, 1.0)
    q_solar = q_solar * (1.0 - 0.50 * ice_frac * lat_ramp)

    # Cloud albedo
    cloud_albedo = cloud_frac * (
        0.35 * (1.0 - conv_frac) + 0.20 * conv_frac
    )
    q_solar = q_solar * (1.0 - cloud_albedo)

    # Outgoing longwave radiation
    olr = A_olr - B_olr * global_temp_offset + B_olr * T_s

    # Cloud greenhouse effect
    cloud_gh = cloud_frac * (
        0.03 * (1.0 - conv_frac) + 0.12 * conv_frac
    )

    # Water-vapor greenhouse effect
    vapor_gh = jnp.where(
        moisture is not None,
        greenhouse_q * jnp.clip(moisture / q_ref, 0.0, 1.0),
        greenhouse_q * jnp.clip(0.8 * q_sat(T_s) / q_ref, 0.0, 1.0),
    )

    effective_olr = olr * (1.0 - cloud_gh) * (1.0 - vapor_gh)
    q_net = q_solar - effective_olr

    return q_net


# ---------------------------------------------------------------------------
# Helpers: 5-point Laplacian (periodic x, Neumann y)
# ---------------------------------------------------------------------------

def _laplacian(field: jnp.ndarray) -> jnp.ndarray:
    """5-point discrete Laplacian, periodic in x, Neumann (zero-flux) in y."""
    # Periodic in x
    left = jnp.roll(field, 1, axis=1)
    right = jnp.roll(field, -1, axis=1)

    # Neumann in y: replicate boundary rows
    up = jnp.concatenate([field[1:2, :], field[1:, :]], axis=0)
    down = jnp.concatenate([field[:-1, :], field[-2:-1, :]], axis=0)

    return left + right + up + down - 4.0 * field


# ---------------------------------------------------------------------------
# 4. Atmosphere time step
# ---------------------------------------------------------------------------

def atmosphere_step(
    air_temp: jnp.ndarray,
    moisture: jnp.ndarray,
    T_s: jnp.ndarray,
    ocean_mask: jnp.ndarray,
    land_temp: jnp.ndarray,
    lat: jnp.ndarray,
    sim_time: float,
    *,
    dt: float,
    kappa_atm: float,
    gamma_oa: float,
    gamma_la: float,
    gamma_ao: float,
    E0: float,
    greenhouse_q: float,
    q_ref: float,
    freshwater_scale_pe: float,
    latent_heat_coeff: float = 800.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advance atmosphere one time step.

    Parameters
    ----------
    air_temp : (ny, nx) atmospheric temperature (deg C)
    moisture : (ny, nx) specific humidity
    T_s : (ny, nx) ocean surface temperature
    ocean_mask : (ny, nx) 1 over ocean, 0 over land
    land_temp : (ny, nx) land surface temperature
    lat : (ny,) latitude in degrees
    sim_time : scalar
    dt, kappa_atm, gamma_oa, gamma_la, gamma_ao, E0 : physics params
    greenhouse_q, q_ref, freshwater_scale_pe : moisture/salinity params
    latent_heat_coeff : W per unit precip (default 800)

    Returns
    -------
    new_air_temp : (ny, nx)
    new_moisture : (ny, nx)
    precip_field : (ny, nx) condensation that fell as precipitation
    ocean_temp_feedback : (ny, nx) additive correction for T_s
    salinity_feedback : (ny, nx) additive correction for salinity
    """
    # --- diffusion ---
    air_diff = kappa_atm * _laplacian(air_temp)
    q_diff = kappa_atm * _laplacian(moisture)

    # --- surface exchange ---
    surf_t = jnp.where(ocean_mask > 0.5, T_s, land_temp)
    gamma = jnp.where(ocean_mask > 0.5, gamma_oa, gamma_la)
    exchange = gamma * (surf_t - air_temp)

    # --- evaporation (ocean only) ---
    evap = E0 * jnp.maximum(0.0, q_sat(surf_t) - moisture) * ocean_mask

    # --- update moisture ---
    q_new = moisture + dt * q_diff + evap

    # --- condensation ---
    q_sat_air = q_sat(air_temp)
    precip = jnp.maximum(0.0, q_new - q_sat_air)
    q_new = jnp.where(q_new > q_sat_air, q_sat_air, q_new)
    q_new = jnp.maximum(q_new, 1e-5)

    # --- latent heat ---
    latent_heat = latent_heat_coeff * precip

    # --- update air temperature ---
    air_new = air_temp + dt * (air_diff + exchange) + latent_heat

    # --- polar boundaries: copy from neighbour row ---
    air_new = air_new.at[0, :].set(air_new[1, :])
    air_new = air_new.at[-1, :].set(air_new[-2, :])
    q_new = q_new.at[0, :].set(q_new[1, :])
    q_new = q_new.at[-1, :].set(q_new[-2, :])

    # --- two-way feedback on ocean ---
    ocean_temp_fb = dt * gamma_ao * (air_new - T_s)

    # --- evaporative cooling of ocean ---
    # Scale by dt for consistency with radiation (which enters as dt*qNet)
    evap_cool = dt * E0 * jnp.maximum(0.0, q_sat(T_s) - q_new) * latent_heat_coeff
    ocean_temp_fb = ocean_temp_fb - evap_cool * ocean_mask

    # --- P-E salinity flux ---
    net_fw = precip - E0 * jnp.maximum(0.0, q_sat(T_s) - q_new)
    salinity_fb = -dt * freshwater_scale_pe * net_fw * ocean_mask

    return air_new, q_new, precip, ocean_temp_fb, salinity_fb


# ---------------------------------------------------------------------------
# 5. Mixed layer depth (latitude-dependent)
# ---------------------------------------------------------------------------

def mixed_layer_depth(lat: jnp.ndarray) -> jnp.ndarray:
    """Variable mixed-layer depth as a function of latitude.

    Parameters
    ----------
    lat : (ny,) latitude in degrees

    Returns
    -------
    mld : (ny,) mixed layer depth in metres
    """
    abs_lat = jnp.abs(lat)

    mld_base = 30.0 + 70.0 * (abs_lat / 80.0) ** 1.5

    # ACC enhancement (Southern Ocean, lat ~ -50)
    d_acc = (lat + 50.0) / 12.0
    mld_acc = jnp.where(
        (lat < -35.0) & (lat > -65.0),
        250.0 * jnp.exp(-d_acc ** 2),
        0.0,
    )

    # Subpolar North Atlantic enhancement (lat ~ 62)
    d_sub = (lat - 62.0) / 8.0
    mld_sub = jnp.where(
        (lat > 50.0) & (lat < 75.0),
        150.0 * jnp.exp(-d_sub ** 2),
        0.0,
    )

    return mld_base + mld_acc + mld_sub
