"""Load observational data and build forcing fields for the simulation.

Supports both:
- JSON fields (data/*.json with inline arrays)
- Binary format (data/bin/*.json metadata + .bin Float32 files)

All data files are at 1024x512, matching the model grid.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from .grid import Grid
from .state import Forcing, Params, SeasonalForcing, State


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _is_north_first(path: str | Path) -> bool:
    """Detect orientation: data/ files are north-first (image convention)."""
    return "/data/" in str(path).replace("\\", "/")


def load_bin_field(bin_dir: str | Path, dataset: str, field: str) -> np.ndarray | None:
    """Load a field from binary format: metadata JSON + .bin Float32 file."""
    bin_dir = Path(bin_dir)
    meta_path = bin_dir / f"{dataset}.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    if "arrays" not in meta or field not in meta["arrays"]:
        return None
    info = meta["arrays"][field]
    bin_path = bin_dir / info["file"]
    if not bin_path.exists():
        return None
    nx, ny = int(meta.get("nx", 1024)), int(meta.get("ny", 512))
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size != nx * ny:
        return None
    arr = raw.reshape(ny, nx)
    # Binary data is north-first (image convention) — flip to south-first
    return arr[::-1, :].copy()


def load_json_field(path: str | Path, field: str) -> tuple[np.ndarray, dict] | None:
    """Load a (ny, nx) array from a JSON file. Returns (array, metadata)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    nx, ny = int(d["nx"]), int(d["ny"])
    if field not in d:
        return None
    arr = np.array(d[field], dtype=np.float32).reshape(ny, nx)
    if _is_north_first(path):
        arr = arr[::-1, :].copy()
    return arr, d


def load_mask(path: str | Path) -> np.ndarray | None:
    """Load hex-packed mask. 1=ocean, 0=land. Returns south-first (ny, nx)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    nx, ny = int(d["nx"]), int(d["ny"])
    hex_str = d["hex"]
    nibbles = np.frombuffer(hex_str.encode("ascii"), dtype=np.uint8)
    is_letter = nibbles >= 97
    vals = np.where(is_letter, nibbles - 87, nibbles - 48).astype(np.uint8)
    bits = np.zeros(nx * ny, dtype=np.uint8)
    bits[0::4] = (vals >> 3) & 1
    bits[1::4] = (vals >> 2) & 1
    bits[2::4] = (vals >> 1) & 1
    bits[3::4] = vals & 1
    mask = bits.reshape(ny, nx).astype(np.float32)
    if _is_north_first(path):
        mask = mask[::-1, :].copy()
    # Ensure polar boundaries are land
    mask[0, :] = 0
    mask[-1, :] = 0
    return mask


def resample_to_grid(
    src: np.ndarray, *, src_lat: np.ndarray, src_lon: np.ndarray, grid: Grid,
) -> jnp.ndarray:
    """Bilinear interpolation from source to model grid."""
    ny_src, nx_src = src.shape
    tgt_lat = np.asarray(grid.lat)
    fj = (tgt_lat - src_lat[0]) / (src_lat[-1] - src_lat[0]) * (ny_src - 1)
    fj = np.clip(fj, 0, ny_src - 1 - 1e-9)
    j0 = np.floor(fj).astype(int)
    j1 = np.minimum(j0 + 1, ny_src - 1)
    tj = (fj - j0).astype(np.float32)

    tgt_lon = np.asarray(grid.lon)
    dlon = (src_lon[-1] - src_lon[0]) / max(nx_src - 1, 1)
    fi = (tgt_lon - src_lon[0]) / dlon
    fi = np.mod(fi, nx_src)
    i0 = np.floor(fi).astype(int) % nx_src
    i1 = (i0 + 1) % nx_src
    ti = (fi - np.floor(fi)).astype(np.float32)

    s00 = src[j0[:, None], i0[None, :]]
    s01 = src[j0[:, None], i1[None, :]]
    s10 = src[j1[:, None], i0[None, :]]
    s11 = src[j1[:, None], i1[None, :]]

    out = (s00 * (1 - tj[:, None]) * (1 - ti[None, :])
           + s01 * (1 - tj[:, None]) * ti[None, :]
           + s10 * tj[:, None] * (1 - ti[None, :])
           + s11 * tj[:, None] * ti[None, :])
    return jnp.asarray(out)


# ---------------------------------------------------------------------------
# High-level: build everything from data directory
# ---------------------------------------------------------------------------

def _load_field_bin_or_json(data_dir: Path, dataset: str, field: str,
                             grid: Grid) -> jnp.ndarray | None:
    """Try binary first, then JSON."""
    bin_dir = data_dir / "bin"
    arr = load_bin_field(bin_dir, dataset, field)
    if arr is not None:
        # Data is already 1024x512 matching grid — check if resample needed
        if arr.shape == (grid.ny, grid.nx):
            return jnp.asarray(arr)
        # Resample
        src_lat = np.linspace(-79.5, 79.5, arr.shape[0])
        src_lon = np.linspace(-180, 180, arr.shape[1], endpoint=False)
        return resample_to_grid(arr, src_lat=src_lat, src_lon=src_lon, grid=grid)

    # Try JSON
    json_path = data_dir / f"{dataset}.json"
    result = load_json_field(json_path, field)
    if result is not None:
        arr, meta = result
        if arr.shape == (grid.ny, grid.nx):
            return jnp.asarray(arr)
        src_lat = np.linspace(float(meta.get("lat0", -79.5)),
                              float(meta.get("lat1", 79.5)), arr.shape[0])
        src_lon = np.linspace(float(meta.get("lon0", -180)),
                              float(meta.get("lon1", 180)), arr.shape[1])
        return resample_to_grid(arr, src_lat=src_lat, src_lon=src_lon, grid=grid)
    return None


def build_wind_curl(data_dir: Path, grid: Grid) -> jnp.ndarray:
    """Load wind curl and RMS-scale to model units.

    Uses zonal-mean RMS matching (like the JS version) to avoid
    amplifying local features when scaling from physical to model units.
    """
    obs_curl = _load_field_bin_or_json(data_dir, "wind_stress", "wind_curl", grid)

    # Analytical fallback
    lat_rad = jnp.deg2rad(grid.lat)[:, None]
    lat_b = grid.lat[:, None]
    sh_boost = jnp.where(lat_b < 0, 2.0, 1.0)
    polar_damp = jnp.where(jnp.abs(lat_b) > 60, 0.7, 1.0)
    analytical = -jnp.cos(3 * lat_rad) * sh_boost * polar_damp * 2.0
    analytical_2d = jnp.broadcast_to(analytical, (grid.ny, grid.nx))

    if obs_curl is None:
        return analytical_2d

    # Zonal-mean RMS matching (same approach as JS generateWindCurlField)
    # Compare zonal means of observed vs analytical at each latitude
    valid = jnp.abs(grid.lat) < 75  # (ny,)
    # Zonal mean of observed (only non-zero cells)
    obs_nonzero = jnp.where(obs_curl != 0, obs_curl, 0.0)
    obs_count = jnp.sum(obs_curl != 0, axis=1)  # (ny,)
    obs_zonal_mean = jnp.where(obs_count > 10,
                                jnp.sum(obs_nonzero, axis=1) / jnp.maximum(obs_count, 1),
                                0.0)  # (ny,)
    # Analytical zonal mean (constant per lat)
    anal_zonal_mean = analytical[:, 0]  # (ny,)

    # RMS of zonal means
    rms_obs2 = jnp.sum(jnp.where(valid, obs_zonal_mean ** 2, 0.0))
    rms_anal2 = jnp.sum(jnp.where(valid, anal_zonal_mean ** 2, 0.0))
    scale = jnp.sqrt(rms_anal2 / jnp.maximum(rms_obs2, 1e-30))

    return obs_curl * scale


def build_ekman(data_dir: Path, grid: Grid) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build Ekman velocity fields from wind stress components."""
    OMEGA = 7.292e-5
    RHO = 1025.0
    H_EK = 50.0

    tau_x = _load_field_bin_or_json(data_dir, "wind_stress", "tau_x", grid)
    tau_y = _load_field_bin_or_json(data_dir, "wind_stress", "tau_y", grid)

    if tau_x is None or tau_y is None:
        # Analytical fallback
        lat_rad = jnp.deg2rad(grid.lat)
        sin_lat = jnp.sin(lat_rad)
        sin_lat = jnp.where(jnp.abs(grid.lat) < 5,
                            jnp.sin(jnp.deg2rad(5.0)) * jnp.sign(grid.lat + 1e-10),
                            sin_lat)
        ve_raw = jnp.cos(3 * lat_rad) / sin_lat
        u_ek = jnp.zeros((grid.ny, grid.nx))
        v_ek = jnp.broadcast_to(ve_raw[:, None] * 0.15, (grid.ny, grid.nx))
        return u_ek, v_ek

    lat_rad = jnp.deg2rad(grid.lat)[:, None]
    f = 2 * OMEGA * jnp.sin(lat_rad)
    f = jnp.where(jnp.abs(grid.lat[:, None]) < 5,
                   2 * OMEGA * jnp.sin(jnp.deg2rad(5.0)) * jnp.sign(grid.lat[:, None] + 1e-10),
                   f)

    u_ek_phys = tau_y / (RHO * f * H_EK)
    v_ek_phys = -tau_x / (RHO * f * H_EK)

    # RMS-scale to nondimensional
    rms = jnp.sqrt(jnp.mean(
        jnp.where(jnp.abs(grid.lat[:, None]) > 5,
                   u_ek_phys ** 2 + v_ek_phys ** 2, 0.0)
    ))
    target_rms = 0.3
    scale = jnp.where(rms > 0, target_rms / rms, 0.0)

    return u_ek_phys * scale, v_ek_phys * scale


def _load_or_fallback(data_dir: Path, dataset: str, field: str,
                      grid: Grid, fallback_val: float) -> jnp.ndarray:
    """Load a field or return a uniform fallback."""
    arr = _load_field_bin_or_json(data_dir, dataset, field, grid)
    if arr is not None:
        return arr
    return jnp.full((grid.ny, grid.nx), fallback_val)


def build_forcing(data_dir: str | Path, grid: Grid) -> Forcing:
    """Build complete Forcing from data directory."""
    data_dir = Path(data_dir)
    shape = (grid.ny, grid.nx)

    # Mask
    mask_arr = load_mask(data_dir / "mask.json")
    if mask_arr is None:
        mask_arr = load_mask(data_dir / "bin" / "mask.json")
    if mask_arr is None:
        mask_arr = np.ones(shape, dtype=np.float32)
        mask_arr[0, :] = 0
        mask_arr[-1, :] = 0
    if mask_arr.shape != shape:
        src_lat = np.linspace(-79.5, 79.5, mask_arr.shape[0])
        src_lon = np.linspace(-180, 180, mask_arr.shape[1], endpoint=False)
        mask_j = np.asarray(resample_to_grid(
            mask_arr, src_lat=src_lat, src_lon=src_lon, grid=grid))
        mask_arr = (mask_j >= 0.5).astype(np.float32)
    ocean_mask = jnp.asarray(mask_arr)

    # Wind curl
    wind_curl = build_wind_curl(data_dir, grid)

    # Salinity climatology
    sal_clim = _load_field_bin_or_json(data_dir, "salinity", "salinity", grid)
    if sal_clim is None:
        lat_rad = jnp.deg2rad(grid.lat)[:, None]
        sal_clim = 34.0 + 2.0 * jnp.cos(2 * lat_rad) - 0.5 * jnp.cos(4 * lat_rad)
        sal_clim = jnp.broadcast_to(sal_clim, shape)

    # Ekman
    ekman_u, ekman_v = build_ekman(data_dir, grid)

    # Depth (bathymetry)
    depth = _load_field_bin_or_json(data_dir, "bathymetry", "depth", grid)
    if depth is None:
        depth = jnp.full(shape, 4000.0)
    else:
        depth = jnp.where(ocean_mask > 0.5,
                          jnp.clip(jnp.where(depth > 0, depth, 200.0), 50.0, 5500.0),
                          0.0)

    # Land temperature
    land_temp = _load_field_bin_or_json(data_dir, "land_surface_temp", "lst", grid)
    if land_temp is None:
        abs_lat = jnp.abs(grid.lat)[:, None]
        land_temp = 28.0 - 0.55 * abs_lat
        land_temp = jnp.broadcast_to(land_temp, shape)

    # SST restoring target (observed climatology)
    T_target = _load_field_bin_or_json(data_dir, "sst", "sst", grid)
    if T_target is None:
        abs_lat = jnp.abs(grid.lat)[:, None]
        T_target = 28.0 - 0.55 * abs_lat - 0.0003 * abs_lat ** 2
        T_target = jnp.broadcast_to(jnp.clip(T_target, -2, 30), shape)
    T_target = jnp.where(ocean_mask > 0.5, T_target, 0.0)

    # Deep T restoring target
    T_deep_target = _load_field_bin_or_json(data_dir, "deep_temp", "temp", grid)
    if T_deep_target is None:
        y_frac = jnp.arange(grid.ny)[:, None] / max(grid.ny - 1, 1)
        T_deep_target = 0.5 + 3.0 * y_frac
        T_deep_target = jnp.broadcast_to(T_deep_target, shape)
    T_deep_target = jnp.where(ocean_mask > 0.5, T_deep_target, 0.0)

    # --- Observed fields for physics alignment ---

    # Mixed layer depth (observed)
    obs_mld = _load_field_bin_or_json(data_dir, "mixed_layer_depth", "mld", grid)
    if obs_mld is None:
        obs_mld = jnp.full(shape, 50.0)
    else:
        obs_mld = jnp.clip(obs_mld, 10.0, 500.0)

    # Cloud fraction (MODIS)
    obs_cloud = _load_field_bin_or_json(data_dir, "cloud_fraction", "cloud_fraction", grid)
    if obs_cloud is None:
        obs_cloud = jnp.full(shape, 0.5)
    else:
        obs_cloud = jnp.clip(obs_cloud, 0.0, 1.0)

    # Surface albedo
    obs_albedo = _load_field_bin_or_json(data_dir, "albedo", "albedo", grid)
    if obs_albedo is None:
        obs_albedo = jnp.full(shape, 0.06)
    else:
        obs_albedo = jnp.clip(obs_albedo, 0.0, 1.0)

    # Sea ice fraction
    obs_sea_ice = _load_field_bin_or_json(data_dir, "sea_ice", "ice_fraction", grid)
    if obs_sea_ice is None:
        obs_sea_ice = jnp.zeros(shape)
    else:
        obs_sea_ice = jnp.clip(obs_sea_ice, 0.0, 1.0)

    # Precipitation (mm/yr)
    obs_precip = _load_or_fallback(data_dir, "precipitation", "precipitation", grid, 0.0)

    # Evaporation
    obs_evap = _load_or_fallback(data_dir, "evaporation", "evaporation", grid, 0.0)

    # Column water vapor (normalized 0-1)
    obs_wv = _load_field_bin_or_json(data_dir, "column_water_vapor", "water_vapor", grid)
    if obs_wv is None:
        # Try the derived humidity field
        obs_wv = _load_field_bin_or_json(data_dir, "water_vapor", "humidity", grid)
    if obs_wv is None:
        obs_wv = jnp.full(shape, 0.5)
    else:
        obs_wv = jnp.clip(obs_wv, 0.0, 1.0)

    # Ocean currents (GODAS or HYCOM)
    obs_u = _load_field_bin_or_json(data_dir, "hycom_surface_currents", "u", grid)
    obs_v = _load_field_bin_or_json(data_dir, "hycom_surface_currents", "v", grid)
    if obs_u is None:
        obs_u = _load_or_fallback(data_dir, "ocean_currents", "u", grid, 0.0)
    if obs_v is None:
        obs_v = _load_or_fallback(data_dir, "ocean_currents", "v", grid, 0.0)

    # NDVI
    obs_ndvi = _load_or_fallback(data_dir, "ndvi", "ndvi", grid, 0.0)

    # Snow cover
    obs_snow = _load_or_fallback(data_dir, "snow_cover", "snow_cover", grid, 0.0)

    # Chlorophyll
    obs_chlorophyll = _load_or_fallback(data_dir, "chlorophyll", "chlor_a", grid, 0.0)

    # Surface pressure
    obs_pressure = _load_or_fallback(data_dir, "surface_pressure", "pressure", grid, 1013.0)

    return Forcing(
        wind_curl=wind_curl,
        ocean_mask=ocean_mask,
        T_target=T_target,
        T_deep_target=T_deep_target,
        sal_climatology=sal_clim,
        ekman_u=ekman_u,
        ekman_v=ekman_v,
        depth_field=depth,
        land_temp=land_temp,
        obs_mld=obs_mld,
        obs_cloud=obs_cloud,
        obs_albedo=obs_albedo,
        obs_sea_ice=obs_sea_ice,
        obs_precip=obs_precip,
        obs_evap=obs_evap,
        obs_water_vapor=obs_wv,
        obs_u=obs_u,
        obs_v=obs_v,
        obs_ndvi=obs_ndvi,
        obs_snow=obs_snow,
        obs_chlorophyll=obs_chlorophyll,
        obs_pressure=obs_pressure,
    )


def build_initial_state(data_dir: str | Path, grid: Grid,
                        forcing: Forcing) -> State:
    """Build initial state from observational data."""
    data_dir = Path(data_dir)
    mask = forcing.ocean_mask

    # SST
    sst = _load_field_bin_or_json(data_dir, "sst", "sst", grid)
    if sst is None:
        abs_lat = jnp.abs(grid.lat)[:, None]
        sst = 28.0 - 0.55 * abs_lat - 0.0003 * abs_lat ** 2
        sst = jnp.broadcast_to(jnp.clip(sst, -2, 30), (grid.ny, grid.nx))
    sst = jnp.where(mask > 0.5, sst, 0.0)

    # Deep temperature
    deep_t = _load_field_bin_or_json(data_dir, "deep_temp", "temp", grid)
    if deep_t is None:
        y_frac = jnp.arange(grid.ny)[:, None] / max(grid.ny - 1, 1)
        deep_t = 0.5 + 3.0 * y_frac
        deep_t = jnp.broadcast_to(deep_t, (grid.ny, grid.nx))
    deep_t = jnp.where(mask > 0.5, deep_t, 0.0)

    # Salinity (surface and deep)
    lat_rad = jnp.deg2rad(grid.lat)[:, None]
    sal_s = 34.0 + 2.0 * jnp.cos(2 * lat_rad) - 0.5 * jnp.cos(4 * lat_rad)
    sal_s = jnp.broadcast_to(sal_s, (grid.ny, grid.nx))
    sal_s = jnp.where(mask > 0.5, sal_s, 0.0)

    sal_d = 34.7 + 0.2 * jnp.cos(2 * lat_rad)
    sal_d = jnp.broadcast_to(sal_d, (grid.ny, grid.nx))
    sal_d = jnp.where(mask > 0.5, sal_d, 0.0)

    # Air temperature
    air_temp = _load_field_bin_or_json(data_dir, "air_temp", "air_temp", grid)
    if air_temp is None:
        air_temp = jnp.where(mask > 0.5, sst,
                             28.0 - 0.55 * jnp.abs(grid.lat)[:, None])
        air_temp = jnp.broadcast_to(air_temp, (grid.ny, grid.nx))

    # Moisture: use observed water vapor if available, else 80% of saturation
    obs_wv = _load_field_bin_or_json(data_dir, "water_vapor", "humidity", grid)
    if obs_wv is not None:
        # Observed humidity field is already in physical units (kg/kg-like)
        moisture = jnp.clip(jnp.asarray(obs_wv), 1e-5, 0.03)
    else:
        q_sat_val = 3.75e-3 * jnp.exp(0.067 * air_temp)
        moisture = 0.80 * q_sat_val

    # Initialize ice from observed sea ice (if available in forcing)
    ice_frac = jnp.clip(forcing.obs_sea_ice, 0.0, 1.0)

    # Start with zero circulation (will spin up from forcing)
    z = jnp.zeros((grid.ny, grid.nx))

    return State(
        psi_s=z, zeta_s=z, psi_d=z, zeta_d=z,
        T_s=sst, S_s=sal_s, T_d=deep_t, S_d=sal_d,
        air_temp=air_temp, moisture=moisture,
        ice_frac=ice_frac,
        sim_time=0.0,
    )


def _load_monthly_json(path: Path, field_key: str, grid: Grid) -> jnp.ndarray | None:
    """Load a (12, ny, nx) monthly climatology from JSON.

    JSON format: {nx, ny, monthly: [[month0_flat], [month1_flat], ...]}
    Data is north-first; we flip to south-first and resample to grid.
    """
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    monthly_data = d.get(field_key)
    if monthly_data is None or len(monthly_data) != 12:
        return None
    nx_src, ny_src = int(d["nx"]), int(d["ny"])
    src_lat = np.linspace(-79.5, 79.5, ny_src)
    src_lon = np.linspace(-180, 180, nx_src, endpoint=False)

    months = []
    for m in range(12):
        arr = np.array(monthly_data[m], dtype=np.float32).reshape(ny_src, nx_src)
        # Flip north-first to south-first
        arr = arr[::-1, :].copy()
        if arr.shape == (grid.ny, grid.nx):
            months.append(jnp.asarray(arr))
        else:
            months.append(resample_to_grid(arr, src_lat=src_lat, src_lon=src_lon, grid=grid))
    return jnp.stack(months, axis=0)  # (12, ny, nx)


def build_seasonal_forcing(data_dir: str | Path, grid: Grid) -> SeasonalForcing:
    """Load monthly climatologies for seasonal forcing cycle."""
    data_dir = Path(data_dir)
    shape12 = (12, grid.ny, grid.nx)

    # SST monthly
    sst_monthly = _load_monthly_json(data_dir / "sst_monthly.json", "monthly", grid)
    if sst_monthly is None:
        sst_monthly = jnp.zeros(shape12)

    # Wind stress monthly (tau_x, tau_y)
    tau_x_monthly = _load_monthly_json(data_dir / "wind_stress_monthly.json", "monthly_tau_x", grid)
    tau_y_monthly = _load_monthly_json(data_dir / "wind_stress_monthly.json", "monthly_tau_y", grid)
    if tau_x_monthly is None:
        tau_x_monthly = jnp.zeros(shape12)
    if tau_y_monthly is None:
        tau_y_monthly = jnp.zeros(shape12)

    # Albedo monthly
    albedo_monthly = _load_monthly_json(data_dir / "albedo_monthly.json", "monthly", grid)
    if albedo_monthly is None:
        albedo_monthly = jnp.zeros(shape12)

    has_data = bool(jnp.any(sst_monthly != 0))

    return SeasonalForcing(
        sst_monthly=sst_monthly,
        tau_x_monthly=tau_x_monthly,
        tau_y_monthly=tau_y_monthly,
        albedo_monthly=albedo_monthly,
        has_data=has_data,
    )
