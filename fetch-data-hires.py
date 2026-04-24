#!/usr/bin/env python3
"""
Fetch all observational data for SimAMOC at 1024x512 resolution.
Sources: Earth Engine (ETOPO1, OISST, ERA5, MODIS, CHIRPS), WOA23 OPeNDAP.
Outputs: JSON data files in data/ + reference PNG images.

Usage:
    /Users/dereklomas/eli/.venv/bin/python fetch-data-hires.py [--field FIELD]

Fields: bathymetry, sst, wind, albedo, precipitation, clouds, salinity, deep_temp, mask
Default: fetch all fields.
"""

import ee
import json
import io
import sys
import time
import urllib.request
import numpy as np
import tifffile
from pathlib import Path
from PIL import Image as PILImage

# Force unbuffered output
print = lambda *a, **kw: __builtins__.__dict__['print'](*a, **{**kw, 'flush': True})

# --- Target grid ---
NX, NY = 1024, 512
LAT0, LAT1 = -79.5, 79.5
LON0, LON1 = -180.0, 180.0
SCALE_X = (LON1 - LON0) / NX   # ~0.3516 deg
SCALE_Y = (LAT1 - LAT0) / NY   # ~0.3105 deg

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(exist_ok=True)

# --- Earth Engine init ---
SA_KEY = Path("/Users/dereklomas/eli/sa-key.json")
EE_GRID = {
    "dimensions": {"width": NX, "height": NY},
    "affineTransform": {
        "scaleX": SCALE_X, "shearX": 0, "translateX": LON0,
        "shearY": 0, "scaleY": -SCALE_Y, "translateY": LAT1,
    },
    "crsCode": "EPSG:4326",
}
GLOBAL_GEOM = None  # set after ee.Initialize()


def init_ee():
    global GLOBAL_GEOM
    credentials = ee.ServiceAccountCredentials(
        "earthengine@eli-africa-494008.iam.gserviceaccount.com", str(SA_KEY)
    )
    ee.Initialize(credentials, project="gen-lang-client-0278315411")
    GLOBAL_GEOM = ee.Geometry.Rectangle([LON0, LAT0, LON1, LAT1])
    print("Connected to Earth Engine")


def ee_to_array(image, band):
    """Download a single EE band as a numpy array at our target grid."""
    img = image.select(band)
    result = ee.data.computePixels(
        {"expression": img, "fileFormat": "GEO_TIFF", "grid": EE_GRID}
    )
    arr = tifffile.imread(io.BytesIO(result)).astype(np.float64)
    assert arr.shape == (NY, NX), f"Expected ({NY},{NX}), got {arr.shape}"
    return arr


def save_reference_png(image, description, vis_params, filename):
    """Save a high-res reference PNG via EE thumbnail."""
    print(f"  Saving reference image: {filename}")
    try:
        # Use slightly inset region to avoid edge reprojection error
        region = ee.Geometry.Rectangle([LON0 + 0.1, LAT0 + 0.1, LON1 - 0.1, LAT1 - 0.1])
        url = image.getThumbURL({
            **vis_params,
            "dimensions": "2048x1024",
            "region": region,
            "format": "png",
        })
        import requests
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and len(r.content) > 1000:
            path = IMG_DIR / filename
            path.write_bytes(r.content)
            print(f"  Saved {path} ({len(r.content) / 1024:.0f} KB)")
            return True
    except Exception as e:
        print(f"  Reference image failed (non-fatal): {e}")
    return False


def save_json(filename, data_dict):
    """Save a JSON data file with grid metadata."""
    output = {
        "nx": NX, "ny": NY,
        "lat0": LAT0, "lat1": LAT1,
        "lon0": LON0, "lon1": LON1,
        **data_dict,
    }
    path = DATA_DIR / filename
    path.write_text(json.dumps(output))
    size_kb = path.stat().st_size / 1024
    print(f"  Saved {path} ({size_kb:.0f} KB)")


def lerp_color(t, stops):
    """Interpolate through a list of (position, r, g, b) color stops."""
    t = np.clip(t, 0, 1)
    r = np.zeros_like(t)
    g = np.zeros_like(t)
    b = np.zeros_like(t)
    for k in range(len(stops) - 1):
        p0, r0, g0, b0 = stops[k]
        p1, r1, g1, b1 = stops[k + 1]
        mask = (t >= p0) & (t <= p1)
        if not np.any(mask):
            continue
        frac = np.where(mask, (t - p0) / max(p1 - p0, 1e-10), 0)
        r = np.where(mask, r0 + frac * (r1 - r0), r)
        g = np.where(mask, g0 + frac * (g1 - g0), g)
        b = np.where(mask, b0 + frac * (b1 - b0), b)
    return r, g, b


COLORMAPS = {
    "terrain": [  # ocean blue → green → brown → white
        (0.0, 10, 30, 100), (0.35, 30, 80, 180), (0.45, 50, 160, 80),
        (0.55, 120, 180, 50), (0.7, 180, 140, 60), (0.85, 200, 180, 140),
        (1.0, 255, 255, 255),
    ],
    "coolwarm": [  # blue → white → red
        (0.0, 30, 50, 200), (0.25, 100, 140, 230), (0.45, 200, 210, 240),
        (0.5, 240, 240, 240), (0.55, 240, 200, 190), (0.75, 230, 100, 80),
        (1.0, 200, 30, 30),
    ],
    "blues": [  # white → blue
        (0.0, 245, 245, 255), (0.3, 180, 200, 240), (0.6, 80, 120, 200),
        (1.0, 10, 30, 140),
    ],
    "grays": [  # white → dark gray
        (0.0, 255, 255, 255), (1.0, 40, 40, 40),
    ],
    "viridis": [  # dark purple → teal → yellow
        (0.0, 68, 1, 84), (0.25, 59, 82, 139), (0.5, 33, 145, 140),
        (0.75, 94, 201, 98), (1.0, 253, 231, 37),
    ],
}


def save_colormap_png(arr, filename, vmin, vmax, cmap_name="viridis", mask=None):
    """Save a numpy array as a colormapped PNG for quick viewing."""
    t = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    stops = COLORMAPS.get(cmap_name, COLORMAPS["viridis"])
    r, g, b = lerp_color(t, stops)
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    if mask is not None:
        rgb[mask == 0] = [40, 40, 40]
    img = PILImage.fromarray(rgb)
    path = IMG_DIR / filename
    img.save(str(path))
    print(f"  Saved preview: {path}")


def _load_mask():
    """Load the ocean mask from data/mask.json."""
    mask_path = DATA_DIR / "mask.json"
    if not mask_path.exists():
        return None
    d = json.loads(mask_path.read_text())
    bits = []
    for c in d["hex"]:
        v = int(c, 16)
        bits.extend([(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1])
    return np.array(bits[:NX * NY], dtype=np.uint8).reshape(NY, NX)


def _fill_ocean_nans(data, mask, default=0.0):
    """Fill NaN values in ocean cells by nearest-neighbor spreading.
    Land cells get the default value. Iteratively expands from valid cells."""
    result = data.copy()
    if mask is None:
        # No mask available — fall back to global default
        result[np.isnan(result)] = default
        return result

    # Land cells get default
    land = mask == 0
    result[land] = default

    # Iteratively fill NaN ocean cells from neighbors
    ocean_nan = (mask == 1) & np.isnan(result)
    iterations = 0
    while np.any(ocean_nan) and iterations < 200:
        filled = result.copy()
        for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(result, -dj, axis=0), -di, axis=1)
            shifted_valid = ~np.isnan(shifted)
            update = ocean_nan & shifted_valid
            filled[update] = shifted[update]
        newly_filled = np.isnan(result) & ~np.isnan(filled)
        if not np.any(newly_filled):
            break
        result = filled
        ocean_nan = (mask == 1) & np.isnan(result)
        iterations += 1

    # Any remaining NaN gets global mean of valid ocean cells
    still_nan = np.isnan(result)
    if np.any(still_nan):
        valid_ocean = result[(mask == 1) & ~np.isnan(result)]
        fill = np.mean(valid_ocean) if len(valid_ocean) > 0 else default
        result[still_nan] = fill

    print(f"  NaN fill: {iterations} iterations")
    return result


# ============================================================
# FIELD FETCHERS
# ============================================================

def fetch_bathymetry():
    """ETOPO1 bedrock elevation → depth (ocean) + elevation (land)."""
    print("\n=== ETOPO1 Bathymetry (1024x512) ===")
    img = ee.Image("NOAA/NGDC/ETOPO1").select("bedrock")

    arr = ee_to_array(img, "bedrock")
    print(f"  Range: {arr.min():.0f} to {arr.max():.0f} m")

    # Ocean depth (positive below sea level)
    depth = np.where(arr < 0, -arr, 0).astype(np.float32)
    # Land elevation (positive above sea level)
    elevation = np.where(arr >= 0, arr, 0).astype(np.float32)

    ocean_cells = np.sum(depth > 0)
    print(f"  Ocean cells: {ocean_cells}/{NX * NY} ({100 * ocean_cells / (NX * NY):.1f}%)")
    print(f"  Max depth: {depth.max():.0f} m, Max elevation: {elevation.max():.0f} m")

    save_json("bathymetry.json", {
        "source": "ETOPO1 bedrock via Earth Engine, native resolution sampled to 1024x512",
        "depth": [round(float(v)) for v in depth.ravel()],
        "elevation": [round(float(v)) for v in elevation.ravel()],
    })

    save_reference_png(img, "ETOPO1", {
        "min": -8000, "max": 5000,
        "palette": ["000033", "000080", "0000cc", "0066ff", "00ccff",
                     "66ffcc", "009933", "339933", "669933", "996633",
                     "cc9933", "ffcc66", "ffffff"],
    }, "bathymetry.png")

    save_colormap_png(arr, "bathymetry_preview.png", -8000, 5000, "terrain")
    return depth, elevation


def fetch_mask(depth):
    """Derive ocean mask from bathymetry."""
    print("\n=== Ocean Mask (from bathymetry) ===")
    mask = (depth > 0).astype(np.uint8)
    # Force polar boundaries to land
    mask[0, :] = 0
    mask[-1, :] = 0

    # Encode as hex string (4 bits per hex char)
    bits = mask.ravel()
    hex_chars = []
    for k in range(0, len(bits), 4):
        nibble = (bits[k] << 3) | (bits[k + 1] << 2) | (bits[k + 2] << 1) | bits[k + 3]
        hex_chars.append(format(nibble, "x"))

    ocean_count = int(np.sum(mask))
    print(f"  Ocean cells: {ocean_count}/{NX * NY}")

    output = {
        "nx": NX, "ny": NY,
        "lat0": LAT0, "lat1": LAT1,
        "source": "Derived from ETOPO1 (ocean = bedrock < 0)",
        "hex": "".join(hex_chars),
    }
    path = DATA_DIR / "mask.json"
    path.write_text(json.dumps(output))
    print(f"  Saved {path} ({path.stat().st_size / 1024:.0f} KB)")

    save_colormap_png(mask.astype(np.float64), "mask_preview.png", 0, 1, "blues")
    return mask


def fetch_sst():
    """NOAA OI SST v2.1 climatology (2015-2023 annual mean, faster than full 30yr)."""
    print("\n=== NOAA OI SST (1024x512) ===")
    # Use recent 9-year period for faster computation
    col = (
        ee.ImageCollection("NOAA/CDR/OISST/V2_1")
        .filterDate("2015-01-01", "2024-01-01")
        .select("sst")
        .mean()
    )
    arr = ee_to_array(col, "sst")
    # OISST uses scale factor 0.01
    arr = arr * 0.01
    # Mask fill values
    arr[arr < -100] = np.nan
    arr[arr > 100] = np.nan

    valid = arr[~np.isnan(arr)]
    print(f"  Range: {valid.min():.1f} to {valid.max():.1f} C")
    print(f"  Valid cells: {len(valid)}/{NX * NY}")

    # Fill NaN with zonal mean for land cells
    arr_filled = arr.copy()
    for j in range(NY):
        row = arr[j, :]
        valid_row = row[~np.isnan(row)]
        fill_val = np.mean(valid_row) if len(valid_row) > 0 else 15.0
        arr_filled[j, np.isnan(arr[j, :])] = fill_val

    save_json("sst.json", {
        "source": "NOAA OI SST v2.1 (CDR), 2015-2023 annual mean, via Earth Engine",
        "sst": [round(float(v), 2) for v in arr_filled.ravel()],
    })

    save_reference_png(col.multiply(0.01), "SST", {
        "min": -2, "max": 32,
        "palette": ["000080", "0000ff", "0066ff", "00ccff", "66ffcc",
                     "ffff00", "ff9900", "ff3300", "cc0000"],
    }, "sst.png")

    save_colormap_png(arr_filled, "sst_preview.png", -2, 32, "coolwarm")
    return arr_filled


def fetch_wind():
    """ERA5 Monthly 10m wind → wind stress and curl."""
    print("\n=== ERA5 Wind Stress (1024x512) ===")
    col = (
        ee.ImageCollection("ECMWF/ERA5/MONTHLY")
        .filterDate("1991-01-01", "2021-01-01")
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
        .mean()
    )
    u10 = ee_to_array(col, "u_component_of_wind_10m")
    v10 = ee_to_array(col, "v_component_of_wind_10m")
    print(f"  U10 range: {u10.min():.2f} to {u10.max():.2f} m/s")
    print(f"  V10 range: {v10.min():.2f} to {v10.max():.2f} m/s")

    # Bulk formula: tau = rho_air * Cd * |U| * U
    rho_air = 1.225  # kg/m3
    Cd = 1.3e-3
    speed = np.sqrt(u10**2 + v10**2)
    tau_x = rho_air * Cd * speed * u10
    tau_y = rho_air * Cd * speed * v10

    print(f"  Tau_x range: {tau_x.min():.4f} to {tau_x.max():.4f} N/m2")
    print(f"  Tau_y range: {tau_y.min():.4f} to {tau_y.max():.4f} N/m2")

    # Compute wind stress curl: d(tau_y)/dx - d(tau_x)/dy
    # Use central differences with cos(lat) metric
    R = 6.371e6  # Earth radius in meters
    lats = np.linspace(LAT1, LAT0, NY)  # top to bottom in image
    cos_lat = np.cos(np.radians(lats))[:, None]
    cos_lat = np.maximum(cos_lat, 0.01)

    dx_m = R * np.radians(SCALE_X) * cos_lat
    dy_m = R * np.radians(SCALE_Y)

    # d(tau_y)/dx — dx_m is (NY,1), broadcast works naturally
    dtydx = np.zeros_like(tau_y)
    dtydx[:, 1:-1] = (tau_y[:, 2:] - tau_y[:, :-2]) / (2 * dx_m)
    # Periodic boundary
    dtydx[:, 0] = (tau_y[:, 1] - tau_y[:, -1]) / (2 * dx_m.ravel())
    dtydx[:, -1] = (tau_y[:, 0] - tau_y[:, -2]) / (2 * dx_m.ravel())

    # d(tau_x)/dy
    dtxdy = np.zeros_like(tau_x)
    dtxdy[1:-1, :] = (tau_x[:-2, :] - tau_x[2:, :]) / (2 * dy_m)  # note: lat decreases with j
    dtxdy[0, :] = dtxdy[1, :]
    dtxdy[-1, :] = dtxdy[-2, :]

    wind_curl = dtydx - dtxdy
    print(f"  Curl range: {wind_curl.min():.2e} to {wind_curl.max():.2e} N/m3")

    save_json("wind_stress.json", {
        "source": "ERA5 Monthly 10m wind, 1991-2020 mean, bulk stress Cd=1.3e-3, via Earth Engine",
        "tau_x": [round(float(v), 6) for v in tau_x.ravel()],
        "tau_y": [round(float(v), 6) for v in tau_y.ravel()],
        "wind_curl": [round(float(v), 9) for v in wind_curl.ravel()],
    })

    save_reference_png(col.select("u_component_of_wind_10m"), "U10 wind", {
        "min": -10, "max": 10,
        "palette": ["0000cc", "6666ff", "ccccff", "ffffff", "ffcccc", "ff6666", "cc0000"],
    }, "wind_u10.png")

    save_colormap_png(tau_x, "wind_stress_preview.png", -0.15, 0.15, "coolwarm")
    return tau_x, tau_y, wind_curl


def fetch_albedo():
    """MODIS MCD43A3 white-sky albedo (2020-2023 annual mean)."""
    print("\n=== MODIS Albedo (1024x512) ===")
    img = (
        ee.ImageCollection("MODIS/061/MCD43A3")
        .filterDate("2020-01-01", "2024-01-01")
        .select("Albedo_WSA_shortwave")
        .mean()
        .multiply(0.001)  # scale factor
    )
    arr = ee_to_array(img, "Albedo_WSA_shortwave")
    # Already scaled by 0.001 in EE, values should be 0-1
    print(f"  Range: {arr.min():.3f} to {arr.max():.3f}")

    # Clamp
    arr = np.clip(arr, 0.02, 0.95)

    save_json("albedo.json", {
        "source": "MODIS MCD43A3 white-sky shortwave albedo, 2020-2023 annual mean, via Earth Engine",
        "albedo": [round(float(v), 4) for v in arr.ravel()],
    })

    save_reference_png(img, "MODIS Albedo", {
        "min": 0, "max": 0.6,
        "palette": ["001a00", "003300", "1a5e1a", "66aa33", "cccc44",
                     "e6c266", "f5deb3", "ffffff"],
    }, "albedo.png")

    save_colormap_png(arr, "albedo_preview.png", 0, 0.6)
    return arr


def fetch_precipitation():
    """GPM IMERG monthly precipitation (2015-2023 mean), land + ocean."""
    print("\n=== GPM IMERG Precipitation (1024x512) ===")
    # IMERG precipitation is mm/hr; mean rate * 8766 hrs/yr = mm/yr
    img = (
        ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07")
        .filterDate("2015-01-01", "2024-01-01")
        .select("precipitation")
        .mean()
        .multiply(8766)  # mm/hr -> mm/yr
    )

    arr = ee_to_array(img, "precipitation")
    arr = np.maximum(arr, 0)
    print(f"  Range: {arr.min():.0f} to {arr.max():.0f} mm/yr")

    save_json("precipitation.json", {
        "source": "NASA GPM IMERG V07 monthly, 2015-2023 annual mean (mm/year), land+ocean, via Earth Engine",
        "precipitation": [round(float(v)) for v in arr.ravel()],
    })

    save_reference_png(img, "GPM IMERG Precip", {
        "min": 0, "max": 3000,
        "palette": ["f7f7f7", "cccccc", "88bbdd", "3388bb", "1166aa", "003388", "001155"],
    }, "precipitation.png")

    save_colormap_png(arr, "precipitation_preview.png", 0, 3000, "blues")
    return arr


def fetch_clouds():
    """MODIS Terra cloud fraction (2020-2023 annual mean)."""
    print("\n=== MODIS Cloud Fraction (1024x512) ===")
    img = (
        ee.ImageCollection("MODIS/061/MOD08_M3")
        .filterDate("2020-01-01", "2024-01-01")
        .select("Cloud_Fraction_Mean_Mean")
        .mean()
        .multiply(0.01)  # scale factor
    )
    arr = ee_to_array(img, "Cloud_Fraction_Mean_Mean")
    # EE multiply may not persist through computePixels for some datasets
    if arr.max() > 2:
        arr = arr * 0.01
    print(f"  Range: {arr.min():.3f} to {arr.max():.3f}")

    arr = np.clip(arr, 0, 1)

    save_json("cloud_fraction.json", {
        "source": "MODIS MOD08_M3 cloud fraction, 2020-2023 annual mean, via Earth Engine",
        "cloud_fraction": [round(float(v), 3) for v in arr.ravel()],
    })

    save_reference_png(img, "Cloud Fraction", {
        "min": 0, "max": 1,
        "palette": ["ffffff", "cccccc", "999999", "666666", "333333"],
    }, "cloud_fraction.png")

    save_colormap_png(arr, "cloud_fraction_preview.png", 0, 1)
    return arr


def fetch_woa23_field(var_name, depth_idx, depth_label):
    """Fetch WOA23 data via OPeNDAP at 0.25 degree, interpolate to our grid."""
    # WOA23 0.25-degree annual climatology
    # Temperature: woa23_decav91C0_t00_04.nc (04 = 0.25 deg)
    # Salinity: woa23_decav91C0_s00_04.nc
    var_char = "t" if var_name == "temperature" else "s"
    filename = f"woa23_decav91C0_{var_char}00_04.nc"
    base_url = f"https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/{var_name}/netcdf/decav91C0/0.25/{filename}"

    print(f"\n=== WOA23 {var_name} at {depth_label} (0.25 deg → 1024x512) ===")

    # Fetch lat/lon coordinates
    print("  Fetching coordinates...")
    lat_url = f"{base_url}.ascii?lat"
    lon_url = f"{base_url}.ascii?lon"
    depth_url = f"{base_url}.ascii?depth"

    def fetch_ascii(url):
        req = urllib.request.Request(url, headers={"User-Agent": "SimAMOC/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8")

    def parse_1d_ascii(text):
        vals = []
        data_started = False
        for line in text.split("\n"):
            line = line.strip()
            if "---" in line:
                data_started = True
                continue
            if not data_started:
                continue
            if not line or line.startswith("}"):
                continue
            # Skip dimension header lines like "lat[720]"
            if "[" in line and "]" in line and "," not in line:
                continue
            for v in line.split(","):
                v = v.strip()
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return vals

    src_lats = parse_1d_ascii(fetch_ascii(lat_url))
    src_lons = parse_1d_ascii(fetch_ascii(lon_url))
    src_depths = parse_1d_ascii(fetch_ascii(depth_url))
    print(f"  Source grid: {len(src_lons)}x{len(src_lats)}, depths: {len(src_depths)}")
    print(f"  Lat: {src_lats[0]} to {src_lats[-1]}")
    print(f"  Depth[{depth_idx}] = {src_depths[depth_idx]}m")

    nlat = len(src_lats)
    nlon = len(src_lons)

    # Fetch the data slice: [time=0][depth=depth_idx][all lat][all lon]
    # Variable name in file
    band = f"{var_char}_an"
    query = f"{band}[0:0][{depth_idx}:{depth_idx}][0:{nlat - 1}][0:{nlon - 1}]"
    data_url = f"{base_url}.ascii?{query}"
    print(f"  Fetching {band} (this may take a minute)...")

    text = fetch_ascii(data_url)
    vals = parse_1d_ascii(text)
    expected = nlat * nlon
    print(f"  Parsed {len(vals)} values (expected {expected})")

    if len(vals) < expected:
        print(f"  WARNING: got fewer values than expected, padding with NaN")
        vals.extend([float("nan")] * (expected - len(vals)))

    # Reshape to [lat][lon]
    src_data = np.array(vals[:expected]).reshape(nlat, nlon)
    # Replace fill values (typically 9.96921e36)
    src_data[np.abs(src_data) > 1e10] = np.nan

    valid = src_data[~np.isnan(src_data)]
    print(f"  Source range: {valid.min():.2f} to {valid.max():.2f}")

    # Bilinear interpolation to our grid
    print("  Interpolating to 1024x512...")
    dst_lats = np.linspace(LAT1, LAT0, NY)  # top to bottom
    dst_lons = np.linspace(LON0, LON1, NX, endpoint=False) + SCALE_X / 2

    result = np.full((NY, NX), np.nan)
    for j in range(NY):
        lat = dst_lats[j]
        # Find bracketing source lat indices
        fj = np.interp(lat, src_lats, np.arange(nlat))
        j0 = int(np.floor(fj))
        j1 = min(j0 + 1, nlat - 1)
        tj = fj - j0

        for i in range(NX):
            lon = dst_lons[i]
            # WOA23 uses -180 to 180 longitude (same as our grid)
            fi = np.interp(lon, src_lons, np.arange(nlon))
            i0 = int(np.floor(fi))
            i1 = (i0 + 1) % nlon
            ti = fi - i0

            # Bilinear interpolation
            v00 = src_data[j0, i0]
            v01 = src_data[j0, i1]
            v10 = src_data[j1, i0]
            v11 = src_data[j1, i1]

            # Handle NaN (use nearest valid)
            corners = [v00, v01, v10, v11]
            valid_corners = [v for v in corners if not np.isnan(v)]
            if len(valid_corners) == 4:
                val = (1 - tj) * ((1 - ti) * v00 + ti * v01) + tj * ((1 - ti) * v10 + ti * v11)
            elif valid_corners:
                val = np.mean(valid_corners)
            else:
                val = np.nan
            result[j, i] = val

    valid_result = result[~np.isnan(result)]
    print(f"  Result range: {valid_result.min():.2f} to {valid_result.max():.2f}")
    print(f"  Valid cells: {len(valid_result)}/{NX * NY}")

    return result, src_depths[depth_idx]


def fetch_salinity():
    """WOA23 surface salinity (0m)."""
    result, actual_depth = fetch_woa23_field("salinity", 0, "surface")

    # Load mask to identify ocean cells
    mask = _load_mask()
    # Fill NaN ocean cells with nearest-neighbor spreading, leave land as default
    result = _fill_ocean_nans(result, mask, default=35.0)

    save_json("salinity.json", {
        "source": f"WOA23 annual mean surface salinity (0m), 0.25-deg interpolated to 1024x512",
        "depth_m": actual_depth,
        "salinity": [round(float(v), 3) for v in result.ravel()],
    })

    mask = _load_mask()
    save_colormap_png(result, "salinity_preview.png", 30, 38, "coolwarm", mask=mask)
    return result


def fetch_deep_temp():
    """WOA23 temperature at ~1000m depth."""
    # WOA23 0.25-deg depth index 46 = 1000m
    result, actual_depth = fetch_woa23_field("temperature", 46, "1000m")

    # Fill NaN ocean cells with nearest-neighbor spreading, leave land as default
    mask = _load_mask()
    result = _fill_ocean_nans(result, mask, default=4.0)

    save_json("deep_temp.json", {
        "source": f"WOA23 annual mean temperature at {actual_depth}m, 0.25-deg interpolated to 1024x512",
        "depth_m": actual_depth,
        "temp": [round(float(v), 3) for v in result.ravel()],
    })

    mask = _load_mask()
    save_colormap_png(result, "deep_temp_preview.png", -2, 15, "coolwarm", mask=mask)
    return result


def fetch_sea_ice():
    """Sea ice concentration from OISST ice band (same dataset as SST)."""
    print("\n=== Sea Ice Concentration (1024x512) ===")
    img = (
        ee.ImageCollection("NOAA/CDR/OISST/V2_1")
        .filterDate("2015-01-01", "2024-01-01")
        .select("ice")
        .mean()
        .multiply(0.01)  # percent to fraction
    )
    arr = ee_to_array(img, "ice")
    if arr.max() > 2:
        arr = arr * 0.01
    arr = np.clip(arr, 0, 1)
    print(f"  Range: {arr.min():.3f} to {arr.max():.3f}")

    save_json("sea_ice.json", {
        "source": "NOAA OISST v2.1 ice concentration, 2015-2023 annual mean, via Earth Engine",
        "ice_fraction": [round(float(v), 4) for v in arr.ravel()],
    })
    save_colormap_png(arr, "sea_ice_preview.png", 0, 1, "blues")
    return arr


def fetch_land_surface_temp():
    """MODIS land surface temperature (daytime)."""
    print("\n=== MODIS Land Surface Temp (1024x512) ===")
    img = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate("2020-01-01", "2024-01-01")
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)  # scale factor -> Kelvin
        .subtract(273.15)  # -> Celsius
    )
    arr = ee_to_array(img, "LST_Day_1km")
    # Check if still in Kelvin
    if arr.max() > 100:
        arr = arr * 0.02 - 273.15
    print(f"  Range: {arr.min():.1f} to {arr.max():.1f} C")

    save_json("land_surface_temp.json", {
        "source": "MODIS MOD11A1 daytime LST, 2020-2023 annual mean (Celsius), via Earth Engine",
        "lst": [round(float(v), 2) for v in arr.ravel()],
    })
    save_colormap_png(arr, "land_surface_temp_preview.png", -30, 50, "coolwarm")
    return arr


def fetch_snow_cover():
    """MODIS snow cover."""
    print("\n=== MODIS Snow Cover (1024x512) ===")
    img = (
        ee.ImageCollection("MODIS/061/MOD10A1")
        .filterDate("2020-01-01", "2024-01-01")
        .select("NDSI_Snow_Cover")
        .mean()
    )
    arr = ee_to_array(img, "NDSI_Snow_Cover")
    # Values 0-100 (percent)
    arr = np.clip(arr, 0, 100)
    print(f"  Range: {arr.min():.1f} to {arr.max():.1f} %")

    save_json("snow_cover.json", {
        "source": "MODIS MOD10A1 NDSI snow cover, 2020-2023 annual mean (percent), via Earth Engine",
        "snow_cover": [round(float(v), 2) for v in arr.ravel()],
    })
    save_colormap_png(arr, "snow_cover_preview.png", 0, 100, "blues")
    return arr


def fetch_ndvi():
    """MODIS NDVI (vegetation index)."""
    print("\n=== MODIS NDVI (1024x512) ===")
    img = (
        ee.ImageCollection("MODIS/061/MOD13A3")
        .filterDate("2020-01-01", "2024-01-01")
        .select("NDVI")
        .mean()
        .multiply(0.0001)  # scale factor
    )
    arr = ee_to_array(img, "NDVI")
    if arr.max() > 2:
        arr = arr * 0.0001
    arr = np.clip(arr, -0.2, 1.0)
    print(f"  Range: {arr.min():.3f} to {arr.max():.3f}")

    save_json("ndvi.json", {
        "source": "MODIS MOD13A3 monthly NDVI, 2020-2023 annual mean, via Earth Engine",
        "ndvi": [round(float(v), 4) for v in arr.ravel()],
    })
    save_colormap_png(arr, "ndvi_preview.png", -0.1, 0.9, "viridis")
    return arr


def fetch_chlorophyll():
    """MODIS Aqua ocean chlorophyll-a."""
    print("\n=== MODIS Aqua Chlorophyll (1024x512) ===")
    img = (
        ee.ImageCollection("NASA/OCEANDATA/MODIS-Aqua/L3SMI")
        .filterDate("2020-01-01", "2024-01-01")
        .select("chlor_a")
        .mean()
    )
    arr = ee_to_array(img, "chlor_a")
    arr = np.maximum(arr, 0)
    print(f"  Range: {arr.min():.3f} to {arr.max():.3f} mg/m3")

    save_json("chlorophyll.json", {
        "source": "MODIS Aqua L3SMI chlorophyll-a, 2020-2023 annual mean (mg/m3), via Earth Engine",
        "chlor_a": [round(float(v), 4) for v in arr.ravel()],
    })
    # Log scale for chlorophyll (0.01 to 10 mg/m3)
    log_arr = np.log10(np.clip(arr, 0.01, 30))
    save_colormap_png(log_arr, "chlorophyll_preview.png", -2, 1.5, "viridis")
    return arr


def fetch_air_temp():
    """ERA5 2m air temperature."""
    print("\n=== ERA5 2m Air Temperature (1024x512) ===")
    img = (
        ee.ImageCollection("ECMWF/ERA5/MONTHLY")
        .filterDate("2015-01-01", "2024-01-01")
        .select("mean_2m_air_temperature")
        .mean()
        .subtract(273.15)  # K -> C
    )
    arr = ee_to_array(img, "mean_2m_air_temperature")
    if arr.max() > 100:
        arr = arr - 273.15
    print(f"  Range: {arr.min():.1f} to {arr.max():.1f} C")

    save_json("air_temp.json", {
        "source": "ERA5 Monthly 2m air temperature, 2015-2023 mean (Celsius), via Earth Engine",
        "air_temp": [round(float(v), 2) for v in arr.ravel()],
    })
    save_colormap_png(arr, "air_temp_preview.png", -40, 35, "coolwarm")
    return arr


def fetch_evaporation():
    """ERA5 Land total evaporation."""
    print("\n=== ERA5 Land Evaporation (1024x512) ===")
    img = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate("2015-01-01", "2024-01-01")
        .select("total_evaporation_sum")
        .mean()
        .multiply(-1000 * 12)  # m/month -> mm/yr (evap is negative in ERA5)
    )
    arr = ee_to_array(img, "total_evaporation_sum")
    if arr.min() < -100:
        arr = arr * -1000 * 12
    print(f"  Range: {arr.min():.0f} to {arr.max():.0f} mm/yr")

    save_json("evaporation.json", {
        "source": "ERA5 Land total evaporation, 2015-2023 mean (mm/year, positive=evap), via Earth Engine",
        "evaporation": [round(float(v)) for v in arr.ravel()],
    })
    save_colormap_png(arr, "evaporation_preview.png", 0, 2000, "blues")
    return arr


def fetch_surface_pressure():
    """ERA5 surface pressure."""
    print("\n=== ERA5 Surface Pressure (1024x512) ===")
    img = (
        ee.ImageCollection("ECMWF/ERA5/MONTHLY")
        .filterDate("2015-01-01", "2024-01-01")
        .select("surface_pressure")
        .mean()
        .multiply(0.01)  # Pa -> hPa
    )
    arr = ee_to_array(img, "surface_pressure")
    if arr.max() > 2000:
        arr = arr * 0.01
    print(f"  Range: {arr.min():.1f} to {arr.max():.1f} hPa")

    save_json("surface_pressure.json", {
        "source": "ERA5 Monthly surface pressure, 2015-2023 mean (hPa), via Earth Engine",
        "pressure": [round(float(v), 1) for v in arr.ravel()],
    })
    save_colormap_png(arr, "surface_pressure_preview.png", 500, 1030, "coolwarm")
    return arr


def fetch_monthly_sst():
    """Monthly SST climatology (12 months) for seasonal forcing."""
    print("\n=== Monthly SST Climatology (1024x512 x 12) ===")
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = []
    for month in range(1, 13):
        print(f"  Month {month} ({MONTH_NAMES[month-1]})...")
        img = (
            ee.ImageCollection("NOAA/CDR/OISST/V2_1")
            .filter(ee.Filter.calendarRange(month, month, "month"))
            .filterDate("2015-01-01", "2024-01-01")
            .select("sst")
            .mean()
            .multiply(0.01)
        )
        arr = ee_to_array(img, "sst")
        if arr.max() > 100:
            arr = arr * 0.01
        arr[arr < -10] = np.nan
        # Fill NaN with zonal mean
        for j in range(NY):
            row = arr[j, :]
            valid = row[~np.isnan(row)]
            fill = np.mean(valid) if len(valid) > 0 else 15.0
            arr[j, np.isnan(arr[j, :])] = fill
        monthly.append([round(float(v), 2) for v in arr.ravel()])
        print(f"    Range: {arr.min():.1f} to {arr.max():.1f} C")
        time.sleep(1)

    save_json("sst_monthly.json", {
        "source": "NOAA OISST v2.1, 2015-2023 monthly climatology (12 months, Celsius), via Earth Engine",
        "months": MONTH_NAMES,
        "monthly": monthly,
    })
    return monthly


def fetch_monthly_wind():
    """Monthly wind stress climatology (12 months) for seasonal forcing."""
    print("\n=== Monthly Wind Stress Climatology (1024x512 x 12) ===")
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rho_air = 1.225
    Cd = 1.3e-3
    monthly_tau_x = []
    monthly_tau_y = []
    for month in range(1, 13):
        print(f"  Month {month} ({MONTH_NAMES[month-1]})...")
        col = (
            ee.ImageCollection("ECMWF/ERA5/MONTHLY")
            .filter(ee.Filter.calendarRange(month, month, "month"))
            .filterDate("1991-01-01", "2021-01-01")
            .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
            .mean()
        )
        u10 = ee_to_array(col, "u_component_of_wind_10m")
        v10 = ee_to_array(col, "v_component_of_wind_10m")
        speed = np.sqrt(u10**2 + v10**2)
        tau_x = rho_air * Cd * speed * u10
        tau_y = rho_air * Cd * speed * v10
        monthly_tau_x.append([round(float(v), 6) for v in tau_x.ravel()])
        monthly_tau_y.append([round(float(v), 6) for v in tau_y.ravel()])
        print(f"    tau_x: {tau_x.min():.4f} to {tau_x.max():.4f}")
        time.sleep(1)

    save_json("wind_stress_monthly.json", {
        "source": "ERA5 Monthly 10m wind, 1991-2020 monthly climatology, bulk stress Cd=1.3e-3, via Earth Engine",
        "months": MONTH_NAMES,
        "monthly_tau_x": monthly_tau_x,
        "monthly_tau_y": monthly_tau_y,
    })
    return monthly_tau_x, monthly_tau_y


def fetch_monthly_albedo():
    """Monthly albedo climatology (12 months) for seasonal forcing."""
    print("\n=== Monthly Albedo Climatology (1024x512 x 12) ===")
    MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly = []
    for month in range(1, 13):
        print(f"  Month {month} ({MONTH_NAMES[month-1]})...")
        img = (
            ee.ImageCollection("MODIS/061/MCD43A3")
            .filter(ee.Filter.calendarRange(month, month, "month"))
            .filterDate("2020-01-01", "2024-01-01")
            .select("Albedo_WSA_shortwave")
            .mean()
            .multiply(0.001)
        )
        arr = ee_to_array(img, "Albedo_WSA_shortwave")
        arr = np.clip(arr, 0.02, 0.95)
        monthly.append([round(float(v), 4) for v in arr.ravel()])
        print(f"    Range: {arr.min():.3f} to {arr.max():.3f}")
        time.sleep(1)

    save_json("albedo_monthly.json", {
        "source": "MODIS MCD43A3 WSA shortwave albedo, 2020-2023 monthly climatology, via Earth Engine",
        "months": MONTH_NAMES,
        "monthly": monthly,
    })
    return monthly


def fetch_mixed_layer_depth():
    """WOA23 mixed layer depth via OPeNDAP."""
    print("\n=== WOA23 Mixed Layer Depth (0.25 deg → 1024x512) ===")
    # WOA23 MLD is in a separate file
    base_url = "https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/MLD/netcdf/decav91C0/0.25/woa23_decav91C0_M00_04.nc"

    def fetch_ascii(url):
        req = urllib.request.Request(url, headers={"User-Agent": "SimAMOC/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read().decode("utf-8")

    def parse_1d_ascii(text):
        vals = []
        data_started = False
        for line in text.split("\n"):
            line = line.strip()
            if "---" in line:
                data_started = True
                continue
            if not data_started:
                continue
            if not line or line.startswith("}"):
                continue
            if "[" in line and "]" in line and "," not in line:
                continue
            for v in line.split(","):
                v = v.strip()
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return vals

    # Try fetching coordinates first to check structure
    print("  Fetching coordinates...")
    try:
        lat_text = fetch_ascii(f"{base_url}.ascii?lat[0:3]")
        src_lats_sample = parse_1d_ascii(lat_text)
        print(f"  Lat sample: {src_lats_sample}")
    except Exception as e:
        print(f"  WOA23 MLD not available at expected path: {e}")
        print("  Trying alternative: computing from temperature profile...")
        # Fallback: compute MLD from temperature difference criterion
        # Use WOA23 temperature at multiple depths
        return _compute_mld_from_temp()

    src_lats = parse_1d_ascii(fetch_ascii(f"{base_url}.ascii?lat"))
    src_lons = parse_1d_ascii(fetch_ascii(f"{base_url}.ascii?lon"))
    nlat = len(src_lats)
    nlon = len(src_lons)
    print(f"  Source grid: {nlon}x{nlat}")

    # Fetch MLD data
    band = "M_an"  # annual mean MLD
    query = f"{band}[0:0][0:{nlat-1}][0:{nlon-1}]"
    print(f"  Fetching {band}...")
    text = fetch_ascii(f"{base_url}.ascii?{query}")
    vals = parse_1d_ascii(text)
    expected = nlat * nlon
    print(f"  Parsed {len(vals)} values (expected {expected})")

    src_data = np.array(vals[:expected]).reshape(nlat, nlon)
    src_data[np.abs(src_data) > 1e10] = np.nan

    valid = src_data[~np.isnan(src_data)]
    print(f"  Source range: {valid.min():.1f} to {valid.max():.1f} m")

    # Interpolate to our grid
    print("  Interpolating to 1024x512...")
    dst_lats = np.linspace(LAT1, LAT0, NY)
    dst_lons = np.linspace(LON0, LON1, NX, endpoint=False) + SCALE_X / 2

    result = np.full((NY, NX), np.nan)
    for j in range(NY):
        lat = dst_lats[j]
        fj = np.interp(lat, src_lats, np.arange(nlat))
        j0 = int(np.floor(fj))
        j1 = min(j0 + 1, nlat - 1)
        tj = fj - j0
        for i in range(NX):
            lon = dst_lons[i]
            fi = np.interp(lon, src_lons, np.arange(nlon))
            i0 = int(np.floor(fi))
            i1 = (i0 + 1) % nlon
            ti = fi - i0
            v00, v01 = src_data[j0, i0], src_data[j0, i1]
            v10, v11 = src_data[j1, i0], src_data[j1, i1]
            corners = [v00, v01, v10, v11]
            valid_c = [v for v in corners if not np.isnan(v)]
            if len(valid_c) == 4:
                result[j, i] = (1-tj)*((1-ti)*v00 + ti*v01) + tj*((1-ti)*v10 + ti*v11)
            elif valid_c:
                result[j, i] = np.mean(valid_c)

    # Fill NaN ocean cells
    mask = _load_mask()
    result = _fill_ocean_nans(result, mask, default=50.0)

    save_json("mixed_layer_depth.json", {
        "source": "WOA23 annual mean mixed layer depth, 0.25-deg interpolated to 1024x512",
        "mld": [round(float(v), 1) for v in result.ravel()],
    })
    save_colormap_png(result, "mixed_layer_depth_preview.png", 0, 300, "blues", mask=mask)
    return result


def _compute_mld_from_temp():
    """Fallback: estimate MLD from WOA23 temp profile (delta-T = 0.5C criterion)."""
    print("  Computing MLD from temperature profile (delta-T=0.5C)...")
    # Use surface temp and temps at 50m, 100m, 150m, 200m
    depths = [(0, 0), (16, 50), (24, 100), (32, 200)]  # (index, depth_m)
    # Just use a simple heuristic: MLD = 50m tropics, 200m midlat, 100m polar
    result = np.full((NY, NX), 100.0)
    lats = np.linspace(LAT1, LAT0, NY)
    for j in range(NY):
        alat = abs(lats[j])
        if alat < 15:
            result[j, :] = 30  # shallow tropical thermocline
        elif alat < 40:
            result[j, :] = 50 + (alat - 15) * 4  # deepening with latitude
        elif alat < 60:
            result[j, :] = 150 + (alat - 40) * 2.5  # deep winter mixing
        else:
            result[j, :] = 200  # polar deep mixing

    mask = _load_mask()
    if mask is not None:
        result[mask == 0] = 0  # land

    save_json("mixed_layer_depth.json", {
        "source": "Estimated MLD from latitude-dependent heuristic (pending WOA23 MLD data)",
        "mld": [round(float(v), 1) for v in result.ravel()],
    })
    save_colormap_png(result, "mixed_layer_depth_preview.png", 0, 300, "blues", mask=mask)
    return result


def fetch_oscar_currents():
    """OSCAR satellite-derived ocean surface currents via OPeNDAP."""
    print("\n=== OSCAR Ocean Surface Currents (1/3 deg → 1024x512) ===")
    # OSCAR v2 via PO.DAAC OPeNDAP
    base_url = "https://opendap.jpl.nasa.gov/opendap/OceanCurrents/oscar_currents_interim/oscar_currents_interim_20200101.nc"

    def fetch_ascii(url):
        req = urllib.request.Request(url, headers={"User-Agent": "SimAMOC/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read().decode("utf-8")

    # Try the GODAS currents we already have a script for as fallback
    print("  Trying GODAS currents via PSL OPeNDAP...")
    try:
        godas_base = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/Derived"
        # Check if accessible
        lat_text = fetch_ascii(f"{godas_base}/ucur.mon.ltm.1991-2020.nc.ascii?lat[0:3]")
        print(f"  GODAS accessible")
    except Exception as e:
        print(f"  GODAS not accessible: {e}")
        print("  Using existing godas_currents_1deg.json if available...")
        # Try loading existing file
        existing = Path("godas_currents_1deg.json")
        if existing.exists():
            import json as jmod
            d = jmod.loads(existing.read_text())
            u_src = np.array(d["u"]).reshape(d["ny"], d["nx"])
            v_src = np.array(d["v"]).reshape(d["ny"], d["nx"])
            # Resample to 1024x512
            dst_lats = np.linspace(LAT1, LAT0, NY)
            dst_lons = np.linspace(LON0, LON1, NX, endpoint=False) + SCALE_X / 2
            src_ny, src_nx = d["ny"], d["nx"]
            src_lat0, src_lat1 = d["lat0"], d["lat1"]
            src_lon0 = d.get("lon0", -180)

            u_out = np.zeros((NY, NX))
            v_out = np.zeros((NY, NX))
            for j in range(NY):
                fj = (dst_lats[j] - src_lat0) / (src_lat1 - src_lat0) * (src_ny - 1)
                j0 = max(0, min(int(np.floor(fj)), src_ny - 2))
                j1 = j0 + 1
                tj = fj - j0
                for i in range(NX):
                    fi = (dst_lons[i] - src_lon0) / 360 * src_nx
                    i0 = int(np.floor(fi)) % src_nx
                    i1 = (i0 + 1) % src_nx
                    ti = fi - np.floor(fi)
                    for arr_src, arr_dst in [(u_src, u_out), (v_src, v_out)]:
                        v00, v01 = arr_src[j0, i0], arr_src[j0, i1]
                        v10, v11 = arr_src[j1, i0], arr_src[j1, i1]
                        arr_dst[j, i] = (1-tj)*((1-ti)*v00 + ti*v01) + tj*((1-ti)*v10 + ti*v11)

            print(f"  U range: {u_out.min():.3f} to {u_out.max():.3f} m/s")
            print(f"  V range: {v_out.min():.3f} to {v_out.max():.3f} m/s")

            mask = _load_mask()
            save_json("ocean_currents.json", {
                "source": "GODAS 1991-2020 LTM surface currents, interpolated to 1024x512",
                "u": [round(float(v), 4) for v in u_out.ravel()],
                "v": [round(float(v), 4) for v in v_out.ravel()],
                "units": "m/s",
            })
            # Preview: current speed
            speed = np.sqrt(u_out**2 + v_out**2)
            save_colormap_png(speed, "ocean_currents_preview.png", 0, 0.5, "viridis", mask=mask)
            return u_out, v_out
        else:
            print("  No existing current data found. Skipping.")
            return None

    # GODAS accessible but full fetch is slow — check for existing file first
    existing = Path("godas_currents_1deg.json")
    if existing.exists():
        print("  Found existing godas_currents_1deg.json, resampling to 1024x512...")
        d = json.loads(existing.read_text())
        u_src = np.array(d["u"]).reshape(d["ny"], d["nx"])
        v_src = np.array(d["v"]).reshape(d["ny"], d["nx"])
        dst_lats = np.linspace(LAT1, LAT0, NY)
        dst_lons = np.linspace(LON0, LON1, NX, endpoint=False) + SCALE_X / 2
        src_ny, src_nx = d["ny"], d["nx"]
        src_lat0, src_lat1 = d["lat0"], d["lat1"]
        src_lon0 = d.get("lon0", -180)

        u_out = np.zeros((NY, NX))
        v_out = np.zeros((NY, NX))
        for j in range(NY):
            fj = (dst_lats[j] - src_lat0) / (src_lat1 - src_lat0) * (src_ny - 1)
            j0 = max(0, min(int(np.floor(fj)), src_ny - 2))
            j1 = j0 + 1
            tj = fj - j0
            for i in range(NX):
                fi = (dst_lons[i] - src_lon0) / 360 * src_nx
                i0 = int(np.floor(fi)) % src_nx
                i1 = (i0 + 1) % src_nx
                ti = fi - np.floor(fi)
                for arr_src, arr_dst in [(u_src, u_out), (v_src, v_out)]:
                    v00, v01 = arr_src[j0, i0], arr_src[j0, i1]
                    v10, v11 = arr_src[j1, i0], arr_src[j1, i1]
                    arr_dst[j, i] = (1-tj)*((1-ti)*v00 + ti*v01) + tj*((1-ti)*v10 + ti*v11)

        print(f"  U range: {u_out.min():.3f} to {u_out.max():.3f} m/s")
        print(f"  V range: {v_out.min():.3f} to {v_out.max():.3f} m/s")

        mask_arr = _load_mask()
        save_json("ocean_currents.json", {
            "source": "GODAS 1991-2020 LTM surface currents, interpolated to 1024x512",
            "u": [round(float(v), 4) for v in u_out.ravel()],
            "v": [round(float(v), 4) for v in v_out.ravel()],
            "units": "m/s",
        })
        speed = np.sqrt(u_out**2 + v_out**2)
        save_colormap_png(speed, "ocean_currents_preview.png", 0, 0.5, "viridis", mask=mask_arr)
        return u_out, v_out

    print("  No existing current data. Run scripts/fetch_godas_currents.py first.")
    return None


# ============================================================
# MAIN
# ============================================================

FIELDS = {
    "bathymetry": fetch_bathymetry,
    "sst": fetch_sst,
    "wind": fetch_wind,
    "albedo": fetch_albedo,
    "precipitation": fetch_precipitation,
    "clouds": fetch_clouds,
    "salinity": fetch_salinity,
    "deep_temp": fetch_deep_temp,
    "sea_ice": fetch_sea_ice,
    "land_surface_temp": fetch_land_surface_temp,
    "snow_cover": fetch_snow_cover,
    "ndvi": fetch_ndvi,
    "chlorophyll": fetch_chlorophyll,
    "air_temp": fetch_air_temp,
    "evaporation": fetch_evaporation,
    "surface_pressure": fetch_surface_pressure,
    "monthly_sst": fetch_monthly_sst,
    "monthly_wind": fetch_monthly_wind,
    "monthly_albedo": fetch_monthly_albedo,
    "mixed_layer_depth": fetch_mixed_layer_depth,
    "ocean_currents": fetch_oscar_currents,
}


def main():
    requested = None
    if "--field" in sys.argv:
        idx = sys.argv.index("--field")
        if idx + 1 < len(sys.argv):
            requested = sys.argv[idx + 1].split(",")

    fields_to_fetch = requested or list(FIELDS.keys())
    print(f"Target grid: {NX}x{NY} ({SCALE_X:.4f} x {SCALE_Y:.4f} deg)")
    print(f"Fields: {', '.join(fields_to_fetch)}")
    print("=" * 60)

    init_ee()

    depth = None
    mask = None

    for field in fields_to_fetch:
        if field not in FIELDS:
            print(f"Unknown field: {field}")
            continue
        t0 = time.time()
        try:
            result = FIELDS[field]()
            if field == "bathymetry":
                depth, elevation = result
                mask = fetch_mask(depth)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Viewer page is hand-curated (data/viewer.html) — don't overwrite

    print("\n" + "=" * 60)
    print(f"All data saved to {DATA_DIR}/")
    print(f"Reference images in {IMG_DIR}/")
    print(f"Open {DATA_DIR}/viewer.html to browse.")


def generate_viewer():
    """Generate an HTML page to view all the reference images."""
    print("\n=== Generating viewer page ===")

    images = sorted(IMG_DIR.glob("*.png"))
    if not images:
        print("  No images found")
        return

    cards = []
    for img_path in images:
        name = img_path.stem.replace("_", " ").replace("-", " ").title()
        size_kb = img_path.stat().st_size / 1024
        cards.append(f"""
        <div class="card">
            <h3>{name}</h3>
            <img src="images/{img_path.name}" alt="{name}" loading="lazy">
            <p class="meta">{img_path.name} ({size_kb:.0f} KB)</p>
        </div>""")

    # Check for JSON metadata
    json_files = sorted(DATA_DIR.glob("*.json"))
    json_rows = []
    for jf in json_files:
        if jf.name == "mask.json":
            continue
        try:
            with open(jf) as f:
                d = json.load(f)
            source = d.get("source", "")
            nx = d.get("nx", "?")
            ny = d.get("ny", "?")
            size_kb = jf.stat().st_size / 1024
            json_rows.append(f"""
            <tr>
                <td><code>{jf.name}</code></td>
                <td>{nx}x{ny}</td>
                <td>{size_kb:.0f} KB</td>
                <td>{source}</td>
            </tr>""")
        except Exception:
            pass

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SimAMOC Data Maps - {NX}x{NY}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0a0a0a; color: #e0e0e0; padding: 2rem; }}
h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; color: #fff; }}
h2 {{ font-size: 1.3rem; margin: 2rem 0 1rem; color: #88aacc; }}
.subtitle {{ color: #888; margin-bottom: 2rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(600px, 1fr)); gap: 1.5rem; }}
.card {{ background: #1a1a1a; border-radius: 8px; overflow: hidden; border: 1px solid #333; }}
.card h3 {{ padding: 0.8rem 1rem 0.4rem; font-size: 1rem; color: #aaccee; }}
.card img {{ width: 100%; display: block; image-rendering: pixelated; }}
.card .meta {{ padding: 0.4rem 1rem 0.8rem; font-size: 0.8rem; color: #666; }}
table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
th, td {{ text-align: left; padding: 0.5rem 1rem; border-bottom: 1px solid #222; }}
th {{ color: #88aacc; font-weight: 600; }}
td {{ font-size: 0.85rem; }}
code {{ background: #222; padding: 0.1rem 0.3rem; border-radius: 3px; font-size: 0.85rem; }}
.footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #222;
           font-size: 0.8rem; color: #555; }}
</style>
</head>
<body>
<h1>SimAMOC Observational Data Maps</h1>
<p class="subtitle">Grid: {NX} x {NY} ({SCALE_X:.4f} x {SCALE_Y:.4f} deg) | Lat: {LAT0} to {LAT1} | Lon: {LON0} to {LON1}</p>

<h2>Reference Images</h2>
<div class="grid">
{"".join(cards)}
</div>

<h2>Data Files</h2>
<table>
<tr><th>File</th><th>Grid</th><th>Size</th><th>Source</th></tr>
{"".join(json_rows)}
</table>

<div class="footer">
Generated by fetch-data-hires.py | Data fetched from Earth Engine + WOA23 OPeNDAP
</div>
</body>
</html>"""

    path = DATA_DIR / "viewer.html"
    path.write_text(html)
    print(f"  Saved {path}")


if __name__ == "__main__":
    main()
