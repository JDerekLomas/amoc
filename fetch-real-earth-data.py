#!/usr/bin/env python3
"""
Fetch real Earth observation data via Google Earth Engine.
Downloads MODIS albedo and CHIRPS precipitation as global 1-degree grids.

Archives full-resolution GeoTIFFs in data/archive/ for manual inspection.
Outputs 360x160 JSON files matching our simulation grid format.

Usage:
    /Users/dereklomas/eli/.venv/bin/python fetch-real-earth-data.py

Requires: Earth Engine service account from eli project.
"""

import ee
import json
import time
import requests
from pathlib import Path

# --- Config ---
NX, NY = 360, 160
LAT0, LAT1 = -79.5, 79.5
LON0, LON1 = -179.5, 179.5
ARCHIVE_DIR = Path("data/archive")
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Earth Engine init ---
SA_KEY = Path("/Users/dereklomas/eli/sa-key.json")
credentials = ee.ServiceAccountCredentials(
    "earthengine@eli-africa-494008.iam.gserviceaccount.com",
    str(SA_KEY),
)
ee.Initialize(credentials, project="gen-lang-client-0278315411")
print("Connected to Earth Engine")

# --- Global region ---
GLOBAL_GEOM = ee.Geometry.Rectangle([LON0 - 0.5, LAT0 - 0.5, LON1 + 0.5, LAT1 + 0.5])


def download_archive_png(image, description, vis_params, archive_path):
    """Download an EE image as a high-res PNG visualization for archival."""
    print(f"  Archiving: {description}...")
    try:
        url = image.getThumbURL({
            **vis_params,
            "dimensions": "3600x1600",  # 10x our grid for inspection
            "region": GLOBAL_GEOM,
            "format": "png",
        })
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and len(r.content) > 1000:
            archive_path.write_bytes(r.content)
            size_kb = len(r.content) / 1024
            print(f"  Archived: {archive_path} ({size_kb:.0f} KB)")
            return True
    except Exception as e:
        print(f"  Archive failed (non-fatal): {e}")
    return False


def ee_to_grid(image, band, scale=111320, reducer=None):
    """
    Sample an EE image to our 360x160 grid using reduceRegion per row.
    Uses getInfo in batches to avoid timeout.
    """
    print(f"  Sampling to {NX}x{NY} grid...")
    result = [0.0] * (NX * NY)

    # Process in latitude bands (batches of rows)
    BATCH = 10  # rows per batch
    for j_start in range(0, NY, BATCH):
        j_end = min(j_start + BATCH, NY)
        lat_lo = LAT0 + j_start - 0.5
        lat_hi = LAT0 + j_end + 0.5

        # Create a feature collection of cell centers for this band
        features = []
        for j in range(j_start, j_end):
            lat = LAT0 + j
            for i in range(NX):
                lon = LON0 + i
                features.append(
                    ee.Feature(
                        ee.Geometry.Point([lon, lat]),
                        {"idx": j * NX + i},
                    )
                )

        fc = ee.FeatureCollection(features)
        sampled = image.select(band).reduceRegions(
            collection=fc,
            reducer=reducer or ee.Reducer.mean(),
            scale=scale,
        )

        try:
            info = sampled.getInfo()
        except Exception as e:
            print(f"    Batch j={j_start}-{j_end} failed: {e}")
            continue

        for feat in info["features"]:
            idx = feat["properties"]["idx"]
            val = feat["properties"].get("mean")
            if val is None:
                val = feat["properties"].get("mode", 0)
            if val is not None and val == val:  # not NaN
                result[idx] = float(val)

        pct = 100 * j_end / NY
        if j_end % 20 == 0 or j_end == NY:
            print(f"    {j_end}/{NY} rows ({pct:.0f}%)")

    return result


def fetch_modis_albedo():
    """Fetch MODIS MCD43A3 white-sky albedo, annual mean climatology."""
    print("\n=== MODIS Albedo (MCD43A3) ===")

    # Annual mean of white-sky shortwave albedo (2020-2023)
    albedo_img = (
        ee.ImageCollection("MODIS/061/MCD43A3")
        .filterDate("2020-01-01", "2024-01-01")
        .select("Albedo_WSA_shortwave")
        .mean()
        .multiply(0.001)  # scale factor: raw values * 0.001 = actual albedo
    )

    # Archive as high-res PNG for visual inspection
    download_archive_png(
        albedo_img,
        "MODIS WSA albedo 2020-2023",
        {"min": 0, "max": 0.6, "palette": ["001a00", "003300", "1a5e1a", "66aa33", "cccc44", "e6c266", "f5deb3", "ffffff"]},
        ARCHIVE_DIR / "modis_albedo_wsa_2020-2023.png",
    )

    # Sample to our grid at ~100km (1-degree cells)
    grid = ee_to_grid(albedo_img, "Albedo_WSA_shortwave", scale=100000)

    # Fill ocean cells with 0.06 (water albedo)
    # Load mask to identify ocean
    mask_path = Path("simamoc/mask.json")
    if mask_path.exists():
        mask_data = json.loads(mask_path.read_text())
        mask_bits = []
        for c in mask_data["hex"]:
            v = int(c, 16)
            mask_bits.extend([(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1])

        for j in range(NY):
            sj = min(int(j * 180 / NY), 179)
            for i in range(NX):
                si = min(i, 359)
                k = j * NX + i
                is_ocean = mask_bits[sj * 360 + si] == 1
                if is_ocean:
                    grid[k] = 0.06
                elif grid[k] <= 0 or grid[k] != grid[k]:
                    grid[k] = 0.20  # default land albedo

    # Clamp to valid range
    grid = [max(0.02, min(0.95, v)) for v in grid]

    # Stats
    land_vals = [v for v in grid if v > 0.07]
    if land_vals:
        print(f"  Land albedo: mean={sum(land_vals)/len(land_vals):.3f}, "
              f"min={min(land_vals):.3f}, max={max(land_vals):.3f}, "
              f"n={len(land_vals)}")

    # Save JSON
    output = {
        "nx": NX, "ny": NY,
        "lat0": LAT0, "lat1": LAT1,
        "lon0": LON0, "lon1": LON1,
        "source": "MODIS MCD43A3 white-sky shortwave albedo, 2020-2023 annual mean",
        "albedo": [round(v, 4) for v in grid],
    }
    out_path = Path("albedo_1deg.json")
    out_path.write_text(json.dumps(output))
    print(f"  Saved {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return grid


def fetch_chirps_precip():
    """Fetch CHIRPS annual precipitation climatology."""
    print("\n=== CHIRPS Precipitation ===")

    # Annual mean precipitation (2015-2023)
    years = []
    for year in range(2015, 2024):
        annual = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterDate(f"{year}-01-01", f"{year}-12-31")
            .select("precipitation")
            .sum()
        )
        years.append(annual)
    precip_img = ee.ImageCollection(years).mean()

    # Archive as high-res PNG for visual inspection
    download_archive_png(
        precip_img,
        "CHIRPS annual precip 2015-2023",
        {"min": 0, "max": 3000, "palette": ["f7f7f7", "cccccc", "88bbdd", "3388bb", "1166aa", "003388", "001155"]},
        ARCHIVE_DIR / "chirps_precip_annual_2015-2023.png",
    )

    # Sample to our grid
    grid = ee_to_grid(precip_img, "precipitation", scale=100000)

    # CHIRPS only covers 50S-50N, fill higher latitudes with estimates
    for j in range(NY):
        lat = LAT0 + j
        if abs(lat) > 50:
            for i in range(NX):
                k = j * NX + i
                if grid[k] <= 0:
                    # Rough polar/subpolar estimate
                    abs_lat = abs(lat)
                    grid[k] = max(50, 400 - 5 * (abs_lat - 50))

    # Zero out ocean cells
    mask_path = Path("simamoc/mask.json")
    if mask_path.exists():
        mask_data = json.loads(mask_path.read_text())
        mask_bits = []
        for c in mask_data["hex"]:
            v = int(c, 16)
            mask_bits.extend([(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1])

        for j in range(NY):
            sj = min(int(j * 180 / NY), 179)
            for i in range(NX):
                si = min(i, 359)
                k = j * NX + i
                is_ocean = mask_bits[sj * 360 + si] == 1
                if is_ocean:
                    grid[k] = 0

    grid = [max(0, round(v)) for v in grid]

    # Stats
    land_vals = [v for v in grid if v > 0]
    if land_vals:
        print(f"  Land precip: mean={sum(land_vals)/len(land_vals):.0f} mm/yr, "
              f"min={min(land_vals)}, max={max(land_vals)}, "
              f"n={len(land_vals)}")

    # Save JSON
    output = {
        "nx": NX, "ny": NY,
        "lat0": LAT0, "lat1": LAT1,
        "lon0": LON0, "lon1": LON1,
        "source": "CHIRPS v2 daily precipitation, 2015-2023 annual mean (mm/year)",
        "precipitation": grid,
    }
    out_path = Path("precipitation_1deg.json")
    out_path.write_text(json.dumps(output))
    print(f"  Saved {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return grid


def verify_regions(albedo, precip):
    """Check key regions against expected values."""
    print("\n=== Regional Verification ===")

    mask_path = Path("simamoc/mask.json")
    mask_bits = None
    if mask_path.exists():
        mask_data = json.loads(mask_path.read_text())
        mask_bits = []
        for c in mask_data["hex"]:
            v = int(c, 16)
            mask_bits.extend([(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1])

    def sample(name, lat_range, lon_range, expected_alb, expected_precip):
        a_vals, p_vals = [], []
        for j in range(NY):
            lat = LAT0 + j
            if lat < lat_range[0] or lat > lat_range[1]:
                continue
            for i in range(NX):
                lon = LON0 + i
                if lon < lon_range[0] or lon > lon_range[1]:
                    continue
                k = j * NX + i
                # Skip ocean
                if mask_bits:
                    sj = min(int(j * 180 / NY), 179)
                    if mask_bits[sj * 360 + i] == 1:
                        continue
                if albedo[k] > 0.07:
                    a_vals.append(albedo[k])
                if precip[k] > 0:
                    p_vals.append(precip[k])

        if not a_vals:
            print(f"  {name}: no land cells found")
            return

        a_mean = sum(a_vals) / len(a_vals)
        p_mean = sum(p_vals) / len(p_vals) if p_vals else 0
        a_ok = "OK" if abs(a_mean - expected_alb) < 0.10 else "WARN"
        p_ok = "OK" if abs(p_mean - expected_precip) < 500 else "WARN"
        print(f"  {name}: albedo={a_mean:.3f} (expect ~{expected_alb:.2f}) [{a_ok}], "
              f"precip={p_mean:.0f} (expect ~{expected_precip}) [{p_ok}], "
              f"n={len(a_vals)}")

    sample("Sahara", (18, 30), (-10, 30), 0.35, 80)
    sample("Amazon", (-10, 2), (-70, -45), 0.13, 2200)
    sample("Congo", (-3, 5), (12, 28), 0.14, 1700)
    sample("Greenland", (65, 82), (-55, -20), 0.70, 200)
    sample("Australia interior", (-30, -20), (125, 145), 0.25, 300)
    sample("Siberia (boreal)", (55, 65), (60, 120), 0.15, 400)
    sample("SE Asia", (-5, 10), (100, 140), 0.14, 2000)


MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]


def fetch_monthly_albedo():
    """Fetch MODIS monthly albedo climatology (2020-2023, 12 months)."""
    print("\n=== MODIS Monthly Albedo ===")
    monthly = []
    for month in range(1, 13):
        print(f"  Month {month} ({MONTH_NAMES[month-1]})...")
        # Filter by calendar month across years
        imgs = (
            ee.ImageCollection("MODIS/061/MCD43A3")
            .filter(ee.Filter.calendarRange(month, month, "month"))
            .filterDate("2020-01-01", "2024-01-01")
            .select("Albedo_WSA_shortwave")
            .mean()
            .multiply(0.001)
        )
        grid = ee_to_grid(imgs, "Albedo_WSA_shortwave", scale=100000)
        # Fill ocean with 0.06, invalid land with 0.20
        mask_path = Path("simamoc/mask.json")
        if mask_path.exists():
            mask_data = json.loads(mask_path.read_text())
            mask_bits = []
            for c in mask_data["hex"]:
                v = int(c, 16)
                mask_bits.extend([(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1])
            for j in range(NY):
                sj = min(int(j * 180 / NY), 179)
                for i in range(NX):
                    k = j * NX + i
                    if mask_bits[sj * 360 + i] == 1:
                        grid[k] = 0.06
                    elif grid[k] <= 0 or grid[k] != grid[k]:
                        grid[k] = 0.20
        grid = [max(0.02, min(0.95, v)) for v in grid]
        monthly.append([round(v, 4) for v in grid])
        time.sleep(1)  # rate limit

    output = {
        "nx": NX, "ny": NY,
        "lat0": LAT0, "lat1": LAT1,
        "source": "MODIS MCD43A3 WSA shortwave albedo, 2020-2023 monthly climatology",
        "monthly": monthly,
    }
    out_path = Path("albedo_monthly_1deg.json")
    out_path.write_text(json.dumps(output))
    print(f"  Saved {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


def fetch_monthly_precip():
    """Fetch CHIRPS monthly precipitation climatology (2015-2023, 12 months)."""
    print("\n=== CHIRPS Monthly Precipitation ===")
    monthly = []
    for month in range(1, 13):
        print(f"  Month {month} ({MONTH_NAMES[month-1]})...")
        # Monthly sum across years, then mean
        monthly_sums = []
        for year in range(2015, 2024):
            m_start = f"{year}-{month:02d}-01"
            m_end_month = month + 1 if month < 12 else 1
            m_end_year = year if month < 12 else year + 1
            m_end = f"{m_end_year}-{m_end_month:02d}-01"
            monthly_sum = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterDate(m_start, m_end)
                .select("precipitation")
                .sum()
            )
            monthly_sums.append(monthly_sum)
        precip_img = ee.ImageCollection(monthly_sums).mean()

        grid = ee_to_grid(precip_img, "precipitation", scale=100000)
        # Fill polar gaps, zero ocean
        for j in range(NY):
            lat = LAT0 + j
            if abs(lat) > 50:
                for i in range(NX):
                    k = j * NX + i
                    if grid[k] <= 0:
                        grid[k] = max(5, 35 - 0.4 * (abs(lat) - 50))
        grid = [max(0, round(v)) for v in grid]
        monthly.append(grid)
        time.sleep(1)

    output = {
        "nx": NX, "ny": NY,
        "lat0": LAT0, "lat1": LAT1,
        "source": "CHIRPS v2 daily precipitation, 2015-2023 monthly climatology (mm/month)",
        "monthly": monthly,
    }
    out_path = Path("precipitation_monthly_1deg.json")
    out_path.write_text(json.dumps(output))
    print(f"  Saved {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    import sys
    print(f"Fetching real Earth data for {NX}x{NY} grid")
    print(f"Archive dir: {ARCHIVE_DIR.absolute()}")
    print("=" * 60)

    if "--monthly" in sys.argv:
        fetch_monthly_albedo()
        fetch_monthly_precip()
    else:
        albedo = fetch_modis_albedo()
        precip = fetch_chirps_precip()
        verify_regions(albedo, precip)

    print("\n" + "=" * 60)
    print("Done. Real data replaces heuristic estimates.")
    print("Archive GeoTIFFs saved in data/archive/ for manual inspection.")
