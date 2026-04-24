#!/usr/bin/env python3
"""
Fetch GODAS surface ocean currents (u, v) via OPeNDAP and produce
a 360x180 (1-degree, -180 to 180, -80 to 80) JSON file for SimAMOC init.

Output: godas_currents_1deg.json with {u: [...], v: [...], nx, ny, lat0, lat1}
"""

import urllib.request
import json
import re
import sys

BASE = "https://psl.noaa.gov/thredds/dodsC/Datasets/godas/Derived"
OUT_NX, OUT_NY = 360, 160  # match SimAMOC grid (LON0=-180, LAT0=-80, LAT1=80)
OUT_LAT0, OUT_LAT1 = -80, 80
OUT_LON0, OUT_LON1 = -180, 180

def fetch_ascii(var_file, query):
    url = f"{BASE}/{var_file}.mon.ltm.1991-2020.nc.ascii?{query}"
    print(f"  Fetching {url[:120]}...", flush=True)
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read().decode('utf-8')

def parse_1d(text):
    """Parse OPeNDAP ASCII 1D array"""
    vals = []
    data_started = False
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('---'):
            data_started = True
            continue
        if not data_started:
            continue
        if not line or line.startswith('}'):
            continue
        # Skip dimension header lines like "lat[418]"
        if re.match(r'^[a-z]+\[\d+\]$', line):
            continue
        for v in line.split(','):
            v = v.strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    return vals

def parse_3d_surface(text, nlat, nlon, ntime=12):
    """Parse OPeNDAP ASCII for [time][lat][lon] (level already sliced to 0)"""
    lines = text.split('\n')
    data_started = False
    vals = []
    for line in lines:
        line = line.strip()
        if line.startswith('---'):
            data_started = True
            continue
        if not data_started:
            continue
        if not line or line.startswith('}'):
            continue
        # Skip array header like "ucur.ucur[12][1][105][360]"
        if re.match(r'^[a-z_]+\.[a-z_]+\[', line):
            continue
        # Skip dimension index lines like "[0][0][0]"
        if re.match(r'^\[[\d\]\[]+$', line):
            continue
        # Data lines contain comma-separated values
        for v in line.split(','):
            v = v.strip()
            if v:
                try:
                    vals.append(float(v))
                except ValueError:
                    pass

    expected = ntime * nlat * nlon
    print(f"  Parsed {len(vals)} values (expected {expected})")

    # Reshape to [time][lat][lon]
    data = []
    idx = 0
    for t in range(ntime):
        month = []
        for j in range(nlat):
            row = vals[idx:idx+nlon]
            month.append(row)
            idx += nlon
        data.append(month)
    return data

def regrid_to_1deg(data_3d, src_lats, src_lons, ntime):
    """Average 12 months, regrid from GODAS grid to our 360x180 grid"""
    nlat_src = len(src_lats)
    nlon_src = len(src_lons)

    out = [0.0] * (OUT_NX * OUT_NY)
    count = [0] * (OUT_NX * OUT_NY)

    for t in range(ntime):
        for jj, lat in enumerate(src_lats):
            if lat < OUT_LAT0 or lat > OUT_LAT1:
                continue
            # Map to output j
            oj = round((lat - OUT_LAT0) / (OUT_LAT1 - OUT_LAT0) * (OUT_NY - 1))
            if oj < 0 or oj >= OUT_NY:
                continue

            for ii, lon in enumerate(src_lons):
                # GODAS uses 0-360 longitude, convert to -180 to 180
                lon180 = lon if lon <= 180 else lon - 360
                oi = round((lon180 - OUT_LON0) / (OUT_LON1 - OUT_LON0) * (OUT_NX - 1))
                oi = oi % OUT_NX

                val = data_3d[t][jj][ii]
                # GODAS missing value is very large
                if abs(val) > 100:
                    continue

                k = oj * OUT_NX + oi
                out[k] += val
                count[k] += 1

    # Average
    for k in range(OUT_NX * OUT_NY):
        if count[k] > 0:
            out[k] /= count[k]
        else:
            out[k] = 0.0  # no data (land or out of GODAS domain)

    return out

def main():
    print("Fetching GODAS surface currents (1991-2020 climatology)...")

    # Fetch coordinates
    print("\n1. Coordinates")
    lat_text = fetch_ascii("ucur", "lat%5B0:4:417%5D")  # stride 4 → ~105 points
    src_lats = parse_1d(lat_text)
    print(f"  Lats: {len(src_lats)} values, {src_lats[0]:.1f} to {src_lats[-1]:.1f}")

    lon_text = fetch_ascii("ucur", "lon%5B0:359%5D")
    src_lons = parse_1d(lon_text)
    print(f"  Lons: {len(src_lons)} values, {src_lons[0]:.1f} to {src_lons[-1]:.1f}")

    nlat = len(src_lats)
    nlon = len(src_lons)

    # Fetch ucur: all months, surface (level=0), strided lat, all lon
    print(f"\n2. Fetching ucur [12 months x surface x {nlat} lat x {nlon} lon]")
    ucur_text = fetch_ascii("ucur", f"ucur%5B0:11%5D%5B0%5D%5B0:4:417%5D%5B0:359%5D")
    ucur_3d = parse_3d_surface(ucur_text, nlat, nlon)

    # Fetch vcur
    print(f"\n3. Fetching vcur [12 months x surface x {nlat} lat x {nlon} lon]")
    vcur_text = fetch_ascii("vcur", f"vcur%5B0:11%5D%5B0%5D%5B0:4:417%5D%5B0:359%5D")
    vcur_3d = parse_3d_surface(vcur_text, nlat, nlon)

    # Regrid to 1-degree
    print("\n4. Regridding to 360x160 (1-degree)...")
    u_out = regrid_to_1deg(ucur_3d, src_lats, src_lons, 12)
    v_out = regrid_to_1deg(vcur_3d, src_lats, src_lons, 12)

    # Round to 4 decimal places to save space
    u_out = [round(v, 4) for v in u_out]
    v_out = [round(v, 4) for v in v_out]

    # Stats
    u_valid = [v for v in u_out if v != 0]
    v_valid = [v for v in v_out if v != 0]
    print(f"  U: {len(u_valid)} ocean cells, range [{min(u_valid):.3f}, {max(u_valid):.3f}] m/s")
    print(f"  V: {len(v_valid)} ocean cells, range [{min(v_valid):.3f}, {max(v_valid):.3f}] m/s")

    # Save
    output = {
        "u": u_out,
        "v": v_out,
        "nx": OUT_NX,
        "ny": OUT_NY,
        "lat0": OUT_LAT0,
        "lat1": OUT_LAT1,
        "lon0": OUT_LON0,
        "lon1": OUT_LON1,
        "source": "GODAS 1991-2020 LTM surface currents",
        "units": "m/s",
    }

    outpath = "godas_currents_1deg.json"
    with open(outpath, 'w') as f:
        json.dump(output, f)

    import os
    size_mb = os.path.getsize(outpath) / 1e6
    print(f"\n  Saved to {outpath} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
