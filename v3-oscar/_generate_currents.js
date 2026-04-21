// Generate synthetic global ocean current velocity field
// Based on known large-scale circulation patterns
// Run with: node _generate_currents.js

const fs = require('fs');

const nx = 360, ny = 180;
const lon0 = -179.5, lat0 = -89.5, step = 1;
const u = new Float64Array(ny * nx);
const v = new Float64Array(ny * nx);

function degToRad(d) { return d * Math.PI / 180; }

// Simple land mask - returns true if point is likely land
function isLand(lat, lon) {
  // Normalize lon to -180..180
  if (lon > 180) lon -= 360;
  if (lon < -180) lon += 360;

  // Antarctica
  if (lat < -60 && lat > -90) {
    // Some ocean near edges
    if (lat > -65) return false;
    return true;
  }

  // Africa
  if (lon >= -20 && lon <= 52 && lat >= -35 && lat <= 37) {
    // Rough Africa shape
    if (lat >= -35 && lat <= -22 && lon >= 15 && lon <= 35) return true;
    if (lat >= -22 && lat <= 0 && lon >= 8 && lon <= 42) return true;
    if (lat >= 0 && lat <= 12 && lon >= -15 && lon <= 50) return true;
    if (lat >= 12 && lat <= 25 && lon >= -17 && lon <= 40) return true;
    if (lat >= 25 && lat <= 37 && lon >= -10 && lon <= 35) return true;
  }

  // Europe
  if (lon >= -10 && lon <= 40 && lat >= 36 && lat <= 71) {
    if (lat >= 36 && lat <= 45 && lon >= -10 && lon <= 3) return true; // Iberia
    if (lat >= 43 && lat <= 51 && lon >= -5 && lon <= 8) return true; // France
    if (lat >= 46 && lat <= 55 && lon >= 5 && lon <= 15) return true; // Central Europe
    if (lat >= 54 && lat <= 58 && lon >= -8 && lon <= 2) return true; // UK
    if (lat >= 56 && lat <= 71 && lon >= 4 && lon <= 30) return true; // Scandinavia
    if (lat >= 36 && lat <= 42 && lon >= 12 && lon <= 20) return true; // Italy
  }

  // North America
  if (lon >= -170 && lon <= -50 && lat >= 15 && lat <= 75) {
    if (lat >= 25 && lat <= 50 && lon >= -130 && lon <= -65) return true;
    if (lat >= 50 && lat <= 60 && lon >= -140 && lon <= -55) return true;
    if (lat >= 60 && lat <= 75 && lon >= -170 && lon <= -55) return true;
    if (lat >= 15 && lat <= 25 && lon >= -105 && lon <= -85) return true; // Mexico
  }

  // Central America
  if (lat >= 7 && lat <= 20 && lon >= -92 && lon <= -77) return true;

  // South America
  if (lon >= -82 && lon <= -34 && lat >= -56 && lat <= 12) {
    if (lat >= -56 && lat <= -40 && lon >= -76 && lon <= -63) return true;
    if (lat >= -40 && lat <= -20 && lon >= -72 && lon <= -40) return true;
    if (lat >= -20 && lat <= 0 && lon >= -80 && lon <= -35) return true;
    if (lat >= 0 && lat <= 12 && lon >= -80 && lon <= -50) return true;
  }

  // Greenland
  if (lat >= 60 && lat <= 84 && lon >= -55 && lon <= -15) return true;

  // Asia
  if (lon >= 40 && lon <= 180 && lat >= 10 && lat <= 75) {
    if (lat >= 40 && lat <= 75 && lon >= 40 && lon <= 180) return true; // Russia/N.Asia
    if (lat >= 20 && lat <= 45 && lon >= 60 && lon <= 135) return true; // Central/East Asia
    if (lat >= 10 && lat <= 25 && lon >= 68 && lon <= 90) return true; // India
    if (lat >= 10 && lat <= 20 && lon >= 95 && lon <= 110) return true; // SE Asia mainland
  }
  // Far east Russia/Kamchatka extended
  if (lat >= 50 && lat <= 65 && lon >= 130 && lon <= 175) return true;

  // Australia
  if (lat >= -40 && lat <= -11 && lon >= 113 && lon <= 155) return true;

  // Indonesia (rough)
  if (lat >= -8 && lat <= 5 && lon >= 95 && lon <= 140) {
    if (lon >= 95 && lon <= 106) return true; // Sumatra
    if (lon >= 106 && lon <= 115 && lat >= -8 && lat <= -1) return true; // Java/Borneo S
    if (lon >= 115 && lon <= 140 && lat >= -5 && lat <= 2) return true; // More islands
  }

  // New Zealand
  if (lat >= -47 && lat <= -34 && lon >= 166 && lon <= 179) return true;

  // Madagascar
  if (lat >= -26 && lat <= -12 && lon >= 43 && lon <= 50) return true;

  // Arabian Peninsula
  if (lat >= 12 && lat <= 30 && lon >= 35 && lon <= 60) return true;

  // Arctic land masses
  if (lat >= 70) {
    if (lon >= 40 && lon <= 180) return true; // Siberian coast
    if (lon >= -180 && lon <= -140) return true; // Alaska north
  }

  return false;
}

// Gaussian bump function
function gauss(x, mu, sigma) {
  const d = (x - mu) / sigma;
  return Math.exp(-0.5 * d * d);
}

// Wind stress curl pattern (simplified)
// Positive in subtropics (anticyclonic), negative in subpolar (cyclonic)
function windStressCurl(lat) {
  // Subtropical: centered ~30N/S, anticyclonic (drives equatorward Sverdrup flow)
  // Subpolar: centered ~55N, cyclonic
  // Trades: ~15N/S
  return -Math.sin(degToRad(lat) * 3); // Simple sinusoidal pattern
}

// Generate currents
for (let j = 0; j < ny; j++) {
  const lat = lat0 + j * step;
  const latRad = degToRad(lat);
  const cosLat = Math.cos(latRad);

  if (Math.abs(cosLat) < 0.01) continue; // Skip poles

  for (let i = 0; i < nx; i++) {
    const lon = lon0 + i * step;
    const idx = j * nx + i;

    if (isLand(lat, lon)) {
      u[idx] = 0;
      v[idx] = 0;
      continue;
    }

    let uu = 0, vv = 0;

    // ===============================
    // 1. ANTARCTIC CIRCUMPOLAR CURRENT
    // ===============================
    // Strong eastward flow around 55S
    const accStrength = 0.35 * gauss(lat, -55, 7);
    uu += accStrength;

    // ===============================
    // 2. SUBTROPICAL GYRES (anticyclonic)
    // ===============================

    // -- North Atlantic subtropical gyre --
    if (lon >= -80 && lon <= 0 && lat >= 15 && lat <= 50) {
      const gyreCenter = {lat: 32, lon: -40};
      const dLat = (lat - gyreCenter.lat) / 18;
      const dLon = (lon - gyreCenter.lon) / 40;
      const r2 = dLat * dLat + dLon * dLon;
      if (r2 < 1) {
        const strength = 0.15 * (1 - r2);
        // Anticyclonic: clockwise in NH
        uu += -strength * dLat * 2;
        vv += strength * dLon * 2;
      }
    }

    // -- South Atlantic subtropical gyre --
    if (lon >= -60 && lon <= 20 && lat >= -45 && lat <= -5) {
      const gyreCenter = {lat: -25, lon: -20};
      const dLat = (lat - gyreCenter.lat) / 20;
      const dLon = (lon - gyreCenter.lon) / 40;
      const r2 = dLat * dLat + dLon * dLon;
      if (r2 < 1) {
        const strength = 0.12 * (1 - r2);
        // Anticyclonic: counter-clockwise in SH
        uu += strength * dLat * 2;
        vv += -strength * dLon * 2;
      }
    }

    // -- North Pacific subtropical gyre --
    if (lon >= 120 || (lon >= -180 && lon <= -100)) {
      let effLon = lon;
      if (effLon < 0) effLon += 360;
      const gyreCenter = {lat: 30, effLon: 200}; // ~160W
      const dLat = (lat - gyreCenter.lat) / 20;
      const dLon = (effLon - gyreCenter.effLon) / 50;
      const r2 = dLat * dLat + dLon * dLon;
      if (r2 < 1 && lat >= 10 && lat <= 50) {
        const strength = 0.15 * (1 - r2);
        uu += -strength * dLat * 2;
        vv += strength * dLon * 2;
      }
    }

    // -- South Pacific subtropical gyre --
    if ((lon >= 150 || (lon >= -180 && lon <= -70)) && lat >= -50 && lat <= -5) {
      let effLon = lon;
      if (effLon < 0) effLon += 360;
      const gyreCenter = {lat: -28, effLon: 230}; // ~130W
      const dLat = (lat - gyreCenter.lat) / 22;
      const dLon = (effLon - gyreCenter.effLon) / 60;
      const r2 = dLat * dLat + dLon * dLon;
      if (r2 < 1) {
        const strength = 0.1 * (1 - r2);
        uu += strength * dLat * 2;
        vv += -strength * dLon * 2;
      }
    }

    // -- Indian Ocean subtropical gyre --
    if (lon >= 20 && lon <= 120 && lat >= -45 && lat <= -5) {
      const gyreCenter = {lat: -25, lon: 70};
      const dLat = (lat - gyreCenter.lat) / 20;
      const dLon = (lon - gyreCenter.lon) / 50;
      const r2 = dLat * dLat + dLon * dLon;
      if (r2 < 1) {
        const strength = 0.12 * (1 - r2);
        uu += strength * dLat * 2;
        vv += -strength * dLon * 2;
      }
    }

    // ===============================
    // 3. WESTERN BOUNDARY CURRENTS
    // ===============================

    // Gulf Stream
    if (lon >= -82 && lon <= -40 && lat >= 25 && lat <= 45) {
      const gsLat = 25 + (lon + 82) * 0.4; // Path curves NE
      const dist = (lat - gsLat) / 3;
      const along = gauss(dist, 0, 1);
      const lonFade = gauss(lon, -65, 15);
      const strength = 1.0 * along * lonFade;
      uu += strength * 0.7;
      vv += strength * 0.5;
    }

    // North Atlantic Drift (extension of Gulf Stream)
    if (lon >= -40 && lon <= 0 && lat >= 42 && lat <= 62) {
      const drift = 0.25 * gauss(lat, 52, 8) * gauss(lon, -20, 20);
      uu += drift * 0.8;
      vv += drift * 0.4;
    }

    // Kuroshio Current
    if (lon >= 120 && lon <= 170 && lat >= 20 && lat <= 42) {
      const ksLat = 22 + (lon - 120) * 0.35;
      const dist = (lat - ksLat) / 3;
      const along = gauss(dist, 0, 1);
      const lonFade = gauss(lon, 145, 15);
      const strength = 0.9 * along * lonFade;
      uu += strength * 0.6;
      vv += strength * 0.5;
    }

    // Brazil Current
    if (lon >= -55 && lon <= -30 && lat >= -40 && lat <= -15) {
      const bcLon = -50 + (lat + 40) * 0.3;
      const dist = (lon - bcLon) / 3;
      const along = gauss(dist, 0, 1);
      const strength = 0.4 * along * gauss(lat, -28, 10);
      uu += strength * 0.2;
      vv += -strength * 0.8;
    }

    // Agulhas Current
    if (lon >= 25 && lon <= 45 && lat >= -40 && lat <= -25) {
      const strength = 0.6 * gauss(lon, 33, 5) * gauss(lat, -33, 5);
      uu += -strength * 0.2;
      vv += -strength * 0.8;
    }

    // East Australian Current
    if (lon >= 148 && lon <= 165 && lat >= -38 && lat <= -20) {
      const strength = 0.4 * gauss(lon, 155, 5) * gauss(lat, -30, 7);
      uu += strength * 0.2;
      vv += -strength * 0.8;
    }

    // ===============================
    // 4. EQUATORIAL CURRENTS
    // ===============================

    // North Equatorial Current (westward, ~15N)
    const necStrength = 0.2 * gauss(lat, 15, 5);
    if (Math.abs(lat - 15) < 10) uu -= necStrength;

    // South Equatorial Current (westward, ~5S to equator)
    const secStrength = 0.25 * gauss(lat, -3, 5);
    if (Math.abs(lat + 3) < 10) uu -= secStrength;

    // Equatorial Countercurrent (eastward, ~5N)
    const eccStrength = 0.2 * gauss(lat, 5, 3);
    if (Math.abs(lat - 5) < 8) uu += eccStrength;

    // ===============================
    // 5. SUBPOLAR GYRES (cyclonic)
    // ===============================

    // North Atlantic subpolar gyre
    if (lon >= -60 && lon <= 0 && lat >= 48 && lat <= 68) {
      const gyreCenter = {lat: 58, lon: -30};
      const dLat = (lat - gyreCenter.lat) / 10;
      const dLon = (lon - gyreCenter.lon) / 30;
      const r2 = dLat * dLat + dLon * dLon;
      if (r2 < 1) {
        const strength = 0.1 * (1 - r2);
        // Cyclonic: counter-clockwise in NH
        uu += strength * dLat * 2;
        vv += -strength * dLon * 2;
      }
    }

    // North Pacific subpolar gyre (Alaska Gyre)
    if ((lon >= 150 || (lon >= -180 && lon <= -140)) && lat >= 42 && lat <= 60) {
      let effLon = lon;
      if (effLon < 0) effLon += 360;
      const gyreCenter = {lat: 52, effLon: 190};
      const dLat = (lat - gyreCenter.lat) / 10;
      const dLon = (effLon - gyreCenter.effLon) / 30;
      const r2 = dLat * dLat + dLon * dLon;
      if (r2 < 1) {
        const strength = 0.1 * (1 - r2);
        uu += strength * dLat * 2;
        vv += -strength * dLon * 2;
      }
    }

    // ===============================
    // 6. EASTERN BOUNDARY / UPWELLING CURRENTS
    // ===============================

    // California Current (equatorward along US west coast)
    if (lon >= -135 && lon <= -115 && lat >= 20 && lat <= 50) {
      const strength = 0.2 * gauss(lon, -125, 5) * gauss(lat, 35, 12);
      uu += -strength * 0.3;
      vv += -strength * 0.7;
    }

    // Canary Current
    if (lon >= -25 && lon <= -5 && lat >= 15 && lat <= 40) {
      const strength = 0.15 * gauss(lon, -15, 7) * gauss(lat, 28, 10);
      uu += -strength * 0.2;
      vv += -strength * 0.7;
    }

    // Benguela Current
    if (lon >= 5 && lon <= 20 && lat >= -35 && lat <= -15) {
      const strength = 0.2 * gauss(lon, 12, 5) * gauss(lat, -25, 8);
      uu += -strength * 0.2;
      vv += strength * 0.7;
    }

    // Peru/Humboldt Current
    if (lon >= -85 && lon <= -70 && lat >= -40 && lat <= -5) {
      const strength = 0.25 * gauss(lon, -78, 5) * gauss(lat, -22, 12);
      uu += -strength * 0.2;
      vv += strength * 0.7;
    }

    // ===============================
    // 7. ADDITIONAL CURRENTS
    // ===============================

    // Labrador Current (southward along Labrador/Newfoundland)
    if (lon >= -60 && lon <= -45 && lat >= 42 && lat <= 60) {
      const strength = 0.25 * gauss(lon, -53, 5) * gauss(lat, 50, 8);
      vv -= strength * 0.8;
      uu -= strength * 0.2;
    }

    // Somali Current (seasonally northward)
    if (lon >= 42 && lon <= 55 && lat >= -5 && lat <= 12) {
      const strength = 0.3 * gauss(lon, 48, 4) * gauss(lat, 4, 6);
      vv += strength * 0.8;
    }

    // Mozambique Current
    if (lon >= 35 && lon <= 45 && lat >= -25 && lat <= -10) {
      const strength = 0.3 * gauss(lon, 40, 4) * gauss(lat, -18, 6);
      vv -= strength * 0.8;
    }

    // Add some noise for realism
    uu += (Math.random() - 0.5) * 0.02;
    vv += (Math.random() - 0.5) * 0.02;

    // Zero out near land (soft mask)
    u[idx] = uu;
    v[idx] = vv;
  }
}

// Smooth the field to remove sharp edges (simple 3x3 box filter, 2 passes)
function smooth(arr) {
  const tmp = new Float64Array(ny * nx);
  for (let pass = 0; pass < 3; pass++) {
    const src = pass === 0 ? arr : tmp;
    for (let j = 1; j < ny - 1; j++) {
      for (let i = 1; i < nx - 1; i++) {
        const lat = lat0 + j * step;
        const lon = lon0 + i * step;
        if (isLand(lat, lon)) { tmp[j * nx + i] = 0; continue; }
        let sum = 0, cnt = 0;
        for (let dj = -1; dj <= 1; dj++) {
          for (let di = -1; di <= 1; di++) {
            const nLat = lat0 + (j + dj) * step;
            const nLon = lon0 + (i + di) * step;
            if (!isLand(nLat, nLon)) {
              sum += src[(j + dj) * nx + (i + di)];
              cnt++;
            }
          }
        }
        tmp[j * nx + i] = cnt > 0 ? sum / cnt : 0;
      }
    }
    for (let k = 0; k < ny * nx; k++) arr[k] = tmp[k];
  }
}

smooth(u);
smooth(v);

// Convert to regular arrays with 3 decimal places
const uArr = Array.from(u).map(x => Math.round(x * 1000) / 1000);
const vArr = Array.from(v).map(x => Math.round(x * 1000) / 1000);

const data = {
  nx, ny, lon0, lat0, step,
  u: uArr,
  v: vArr
};

const json = JSON.stringify(data);
fs.writeFileSync('/Users/dereklomas/lukebarrington/amoc/v3-oscar/currents.json', json);
console.log(`Written currents.json: ${(json.length / 1024 / 1024).toFixed(1)} MB`);
console.log(`Grid: ${nx}x${ny} = ${nx * ny} cells`);

// Quick stats
let maxSpeed = 0;
for (let k = 0; k < ny * nx; k++) {
  const spd = Math.sqrt(uArr[k] * uArr[k] + vArr[k] * vArr[k]);
  if (spd > maxSpeed) maxSpeed = spd;
}
console.log(`Max speed: ${maxSpeed.toFixed(3)} m/s`);
