/**
 * SimAMOC v2 — State Management
 *
 * All model state lives in flat typed arrays. Physics operates on state.
 * State can be serialized, snapshotted, and restored.
 *
 * Grid: equirectangular, 2:1 aspect ratio (NX = 2*NY).
 * Convention: row 0 = south (LAT0), row NY-1 = north (LAT1).
 * Longitude wraps periodically.
 */

export const DEFAULT_NX = 256;
export const DEFAULT_NY = 128;
export const LAT0 = -79.5;
export const LAT1 = 79.5;
export const LON0 = -180;
export const LON1 = 180;

export function createGrid(nx = DEFAULT_NX, ny = DEFAULT_NY) {
  const dx = (LON1 - LON0) / nx;          // degrees per cell in x
  const dy = (LAT1 - LAT0) / (ny - 1);    // degrees per cell in y
  const dxRad = dx * Math.PI / 180;
  const dyRad = dy * Math.PI / 180;
  const R = 6.371e6;                        // Earth radius (m)
  const dxM = R * dxRad;                    // meters per cell at equator
  const dyM = R * dyRad;

  // Precompute latitude and cos(lat) for each row
  const lat = new Float32Array(ny);
  const cosLat = new Float32Array(ny);
  for (let j = 0; j < ny; j++) {
    lat[j] = LAT0 + j * dy;
    cosLat[j] = Math.cos(lat[j] * Math.PI / 180);
  }

  return {
    nx, ny, n: nx * ny,
    dx, dy, dxRad, dyRad, dxM, dyM, R,
    lat, cosLat,
    LAT0, LAT1, LON0, LON1,

    // Index helpers
    idx(i, j) { return j * nx + i; },
    iWrap(i) { return ((i % nx) + nx) % nx; },
    lonAt(i) { return LON0 + i * dx; },
    latAt(j) { return lat[j]; },
  };
}

/**
 * Create the full model state. All fields are flat Float32Arrays.
 */
export function createState(grid) {
  const { n } = grid;
  return {
    // Ocean surface
    temp: new Float32Array(n),        // SST (°C)
    sal: new Float32Array(n),         // salinity (PSU)
    psi: new Float32Array(n),         // streamfunction
    zeta: new Float32Array(n),        // vorticity

    // Ocean deep layer
    deepTemp: new Float32Array(n),    // deep temperature (°C)
    deepSal: new Float32Array(n),     // deep salinity (PSU)
    deepPsi: new Float32Array(n),     // deep streamfunction
    deepZeta: new Float32Array(n),    // deep vorticity

    // Atmosphere
    airTemp: new Float32Array(n),     // surface air temperature (°C)
    humidity: new Float32Array(n),    // specific humidity (kg/kg)
    precip: new Float32Array(n),      // precipitation rate (mm/day)
    evap: new Float32Array(n),        // evaporation rate (mm/day)
    cloudFrac: new Float32Array(n),   // cloud fraction (0-1)

    // Ice
    iceFrac: new Float32Array(n),     // sea ice fraction (0-1)

    // Masks and geometry (Uint8)
    mask: new Uint8Array(n),          // 1=ocean, 0=land

    // Forcing (read-only, loaded from data)
    windCurlTau: new Float32Array(n), // wind stress curl (N/m³)
    obsSSTTarget: new Float32Array(n),// observed SST for init/validation
    obsSalTarget: new Float32Array(n),// observed salinity for restoring

    // Simulation metadata
    step: 0,
    simTime: 0,
  };
}

/**
 * Snapshot state to a serializable object.
 */
export function snapshot(state) {
  const snap = { step: state.step, simTime: state.simTime };
  for (const [key, val] of Object.entries(state)) {
    if (val instanceof Float32Array || val instanceof Uint8Array) {
      snap[key] = Array.from(val);
    }
  }
  return snap;
}

/**
 * Restore state from a snapshot.
 */
export function restore(state, snap) {
  state.step = snap.step || 0;
  state.simTime = snap.simTime || 0;
  for (const [key, val] of Object.entries(snap)) {
    if (Array.isArray(val) && state[key]) {
      state[key].set(val);
    }
  }
}
