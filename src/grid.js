// Grid descriptor and state array factory
// Flat typed arrays, south-first orientation (row 0 = southernmost latitude)
// Matches observational data convention (lat0=-79.5 at j=0, lat1=79.5 at j=ny-1)

import { EARTH, SIMULATION } from './params.js';

export class Grid {
  constructor(nx = SIMULATION.nx, ny = SIMULATION.ny,
              latMin = SIMULATION.latMin, latMax = SIMULATION.latMax) {
    this.nx = nx;
    this.ny = ny;
    this.size = nx * ny;

    // Latitude array (south-first: index 0 = latMin)
    this.lat = new Float64Array(ny);
    this.lon = new Float64Array(nx);
    this.dlat = (latMax - latMin) / ny;
    this.dlon = 360 / nx;

    for (let j = 0; j < ny; j++) {
      this.lat[j] = latMin + (j + 0.5) * this.dlat;
    }
    for (let i = 0; i < nx; i++) {
      this.lon[i] = (i + 0.5) * this.dlon;
    }

    // Precompute trig and metric terms
    this.cosLat = new Float64Array(ny);
    this.sinLat = new Float64Array(ny);
    this.f = new Float64Array(ny);        // Coriolis parameter
    this.beta = new Float64Array(ny);     // df/dy
    this.dy = new Float64Array(ny);       // cell height in meters
    this.dx = new Float64Array(ny);       // cell width in meters (varies with lat)

    const dlatRad = (this.dlat * Math.PI) / 180;
    const dlonRad = (this.dlon * Math.PI) / 180;

    for (let j = 0; j < ny; j++) {
      const latRad = (this.lat[j] * Math.PI) / 180;
      this.cosLat[j] = Math.cos(latRad);
      this.sinLat[j] = Math.sin(latRad);
      this.f[j] = 2 * EARTH.omega * this.sinLat[j];
      this.beta[j] = (2 * EARTH.omega * this.cosLat[j]) / EARTH.radius;
      this.dy[j] = EARTH.radius * dlatRad;
      this.dx[j] = EARTH.radius * this.cosLat[j] * dlonRad;
    }
  }

  // Index into flat array
  idx(i, j) {
    return j * this.nx + i;
  }

  // Periodic in x, clamped in y
  wrap(i, j) {
    i = ((i % this.nx) + this.nx) % this.nx;
    j = Math.max(0, Math.min(this.ny - 1, j));
    return this.idx(i, j);
  }

  // Create a new Float32Array for this grid
  createField(fill = 0) {
    const arr = new Float32Array(this.size);
    if (fill !== 0) arr.fill(fill);
    return arr;
  }
}
