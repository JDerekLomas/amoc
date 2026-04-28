// Load observational data from root-level *_1deg.json files
// These are 360x160 grids (lat -79.5 to 79.5) with plain JSON arrays
// Mask is 360x180 hex-encoded

// Resample a 360x160 field to target grid (nearest-neighbor)
function resampleField(src, srcNx, srcNy, dstNx, dstNy) {
  const dst = new Float32Array(dstNx * dstNy);
  for (let dj = 0; dj < dstNy; dj++) {
    const sj = Math.min(Math.floor(dj * srcNy / dstNy), srcNy - 1);
    for (let di = 0; di < dstNx; di++) {
      const si = Math.min(Math.floor(di * srcNx / dstNx), srcNx - 1);
      dst[dj * dstNx + di] = src[sj * srcNx + si];
    }
  }
  return dst;
}

// Load a scalar field from a 1deg JSON file
export async function loadField1deg(url, key, grid) {
  const data = await fetch(url).then(r => r.json());
  const srcNx = data.nx;
  const srcNy = data.ny;
  const arr = data[key];
  if (!arr) throw new Error(`Key "${key}" not found in ${url}`);

  const src = new Float32Array(arr);
  if (srcNx === grid.nx && srcNy === grid.ny) return src;
  return resampleField(src, srcNx, srcNy, grid.nx, grid.ny);
}

// Load wind stress (has tau_x, tau_y as separate arrays)
export async function loadWindStress(url, grid) {
  const data = await fetch(url).then(r => r.json());
  const srcNx = data.nx;
  const srcNy = data.ny;

  let tauX = new Float32Array(data.tau_x);
  let tauY = new Float32Array(data.tau_y);

  if (srcNx !== grid.nx || srcNy !== grid.ny) {
    tauX = resampleField(tauX, srcNx, srcNy, grid.nx, grid.ny);
    tauY = resampleField(tauY, srcNx, srcNy, grid.nx, grid.ny);
  }
  return { tauX, tauY };
}

// Load land/ocean mask from hex-encoded bitmask (360x180)
export async function loadMask(url, grid) {
  const data = await fetch(url).then(r => r.json());
  const srcNx = data.nx;
  const srcNy = data.ny;

  // Decode hex to bit array
  const hex = data.hex;
  const bits = new Uint8Array(srcNx * srcNy);
  for (let c = 0; c < hex.length; c++) {
    const v = parseInt(hex[c], 16);
    const base = c * 4;
    if (base < bits.length) bits[base] = (v >> 3) & 1;
    if (base + 1 < bits.length) bits[base + 1] = (v >> 2) & 1;
    if (base + 2 < bits.length) bits[base + 2] = (v >> 1) & 1;
    if (base + 3 < bits.length) bits[base + 3] = v & 1;
  }

  // Resample if needed
  const mask = grid.createField();
  for (let dj = 0; dj < grid.ny; dj++) {
    const sj = Math.min(Math.floor(dj * srcNy / grid.ny), srcNy - 1);
    for (let di = 0; di < grid.nx; di++) {
      const si = Math.min(Math.floor(di * srcNx / grid.nx), srcNx - 1);
      mask[dj * grid.nx + di] = bits[sj * srcNx + si];
    }
  }

  // Enforce polar boundaries as land
  for (let i = 0; i < grid.nx; i++) {
    mask[i] = 0;
    mask[(grid.ny - 1) * grid.nx + i] = 0;
  }

  return mask;
}
