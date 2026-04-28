// Perceptually uniform colormaps
// Each is an array of [r, g, b] stops (0-255), sampled at even intervals

// Thermal (cmocean) - for SST
const THERMAL_STOPS = [
  [4, 35, 51], [9, 56, 82], [13, 75, 107], [26, 94, 120],
  [46, 112, 121], [66, 129, 116], [91, 144, 105], [122, 158, 89],
  [157, 170, 72], [193, 180, 60], [227, 189, 54], [249, 202, 62],
  [253, 219, 96], [254, 237, 144], [254, 253, 195],
];

// RdBu_r (diverging) - for anomalies
const RDBU_STOPS = [
  [103, 0, 31], [178, 24, 43], [214, 96, 77], [239, 138, 98],
  [253, 187, 132], [253, 219, 199], [247, 247, 247],
  [209, 229, 240], [146, 197, 222], [67, 147, 195],
  [33, 102, 172], [5, 48, 97],
];

// Viridis - general sequential
const VIRIDIS_STOPS = [
  [68, 1, 84], [72, 26, 108], [71, 47, 126], [65, 68, 135],
  [57, 86, 140], [49, 104, 142], [42, 120, 142], [35, 136, 141],
  [31, 152, 139], [34, 168, 132], [53, 183, 121], [83, 197, 104],
  [122, 209, 81], [168, 219, 52], [217, 226, 25], [253, 231, 37],
];

function interpolateColormap(stops, t) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1;
  const idx = t * n;
  const i0 = Math.floor(idx);
  const i1 = Math.min(i0 + 1, n);
  const f = idx - i0;
  return [
    stops[i0][0] + f * (stops[i1][0] - stops[i0][0]),
    stops[i0][1] + f * (stops[i1][1] - stops[i0][1]),
    stops[i0][2] + f * (stops[i1][2] - stops[i0][2]),
  ];
}

// Generate lookup table (256 entries, RGBA)
function buildLUT(stops) {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = interpolateColormap(stops, i / 255);
    lut[i * 4] = r;
    lut[i * 4 + 1] = g;
    lut[i * 4 + 2] = b;
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

export const COLORMAPS = {
  thermal: buildLUT(THERMAL_STOPS),
  rdbu: buildLUT(RDBU_STOPS),
  viridis: buildLUT(VIRIDIS_STOPS),
};

// Map a value to RGBA using a LUT
export function mapColor(lut, value, min, max) {
  const t = (value - min) / (max - min);
  const idx = Math.max(0, Math.min(255, Math.round(t * 255)));
  return [lut[idx * 4], lut[idx * 4 + 1], lut[idx * 4 + 2], lut[idx * 4 + 3]];
}
