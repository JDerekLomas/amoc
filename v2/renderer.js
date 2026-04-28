/**
 * SimAMOC v2 — CPU Renderer
 *
 * Renders model state to a canvas using ImageData.
 * Colormaps for SST, salinity, streamfunction, speed, moisture, P-E.
 */

const COLORMAPS = {
  temp: { // Blue-Cyan-Green-Yellow-Red
    stops: [-2, 5, 15, 25, 32],
    colors: [[40,60,140], [40,140,200], [60,180,120], [220,200,60], [200,40,40]],
  },
  sal: { // Purple-Blue-Green-Yellow
    stops: [30, 33, 35, 37, 40],
    colors: [[80,40,120], [40,80,180], [40,160,120], [200,200,60], [200,60,60]],
  },
  psi: { // Blue-White-Red (diverging)
    stops: [-0.3, -0.05, 0, 0.05, 0.3],
    colors: [[40,60,200], [120,160,220], [200,200,200], [220,140,100], [200,40,40]],
  },
  speed: { // Black-Blue-Cyan-Yellow-White
    stops: [0, 0.3, 0.8, 1.5, 3.0],
    colors: [[10,10,30], [20,60,160], [40,180,200], [220,200,60], [255,255,255]],
  },
  humidity: { // Brown-Yellow-Green-Blue
    stops: [0, 0.005, 0.01, 0.015, 0.025],
    colors: [[120,80,40], [200,180,60], [60,160,60], [40,120,200], [40,60,180]],
  },
  pe: { // Red(dry)-White-Blue(wet) diverging
    stops: [-0.01, -0.002, 0, 0.002, 0.01],
    colors: [[180,40,40], [200,140,120], [200,200,200], [120,140,200], [40,40,180]],
  },
};

function lerpColor(stops, colors, value) {
  if (value <= stops[0]) return colors[0];
  if (value >= stops[stops.length - 1]) return colors[colors.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (value <= stops[i + 1]) {
      const t = (value - stops[i]) / (stops[i + 1] - stops[i]);
      return [
        colors[i][0] + t * (colors[i + 1][0] - colors[i][0]),
        colors[i][1] + t * (colors[i + 1][1] - colors[i][1]),
        colors[i][2] + t * (colors[i + 1][2] - colors[i][2]),
      ];
    }
  }
  return colors[colors.length - 1];
}

/**
 * Render a field to canvas.
 * @param {string} fieldName - 'temp', 'sal', 'psi', 'speed', 'humidity', 'pe'
 */
export function render(canvas, state, grid, fieldName = 'temp') {
  const { nx, ny } = grid;
  const ctx = canvas.getContext('2d');
  canvas.width = nx;
  canvas.height = ny;
  const img = ctx.createImageData(nx, ny);
  const { mask, psi } = state;
  const invDx = 1 / grid.dx, invDy = 1 / grid.dy;

  const cm = COLORMAPS[fieldName] || COLORMAPS.temp;

  for (let j = 0; j < ny; j++) {
    // Flip: canvas row 0 = top = north, state row 0 = south
    const canvasJ = ny - 1 - j;
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const pixel = (canvasJ * nx + i) * 4;

      if (!mask[k]) {
        // Land: dark gray-green
        img.data[pixel] = 30;
        img.data[pixel + 1] = 35;
        img.data[pixel + 2] = 25;
        img.data[pixel + 3] = 255;
        continue;
      }

      let value;
      switch (fieldName) {
        case 'temp': value = state.temp[k]; break;
        case 'sal': value = state.sal[k]; break;
        case 'psi': value = state.psi[k]; break;
        case 'humidity': value = state.humidity[k]; break;
        case 'pe': value = state.precip[k] - state.evap[k]; break;
        case 'speed': {
          const ip = grid.iWrap(i + 1), im = grid.iWrap(i - 1);
          const u = j > 0 && j < ny - 1 ? -(psi[(j+1)*nx+i] - psi[(j-1)*nx+i]) * 0.5 * invDy : 0;
          const v = (psi[j*nx+ip] - psi[j*nx+im]) * 0.5 * invDx;
          value = Math.sqrt(u*u + v*v);
          break;
        }
        default: value = state.temp[k];
      }

      const [r, g, b] = lerpColor(cm.stops, cm.colors, value);
      img.data[pixel] = r;
      img.data[pixel + 1] = g;
      img.data[pixel + 2] = b;
      img.data[pixel + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
}
