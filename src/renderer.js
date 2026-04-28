// Canvas2D renderer
// Renders scalar fields with colormaps + optional current particle overlay

import { COLORMAPS } from './colormaps.js';

export class Renderer {
  constructor(canvas, grid) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.grid = grid;

    // Rendering state
    this.activeLayer = 'sst';
    this.showCurrents = true;
    this.colormap = 'thermal';

    // Layer config
    this.layers = {
      sst:    { cmap: 'thermal', min: -2,   max: 30,  unit: 'C',    label: 'Sea Surface Temperature' },
      psi:    { cmap: 'rdbu',    min: -30,  max: 30,  unit: 'm2/s', label: 'Streamfunction' },
      speed:  { cmap: 'viridis', min: 0,    max: 0.5, unit: 'm/s',  label: 'Current Speed' },
      qnet:   { cmap: 'rdbu',    min: -150, max: 150, unit: 'W/m2', label: 'Net Heat Flux' },
      sal:    { cmap: 'viridis', min: 32,   max: 38,  unit: 'PSU',  label: 'Surface Salinity' },
      vort:   { cmap: 'rdbu',    min: -1e-5,max: 1e-5,unit: '1/s',  label: 'Vorticity' },
      section:{ cmap: 'thermal', min: 0,    max: 25,  unit: 'C',    label: 'Atlantic Cross-Section' },
    };

    // Cross-section mode
    this.sectionMode = false;

    // Particle system for currents
    this.particles = [];
    this.numParticles = 3000;
    this._initParticles();

    // Work arrays
    this._speedField = grid.createField();

    // Image buffer
    this.imageData = null;
    this.offscreen = null;
    this.offCtx = null;

    // Size to fill container
    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  _resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    // Maintain 2:1 aspect ratio
    let w = rect.width;
    let h = w / 2;
    if (h > rect.height) {
      h = rect.height;
      w = h * 2;
    }
    this.canvas.width = w;
    this.canvas.height = h;
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';
    this.scaleX = w / this.grid.nx;
    this.scaleY = h / this.grid.ny;
    this.imageData = this.ctx.createImageData(this.grid.nx, this.grid.ny);
    this.offscreen = new OffscreenCanvas(this.grid.nx, this.grid.ny);
    this.offCtx = this.offscreen.getContext('2d');
  }

  _initParticles() {
    this.particles = [];
    for (let i = 0; i < this.numParticles; i++) {
      this.particles.push({
        x: Math.random() * this.grid.nx,
        y: Math.random() * this.grid.ny,
        age: Math.floor(Math.random() * 60),
        maxAge: 40 + Math.floor(Math.random() * 40),
      });
    }
  }

  render(ocean, atmosphere) {
    if (this.sectionMode) {
      this._renderSection(ocean);
      return;
    }

    const { ctx, grid, canvas } = this;
    const { nx, ny } = grid;
    const layer = this.layers[this.activeLayer];
    const lut = COLORMAPS[layer.cmap];

    // Get the field to render
    let field;
    switch (this.activeLayer) {
      case 'sst': field = ocean.T; break;
      case 'psi': field = ocean.psi; break;
      case 'speed': {
        field = this._speedField;
        for (let k = 0; k < grid.size; k++) {
          field[k] = Math.sqrt(ocean.u[k] ** 2 + ocean.v[k] ** 2);
        }
        break;
      }
      case 'sal': field = ocean.S; break;
      case 'qnet': field = ocean.Qnet; break;
      case 'vort': field = ocean.zeta; break;
      default: field = ocean.T;
    }

    // Render field to ImageData
    const data = this.imageData.data;
    const range = layer.max - layer.min;

    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const k = j * nx + i;
        // Flip vertically: canvas row 0 = top = north = data row ny-1
        const canvasJ = ny - 1 - j;
        const px = (canvasJ * nx + i) * 4;

        if (ocean.mask[k] < 0.5) {
          // Land: dark gray
          data[px] = 30;
          data[px + 1] = 32;
          data[px + 2] = 34;
          data[px + 3] = 255;
        } else {
          const t = (field[k] - layer.min) / range;
          const idx = Math.max(0, Math.min(255, Math.round(t * 255))) * 4;
          data[px] = lut[idx];
          data[px + 1] = lut[idx + 1];
          data[px + 2] = lut[idx + 2];
          data[px + 3] = 255;
        }
      }
    }

    // Draw scaled
    this.offCtx.putImageData(this.imageData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.offscreen, 0, 0, canvas.width, canvas.height);

    // Current particles
    if (this.showCurrents && this.activeLayer === 'sst') {
      this._renderParticles(ocean, ctx);
    }

    // Colorbar
    this._renderColorbar(ctx, lut, layer);
  }

  _renderParticles(ocean, ctx) {
    const { nx, ny } = this.grid;
    const { scaleX, scaleY } = this;
    const speed = 50000; // visual speed multiplier (SI units: u~0.01 m/s, dx~1e5 m)

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 0.8;

    for (const p of this.particles) {
      const i = Math.floor(p.x);
      const j = Math.floor(p.y);

      if (i < 0 || i >= nx || j < 0 || j >= ny) {
        this._resetParticle(p);
        continue;
      }

      const k = j * nx + i;
      if (ocean.mask[k] < 0.5) {
        this._resetParticle(p);
        continue;
      }

      const u = ocean.u[k];
      const v = ocean.v[k];

      // Move particle (in grid coordinates)
      const dxj = this.grid.dx[j];
      const dyj = this.grid.dy[j];
      const newX = p.x + (u / dxj) * nx * speed;
      const newY = p.y + (v / dyj) * ny * speed;  // south-first: v>0 = northward = larger j

      // Draw trail (flip y for canvas: canvas top = north = high j)
      const alpha = Math.max(0, 1 - p.age / p.maxAge) * 0.5;
      ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(p.x * scaleX, (ny - p.y) * scaleY);
      ctx.lineTo(newX * scaleX, (ny - newY) * scaleY);
      ctx.stroke();

      p.x = ((newX % nx) + nx) % nx;
      p.y = newY;
      p.age++;

      if (p.age > p.maxAge || p.y < 0 || p.y >= ny) {
        this._resetParticle(p);
      }
    }
  }

  _resetParticle(p) {
    p.x = Math.random() * this.grid.nx;
    p.y = Math.random() * this.grid.ny;
    p.age = 0;
    p.maxAge = 40 + Math.floor(Math.random() * 40);
  }

  _renderColorbar(ctx, lut, layer) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const barW = 12;
    const barH = Math.min(200, h * 0.4);
    const x = w - barW - 40;
    const y = (h - barH) / 2;

    // Bar background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(x - 4, y - 20, barW + 50, barH + 40);

    // Color gradient
    for (let i = 0; i < barH; i++) {
      const t = 1 - i / barH;  // top = max
      const idx = Math.round(t * 255) * 4;
      ctx.fillStyle = `rgb(${lut[idx]}, ${lut[idx + 1]}, ${lut[idx + 2]})`;
      ctx.fillRect(x, y + i, barW, 1);
    }

    // Labels
    ctx.fillStyle = '#aaa';
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`${layer.max}`, x + barW + 4, y + 8);
    ctx.fillText(`${layer.min}`, x + barW + 4, y + barH);
    ctx.fillText(layer.unit, x + barW + 4, y + barH / 2 + 4);
  }

  setLayer(name) {
    if (name === 'section') {
      this.sectionMode = true;
      this.activeLayer = 'section';
    } else if (this.layers[name]) {
      this.sectionMode = false;
      this.activeLayer = name;
    }
  }

  // Atlantic meridional cross-section: depth (y-axis) vs latitude (x-axis)
  // Shows temperature profile and MOC streamfunction contours
  _renderSection(ocean) {
    const { ctx, canvas, grid } = this;
    const { nx, ny, lat } = grid;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, w, h);

    // Define Atlantic longitude range (roughly 280-360 in 0-360 coords, plus 0-10)
    // In the data: lon index for Atlantic
    const atlI0 = Math.round(280 / grid.dlon);  // ~280°E = 80°W
    const atlI1 = Math.round(350 / grid.dlon);  // ~350°E = 10°W

    // Vertical structure: 2 layers → interpolate to a smooth profile
    // Depths: surface (0-H_surface), deep (H_surface to 4000m)
    const nDepth = 80;  // vertical pixels in the section
    const maxDepth = 4000;
    const surfaceDepth = 50;  // OCEAN.mixedLayerDepth

    // Section dimensions on canvas
    const margin = { left: 60, right: 20, top: 30, bottom: 40 };
    const secW = w - margin.left - margin.right;
    const secH = h - margin.top - margin.bottom;

    const lut = COLORMAPS.thermal;

    // Draw temperature section
    for (let pj = 0; pj < ny; pj++) {
      // x position on canvas (latitude axis)
      const x0 = margin.left + (pj / ny) * secW;
      const x1 = margin.left + ((pj + 1) / ny) * secW;
      const cellW = x1 - x0;

      // Average temperature across Atlantic longitudes
      let surfT = 0, deepT = 0, count = 0;
      for (let i = atlI0; i < atlI1; i++) {
        const ii = i % nx;
        const k = grid.idx(ii, pj);
        if (ocean.mask[k] > 0.5) {
          surfT += ocean.T[k];
          deepT += ocean.Tdeep[k];
          count++;
        }
      }
      if (count === 0) {
        // Land column — draw as dark
        ctx.fillStyle = '#1e2024';
        ctx.fillRect(x0, margin.top, cellW + 1, secH);
        continue;
      }
      surfT /= count;
      deepT /= count;

      // Draw vertical profile with smooth interpolation
      for (let dj = 0; dj < nDepth; dj++) {
        const depthFrac = dj / nDepth;
        const depth = depthFrac * maxDepth;
        const y0 = margin.top + depthFrac * secH;
        const y1 = margin.top + ((dj + 1) / nDepth) * secH;

        // Smooth transition between surface and deep
        // Use exponential profile: T(z) = Tdeep + (Tsurf - Tdeep) * exp(-z/scale)
        const scale = 300;  // thermocline e-folding depth (m)
        const t = deepT + (surfT - deepT) * Math.exp(-depth / scale);

        // Map to color
        const tNorm = Math.max(0, Math.min(1, (t - 0) / 25));
        const idx = Math.round(tNorm * 255) * 4;
        ctx.fillStyle = `rgb(${lut[idx]},${lut[idx + 1]},${lut[idx + 2]})`;
        ctx.fillRect(x0, y0, cellW + 1, y1 - y0 + 1);
      }
    }

    // Draw MOC streamfunction as contour lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();

    // MOC line at mid-depth (~1000m)
    const mocDepthFrac = 1000 / maxDepth;
    const mocY = margin.top + mocDepthFrac * secH;

    // Smooth MOC with a 5-point running average before plotting
    const smoothMoc = new Float32Array(ny);
    const hw = 3;
    for (let j = 0; j < ny; j++) {
      let sum = 0, cnt = 0;
      for (let jj = Math.max(0, j - hw); jj <= Math.min(ny - 1, j + hw); jj++) {
        sum += ocean.moc[jj]; cnt++;
      }
      smoothMoc[j] = sum / cnt;
    }

    // Plot MOC values as a line
    let started = false;
    for (let pj = 0; pj < ny; pj++) {
      const x = margin.left + (pj + 0.5) / ny * secW;
      const mocVal = smoothMoc[pj];
      // Scale: clamp to ±30 Sv, map to section height
      const clampedMoc = Math.max(-30, Math.min(30, mocVal));
      const yOff = -clampedMoc * (secH * 0.01);
      const y = mocY + yOff;

      if (!started) { ctx.moveTo(x, y); started = true; }
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Zero line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(margin.left, mocY);
    ctx.lineTo(margin.left + secW, mocY);
    ctx.stroke();

    // Axes labels
    ctx.fillStyle = '#888';
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';

    // Latitude ticks
    for (let latVal = -60; latVal <= 80; latVal += 20) {
      const frac = (latVal - grid.lat[0]) / (grid.lat[ny - 1] - grid.lat[0]);
      const x = margin.left + frac * secW;
      ctx.fillText(`${latVal}°`, x, h - margin.bottom + 16);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)';
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, margin.top + secH);
      ctx.stroke();
    }

    // Depth ticks
    ctx.textAlign = 'right';
    for (const d of [0, 500, 1000, 2000, 3000, 4000]) {
      const y = margin.top + (d / maxDepth) * secH;
      ctx.fillText(`${d}m`, margin.left - 6, y + 4);
    }

    // Title
    ctx.fillStyle = '#aaa';
    ctx.font = '12px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Atlantic Meridional Cross-Section — Temperature + MOC (dashed)', w / 2, margin.top - 10);

    // MOC label
    ctx.fillStyle = '#666';
    ctx.font = '10px Inter, system-ui, sans-serif';
    ctx.textAlign = 'left';
    const maxMoc = Math.max(...ocean.moc);
    const minMoc = Math.min(...ocean.moc);
    ctx.fillText(`MOC: ${minMoc.toFixed(1)} to ${maxMoc.toFixed(1)} Sv`, margin.left, h - 6);
  }
}
