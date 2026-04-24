// ============================================================
// RENDERER — Canvas rendering, GPU render pipeline, colormaps
// ============================================================
// Extracted from index.html (Phase 3 of model/UI separation)
// Depends on model.js globals (fields, params, grid, particles).
// Depends on gpu-solver.js globals (gpuDevice, gpuPsiBuf, etc.) for GPU render pipeline.

// ============================================================
// RENDERING, GPU SOLVER, UI — uses model.js globals
// ============================================================

// Canvas refs
const simCanvas = document.getElementById('sim');
const ctx = simCanvas.getContext('2d');
const W = simCanvas.width, H = simCanvas.height;
// Map rendering (LAND_POLYS, maskSrcBits, cellW, cellH are in model.js)
const mapCanvas = document.createElement('canvas');
mapCanvas.width = W; mapCanvas.height = H;
const mapCtx = mapCanvas.getContext('2d');

// ============================================================
// MAP UNDERLAY
// ============================================================
function lonToX(lon) { return ((lon - LON0) / (LON1 - LON0)) * W; }
function latToY(lat) { return (1 - (lat - LAT0) / (LAT1 - LAT0)) * H; }

function drawMapUnderlay() {
  mapCtx.clearRect(0, 0, W, H);
  mapCtx.fillStyle = '#0a1420';
  mapCtx.fillRect(0, 0, W, H);

  // Draw land from the MASK with elevation-based coloring
  if (mask && NX && NY) {
    var cw_ = W / NX, ch_ = H / NY;
    for (var mj = 0; mj < NY; mj++) {
      for (var mi = 0; mi < NX; mi++) {
        var mk = mj * NX + mi;
        if (!mask[mk]) {
          if (remappedElevation) {
            var elev = remappedElevation[mk] || 0;
            // Elevation colormap: green lowlands → brown hills → gray mountains → white peaks
            var r, g, b;
            if (elev < 100) {
              // Low: dark green (#1a3e20 → #2a5e30)
              var t = elev / 100;
              r = 26 + 16 * t; g = 62 + 32 * t; b = 32 + 16 * t;
            } else if (elev < 500) {
              // Mid: green to tan (#2a5e30 → #8a7a50)
              var t = (elev - 100) / 400;
              r = 42 + 96 * t; g = 94 - 16 * t; b = 48 + 32 * t;
            } else if (elev < 2000) {
              // High: tan to brown (#8a7a50 → #6a5040)
              var t = (elev - 500) / 1500;
              r = 138 - 32 * t; g = 122 - 42 * t; b = 80 - 16 * t;
            } else if (elev < 4000) {
              // Mountain: brown to gray (#6a5040 → #9a9090)
              var t = (elev - 2000) / 2000;
              r = 106 + 48 * t; g = 80 + 64 * t; b = 64 + 80 * t;
            } else {
              // Peak: gray to white (#9a9090 → #d0d0d0)
              var t = Math.min(1, (elev - 4000) / 4000);
              r = 154 + 54 * t; g = 144 + 64 * t; b = 144 + 64 * t;
            }
            mapCtx.fillStyle = 'rgb(' + Math.floor(r) + ',' + Math.floor(g) + ',' + Math.floor(b) + ')';
          } else {
            mapCtx.fillStyle = '#1a2e20';
          }
          mapCtx.fillRect(mi * cw_, (NY - 1 - mj) * ch_, cw_ + 0.5, ch_ + 0.5);
        }
      }
    }
  }
  mapCtx.strokeStyle = 'rgba(255,255,255,0.04)';
  mapCtx.lineWidth = 0.5;
  for (var lat = -60; lat <= 60; lat += 30) {
    var y = latToY(lat);
    mapCtx.beginPath(); mapCtx.moveTo(0, y); mapCtx.lineTo(W, y); mapCtx.stroke();
    mapCtx.fillStyle = 'rgba(255,255,255,0.1)'; mapCtx.font = '7px system-ui'; mapCtx.textAlign = 'right';
    mapCtx.fillText((lat >= 0 ? lat + '\u00b0N' : Math.abs(lat) + '\u00b0S'), W - 3, y - 2);
  }
  for (var lon = -120; lon <= 120; lon += 60) {
    var x = lonToX(lon);
    mapCtx.beginPath(); mapCtx.moveTo(x, 0); mapCtx.lineTo(x, H); mapCtx.stroke();
    mapCtx.fillStyle = 'rgba(255,255,255,0.1)'; mapCtx.font = '7px system-ui'; mapCtx.textAlign = 'center';
    mapCtx.fillText((lon <= 0 ? Math.abs(lon) + '\u00b0W' : lon + '\u00b0E'), x, H - 3);
  }
}

// Shader strings (timestepShaderCode, poissonShaderCode, enforceBCShaderCode,
// deepTimestepShaderCode, temperatureShaderCode) are in model.js

// ============================================================
// WebGPU RENDER SHADERS (fullscreen quad)
// ============================================================
// NOTE: Compute shader strings are in model.js
var renderVertexShaderCode = [
'@vertex',
'fn main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {',
'  // Fullscreen triangle: 3 vertices, no buffer needed',
'  var pos = array<vec2f, 3>(',
'    vec2f(-1.0, -1.0),',
'    vec2f( 3.0, -1.0),',
'    vec2f(-1.0,  3.0)',
'  );',
'  return vec4f(pos[vi], 0.0, 1.0);',
'}'
].join('\n');

var renderFragmentShaderCode = [
'struct RenderParams {',
'  nx: u32, ny: u32,',
'  fieldMode: u32, _pad: u32,',
'  absMax: f32, maxSpd: f32,',
'  canvasW: f32, canvasH: f32,',
'  simTime: f32, _pad2: u32, _pad3: u32, _pad4: u32,',
'};',
'',
'@group(0) @binding(0) var<storage, read> psi: array<f32>;',
'@group(0) @binding(1) var<storage, read> zeta: array<f32>;',
'@group(0) @binding(2) var<storage, read> temp: array<f32>;',
'@group(0) @binding(3) var<storage, read> mask: array<u32>;',
'@group(0) @binding(4) var<uniform> rp: RenderParams;',
'',
'fn psiColor(val: f32, absMax: f32) -> vec3f {',
'  let t = clamp(val / (absMax + 1e-30), -1.0, 1.0);',
'  if (t < 0.0) {',
'    let s = -t;',
'    return vec3f((40.0 + 60.0 * s) / 255.0, (80.0 + 100.0 * s) / 255.0, (160.0 + 95.0 * s) / 255.0);',
'  }',
'  return vec3f((200.0 + 55.0 * t) / 255.0, (100.0 - 40.0 * t) / 255.0, (80.0 - 40.0 * t) / 255.0);',
'}',
'',
'fn vortColor(val: f32, absMax: f32) -> vec3f {',
'  let t = clamp(val / (absMax + 1e-30), -1.0, 1.0);',
'  if (t < 0.0) {',
'    let s = -t;',
'    return vec3f((30.0 + 20.0 * s) / 255.0, (60.0 + 140.0 * s) / 255.0, (40.0 + 60.0 * s) / 255.0);',
'  }',
'  return vec3f((180.0 + 75.0 * t) / 255.0, (60.0 + 40.0 * t) / 255.0, (120.0 + 60.0 * t) / 255.0);',
'}',
'',
'fn speedColor(spd: f32, maxSpd: f32) -> vec3f {',
'  let t = clamp(spd / (maxSpd + 1e-30), 0.0, 1.0);',
'  if (t < 0.25) { let s = t / 0.25; return vec3f((10.0 + 10.0 * s) / 255.0, (15.0 + 35.0 * s) / 255.0, (40.0 + 60.0 * s) / 255.0); }',
'  if (t < 0.5) { let s = (t - 0.25) / 0.25; return vec3f((20.0 + 30.0 * s) / 255.0, (50.0 + 80.0 * s) / 255.0, (100.0 + 80.0 * s) / 255.0); }',
'  if (t < 0.75) { let s = (t - 0.5) / 0.25; return vec3f((50.0 + 120.0 * s) / 255.0, (130.0 + 80.0 * s) / 255.0, (180.0 - 60.0 * s) / 255.0); }',
'  let s = (t - 0.75) / 0.25;',
'  return vec3f((170.0 + 80.0 * s) / 255.0, (210.0 + 30.0 * s) / 255.0, (120.0 - 80.0 * s) / 255.0);',
'}',
'',
'fn tempColor(tIn: f32) -> vec3f {',
'  let t = clamp(tIn, -5.0, 30.0);',
'  if (t < 5.0) {',
'    let s = (t - (-5.0)) / 10.0;',
'    return vec3f((20.0 + 10.0 * s) / 255.0, (30.0 + 150.0 * s) / 255.0, (140.0 + 80.0 * s) / 255.0);',
'  }',
'  if (t < 15.0) {',
'    let s = (t - 5.0) / 10.0;',
'    return vec3f((30.0 - 10.0 * s) / 255.0, (180.0 + 20.0 * s) / 255.0, (220.0 - 120.0 * s) / 255.0);',
'  }',
'  if (t < 22.0) {',
'    let s = (t - 15.0) / 7.0;',
'    return vec3f((20.0 + 210.0 * s) / 255.0, (200.0 + 30.0 * s) / 255.0, (100.0 - 60.0 * s) / 255.0);',
'  }',
'  if (t < 26.0) {',
'    let s = (t - 22.0) / 4.0;',
'    return vec3f((230.0 + 20.0 * s) / 255.0, (230.0 - 80.0 * s) / 255.0, (40.0 - 10.0 * s) / 255.0);',
'  }',
'  let s = (t - 26.0) / 4.0;',
'  return vec3f(250.0 / 255.0, (150.0 - 100.0 * s) / 255.0, (30.0 - 30.0 * s) / 255.0);',
'}',
'',
'@fragment',
'fn main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {',
'  let nx = rp.nx;',
'  let ny = rp.ny;',
'  // Map fragment position to grid cell',
'  let fi = fragCoord.x / rp.canvasW * f32(nx);',
'  // Flip Y: top of canvas = high j (north)',
'  let fj = (1.0 - fragCoord.y / rp.canvasH) * f32(ny);',
'  let i = u32(clamp(fi, 0.0, f32(nx) - 1.0));',
'  let j = u32(clamp(fj, 0.0, f32(ny) - 1.0));',
'  let k = j * nx + i;',
'',
'  // Land: seasonal temperature coloring',
'  if (mask[k] == 0u) {',
'    let lat = -80.0 + f32(j) / f32(ny - 1u) * 160.0;',
'    let latRad = lat * 3.14159265 / 180.0;',
'    let yearPhase = 2.0 * 3.14159265 * rp.simTime / 10.0;',
'    let decl = 23.44 * sin(yearPhase) * 3.14159265 / 180.0;',
'    let cosZ = cos(latRad) * cos(decl) + sin(latRad) * sin(decl);',
'    let landT = 35.0 * max(0.0, cosZ) - 15.0;',
'    // Dark muted earth tones',
'    let tN = clamp((landT + 20.0) / 50.0, 0.0, 1.0);',
'    let lr = (20.0 + 40.0 * tN) / 255.0;',
'    let lg = (30.0 + 30.0 * tN) / 255.0;',
'    let lb = (15.0 + 10.0 * tN) / 255.0;',
'    // Snow cover when land temp < -5',
'    if (landT < -5.0) {',
'      let snow = clamp((-5.0 - landT) / 15.0, 0.0, 1.0);',
'      return vec4f(mix(vec3f(lr, lg, lb), vec3f(0.75, 0.8, 0.85), snow), 0.85);',
'    }',
'    return vec4f(lr, lg, lb, 0.85);',
'  }',
'',
'  var col: vec3f;',
'  if (rp.fieldMode == 0u) {',
'    // Streamfunction',
'    col = psiColor(psi[k], rp.absMax);',
'  } else if (rp.fieldMode == 1u) {',
'    // Vorticity',
'    col = vortColor(zeta[k], rp.absMax);',
'  } else if (rp.fieldMode == 2u) {',
'    // Speed: compute from psi gradients with periodic x',
'    if (j < 1u || j >= ny - 1u) {',
'      col = vec3f(0.04, 0.06, 0.16);',
'    } else {',
'      let sip1 = select(i + 1u, 0u, i == nx - 1u);',
'      let sim1 = select(i - 1u, nx - 1u, i == 0u);',
'      let invDx = f32(nx - 1u);',
'      let invDy = f32(ny - 1u);',
'      let u = -(psi[(j + 1u) * nx + i] - psi[(j - 1u) * nx + i]) * 0.5 * invDy;',
'      let v = (psi[j * nx + sip1] - psi[j * nx + sim1]) * 0.5 * invDx;',
'      let spd = sqrt(u * u + v * v);',
'      col = speedColor(spd, rp.maxSpd);',
'    }',
'  } else {',
'    // Temperature',
'    col = tempColor(temp[k]);',
'  }',
'',
'  return vec4f(col, 190.0 / 255.0);',
'}'
].join('\n');

// ============================================================
// WebGPU SOLVER
// ============================================================
// GPU compute solver (initWebGPU, gpuRunSteps, gpuReadback, gpuReset,
// uploadParams, updateGPUBuffersAfterPaint, rebuildBindGroups) is in gpu-solver.js

// GPU Render pipeline state
var gpuRenderPipeline = null;
var gpuRenderParamsBuf = null;
var gpuRenderBindGroup = null;
var gpuCanvasCtx = null;
var gpuRenderFormat = null;
var gpuRenderEnabled = false;
// Paint tool: push CPU-side changes to GPU
function initGPURenderPipeline() {
  var renderCanvas = document.getElementById('gpu-render-canvas');
  gpuCanvasCtx = renderCanvas.getContext('webgpu');
  if (!gpuCanvasCtx) { console.warn('Could not get webgpu context for render canvas'); return false; }

  gpuRenderFormat = navigator.gpu.getPreferredCanvasFormat();
  gpuCanvasCtx.configure({
    device: gpuDevice,
    format: gpuRenderFormat,
    alphaMode: 'premultiplied'
  });

  // Render params uniform: nx, ny, fieldMode, pad, absMax, maxSpd, canvasW, canvasH
  gpuRenderParamsBuf = gpuDevice.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  var vertModule = gpuDevice.createShaderModule({ code: renderVertexShaderCode });
  var fragModule = gpuDevice.createShaderModule({ code: renderFragmentShaderCode });

  // Create bind group layout explicitly so we can share buffers
  var renderBGLayout = gpuDevice.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    ]
  });

  var renderPipelineLayout = gpuDevice.createPipelineLayout({ bindGroupLayouts: [renderBGLayout] });

  gpuRenderPipeline = gpuDevice.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: { module: vertModule, entryPoint: 'main' },
    fragment: {
      module: fragModule,
      entryPoint: 'main',
      targets: [{
        format: gpuRenderFormat,
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' }
        }
      }]
    },
    primitive: { topology: 'triangle-list' }
  });

  gpuRenderBindGroup = gpuDevice.createBindGroup({
    layout: renderBGLayout,
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuTempBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuRenderParamsBuf } },
    ]
  });

  gpuRenderEnabled = true;
  return true;
}

function gpuRenderField() {
  if (!gpuRenderEnabled) return;

  // Compute absMax/maxSpd on GPU-side via a quick scan of the last readback data
  var absMax = 1.0;
  var maxSpd = 1.0;
  var fieldMode = 3; // default temp
  if (showField === 'psi') {
    fieldMode = 0;
    absMax = 0;
    for (var k = 0; k < NX * NY; k++) { var a = Math.abs(psi[k]); if (a > absMax) absMax = a; }
    if (absMax < 1e-30) absMax = 1;
  } else if (showField === 'vort') {
    fieldMode = 1;
    absMax = 0;
    for (var k = 0; k < NX * NY; k++) { var a = Math.abs(zeta[k]); if (a > absMax) absMax = a; }
    if (absMax < 1e-30) absMax = 1;
  } else if (showField === 'speed') {
    fieldMode = 2;
    maxSpd = 0;
    for (var j = 1; j < NY - 1; j++) for (var i = 1; i < NX - 1; i++) {
      var vel = getVel(i, j);
      var s = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (s > maxSpd) maxSpd = s;
    }
    if (maxSpd < 1e-30) maxSpd = 1;
  }

  // Upload render params
  var buf = new ArrayBuffer(48);
  var u32v = new Uint32Array(buf);
  var f32v = new Float32Array(buf);
  u32v[0] = NX;
  u32v[1] = NY;
  u32v[2] = fieldMode;
  u32v[3] = 0; // pad
  f32v[4] = absMax;
  f32v[5] = maxSpd;
  var rc = document.getElementById('gpu-render-canvas');
  f32v[6] = rc ? rc.width : 960.0;
  f32v[7] = rc ? rc.height : 427.0;
  f32v[8] = simTime;
  u32v[9] = 0; u32v[10] = 0; u32v[11] = 0; // pad
  gpuDevice.queue.writeBuffer(gpuRenderParamsBuf, 0, buf);

  var textureView = gpuCanvasCtx.getCurrentTexture().createView();
  var encoder = gpuDevice.createCommandEncoder();
  var pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: textureView,
      clearValue: { r: 0, g: 0, b: 0, a: 0 },
      loadOp: 'clear',
      storeOp: 'store'
    }]
  });
  pass.setPipeline(gpuRenderPipeline);
  pass.setBindGroup(0, gpuRenderBindGroup);
  pass.draw(3);
  pass.end();
  gpuDevice.queue.submit([encoder.finish()]);
}



// ============================================================
// RENDERING
// ============================================================
function psiToRGB(val, absMax) {
  var t = Math.max(-1, Math.min(1, val / (absMax + 1e-30)));
  if (t < 0) {
    var s = -t;
    return [Math.floor(40 + 60 * s), Math.floor(80 + 100 * s), Math.floor(160 + 95 * s)];
  }
  return [Math.floor(200 + 55 * t), Math.floor(100 - 40 * t), Math.floor(80 - 40 * t)];
}

function vortToRGB(val, absMax) {
  var t = Math.max(-1, Math.min(1, val / (absMax + 1e-30)));
  if (t < 0) {
    var s = -t;
    return [Math.floor(30 + 20 * s), Math.floor(60 + 140 * s), Math.floor(40 + 60 * s)];
  }
  return [Math.floor(180 + 75 * t), Math.floor(60 + 40 * t), Math.floor(120 + 60 * t)];
}

function speedToRGB(spd, maxSpd) {
  var t = Math.min(1, spd / (maxSpd + 1e-30));
  if (t < 0.25) { var s = t / 0.25; return [Math.floor(10 + 10 * s), Math.floor(15 + 35 * s), Math.floor(40 + 60 * s)]; }
  if (t < 0.5) { var s = (t - 0.25) / 0.25; return [Math.floor(20 + 30 * s), Math.floor(50 + 80 * s), Math.floor(100 + 80 * s)]; }
  if (t < 0.75) { var s = (t - 0.5) / 0.25; return [Math.floor(50 + 120 * s), Math.floor(130 + 80 * s), Math.floor(180 - 60 * s)]; }
  var s = (t - 0.75) / 0.25;
  return [Math.floor(170 + 80 * s), Math.floor(210 + 30 * s), Math.floor(120 - 80 * s)];
}

function salToRGB(s) {
  // Salinity colormap: 30-38 PSU
  // Fresh (brown/tan) → reference 35 (white) → salty (deep purple)
  var t = Math.max(0, Math.min(1, (s - 30) / 8)); // 0 at 30 PSU, 1 at 38 PSU
  var mid = (35 - 30) / 8; // 0.625
  if (t < mid) {
    var f = t / mid;
    return [Math.floor(140 + 115 * f), Math.floor(100 + 120 * f), Math.floor(60 + 140 * f)]; // brown → white
  } else {
    var f = (t - mid) / (1 - mid);
    return [Math.floor(255 - 155 * f), Math.floor(220 - 140 * f), Math.floor(200 - 60 * f)]; // white → purple
  }
}

function densityToRGB(temp, sal) {
  // Density anomaly: ρ' = -α(T-15) + β(S-35)
  // Light (warm+fresh = buoyant) to dark (cold+salty = dense)
  var rho = -0.05 * (temp - 15) + 0.8 * ((sal || 35) - 35);
  // Normalize: light water (~-1.5) to dense water (~+1.5)
  var t = Math.max(0, Math.min(1, (rho + 1.5) / 3.0));
  // Light=warm yellow, heavy=deep blue
  if (t < 0.5) {
    var f = t / 0.5;
    return [Math.floor(255 - 55 * f), Math.floor(240 - 80 * f), Math.floor(120 + 40 * f)]; // yellow → teal
  } else {
    var f = (t - 0.5) / 0.5;
    return [Math.floor(200 - 170 * f), Math.floor(160 - 130 * f), Math.floor(160 - 40 * f)]; // teal → dark blue
  }
}

function cloudFracToRGB(cf) {
  // Cloud fraction (0-1) colormap: deep blue (clear sky) → white (overcast)
  cf = Math.max(0, Math.min(1, cf));
  var r = Math.floor(20 + 235 * cf);
  var g = Math.floor(30 + 225 * cf);
  var b = Math.floor(60 + 195 * cf);
  return [r, g, b];
}

function depthToRGB(d) {
  // Light blue (shallow) to dark navy (deep)
  var t = Math.min(1, Math.max(0, d / 4000));
  t = Math.sqrt(t); // expand shallow range
  var r = Math.floor(120 - 100 * t);
  var g = Math.floor(200 - 160 * t);
  var b = Math.floor(220 - 140 * t);
  return [r, g, b];
}

function tempToRGB(t) {
  // Thermal colormap with ICE: white/ice < -1.8, deep blue -1.8 to 5, cyan to green to yellow to red
  if (t > 30) t = 30;
  // Ice: below freezing point of seawater (-1.8C) = white/light blue
  if (t < -1.8) {
    var s = Math.max(0, Math.min(1, (t - (-10)) / 8.2)); // -10 to -1.8
    return [Math.floor(180 + 60 * s), Math.floor(200 + 40 * s), Math.floor(220 + 30 * s)]; // white-ish to pale blue
  }
  if (t < 2) {
    var s = (t - (-1.8)) / 3.8; // -1.8 to 2
    return [Math.floor(240 - 210 * s), Math.floor(240 - 170 * s), Math.floor(250 - 100 * s)]; // pale blue to deep blue
  }
  if (t < 8) {
    var s = (t - 2) / 6;
    return [Math.floor(30 - 10 * s), Math.floor(70 + 130 * s), Math.floor(150 + 70 * s)]; // deep blue to cyan
  }
  if (t < 15) {
    var s = (t - 8) / 7;
    return [Math.floor(20 + 20 * s), Math.floor(200 + 10 * s), Math.floor(220 - 130 * s)]; // cyan to teal/green
  }
  if (t < 22) {
    var s = (t - 15) / 7;
    return [Math.floor(40 + 190 * s), Math.floor(210 + 20 * s), Math.floor(90 - 50 * s)]; // green to yellow
  }
  if (t < 27) {
    var s = (t - 22) / 5;
    return [Math.floor(230 + 20 * s), Math.floor(230 - 100 * s), Math.floor(40 - 10 * s)]; // yellow to orange
  }
  var s = (t - 27) / 3;
  return [Math.floor(250), Math.floor(130 - 80 * s), Math.floor(30 - 20 * s)]; // orange to red
}

// Use ImageData for fast field rendering
var fieldCanvas = document.createElement('canvas');
var fieldCtx;

function initFieldCanvas() {
  fieldCanvas.width = NX;
  fieldCanvas.height = NY;
  fieldCtx = fieldCanvas.getContext('2d');
}

// Land temperature with thermal inertia + altitude lapse rate + albedo
var landTempField = null;
var landCanvas = null, landCtx_ = null, landTmpCanvas = null;
var lastLandTime = -999;

function getLandAlbedoAtK(k) {
  if (!remappedAlbedo) return 0.20;
  var v = remappedAlbedo[k];
  return (v != null && !isNaN(v)) ? v : 0.20;
}

function initLandTemp() {
  if (landTempField && landTempField.length === NX * NY) return;
  landTempField = new Float32Array(NX * NY);
  for (var j = 0; j < NY; j++) {
    var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    var cosZ = Math.max(0, Math.cos(lat * Math.PI / 180));
    for (var i = 0; i < NX; i++) {
      var k = j * NX + i;
      if (mask[k]) continue;
      var elev = remappedElevation ? (remappedElevation[k] || 0) : 0;
      var alb = getLandAlbedoAtK(k);
      var baseT = 50 * cosZ * (1 - alb) / 0.80 - 20;
      landTempField[k] = baseT - 6.5 * elev / 1000;
    }
  }
}

function drawSeasonalLand() {
  if (!mask || !NX || !NY) return;
  if (!landTempField) initLandTemp();

  if (!landCanvas) {
    landCanvas = document.createElement('canvas');
    landCanvas.width = W; landCanvas.height = H;
    landCtx_ = landCanvas.getContext('2d');
    landTmpCanvas = document.createElement('canvas');
    landTmpCanvas.width = NX; landTmpCanvas.height = NY;
  }

  // Update smoothly — every 0.05 sim time units (~18 updates per year)
  var timeDelta = simTime - lastLandTime;
  if (Math.abs(timeDelta) > 0.05 || lastLandTime < 0) {
    lastLandTime = simTime;

    var yearPhase = 2 * Math.PI * (simTime % T_YEAR) / T_YEAR;
    var decl = 23.44 * Math.sin(yearPhase) * Math.PI / 180;
    // Relax land temp toward solar equilibrium (thermal inertia + albedo)
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var latRad = lat * Math.PI / 180;
      var cosZ = Math.cos(latRad) * Math.cos(decl) + Math.sin(latRad) * Math.sin(decl);
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        if (mask[k]) continue;
        var elev = remappedElevation ? (remappedElevation[k] || 0) : 0;
        var alb = getLandAlbedoAtK(k);
        var solarT = 50 * Math.max(0, cosZ) * (1 - alb) / 0.80 - 20;
        var targetT = solarT - 6.5 * elev / 1000;
        landTempField[k] += 0.08 * (targetT - landTempField[k]);
      }
    }

    // Render to pixel buffer
    var tmpCtx = landTmpCanvas.getContext('2d');
    var imgData = tmpCtx.createImageData(NX, NY);
    var px = imgData.data;
    for (var mj = 0; mj < NY; mj++) {
      var dstRow = NY - 1 - mj;
      for (var mi = 0; mi < NX; mi++) {
        var mk = mj * NX + mi;
        var idx = (dstRow * NX + mi) * 4;
        if (!mask[mk]) {
          var rgb = tempToRGB(landTempField[mk]);
          px[idx] = rgb[0]; px[idx+1] = rgb[1]; px[idx+2] = rgb[2]; px[idx+3] = 255;
        }
      }
    }
    tmpCtx.putImageData(imgData, 0, 0);
    landCtx_.clearRect(0, 0, W, H);
    landCtx_.imageSmoothingEnabled = false;
    landCtx_.drawImage(landTmpCanvas, 0, 0, W, H);
  }

  ctx.drawImage(landCanvas, 0, 0);
}

function drawColorLegend() {
  var lx = W - 30, ly = 20, lh = 140, lw = 12;
  var title = '', labels = [];

  if (showField === "temp" || showField === "deeptemp" || showField === "airtemp") {
    for (var li = 0; li < lh; li++) {
      var t_ = -10 + 40 * (lh - li) / lh;
      var c = tempToRGB(t_);
      ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    title = showField === "deeptemp" ? "Deep \u00b0C" : showField === "airtemp" ? "Air \u00b0C" : "SST \u00b0C";
    labels = [[0, "30\u00b0"], [0.375, "15\u00b0"], [0.75, "0\u00b0"], [1, "-10\u00b0"]];
  } else if (showField === "speed") {
    for (var li = 0; li < lh; li++) {
      var c = speedToRGB(3.0 * (lh - li) / lh, 3.0);
      ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    title = "Speed"; labels = [[0, "Fast"], [0.5, "Med"], [1, "Still"]];
  } else if (showField === "sal") {
    for (var li = 0; li < lh; li++) {
      var c = salToRGB(37 - 5 * li / lh);
      ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    title = "PSU"; labels = [[0, "37"], [0.5, "34.5"], [1, "32"]];
  } else if (showField === "clouds" || showField === "obsclouds") {
    for (var li = 0; li < lh; li++) {
      var cf = 0.75 * (lh - li) / lh;
      var c = typeof cloudFracToRGB === 'function' ? cloudFracToRGB(cf) : [Math.floor(20+235*cf), Math.floor(30+225*cf), Math.floor(60+195*cf)];
      ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    title = showField === "obsclouds" ? "Obs Clouds" : "Clouds";
    labels = [[0, "75%"], [0.33, "50%"], [0.67, "25%"], [1, "Clear"]];
  } else if (showField === "depth") {
    for (var li = 0; li < lh; li++) {
      var c = depthToRGB(4000 * li / lh);
      ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    title = "Depth"; labels = [[0, "0 m"], [0.5, "2000"], [1, "4000"]];
  } else if (showField === "density") {
    for (var li = 0; li < lh; li++) {
      var frac = li / lh;
      var c = densityToRGB(30 - 40 * frac, 34 + 3 * frac);
      ctx.fillStyle = "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    title = "Density"; labels = [[0, "Light"], [0.5, "Mid"], [1, "Dense"]];
  } else if (showField === "psi" || showField === "deepflow") {
    for (var li = 0; li < lh; li++) {
      var frac = (lh - li) / lh;
      ctx.fillStyle = "rgb(" + Math.floor(frac > 0.5 ? 255*(frac-0.5)*2 : 0) + ",20," + Math.floor(frac < 0.5 ? 255*(0.5-frac)*2 : 0) + ")";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    title = showField === "psi" ? "\u03C8" : "Deep \u03C8"; labels = [[0, "CW"], [0.5, "0"], [1, "CCW"]];
  } else if (showField === "vort") {
    for (var li = 0; li < lh; li++) {
      var frac = (lh - li) / lh;
      ctx.fillStyle = "rgb(" + Math.floor(frac > 0.5 ? 200*(frac-0.5)*2 : 0) + "," + Math.floor(frac < 0.5 ? 180*(0.5-frac)*2 : 0) + ",30)";
      ctx.fillRect(lx, ly + li, lw, 1);
    }
    title = "Vorticity"; labels = [[0, "+"], [0.5, "0"], [1, "\u2212"]];
  }

  if (title) {
    ctx.strokeStyle = "rgba(255,255,255,.15)"; ctx.strokeRect(lx, ly, lw, lh);
    ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.font = "8px system-ui"; ctx.textAlign = "left";
    ctx.fillText(title, lx - 2, ly - 4);
    for (var ll = 0; ll < labels.length; ll++) {
      ctx.fillStyle = "rgba(255,255,255,.5)";
      ctx.fillText(labels[ll][1], lx + lw + 3, ly + labels[ll][0] * lh + 4);
    }
  }
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  drawSeasonalLand();

  // Compute field image
  var imgData = fieldCtx.createImageData(NX, NY);
  var data = imgData.data;

  var absMax = 0;
  var maxSpd = 0;
  var k;

  if (showField === 'psi') {
    for (k = 0; k < NX * NY; k++) { var a = Math.abs(psi[k]); if (a > absMax) absMax = a; }
  } else if (showField === 'deepflow') {
    if (deepPsi) for (k = 0; k < NX * NY; k++) { var a = Math.abs(deepPsi[k]); if (a > absMax) absMax = a; }
  } else if (showField === 'vort') {
    for (k = 0; k < NX * NY; k++) { var a = Math.abs(zeta[k]); if (a > absMax) absMax = a; }
  } else if (showField === 'temp' || showField === 'deeptemp') {
    // temp uses fixed colormap, no normalization needed
  } else {
    for (var j = 1; j < NY - 1; j++) for (var i = 1; i < NX - 1; i++) {
      var vel = getVel(i, j);
      var s = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (s > maxSpd) maxSpd = s;
    }
  }
  if (absMax < 1e-30) absMax = 1;
  if (maxSpd < 1e-30) maxSpd = 1;

  for (var j = 0; j < NY; j++) {
    for (var i = 0; i < NX; i++) {
      var srcK = j * NX + i;
      // Flip vertically: row 0 of image = top = j=NY-1
      var dstRow = NY - 1 - j;
      var dstIdx = (dstRow * NX + i) * 4;

      if (!mask[srcK]) {
        // Cloud view: show land cloud fraction from precipitation data
        if (showField === 'clouds' && cloudField && cloudField[srcK] > 0) {
          var rgb = cloudFracToRGB(cloudField[srcK]);
          data[dstIdx] = rgb[0]; data[dstIdx + 1] = rgb[1]; data[dstIdx + 2] = rgb[2]; data[dstIdx + 3] = 200;
          continue;
        }
        if (showField === 'obsclouds' && obsCloudField && obsCloudField[srcK] > 0) {
          var rgb = cloudFracToRGB(obsCloudField[srcK]);
          data[dstIdx] = rgb[0]; data[dstIdx + 1] = rgb[1]; data[dstIdx + 2] = rgb[2]; data[dstIdx + 3] = 200;
          continue;
        }
        // Air temp view: show land temp from landTempField
        if (showField === 'airtemp' && landTempField && landTempField[srcK] !== 0) {
          var rgb = tempToRGB(landTempField[srcK]);
          data[dstIdx] = rgb[0]; data[dstIdx + 1] = rgb[1]; data[dstIdx + 2] = rgb[2]; data[dstIdx + 3] = 200;
          continue;
        }
        // Land: show seasonal temperature or elevation
        var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
        var latRad = lat * Math.PI / 180;
        var yearPhase = 2 * Math.PI * simTime / T_YEAR;
        var decl = 23.44 * Math.sin(yearPhase) * Math.PI / 180;
        var cosZ = Math.cos(latRad) * Math.cos(decl) + Math.sin(latRad) * Math.sin(decl);
        var landT = 50 * Math.max(0, cosZ) - 20; // seasonal land temperature

        // Blend elevation (static) with seasonal temp (dynamic)
        var elev = remappedElevation ? (remappedElevation[srcK] || 0) : 0;

        // Color: warm land = green/brown, cold land = white/gray, high elevation = lighter
        var r, g, b;
        var elevFade = Math.min(1, elev / 3000); // 0 at sea level, 1 at 3000m+
        if (landT < -5) {
          // Frozen: white-blue, lighter at high elevation
          var f = Math.max(0, Math.min(1, (landT + 20) / 15));
          r = Math.floor(180 + 60 * f + 15 * elevFade);
          g = Math.floor(190 + 50 * f + 15 * elevFade);
          b = Math.floor(210 + 30 * f + 15 * elevFade);
        } else if (landT < 10) {
          // Cool: gray-green, browner at high elevation
          var f = (landT + 5) / 15;
          r = Math.floor(100 + 40 * f + 40 * elevFade);
          g = Math.floor(110 + 50 * f - 20 * elevFade);
          b = Math.floor(90 - 20 * f + 30 * elevFade);
        } else if (landT < 25) {
          // Warm: green, browner at high elevation
          var f = (landT - 10) / 15;
          r = Math.floor(140 + 50 * f + 30 * elevFade);
          g = Math.floor(160 - 30 * f - 40 * elevFade);
          b = Math.floor(70 - 20 * f + 20 * elevFade);
        } else {
          // Hot: tan/brown desert
          r = Math.floor(190 + 20 * elevFade);
          g = Math.floor(160 - 30 * elevFade);
          b = Math.floor(80 + 20 * elevFade);
        }
        data[dstIdx] = r; data[dstIdx + 1] = g; data[dstIdx + 2] = b; data[dstIdx + 3] = 220;
        continue;
      }

      var rgb;
      if (showField === 'psi') rgb = psiToRGB(psi[srcK], absMax);
      else if (showField === 'vort') rgb = vortToRGB(zeta[srcK], absMax);
      else if (showField === 'temp') rgb = tempToRGB(temp[srcK]);
      else if (showField === 'deeptemp') rgb = tempToRGB(deepTemp ? deepTemp[srcK] : 0);
      else if (showField === 'deepflow') rgb = psiToRGB(deepPsi ? deepPsi[srcK] : 0, absMax);
      else if (showField === 'sal') rgb = salToRGB(sal ? sal[srcK] : 35);
      else if (showField === 'density') rgb = densityToRGB(temp[srcK], sal ? sal[srcK] : 35);
      else if (showField === 'depth') rgb = depthToRGB(depth ? depth[srcK] : 0);
      else if (showField === 'clouds') { rgb = cloudField ? cloudFracToRGB(cloudField[srcK]) : [30, 40, 70]; }
      else if (showField === 'obsclouds') { rgb = obsCloudField ? cloudFracToRGB(obsCloudField[srcK]) : [30, 40, 70]; }
      else if (showField === 'airtemp') { rgb = airTemp ? tempToRGB(airTemp[srcK]) : tempToRGB(temp[srcK]); }
      else {
        var vel = getVel(i, j);
        rgb = speedToRGB(Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]), maxSpd);
      }
      data[dstIdx] = rgb[0]; data[dstIdx + 1] = rgb[1]; data[dstIdx + 2] = rgb[2]; data[dstIdx + 3] = 190;
    }
  }
  fieldCtx.putImageData(imgData, 0, 0);

  // Draw field — the mask IS the coastline. Editable like SimEarth.
  // Land cells have alpha=0 in ImageData so only ocean shows through.
  ctx.imageSmoothingEnabled = false; // crisp grid cells
  var fieldDx = -W / (2 * (NX - 1));
  var fieldDy = -H / (2 * (NY - 1));
  var fieldDw = W * NX / (NX - 1);
  var fieldDh = H * NY / (NY - 1);
  ctx.drawImage(fieldCanvas, fieldDx, fieldDy, fieldDw, fieldDh);

  // Contour lines for streamfunction
  if (showField === 'psi' && absMax > 1e-20) {
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 0.5;
    var nContours = 16;
    for (var c = 1; c < nContours; c++) {
      var level = -absMax + 2 * absMax * c / nContours;
      for (var j = 0; j < NY - 1; j++) for (var i = 0; i < NX - 1; i++) {
        var v00 = psi[j * NX + i] - level, v10 = psi[j * NX + i + 1] - level;
        var v01 = psi[(j + 1) * NX + i] - level, v11 = psi[(j + 1) * NX + i + 1] - level;
        var s00 = v00 > 0 ? 1 : 0, s10 = v10 > 0 ? 1 : 0, s01 = v01 > 0 ? 1 : 0, s11 = v11 > 0 ? 1 : 0;
        var sum = s00 + s10 + s01 + s11;
        if (sum > 0 && sum < 4) {
          var cx_ = i * cellW + cellW / 2, cy_ = (NY - 1 - j) * cellH - cellH / 2;
          ctx.beginPath(); ctx.arc(cx_, cy_, 0.5, 0, Math.PI * 2); ctx.stroke();
        }
      }
    }
  }

  // Particles
  if (showParticles) {
    for (var p = 0; p < NP; p++) {
      var x = px[p] * cellW, y = (NY - 1 - py[p]) * cellH;
      var vel = getVel(px[p], py[p]);
      var spd = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      var alpha = Math.min(1, page_[p] / 20) * Math.min(1, (MAX_AGE - page_[p]) / 20);
      var bright = Math.min(1, spd / (maxSpd || 1) * 3);
      ctx.fillStyle = 'rgba(' + Math.floor(200 + 55 * bright) + ',' + Math.floor(220 + 35 * bright) + ',' + Math.floor(240 + 15 * bright) + ',' + (alpha * 0.6) + ')';
      ctx.fillRect(x - 0.5, y - 0.5, 1.5, 1.5);
    }
  }

  // Basin boundary
  ctx.strokeStyle = '#2a4050';
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, W, H);

  // Geographic labels
  ctx.fillStyle = 'rgba(200,220,240,0.15)';
  ctx.font = '9px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('N. Atlantic', lonToX(-40), latToY(35));
  ctx.fillText('N. Pacific', lonToX(-160), latToY(35));
  ctx.fillText('Indian', lonToX(75), latToY(-15));
  ctx.fillText('S. Ocean', lonToX(0), latToY(-60));

  drawColorLegend();
  // Western boundary annotation
  if (totalSteps > 500) {
    var maxWestVel = 0;
    for (var j = Math.floor(NY / 4); j < Math.floor(NY * 3 / 4); j++) {
      var vel = getVel(3, j);
      var sv = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (sv > maxWestVel) maxWestVel = sv;
    }
    if (maxWestVel > 0.5) {
      ctx.fillStyle = 'rgba(200,160,100,0.5)';
      ctx.font = '9px system-ui';
      ctx.save(); ctx.translate(20, H / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText('Western Boundary Current', 0, 0);
      ctx.restore();
    }
  }
}

// Draw overlay: map underlay, particles, contours, labels (for GPU render mode)
// The 2D canvas is transparent, sitting on top of the WebGPU canvas
function drawOverlay() {
  ctx.clearRect(0, 0, W, H);

  drawSeasonalLand();

  // Cloud layer: semi-transparent white based on cloud fraction
  if (temp && showField === 'temp') {
    var cloudAlpha = 0.35;
    for (var cj = 0; cj < NY; cj++) {
      var clat = LAT0 + (cj / (NY - 1)) * (LAT1 - LAT0);
      var clatRad = clat * Math.PI / 180;
      for (var ci = 0; ci < NX; ci++) {
        var ck = cj * NX + ci;
        var cf;
        if (!mask[ck]) {
          // Land clouds from cloudField (precipitation-derived)
          cf = (cloudField && cloudField[ck] > 0) ? cloudField[ck] : 0;
        } else {
          var sst = temp[ck];
          var cb = 0.25 + 0.15 * Math.cos(2 * clatRad);
          var cv = 0.15 * Math.max(0, Math.min(1, (sst - 15) / 15));
          var cp = 0.10 * Math.max(0, Math.min(1, (Math.abs(clat) - 50) / 30));
          cf = Math.max(0.05, Math.min(0.75, cb + cv + cp));
        }
        if (cf > 0.15) {
          var cx = ci * cellW;
          var cy = (NY - 1 - cj) * cellH;
          var alpha = cf * cloudAlpha;
          ctx.fillStyle = 'rgba(255,255,255,' + alpha.toFixed(3) + ')';
          ctx.fillRect(cx, cy, cellW + 0.5, cellH + 0.5);
        }
      }
    }
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;
  for (var lat = -60; lat <= 60; lat += 30) {
    var gy = latToY(lat);
    ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.1)'; ctx.font = '7px system-ui'; ctx.textAlign = 'right';
    ctx.fillText((lat >= 0 ? lat + '\u00b0N' : Math.abs(lat) + '\u00b0S'), W - 3, gy - 2);
  }
  for (var lon = -120; lon <= 120; lon += 60) {
    var gx = lonToX(lon);
    ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.1)'; ctx.font = '7px system-ui'; ctx.textAlign = 'center';
    ctx.fillText((lon <= 0 ? Math.abs(lon) + '\u00b0W' : lon + '\u00b0E'), gx, H - 3);
  }

  // Contour lines for streamfunction
  if (showField === 'psi') {
    var absMax = 0;
    for (var k = 0; k < NX * NY; k++) { var a = Math.abs(psi[k]); if (a > absMax) absMax = a; }
    if (absMax > 1e-20) {
      ctx.strokeStyle = 'rgba(255,255,255,0.12)';
      ctx.lineWidth = 0.5;
      var nContours = 16;
      for (var c = 1; c < nContours; c++) {
        var level = -absMax + 2 * absMax * c / nContours;
        for (var j = 0; j < NY - 1; j++) for (var i = 0; i < NX - 1; i++) {
          var v00 = psi[j * NX + i] - level, v10 = psi[j * NX + i + 1] - level;
          var v01 = psi[(j + 1) * NX + i] - level, v11 = psi[(j + 1) * NX + i + 1] - level;
          var s00 = v00 > 0 ? 1 : 0, s10 = v10 > 0 ? 1 : 0, s01 = v01 > 0 ? 1 : 0, s11 = v11 > 0 ? 1 : 0;
          var sum = s00 + s10 + s01 + s11;
          if (sum > 0 && sum < 4) {
            var cx_ = i * cellW + cellW / 2, cy_ = (NY - 1 - j) * cellH - cellH / 2;
            ctx.beginPath(); ctx.arc(cx_, cy_, 0.5, 0, Math.PI * 2); ctx.stroke();
          }
        }
      }
    }
  }

  // Particles
  var maxSpd = 1;
  if (showParticles) {
    for (var p = 0; p < NP; p++) {
      var x = px[p] * cellW, y = (NY - 1 - py[p]) * cellH;
      var vel = getVel(px[p], py[p]);
      var spd = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (spd > maxSpd) maxSpd = spd;
    }
    for (var p = 0; p < NP; p++) {
      var x = px[p] * cellW, y = (NY - 1 - py[p]) * cellH;
      var vel = getVel(px[p], py[p]);
      var spd = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      var alpha = Math.min(1, page_[p] / 20) * Math.min(1, (MAX_AGE - page_[p]) / 20);
      var bright = Math.min(1, spd / (maxSpd || 1) * 3);
      ctx.fillStyle = 'rgba(' + Math.floor(200 + 55 * bright) + ',' + Math.floor(220 + 35 * bright) + ',' + Math.floor(240 + 15 * bright) + ',' + (alpha * 0.6) + ')';
      ctx.fillRect(x - 0.5, y - 0.5, 1.5, 1.5);
    }
  }

  // Basin boundary
  ctx.strokeStyle = '#2a4050';
  ctx.lineWidth = 1;
  ctx.strokeRect(0, 0, W, H);

  // Geographic labels
  ctx.fillStyle = 'rgba(200,220,240,0.15)';
  ctx.font = '9px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('N. Atlantic', lonToX(-40), latToY(35));
  ctx.fillText('N. Pacific', lonToX(-160), latToY(35));
  ctx.fillText('Indian', lonToX(75), latToY(-15));
  ctx.fillText('S. Ocean', lonToX(0), latToY(-60));

  drawColorLegend();
  // Western boundary annotation
  if (totalSteps > 500) {
    var maxWestVel = 0;
    for (var j = Math.floor(NY / 4); j < Math.floor(NY * 3 / 4); j++) {
      var vel = getVel(3, j);
      var sv = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
      if (sv > maxWestVel) maxWestVel = sv;
    }
    if (maxWestVel > 0.5) {
      ctx.fillStyle = 'rgba(200,160,100,0.5)';
      ctx.font = '9px system-ui';
      ctx.save(); ctx.translate(20, H / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText('Western Boundary Current', 0, 0);
      ctx.restore();
    }
  }
}

// Velocity profile
var profCanvas = document.getElementById('profile');
var profCtx = profCanvas.getContext('2d');

function drawProfile() {
  var dpr = devicePixelRatio || 1;
  var r = profCanvas.getBoundingClientRect();
  profCanvas.width = r.width * dpr; profCanvas.height = r.height * dpr;
  profCtx.scale(dpr, dpr);
  var w = r.width, h = r.height;
  profCtx.clearRect(0, 0, w, h);

  var jMid = Math.floor(NY / 2);
  var maxV = 0;
  var vals = [];
  for (var i = 0; i < NX; i++) {
    var vel = getVel(i, jMid);
    vals.push(vel[1]);
    if (Math.abs(vel[1]) > maxV) maxV = Math.abs(vel[1]);
  }
  if (maxV < 1e-30) maxV = 1;

  var m = { l: 30, r: 6, t: 8, b: 18 }, pw = w - m.l - m.r, ph = h - m.t - m.b;

  profCtx.strokeStyle = '#1a2838'; profCtx.lineWidth = 1;
  profCtx.beginPath(); profCtx.moveTo(m.l, m.t); profCtx.lineTo(m.l, h - m.b); profCtx.lineTo(w - m.r, h - m.b); profCtx.stroke();

  var zy = m.t + ph / 2;
  profCtx.strokeStyle = '#2a3848'; profCtx.setLineDash([3, 3]);
  profCtx.beginPath(); profCtx.moveTo(m.l, zy); profCtx.lineTo(w - m.r, zy); profCtx.stroke(); profCtx.setLineDash([]);

  profCtx.fillStyle = '#4a7090'; profCtx.font = '7px system-ui';
  profCtx.textAlign = 'center'; profCtx.fillText('x (west to east)', w / 2, h - 2);
  profCtx.textAlign = 'right'; profCtx.fillText('v', m.l - 4, m.t + ph / 2 + 3);

  profCtx.strokeStyle = '#50b0e0'; profCtx.lineWidth = 1.5;
  profCtx.beginPath();
  for (var i = 0; i < NX; i++) {
    var x = m.l + (i / (NX - 1)) * pw;
    var y = zy - (vals[i] / maxV) * (ph / 2) * 0.9;
    if (i === 0) profCtx.moveTo(x, y); else profCtx.lineTo(x, y);
  }
  profCtx.stroke();

  profCtx.fillStyle = 'rgba(200,140,60,0.15)';
  profCtx.fillRect(m.l, m.t, pw * 0.08, ph);
}

// Radiative balance profile
var radCanvas = document.getElementById('rad-profile');
var radCtx = radCanvas.getContext('2d');

function drawRadProfile() {
  var dpr = devicePixelRatio || 1;
  var r = radCanvas.getBoundingClientRect();
  radCanvas.width = r.width * dpr; radCanvas.height = r.height * dpr;
  radCtx.scale(dpr, dpr);
  var w = r.width, h = r.height;
  radCtx.clearRect(0, 0, w, h);

  var m = { l: 30, r: 6, t: 8, b: 18 }, pw = w - m.l - m.r, ph = h - m.t - m.b;

  // Compute solar, OLR, and net for each latitude row (zonally averaged)
  var yearPhase = 2 * Math.PI * simTime / T_YEAR;
  var decl = 23.44 * Math.sin(yearPhase) * Math.PI / 180;

  var solar = [], olrArr = [], netArr = [], meanT = [];
  var maxVal = 0.1;

  for (var j = 0; j < NY; j++) {
    var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    var latRad = lat * Math.PI / 180;
    var cosZ = Math.cos(latRad) * Math.cos(decl) + Math.sin(latRad) * Math.sin(decl);

    // Zonal mean temperature
    var tSum = 0, tCount = 0;
    for (var i = 0; i < NX; i++) {
      var k = j * NX + i;
      if (mask[k]) { tSum += temp[k]; tCount++; }
    }
    var avgT = tCount > 0 ? tSum / tCount : 0;
    meanT.push(avgT);

    // Ice fraction for this temperature
    var iceT = Math.max(0, Math.min(1, (avgT + 3) / 5));
    var iceFrac = 1 - iceT * iceT * (3 - 2 * iceT);
    var albedoFactor = 0.15 + 0.85 * (1 - iceFrac);

    var qs = S_solar * Math.max(0, cosZ) * albedoFactor;
    var olr = A_olr + B_olr * avgT;
    var net = qs - olr;
    solar.push(qs);
    olrArr.push(olr);
    netArr.push(net);

    var mv = Math.max(Math.abs(qs), Math.abs(olr), Math.abs(net));
    if (mv > maxVal) maxVal = mv;
  }

  // Axes
  radCtx.strokeStyle = '#1a2838'; radCtx.lineWidth = 1;
  radCtx.beginPath(); radCtx.moveTo(m.l, m.t); radCtx.lineTo(m.l, h - m.b); radCtx.lineTo(w - m.r, h - m.b); radCtx.stroke();

  // Zero line
  var zy = m.t + ph * (maxVal / (2 * maxVal));
  radCtx.strokeStyle = '#2a3848'; radCtx.setLineDash([3, 3]);
  radCtx.beginPath(); radCtx.moveTo(m.l, zy); radCtx.lineTo(w - m.r, zy); radCtx.stroke(); radCtx.setLineDash([]);

  // Labels
  radCtx.fillStyle = '#4a7090'; radCtx.font = '7px system-ui';
  radCtx.textAlign = 'center'; radCtx.fillText('80\u00b0S                    latitude                    80\u00b0N', w / 2, h - 2);
  radCtx.textAlign = 'right'; radCtx.fillText('0', m.l - 3, zy + 3);

  function plotLine(arr, color) {
    radCtx.strokeStyle = color; radCtx.lineWidth = 1.5;
    radCtx.beginPath();
    for (var j2 = 0; j2 < NY; j2++) {
      var x = m.l + (j2 / (NY - 1)) * pw;
      var y = zy - (arr[j2] / maxVal) * (ph / 2) * 0.9;
      if (j2 === 0) radCtx.moveTo(x, y); else radCtx.lineTo(x, y);
    }
    radCtx.stroke();
  }

  plotLine(solar, '#e8a040');    // Solar: warm orange
  plotLine(olrArr, '#e05050');   // OLR: red
  plotLine(netArr, '#40c080');   // Net: green

  // Legend
  radCtx.font = '7px system-ui'; radCtx.textAlign = 'left';
  var lx = m.l + 4, ly = m.t + 8;
  radCtx.fillStyle = '#e8a040'; radCtx.fillText('Solar', lx, ly);
  radCtx.fillStyle = '#e05050'; radCtx.fillText('OLR', lx + 30, ly);
  radCtx.fillStyle = '#40c080'; radCtx.fillText('Net', lx + 55, ly);
}

// ============================================================
// AMOC TIMESERIES CHART
// ============================================================
var amocHistory = [];
var AMOC_HISTORY_LEN = 200;
var amocCanvas, amocCtx;
var rapidAmocMean = null; // loaded from RAPID CSV

function initAmocChart() {
  amocCanvas = document.getElementById('amoc-chart');
  if (!amocCanvas) return;
  amocCtx = amocCanvas.getContext('2d');
  // Load RAPID AMOC data
  fetch('../earth-data/timeseries/rapid_amoc_monthly.csv').then(function(r) { return r.text(); }).then(function(txt) {
    var lines = txt.split('\n');
    var sum = 0, n = 0;
    for (var i = 0; i < lines.length; i++) {
      if (lines[i].startsWith('#') || lines[i].startsWith('Year')) continue;
      var parts = lines[i].split(',');
      var sv = parseFloat(parts[2]);
      if (!isNaN(sv)) { sum += sv; n++; }
    }
    if (n > 0) rapidAmocMean = sum / n;
    console.log('RAPID AMOC mean: ' + (rapidAmocMean ? rapidAmocMean.toFixed(1) : 'N/A') + ' Sv (' + n + ' months)');
  }).catch(function() {});
}

function pushAmocSample() {
  amocHistory.push(amocStrength);
  if (amocHistory.length > AMOC_HISTORY_LEN) amocHistory.shift();
}

function drawAmocChart() {
  if (!amocCanvas || !amocCtx) return;
  var dpr = devicePixelRatio || 1;
  var r = amocCanvas.getBoundingClientRect();
  amocCanvas.width = r.width * dpr; amocCanvas.height = r.height * dpr;
  amocCtx.scale(dpr, dpr);
  var w = r.width, h = r.height;
  amocCtx.clearRect(0, 0, w, h);

  var m = { l: 30, r: 6, t: 8, b: 18 }, pw = w - m.l - m.r, ph = h - m.t - m.b;
  var hist = amocHistory;
  if (hist.length < 2) return;

  // Auto-scale Y axis
  var yMin = Infinity, yMax = -Infinity;
  for (var i = 0; i < hist.length; i++) {
    if (hist[i] < yMin) yMin = hist[i];
    if (hist[i] > yMax) yMax = hist[i];
  }
  // Include RAPID mean in range if available
  if (rapidAmocMean !== null) {
    if (rapidAmocMean < yMin) yMin = rapidAmocMean * 0.8;
    if (rapidAmocMean > yMax) yMax = rapidAmocMean * 1.2;
  }
  var yPad = Math.max(0.01, (yMax - yMin) * 0.15);
  yMin -= yPad; yMax += yPad;
  if (yMax - yMin < 0.02) { yMax += 0.01; yMin -= 0.01; }

  function yToScreen(v) { return m.t + ph * (1 - (v - yMin) / (yMax - yMin)); }

  // Axes
  amocCtx.strokeStyle = '#1a2838'; amocCtx.lineWidth = 1;
  amocCtx.beginPath(); amocCtx.moveTo(m.l, m.t); amocCtx.lineTo(m.l, h - m.b); amocCtx.lineTo(w - m.r, h - m.b); amocCtx.stroke();

  // Zero line
  if (yMin < 0 && yMax > 0) {
    amocCtx.strokeStyle = '#2a3848'; amocCtx.setLineDash([3, 3]);
    amocCtx.beginPath(); amocCtx.moveTo(m.l, yToScreen(0)); amocCtx.lineTo(w - m.r, yToScreen(0)); amocCtx.stroke();
    amocCtx.setLineDash([]);
  }

  // RAPID observed mean line
  if (rapidAmocMean !== null) {
    var ry = yToScreen(rapidAmocMean);
    amocCtx.strokeStyle = '#e8a040'; amocCtx.lineWidth = 1; amocCtx.setLineDash([4, 3]);
    amocCtx.beginPath(); amocCtx.moveTo(m.l, ry); amocCtx.lineTo(w - m.r, ry); amocCtx.stroke();
    amocCtx.setLineDash([]);
    amocCtx.fillStyle = '#e8a040'; amocCtx.font = '7px system-ui'; amocCtx.textAlign = 'left';
    amocCtx.fillText('RAPID ' + rapidAmocMean.toFixed(1) + ' Sv', m.l + 3, ry - 3);
  }

  // Simulated AMOC line
  var lastVal = hist[hist.length - 1];
  var color = lastVal > 0.01 ? '#4aba70' : (lastVal < -0.01 ? '#e06050' : '#4a9ec8');
  amocCtx.strokeStyle = color; amocCtx.lineWidth = 1.5;
  amocCtx.beginPath();
  for (var j = 0; j < hist.length; j++) {
    var x = m.l + (j / (AMOC_HISTORY_LEN - 1)) * pw;
    var y = yToScreen(hist[j]);
    if (j === 0) amocCtx.moveTo(x, y); else amocCtx.lineTo(x, y);
  }
  amocCtx.stroke();

  // Labels
  amocCtx.fillStyle = '#4a7090'; amocCtx.font = '7px system-ui';
  amocCtx.textAlign = 'right';
  amocCtx.fillText(yMax.toFixed(3), m.l - 3, m.t + 8);
  amocCtx.fillText(yMin.toFixed(3), m.l - 3, h - m.b - 2);
  amocCtx.textAlign = 'center';
  amocCtx.fillText('Simulation time \u2192', w / 2, h - 2);
}

