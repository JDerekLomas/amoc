// ============================================================
// GPU COMPUTE SOLVER — WebGPU pipeline management
// ============================================================
// Extracted from index.html (Phase 2 of model/UI separation)
// Depends on model.js globals (params, fields, shaders, grid constants).
// GPU render pipeline (initGPURenderPipeline, gpuRenderField) stays in index.html.

var gpuDevice = null;
var gpuFFTPoissonSolve = null; // assigned inside initWebGPU, used by gpuRunSteps
var gpuPsiBuf, gpuZetaBuf, gpuZetaNewBuf, gpuMaskBuf, gpuParamsBuf;
var gpuReadbackBuf, gpuZetaReadbackBuf;
var gpuTempBuf, gpuTempNewBuf, gpuTempReadbackBuf;
var gpuDeepTempBuf, gpuDeepTempNewBuf, gpuDeepTempReadbackBuf;
var gpuDeepPsiBuf, gpuDeepZetaBuf, gpuDeepZetaNewBuf, gpuDeepPsiReadbackBuf;
var gpuDepthBuf;
var gpuSalClimBuf, gpuWindCurlBuf, gpuEkmanBuf;
var gpuSnowBuf, gpuSeaIceBuf, gpuEvapBuf, gpuPrecipBuf;
var gpuTimestepPipeline, gpuPoissonPipeline, gpuEnforceBCPipeline, gpuTemperaturePipeline, gpuDeepTimestepPipeline;
var gpuTimestepBindGroup, gpuPoissonBindGroup, gpuEnforceBCBindGroup, gpuTemperatureBindGroup;
var gpuSwapTimestepBindGroup; // for after swap
var gpuSwapTemperatureBindGroup;

var readbackFrameCounter = 0;
var READBACK_INTERVAL = 2; // read back frequently for stability checks

async function initWebGPU() {
  if (!navigator.gpu) return false;
  var adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return false;
  gpuDevice = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: GPU_NX * GPU_NY * 4 * 4,  // stacked T+S buffers
      maxBufferSize: GPU_NX * GPU_NY * 4 * 4,
      maxStorageBuffersPerShaderStage: 14  // temperature shader needs 9 storage buffers
    }
  });
  if (!gpuDevice) return false;

  NX = GPU_NX; NY = GPU_NY;
  dx = 1.0 / (NX - 1); dy = 1.0 / (NY - 1);
  invDx = 1 / dx; invDy = 1 / dy;
  invDx2 = invDx * invDx; invDy2 = invDy * invDy;
  cellW = W / NX; cellH = H / NY;

  mask = buildMask(NX, NY);
  var maskU32 = buildMaskU32(mask, NX, NY);

  var bufSize = NX * NY * 4;

  // Create GPU buffers
  gpuPsiBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuZetaBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuZetaNewBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuMaskBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuParamsBuf = gpuDevice.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // 40 fields = 160 bytes (includes salinity params)
  gpuReadbackBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpuZetaReadbackBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  // Stacked layout: first NX*NY = temperature, second NX*NY = salinity (2x size)
  var tracerBufSize = bufSize * 2;
  gpuTempBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuTempNewBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuTempReadbackBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpuDeepTempBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuDeepTempNewBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuDeepTempReadbackBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpuDeepPsiBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuDeepZetaBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuDeepZetaNewBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuDeepPsiReadbackBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  gpuDepthBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuSalClimBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuWindCurlBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuEkmanBuf = gpuDevice.createBuffer({ size: tracerBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); // stacked u_ek + v_ek
  gpuSnowBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuSeaIceBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuEvapBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuPrecipBuf = gpuDevice.createBuffer({ size: bufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  // FFT Poisson solver buffers (complex: real + imaginary, row-major and mode-major)
  var fftBufSize = NX * NY * 4;
  var gpuFFTReA = gpuDevice.createBuffer({ size: fftBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  var gpuFFTImA = gpuDevice.createBuffer({ size: fftBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  var gpuFFTReB = gpuDevice.createBuffer({ size: fftBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  var gpuFFTImB = gpuDevice.createBuffer({ size: fftBufSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  var gpuFFTParamsBuf = gpuDevice.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); // nx, ny, passStride, direction
  // Precomputed cos(lat) per row for tridiagonal solver
  var cosLatData = new Float32Array(NY);
  for (var cj = 0; cj < NY; cj++) {
    var clat = LAT0 + (cj / (NY - 1)) * (LAT1 - LAT0);
    cosLatData[cj] = Math.max(Math.cos(clat * Math.PI / 180), 0.087);
  }
  var gpuCosLatBuf = gpuDevice.createBuffer({ size: NY * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gpuDevice.queue.writeBuffer(gpuCosLatBuf, 0, cosLatData);

  // Upload mask
  gpuDevice.queue.writeBuffer(gpuMaskBuf, 0, maskU32);

  // Create pipelines
  gpuTimestepPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: timestepShaderCode }), entryPoint: 'main' }
  });

  gpuPoissonPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: poissonShaderCode }), entryPoint: 'main' }
  });

  gpuEnforceBCPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: enforceBCShaderCode }), entryPoint: 'main' }
  });

  gpuTemperaturePipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: temperatureShaderCode }), entryPoint: 'main' }
  });

  gpuDeepTimestepPipeline = gpuDevice.createComputePipeline({
    layout: 'auto',
    compute: { module: gpuDevice.createShaderModule({ code: deepTimestepShaderCode }), entryPoint: 'main' }
  });

  // FFT Poisson pipelines
  var gpuFFTBitRevPipeline = gpuDevice.createComputePipeline({
    layout: 'auto', compute: { module: gpuDevice.createShaderModule({ code: fftBitRevShaderCode }), entryPoint: 'main' }
  });
  var gpuFFTButterflyPipeline = gpuDevice.createComputePipeline({
    layout: 'auto', compute: { module: gpuDevice.createShaderModule({ code: fftButterflyShaderCode }), entryPoint: 'main' }
  });
  var gpuFFTTridiagPipeline = gpuDevice.createComputePipeline({
    layout: 'auto', compute: { module: gpuDevice.createShaderModule({ code: fftTridiagShaderCode }), entryPoint: 'main' }
  });
  var gpuFFTTransposePipeline = gpuDevice.createComputePipeline({
    layout: 'auto', compute: { module: gpuDevice.createShaderModule({ code: fftTransposeShaderCode }), entryPoint: 'main' }
  });
  var gpuFFTScaleMaskPipeline = gpuDevice.createComputePipeline({
    layout: 'auto', compute: { module: gpuDevice.createShaderModule({ code: fftScaleMaskShaderCode }), entryPoint: 'main' }
  });

  // FFT bind groups
  var fftButterflyBG = gpuDevice.createBindGroup({
    layout: gpuFFTButterflyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReA } },
      { binding: 1, resource: { buffer: gpuFFTImA } },
      { binding: 2, resource: { buffer: gpuFFTParamsBuf } }
    ]
  });
  var fftBitRevBG = gpuDevice.createBindGroup({
    layout: gpuFFTBitRevPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReA } },
      { binding: 1, resource: { buffer: gpuFFTImA } },
      { binding: 2, resource: { buffer: gpuFFTParamsBuf } }
    ]
  });
  var fftTransposeReBG = gpuDevice.createBindGroup({
    layout: gpuFFTTransposePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReA } },
      { binding: 1, resource: { buffer: gpuFFTReB } },
      { binding: 2, resource: { buffer: gpuFFTParamsBuf } }
    ]
  });
  var fftTransposeImBG = gpuDevice.createBindGroup({
    layout: gpuFFTTransposePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTImA } },
      { binding: 1, resource: { buffer: gpuFFTImB } },
      { binding: 2, resource: { buffer: gpuFFTParamsBuf } }
    ]
  });
  var fftTridiagBG = gpuDevice.createBindGroup({
    layout: gpuFFTTridiagPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReB } },  // mode-major zeta_hat real
      { binding: 1, resource: { buffer: gpuFFTImB } },  // mode-major zeta_hat imag
      { binding: 2, resource: { buffer: gpuFFTReA } },  // output psi_hat real (reuse A)
      { binding: 3, resource: { buffer: gpuFFTImA } },  // output psi_hat imag (reuse A)
      { binding: 4, resource: { buffer: gpuFFTParamsBuf } },
      { binding: 5, resource: { buffer: gpuCosLatBuf } }
    ]
  });
  // Transpose back: mode-major A → row-major B
  var fftTransposeBackReBG = gpuDevice.createBindGroup({
    layout: gpuFFTTransposePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReA } },
      { binding: 1, resource: { buffer: gpuFFTReB } },
      { binding: 2, resource: { buffer: gpuFFTParamsBuf } }
    ]
  });
  var fftTransposeBackImBG = gpuDevice.createBindGroup({
    layout: gpuFFTTransposePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTImA } },
      { binding: 1, resource: { buffer: gpuFFTImB } },
      { binding: 2, resource: { buffer: gpuFFTParamsBuf } }
    ]
  });
  // Inverse FFT uses B buffers
  var fftInvButterflyBG = gpuDevice.createBindGroup({
    layout: gpuFFTButterflyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReB } },
      { binding: 1, resource: { buffer: gpuFFTImB } },
      { binding: 2, resource: { buffer: gpuFFTParamsBuf } }
    ]
  });
  var fftInvBitRevBG = gpuDevice.createBindGroup({
    layout: gpuFFTBitRevPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReB } },
      { binding: 1, resource: { buffer: gpuFFTImB } },
      { binding: 2, resource: { buffer: gpuFFTParamsBuf } }
    ]
  });

  // Precompute per-pass param buffers and bind groups for FFT
  var logNX = Math.round(Math.log2(NX)); // 9 for NX=512
  var fftN = NX * NY;
  var fftWG = Math.ceil(fftN / 64);
  var fftWGHalf = Math.ceil((NX / 2 * NY) / 64);

  function makeFFTParamBuf(nx, ny, stride, dir) {
    var buf = gpuDevice.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    var data = new ArrayBuffer(16);
    new Uint32Array(data, 0, 3).set([nx, ny, stride]);
    new Float32Array(data, 12, 1)[0] = dir;
    gpuDevice.queue.writeBuffer(buf, 0, data);
    return buf;
  }

  // Forward FFT params + bind groups (A buffers)
  var fftFwdBitRevParamBuf = makeFFTParamBuf(NX, NY, 0, -1.0);
  var fftFwdBitRevBG = gpuDevice.createBindGroup({
    layout: gpuFFTBitRevPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: gpuFFTReA } }, { binding: 1, resource: { buffer: gpuFFTImA } }, { binding: 2, resource: { buffer: fftFwdBitRevParamBuf } }]
  });
  var fftFwdButterflyBGs = [];
  for (var p = 0; p < logNX; p++) {
    var pb = makeFFTParamBuf(NX, NY, 1 << p, -1.0);
    fftFwdButterflyBGs.push(gpuDevice.createBindGroup({
      layout: gpuFFTButterflyPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: gpuFFTReA } }, { binding: 1, resource: { buffer: gpuFFTImA } }, { binding: 2, resource: { buffer: pb } }]
    }));
  }

  // Transpose params: forward (NX,NY) and reverse (NY,NX)
  var fftTransFwdParamBuf = makeFFTParamBuf(NX, NY, 0, 0);
  var fftTransRevParamBuf = makeFFTParamBuf(NY, NX, 0, 0);

  var fftTransFwdReBG = gpuDevice.createBindGroup({
    layout: gpuFFTTransposePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: gpuFFTReA } }, { binding: 1, resource: { buffer: gpuFFTReB } }, { binding: 2, resource: { buffer: fftTransFwdParamBuf } }]
  });
  var fftTransFwdImBG = gpuDevice.createBindGroup({
    layout: gpuFFTTransposePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: gpuFFTImA } }, { binding: 1, resource: { buffer: gpuFFTImB } }, { binding: 2, resource: { buffer: fftTransFwdParamBuf } }]
  });

  // Tridiagonal params
  var fftTridiagParamBuf = makeFFTParamBuf(NX, NY, 0, 0);
  var fftTridiagBGReal = gpuDevice.createBindGroup({
    layout: gpuFFTTridiagPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReB } }, { binding: 1, resource: { buffer: gpuFFTImB } },
      { binding: 2, resource: { buffer: gpuFFTReA } }, { binding: 3, resource: { buffer: gpuFFTImA } },
      { binding: 4, resource: { buffer: fftTridiagParamBuf } }, { binding: 5, resource: { buffer: gpuCosLatBuf } }
    ]
  });

  // Reverse transpose: A → B (mode-major to row-major)
  var fftTransRevReBG = gpuDevice.createBindGroup({
    layout: gpuFFTTransposePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: gpuFFTReA } }, { binding: 1, resource: { buffer: gpuFFTReB } }, { binding: 2, resource: { buffer: fftTransRevParamBuf } }]
  });
  var fftTransRevImBG = gpuDevice.createBindGroup({
    layout: gpuFFTTransposePipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: gpuFFTImA } }, { binding: 1, resource: { buffer: gpuFFTImB } }, { binding: 2, resource: { buffer: fftTransRevParamBuf } }]
  });

  // Inverse FFT params + bind groups (B buffers)
  var fftInvBitRevParamBuf = makeFFTParamBuf(NX, NY, 0, 1.0);
  var fftInvBitRevBGReal = gpuDevice.createBindGroup({
    layout: gpuFFTBitRevPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: gpuFFTReB } }, { binding: 1, resource: { buffer: gpuFFTImB } }, { binding: 2, resource: { buffer: fftInvBitRevParamBuf } }]
  });
  var fftInvButterflyBGs = [];
  for (var p = 0; p < logNX; p++) {
    var pb = makeFFTParamBuf(NX, NY, 1 << p, 1.0);
    fftInvButterflyBGs.push(gpuDevice.createBindGroup({
      layout: gpuFFTButterflyPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: gpuFFTReB } }, { binding: 1, resource: { buffer: gpuFFTImB } }, { binding: 2, resource: { buffer: pb } }]
    }));
  }

  // Scale+mask bind groups (created per-call since psiDstBuf varies)
  var fftScaleMaskParamBuf = makeFFTParamBuf(NX, NY, 0, 0);
  var fftScaleMaskBG_surface = gpuDevice.createBindGroup({
    layout: gpuFFTScaleMaskPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReB } }, { binding: 1, resource: { buffer: gpuPsiBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } }, { binding: 3, resource: { buffer: fftScaleMaskParamBuf } }
    ]
  });
  var fftScaleMaskBG_deep = gpuDevice.createBindGroup({
    layout: gpuFFTScaleMaskPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuFFTReB } }, { binding: 1, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } }, { binding: 3, resource: { buffer: fftScaleMaskParamBuf } }
    ]
  });

  // Pre-create zero buffer for clearing imaginary part
  var gpuFFTZeroBuf = gpuDevice.createBuffer({ size: fftN * 4, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  gpuDevice.queue.writeBuffer(gpuFFTZeroBuf, 0, new Float32Array(fftN));

  gpuFFTPoissonSolve = function(encoder, zetaSrcBuf, psiDstBuf) {
    // Copy zeta to real, zero imaginary
    encoder.copyBufferToBuffer(zetaSrcBuf, 0, gpuFFTReA, 0, fftN * 4);
    encoder.copyBufferToBuffer(gpuFFTZeroBuf, 0, gpuFFTImA, 0, fftN * 4);

    // Forward FFT: bit-reversal + butterfly passes on A buffers
    var p0 = encoder.beginComputePass(); p0.setPipeline(gpuFFTBitRevPipeline); p0.setBindGroup(0, fftFwdBitRevBG); p0.dispatchWorkgroups(fftWG); p0.end();
    for (var p = 0; p < logNX; p++) {
      var bp = encoder.beginComputePass(); bp.setPipeline(gpuFFTButterflyPipeline); bp.setBindGroup(0, fftFwdButterflyBGs[p]); bp.dispatchWorkgroups(fftWGHalf); bp.end();
    }

    // Transpose A → B (row-major to mode-major)
    var t1 = encoder.beginComputePass(); t1.setPipeline(gpuFFTTransposePipeline); t1.setBindGroup(0, fftTransFwdReBG); t1.dispatchWorkgroups(fftWG); t1.end();
    var t2 = encoder.beginComputePass(); t2.setPipeline(gpuFFTTransposePipeline); t2.setBindGroup(0, fftTransFwdImBG); t2.dispatchWorkgroups(fftWG); t2.end();

    // Tridiagonal solve: B (input) → A (output)
    var tr = encoder.beginComputePass(); tr.setPipeline(gpuFFTTridiagPipeline); tr.setBindGroup(0, fftTridiagBGReal); tr.dispatchWorkgroups(NX); tr.end();

    // Transpose A → B (mode-major back to row-major, swapped dims)
    var t3 = encoder.beginComputePass(); t3.setPipeline(gpuFFTTransposePipeline); t3.setBindGroup(0, fftTransRevReBG); t3.dispatchWorkgroups(fftWG); t3.end();
    var t4 = encoder.beginComputePass(); t4.setPipeline(gpuFFTTransposePipeline); t4.setBindGroup(0, fftTransRevImBG); t4.dispatchWorkgroups(fftWG); t4.end();

    // Inverse FFT: bit-reversal + butterfly passes on B buffers
    var br = encoder.beginComputePass(); br.setPipeline(gpuFFTBitRevPipeline); br.setBindGroup(0, fftInvBitRevBGReal); br.dispatchWorkgroups(fftWG); br.end();
    for (var p = 0; p < logNX; p++) {
      var ibp = encoder.beginComputePass(); ibp.setPipeline(gpuFFTButterflyPipeline); ibp.setBindGroup(0, fftInvButterflyBGs[p]); ibp.dispatchWorkgroups(fftWGHalf); ibp.end();
    }

    // Scale by 1/NX and mask land
    var smBG = (psiDstBuf === gpuPsiBuf) ? fftScaleMaskBG_surface : fftScaleMaskBG_deep;
    var sm = encoder.beginComputePass(); sm.setPipeline(gpuFFTScaleMaskPipeline); sm.setBindGroup(0, smBG); sm.dispatchWorkgroups(fftWG); sm.end();
  };

  // Debug: run FFT stages one at a time with readback
  window.debugGPUFFT = async function() {
    // Write uniform zeta=1 to gpuZetaBuf
    var testZeta = new Float32Array(fftN);
    for (var j = 1; j < NY - 1; j++) for (var i = 0; i < NX; i++) testZeta[j * NX + i] = 1.0;
    gpuDevice.queue.writeBuffer(gpuZetaBuf, 0, testZeta);

    var readBuf = gpuDevice.createBuffer({ size: fftN * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    async function readBuffer(srcBuf, label) {
      var enc = gpuDevice.createCommandEncoder();
      enc.copyBufferToBuffer(srcBuf, 0, readBuf, 0, fftN * 4);
      gpuDevice.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      var data = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      var max = 0, nonZero = 0, sum = 0, nanCount = 0, infCount = 0;
      for (var k = 0; k < data.length; k++) {
        if (isNaN(data[k])) { nanCount++; continue; }
        if (!isFinite(data[k])) { infCount++; continue; }
        if (Math.abs(data[k]) > 1e-10) nonZero++;
        if (Math.abs(data[k]) > max) max = Math.abs(data[k]);
        sum += data[k];
      }
      var msg = label + ': max=' + max.toExponential(3) + ', nz=' + nonZero + ', nan=' + nanCount + ', inf=' + infCount;
      console.log(msg);
      // Show on screen
      var el = document.getElementById('gpu-fft-debug');
      if (!el) { el = document.createElement('pre'); el.id = 'gpu-fft-debug'; el.style.cssText = 'position:fixed;top:10px;left:10px;z-index:9999;background:rgba(0,0,0,0.9);color:#0f0;padding:12px;font-size:11px;max-height:90vh;overflow:auto;border:1px solid #0f0;pointer-events:none'; document.body.appendChild(el); }
      el.textContent += msg + '\n';
      return { max, nonZero, sum, sample: [data[0], data[80*NX+256], data[fftN-1]] };
    }

    // Stage 0: Copy zeta to ReA, zero ImA
    var e0 = gpuDevice.createCommandEncoder();
    e0.copyBufferToBuffer(gpuZetaBuf, 0, gpuFFTReA, 0, fftN * 4);
    e0.copyBufferToBuffer(gpuFFTZeroBuf, 0, gpuFFTImA, 0, fftN * 4);
    gpuDevice.queue.submit([e0.finish()]);
    await gpuDevice.queue.onSubmittedWorkDone();
    var s0 = await readBuffer(gpuFFTReA, '0-CopyZeta');

    // Stage 1: Bit-reversal
    var e1 = gpuDevice.createCommandEncoder();
    var p0 = e1.beginComputePass(); p0.setPipeline(gpuFFTBitRevPipeline); p0.setBindGroup(0, fftFwdBitRevBG); p0.dispatchWorkgroups(fftWG); p0.end();
    gpuDevice.queue.submit([e1.finish()]);
    await gpuDevice.queue.onSubmittedWorkDone();
    var s1 = await readBuffer(gpuFFTReA, '1-BitRev');

    // Stage 2: All butterfly passes
    for (var p = 0; p < logNX; p++) {
      var eb = gpuDevice.createCommandEncoder();
      var bp = eb.beginComputePass(); bp.setPipeline(gpuFFTButterflyPipeline); bp.setBindGroup(0, fftFwdButterflyBGs[p]); bp.dispatchWorkgroups(fftWGHalf); bp.end();
      gpuDevice.queue.submit([eb.finish()]);
      await gpuDevice.queue.onSubmittedWorkDone();
    }
    var s2 = await readBuffer(gpuFFTReA, '2-FFT');

    // Stage 3: Transpose ReA → ReB
    var e3 = gpuDevice.createCommandEncoder();
    var t1 = e3.beginComputePass(); t1.setPipeline(gpuFFTTransposePipeline); t1.setBindGroup(0, fftTransFwdReBG); t1.dispatchWorkgroups(fftWG); t1.end();
    var t2 = e3.beginComputePass(); t2.setPipeline(gpuFFTTransposePipeline); t2.setBindGroup(0, fftTransFwdImBG); t2.dispatchWorkgroups(fftWG); t2.end();
    gpuDevice.queue.submit([e3.finish()]);
    await gpuDevice.queue.onSubmittedWorkDone();
    var s3 = await readBuffer(gpuFFTReB, '3-Transpose');

    // Stage 4: Tridiagonal B→A
    var e4 = gpuDevice.createCommandEncoder();
    var tr = e4.beginComputePass(); tr.setPipeline(gpuFFTTridiagPipeline); tr.setBindGroup(0, fftTridiagBGReal); tr.dispatchWorkgroups(NX); tr.end();
    gpuDevice.queue.submit([e4.finish()]);
    await gpuDevice.queue.onSubmittedWorkDone();
    var s4 = await readBuffer(gpuFFTReA, '4-Tridiag');

    // Stage 5: Reverse transpose A→B
    var e5 = gpuDevice.createCommandEncoder();
    var t3 = e5.beginComputePass(); t3.setPipeline(gpuFFTTransposePipeline); t3.setBindGroup(0, fftTransRevReBG); t3.dispatchWorkgroups(fftWG); t3.end();
    var t4 = e5.beginComputePass(); t4.setPipeline(gpuFFTTransposePipeline); t4.setBindGroup(0, fftTransRevImBG); t4.dispatchWorkgroups(fftWG); t4.end();
    gpuDevice.queue.submit([e5.finish()]);
    await gpuDevice.queue.onSubmittedWorkDone();
    var s5 = await readBuffer(gpuFFTReB, '5-RevTranspose');

    // Stage 6: Inverse bit-reversal
    var e6a = gpuDevice.createCommandEncoder();
    var ibr = e6a.beginComputePass(); ibr.setPipeline(gpuFFTBitRevPipeline); ibr.setBindGroup(0, fftInvBitRevBGReal); ibr.dispatchWorkgroups(fftWG); ibr.end();
    gpuDevice.queue.submit([e6a.finish()]);
    await gpuDevice.queue.onSubmittedWorkDone();
    var s6a = await readBuffer(gpuFFTReB, '6a-InvBitRev(Re)');
    await readBuffer(gpuFFTImB, '6a-InvBitRev(Im)');
    // Stage 6b: Inverse butterfly passes
    for (var p = 0; p < logNX; p++) {
      var eib = gpuDevice.createCommandEncoder();
      var ibp = eib.beginComputePass(); ibp.setPipeline(gpuFFTButterflyPipeline); ibp.setBindGroup(0, fftInvButterflyBGs[p]); ibp.dispatchWorkgroups(fftWGHalf); ibp.end();
      gpuDevice.queue.submit([eib.finish()]);
      await gpuDevice.queue.onSubmittedWorkDone();
      if (p === 0) await readBuffer(gpuFFTReB, '6b-InvBfly0');
    }
    var s6 = await readBuffer(gpuFFTReB, '6-InvFFT');

    // Stage 7: Scale+mask
    var e7 = gpuDevice.createCommandEncoder();
    var sm = e7.beginComputePass(); sm.setPipeline(gpuFFTScaleMaskPipeline); sm.setBindGroup(0, fftScaleMaskBG_surface); sm.dispatchWorkgroups(fftWG); sm.end();
    gpuDevice.queue.submit([e7.finish()]);
    await gpuDevice.queue.onSubmittedWorkDone();
    var s7 = await readBuffer(gpuPsiBuf, '7-ScaleMask');

    readBuf.destroy();
    return { s0, s1, s2, s3, s4, s5, s6, s7 };
  };

  // Build bind groups
  rebuildBindGroups();

  // CPU-side readback arrays — initialize with Earth-like conditions
  psi = new Float32Array(NX * NY);
  zeta = new Float32Array(NX * NY);
  temp = new Float32Array(NX * NY);
  deepTemp = new Float32Array(NX * NY);
  sal = new Float32Array(NX * NY);
  deepSal = new Float32Array(NX * NY);

  deepPsi = new Float32Array(NX * NY);
  deepZeta = new Float32Array(NX * NY);

  // Pre-remap observation data to model grid (elevation, albedo, precip)
  buildRemappedFields();

  // Generate bathymetry from distance to coast
  generateDepthField();
  gpuDevice.queue.writeBuffer(gpuDepthBuf, 0, depth);

  // Salinity climatology (WOA23 or zonal formula)
  generateSalClimatologyField();
  gpuDevice.queue.writeBuffer(gpuSalClimBuf, 0, salClimatologyField);

  // Wind stress curl (NCEP or zeros for analytical fallback)
  generateWindCurlField();
  gpuDevice.queue.writeBuffer(gpuWindCurlBuf, 0, windCurlFieldData);

  // Observed cloud fraction for validation view
  generateObsCloudField();

  // Ekman velocity field for wind-driven heat transport
  generateEkmanField();
  gpuDevice.queue.writeBuffer(gpuEkmanBuf, 0, ekmanField);

  // Snow cover, sea ice, evaporation, precipitation for radiation + salinity
  generateSnowField();
  gpuDevice.queue.writeBuffer(gpuSnowBuf, 0, snowField);
  generateSeaIceField();
  gpuDevice.queue.writeBuffer(gpuSeaIceBuf, 0, seaIceField);
  generateEvapField();
  gpuDevice.queue.writeBuffer(gpuEvapBuf, 0, evapField);
  generatePrecipField();
  gpuDevice.queue.writeBuffer(gpuPrecipBuf, 0, precipOceanField);

  // Stommel analytical solution: western boundary current from the start
  initStommelSolution();
  gpuDevice.queue.writeBuffer(gpuPsiBuf, 0, psi);
  gpuDevice.queue.writeBuffer(gpuZetaBuf, 0, zeta);
  gpuDevice.queue.writeBuffer(gpuDeepPsiBuf, 0, deepPsi);
  gpuDevice.queue.writeBuffer(gpuDeepZetaBuf, 0, deepZeta);

  // Realistic temperature + salinity from observations
  initTemperatureField();
  // Pack T+S into stacked buffers for GPU
  var surfTracer = new Float32Array(NX * NY * 2);
  var deepTracer = new Float32Array(NX * NY * 2);
  for (var tk = 0; tk < NX * NY; tk++) {
    surfTracer[tk] = temp[tk];
    surfTracer[tk + NX * NY] = sal[tk];
    deepTracer[tk] = deepTemp[tk];
    deepTracer[tk + NX * NY] = deepSal[tk];
  }
  gpuDevice.queue.writeBuffer(gpuTempBuf, 0, surfTracer);
  gpuDevice.queue.writeBuffer(gpuDeepTempBuf, 0, deepTracer);

  return true;
}

function updateGPUBuffersAfterPaint() {
  if (!gpuDevice) return;
  // Re-upload mask
  var maskU32 = new Uint32Array(NX * NY);
  for (var k = 0; k < NX * NY; k++) maskU32[k] = mask[k];
  gpuDevice.queue.writeBuffer(gpuMaskBuf, 0, maskU32);
  // Re-upload zeta (for heat/cold/wind painting)
  var zetaF32 = new Float32Array(NX * NY);
  for (var k2 = 0; k2 < NX * NY; k2++) zetaF32[k2] = zeta[k2];
  gpuDevice.queue.writeBuffer(gpuZetaBuf, 0, zetaF32);
  // Re-upload psi (zeroed on new land)
  var psiF32 = new Float32Array(NX * NY);
  for (var k3 = 0; k3 < NX * NY; k3++) psiF32[k3] = psi[k3];
  gpuDevice.queue.writeBuffer(gpuPsiBuf, 0, psiF32);
  // Re-upload temperature
  var tempF32 = new Float32Array(NX * NY);
  for (var k4 = 0; k4 < NX * NY; k4++) tempF32[k4] = temp[k4];
  gpuDevice.queue.writeBuffer(gpuTempBuf, 0, tempF32);
  // Regenerate depth from mask and upload
  if (gpuDepthBuf) {
    generateDepthField();
    gpuDevice.queue.writeBuffer(gpuDepthBuf, 0, depth);
  }
  // Re-upload deep temperature
  if (deepTemp) {
    var deepF32 = new Float32Array(NX * NY);
    for (var k5 = 0; k5 < NX * NY; k5++) deepF32[k5] = deepTemp[k5];
    gpuDevice.queue.writeBuffer(gpuDeepTempBuf, 0, deepF32);
  }
}

function rebuildBindGroups() {
  // Timestep: reads psi, zeta, temp, deepPsi -> writes zetaNew
  gpuTimestepBindGroup = gpuDevice.createBindGroup({
    layout: gpuTimestepPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuZetaNewBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuTempBuf } },
      { binding: 6, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 7, resource: { buffer: gpuWindCurlBuf } },
    ]
  });

  // Swapped version
  gpuSwapTimestepBindGroup = gpuDevice.createBindGroup({
    layout: gpuTimestepPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuZetaBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuTempBuf } },
      { binding: 6, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 7, resource: { buffer: gpuWindCurlBuf } },
    ]
  });

  // Poisson: reads/writes psi, reads zeta
  // Red-Black SOR: two params buffers with different color flags
  var redBuf = new ArrayBuffer(160);
  var redU32 = new Uint32Array(redBuf);
  new Float32Array(redBuf).set(new Float32Array(gpuParamsBuf.size ? 40 : 40)); // will be overwritten by uploadParams
  redU32[32] = 0; // color = red
  gpuParamsRedBuf = gpuDevice.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  var blackBuf = new ArrayBuffer(160);
  var blackU32 = new Uint32Array(blackBuf);
  blackU32[32] = 1; // color = black
  gpuParamsBlackBuf = gpuDevice.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  gpuPoissonBindGroupRed = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsRedBuf } },
    ]
  });
  gpuPoissonBindGroupBlack = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBlackBuf } },
    ]
  });
  gpuPoissonBindGroupSwapRed = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsRedBuf } },
    ]
  });
  gpuPoissonBindGroupSwapBlack = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBlackBuf } },
    ]
  });

  // Deep Poisson reuses the same red/black approach
  gpuDeepPoissonBindGroupRed = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsRedBuf } },
    ]
  });
  gpuDeepPoissonBindGroupBlack = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBlackBuf } },
    ]
  });

  // EnforceBC: reads/writes psi and zeta, reads mask
  gpuEnforceBCBindGroup = gpuDevice.createBindGroup({
    layout: gpuEnforceBCPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });

  gpuEnforceBCBindGroupSwap = gpuDevice.createBindGroup({
    layout: gpuEnforceBCPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });

  // Temperature: reads psi, tempIn, deepTempIn, depth, ekman -> writes tempOut, deepTempOut
  gpuTemperatureBindGroup = gpuDevice.createBindGroup({
    layout: gpuTemperaturePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuTempBuf } },
      { binding: 2, resource: { buffer: gpuTempNewBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuDeepTempBuf } },
      { binding: 6, resource: { buffer: gpuDeepTempNewBuf } },
      { binding: 7, resource: { buffer: gpuDepthBuf } },
      { binding: 8, resource: { buffer: gpuSalClimBuf } },
      { binding: 9, resource: { buffer: gpuEkmanBuf } },
      { binding: 10, resource: { buffer: gpuSnowBuf } },
      { binding: 11, resource: { buffer: gpuSeaIceBuf } },
      { binding: 12, resource: { buffer: gpuEvapBuf } },
      { binding: 13, resource: { buffer: gpuPrecipBuf } },
    ]
  });

  // Swapped: reads psi, tempNew, deepTempNew, depth -> writes temp, deepTemp
  gpuSwapTemperatureBindGroup = gpuDevice.createBindGroup({
    layout: gpuTemperaturePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuPsiBuf } },
      { binding: 1, resource: { buffer: gpuTempNewBuf } },
      { binding: 2, resource: { buffer: gpuTempBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuDeepTempNewBuf } },
      { binding: 6, resource: { buffer: gpuDeepTempBuf } },
      { binding: 7, resource: { buffer: gpuDepthBuf } },
      { binding: 8, resource: { buffer: gpuSalClimBuf } },
      { binding: 9, resource: { buffer: gpuEkmanBuf } },
      { binding: 10, resource: { buffer: gpuSnowBuf } },
      { binding: 11, resource: { buffer: gpuSeaIceBuf } },
      { binding: 12, resource: { buffer: gpuEvapBuf } },
      { binding: 13, resource: { buffer: gpuPrecipBuf } },
    ]
  });
  // Deep timestep: reads deepPsi, deepZeta, surfacePsi -> writes deepZetaNew
  gpuDeepTimestepBindGroup = gpuDevice.createBindGroup({
    layout: gpuDeepTimestepPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuDeepZetaNewBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuPsiBuf } },
      { binding: 6, resource: { buffer: gpuDeepTempBuf } },
    ]
  });
  gpuSwapDeepTimestepBindGroup = gpuDevice.createBindGroup({
    layout: gpuDeepTimestepPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 3, resource: { buffer: gpuMaskBuf } },
      { binding: 4, resource: { buffer: gpuParamsBuf } },
      { binding: 5, resource: { buffer: gpuPsiBuf } },
      { binding: 6, resource: { buffer: gpuDeepTempBuf } },
    ]
  });

  // Deep Poisson: reuses pipeline, different buffers
  gpuDeepPoissonBindGroup = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });
  gpuDeepPoissonBindGroupSwap = gpuDevice.createBindGroup({
    layout: gpuPoissonPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });

  // Deep enforceBC: reuses pipeline, different buffers
  gpuDeepEnforceBCBindGroup = gpuDevice.createBindGroup({
    layout: gpuEnforceBCPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });
  gpuDeepEnforceBCBindGroupSwap = gpuDevice.createBindGroup({
    layout: gpuEnforceBCPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: gpuDeepPsiBuf } },
      { binding: 1, resource: { buffer: gpuDeepZetaNewBuf } },
      { binding: 2, resource: { buffer: gpuMaskBuf } },
      { binding: 3, resource: { buffer: gpuParamsBuf } },
    ]
  });
}

var gpuPoissonBindGroupSwap, gpuEnforceBCBindGroupSwap;
var gpuDeepTimestepBindGroup, gpuSwapDeepTimestepBindGroup;
var gpuDeepPoissonBindGroup, gpuDeepPoissonBindGroupSwap;
var gpuDeepEnforceBCBindGroup, gpuDeepEnforceBCBindGroupSwap;
// Red-Black SOR Poisson bind groups + params buffers
var gpuParamsRedBuf, gpuParamsBlackBuf;
var gpuPoissonBindGroupRed, gpuPoissonBindGroupBlack;
var gpuPoissonBindGroupSwapRed, gpuPoissonBindGroupSwapBlack;
var gpuDeepPoissonBindGroupRed, gpuDeepPoissonBindGroupBlack;

function uploadParams() {
  // Params struct: nx(u32), ny(u32), dx(f32), dy(f32), dt(f32), beta(f32), r(f32), A(f32),
  //   windStrength(f32), doubleGyre(u32), alphaT(f32), simTime(f32), yearSpeed(f32), freshwater(f32), pad, pad
  var buf = new ArrayBuffer(160);
  var u32 = new Uint32Array(buf);
  var f32 = new Float32Array(buf);
  u32[0] = NX;
  u32[1] = NY;
  f32[2] = dx;
  f32[3] = dy;
  f32[4] = dt;
  f32[5] = beta;
  f32[6] = r_friction;
  f32[7] = A_visc;
  f32[8] = windStrength;
  u32[9] = doubleGyre ? 1 : 0;
  f32[10] = alpha_T;
  f32[11] = simTime;
  f32[12] = yearSpeed;
  f32[13] = freshwaterForcing;
  f32[14] = globalTempOffset;
  f32[15] = gamma_mix;
  f32[16] = gamma_deep_form;
  f32[17] = kappa_deep;
  f32[18] = H_surface;
  f32[19] = H_deep;
  f32[20] = F_couple_s; f32[21] = F_couple_d;
  f32[22] = S_solar; f32[23] = A_olr;
  f32[24] = B_olr; f32[25] = kappa_diff;
  f32[26] = r_deep; f32[27] = 0.02; // landHeatK
  // Salinity parameters
  f32[28] = beta_S;             // haline contraction coefficient
  f32[29] = kappa_sal;          // salinity diffusion
  f32[30] = kappa_deep_sal;     // deep salinity diffusion
  f32[31] = salRestoringRate;   // surface salinity restoring rate
  u32[32] = 0; // _padS0 (SOR color flag)
  f32[33] = evapScale;
  f32[34] = peScale;
  f32[35] = snowAlbedoScale;
  // Note: in WGSL buf must be 16-byte aligned, 160 = 40 * 4 bytes, OK
  // But uniform buffer size must match struct size exactly
  // 36 used fields * 4 = 144 bytes, pad to 160 for alignment
  gpuDevice.queue.writeBuffer(gpuParamsBuf, 0, buf);
  // Red-black SOR params: copy main params, then set color flag
  if (gpuParamsRedBuf) {
    u32[32] = 0; // red
    gpuDevice.queue.writeBuffer(gpuParamsRedBuf, 0, buf);
    u32[32] = 1; // black
    gpuDevice.queue.writeBuffer(gpuParamsBlackBuf, 0, buf);
    u32[32] = 0; // restore
  }
}

var POISSON_ITERS = 12;        // grid Laplacian (no cos(lat)) — converges fast with ψ as initial guess
var DEEP_POISSON_ITERS = 6;    // deep layer

function gpuRunSteps(nSteps) {
  uploadParams();

  var wgX = Math.ceil(NX / 8);
  var wgY = Math.ceil(NY / 8);
  var wgLinear = Math.ceil((NX * NY) / 64);

  var encoder = gpuDevice.createCommandEncoder();

  for (var s = 0; s < nSteps; s++) {
    var isEven = (s % 2 === 0);

    // Timestep: compute zetaNew from zeta and psi (reads temp for buoyancy)
    var tsPass = encoder.beginComputePass();
    tsPass.setPipeline(gpuTimestepPipeline);
    tsPass.setBindGroup(0, isEven ? gpuTimestepBindGroup : gpuSwapTimestepBindGroup);
    tsPass.dispatchWorkgroups(wgX, wgY);
    tsPass.end();

    // Temperature step: advect and force temperature
    var tempPass = encoder.beginComputePass();
    tempPass.setPipeline(gpuTemperaturePipeline);
    tempPass.setBindGroup(0, isEven ? gpuTemperatureBindGroup : gpuSwapTemperatureBindGroup);
    tempPass.dispatchWorkgroups(wgX, wgY);
    tempPass.end();

    // After timestep, the "new" zeta is in the opposite buffer.
    // Enforce BC on psi and the new zeta
    var bcPass = encoder.beginComputePass();
    bcPass.setPipeline(gpuEnforceBCPipeline);
    bcPass.setBindGroup(0, isEven ? gpuEnforceBCBindGroupSwap : gpuEnforceBCBindGroup);
    bcPass.dispatchWorkgroups(wgLinear);
    bcPass.end();

    // Surface Poisson solve: red-black SOR (iterative, hardware-portable)
    for (var pi = 0; pi < POISSON_ITERS; pi++) {
      var redPass = encoder.beginComputePass();
      redPass.setPipeline(gpuPoissonPipeline);
      redPass.setBindGroup(0, isEven ? gpuPoissonBindGroupRed : gpuPoissonBindGroupSwapRed);
      redPass.dispatchWorkgroups(wgX, wgY);
      redPass.end();
      var blackPass = encoder.beginComputePass();
      blackPass.setPipeline(gpuPoissonPipeline);
      blackPass.setBindGroup(0, isEven ? gpuPoissonBindGroupBlack : gpuPoissonBindGroupSwapBlack);
      blackPass.dispatchWorkgroups(wgX, wgY);
      blackPass.end();
    }

    // Deep layer timestep
    var deepTsPass = encoder.beginComputePass();
    deepTsPass.setPipeline(gpuDeepTimestepPipeline);
    deepTsPass.setBindGroup(0, isEven ? gpuDeepTimestepBindGroup : gpuSwapDeepTimestepBindGroup);
    deepTsPass.dispatchWorkgroups(wgX, wgY);
    deepTsPass.end();

    // Deep enforceBC
    var deepBcPass = encoder.beginComputePass();
    deepBcPass.setPipeline(gpuEnforceBCPipeline);
    deepBcPass.setBindGroup(0, isEven ? gpuDeepEnforceBCBindGroupSwap : gpuDeepEnforceBCBindGroup);
    deepBcPass.dispatchWorkgroups(wgLinear);
    deepBcPass.end();

    // Copy deep zeta to primary buffer if in swapped state (even steps)
    if (isEven) {
      encoder.copyBufferToBuffer(gpuDeepZetaNewBuf, 0, gpuDeepZetaBuf, 0, NX * NY * 4);
    }

    // Deep Poisson solve: red-black SOR (iterative)
    for (var dpi = 0; dpi < DEEP_POISSON_ITERS; dpi++) {
      var deepRedPass = encoder.beginComputePass();
      deepRedPass.setPipeline(gpuPoissonPipeline);
      deepRedPass.setBindGroup(0, gpuDeepPoissonBindGroupRed);
      deepRedPass.dispatchWorkgroups(wgX, wgY);
      deepRedPass.end();
      var deepBlackPass = encoder.beginComputePass();
      deepBlackPass.setPipeline(gpuPoissonPipeline);
      deepBlackPass.setBindGroup(0, gpuDeepPoissonBindGroupBlack);
      deepBlackPass.dispatchWorkgroups(wgX, wgY);
      deepBlackPass.end();
    }
  }

  // After all steps, ensure final results are in primary buffers
  if (nSteps % 2 !== 0) {
    encoder.copyBufferToBuffer(gpuZetaNewBuf, 0, gpuZetaBuf, 0, NX * NY * 4);
    encoder.copyBufferToBuffer(gpuTempNewBuf, 0, gpuTempBuf, 0, NX * NY * 4 * 2);   // T + S stacked
    encoder.copyBufferToBuffer(gpuDeepTempNewBuf, 0, gpuDeepTempBuf, 0, NX * NY * 4 * 2); // Td + Sd stacked
    encoder.copyBufferToBuffer(gpuDeepZetaNewBuf, 0, gpuDeepZetaBuf, 0, NX * NY * 4);
  }

  // Readback when needed
  var needReadback = gpuRenderEnabled ? (readbackFrameCounter % READBACK_INTERVAL === 0) : true;
  if (needReadback) {
    encoder.copyBufferToBuffer(gpuPsiBuf, 0, gpuReadbackBuf, 0, NX * NY * 4);
    encoder.copyBufferToBuffer(gpuZetaBuf, 0, gpuZetaReadbackBuf, 0, NX * NY * 4);
    encoder.copyBufferToBuffer(gpuTempBuf, 0, gpuTempReadbackBuf, 0, NX * NY * 4 * 2);   // T + S
    encoder.copyBufferToBuffer(gpuDeepTempBuf, 0, gpuDeepTempReadbackBuf, 0, NX * NY * 4 * 2); // Td + Sd
    encoder.copyBufferToBuffer(gpuDeepPsiBuf, 0, gpuDeepPsiReadbackBuf, 0, NX * NY * 4);
  }

  gpuDevice.queue.submit([encoder.finish()]);
  totalSteps += nSteps;
  simTime += nSteps * dt * yearSpeed;
}
var readbackPending = false;

async function gpuReadback() {
  if (readbackPending) return;
  readbackPending = true;
  try {
    await gpuReadbackBuf.mapAsync(GPUMapMode.READ);
    var data = new Float32Array(gpuReadbackBuf.getMappedRange().slice(0));
    gpuReadbackBuf.unmap();
    psi = data;

    await gpuZetaReadbackBuf.mapAsync(GPUMapMode.READ);
    var zData = new Float32Array(gpuZetaReadbackBuf.getMappedRange().slice(0));
    gpuZetaReadbackBuf.unmap();
    zeta = zData;

    // Stacked readback: first NX*NY = temperature, second NX*NY = salinity
    await gpuTempReadbackBuf.mapAsync(GPUMapMode.READ);
    var tSData = new Float32Array(gpuTempReadbackBuf.getMappedRange().slice(0));
    gpuTempReadbackBuf.unmap();
    temp = tSData.subarray(0, NX * NY);
    sal = tSData.subarray(NX * NY, NX * NY * 2);

    await gpuDeepTempReadbackBuf.mapAsync(GPUMapMode.READ);
    var dSData = new Float32Array(gpuDeepTempReadbackBuf.getMappedRange().slice(0));
    gpuDeepTempReadbackBuf.unmap();
    deepTemp = dSData.subarray(0, NX * NY);
    deepSal = dSData.subarray(NX * NY, NX * NY * 2);

    await gpuDeepPsiReadbackBuf.mapAsync(GPUMapMode.READ);
    var dpData = new Float32Array(gpuDeepPsiReadbackBuf.getMappedRange().slice(0));
    gpuDeepPsiReadbackBuf.unmap();
    deepPsi = dpData;
  } catch (e) {
    // readback failed, skip this frame
  }
  readbackPending = false;
}

function gpuReset() {
  psi = new Float32Array(NX * NY);
  zeta = new Float32Array(NX * NY);
  temp = new Float32Array(NX * NY);
  deepTemp = new Float32Array(NX * NY);
  sal = new Float32Array(NX * NY);
  deepSal = new Float32Array(NX * NY);
  initStommelSolution();
  initTemperatureField();
  gpuDevice.queue.writeBuffer(gpuPsiBuf, 0, psi);
  gpuDevice.queue.writeBuffer(gpuZetaBuf, 0, zeta);
  gpuDevice.queue.writeBuffer(gpuZetaNewBuf, 0, new Float32Array(NX * NY));
  // Pack T+S stacked for GPU
  var surfTr = new Float32Array(NX * NY * 2);
  var deepTr = new Float32Array(NX * NY * 2);
  for (var rk = 0; rk < NX * NY; rk++) {
    surfTr[rk] = temp[rk]; surfTr[rk + NX * NY] = sal[rk];
    deepTr[rk] = deepTemp[rk]; deepTr[rk + NX * NY] = deepSal[rk];
  }
  gpuDevice.queue.writeBuffer(gpuTempBuf, 0, surfTr);
  gpuDevice.queue.writeBuffer(gpuTempNewBuf, 0, new Float32Array(NX * NY * 2));
  gpuDevice.queue.writeBuffer(gpuDeepTempBuf, 0, deepTr);
  gpuDevice.queue.writeBuffer(gpuDeepTempNewBuf, 0, new Float32Array(NX * NY * 2));
  deepPsi = new Float32Array(NX * NY);
  deepZeta = new Float32Array(NX * NY);
  gpuDevice.queue.writeBuffer(gpuDeepPsiBuf, 0, deepPsi);
  gpuDevice.queue.writeBuffer(gpuDeepZetaBuf, 0, deepZeta);
  gpuDevice.queue.writeBuffer(gpuDeepZetaNewBuf, 0, new Float32Array(NX * NY));
  // Atmosphere (CPU-side)
  airTemp = new Float32Array(NX * NY);
  for (var ai = 0; ai < NX * NY; ai++) {
    if (mask[ai]) airTemp[ai] = temp[ai];
    else { var aj = Math.floor(ai / NX); airTemp[ai] = 28 - 0.55 * Math.abs(LAT0 + (aj/(NY-1))*(LAT1-LAT0)); }
  }
  totalSteps = 0;
  simTime = 0;
  readbackFrameCounter = 0;
}

