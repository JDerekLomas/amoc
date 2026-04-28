struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: FFTParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let idx = id.x;
  if (idx >= p.nx * p.ny) { return; }
  // Row-major to mode-major: src[j*nx+i] -> dst[i*ny+j]
  let j = idx / p.nx;
  let i = idx % p.nx;
  dst[i * p.ny + j] = src[j * p.nx + i];
}
