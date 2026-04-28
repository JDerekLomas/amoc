struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };
@group(0) @binding(0) var<storage, read> re: array<f32>;
@group(0) @binding(1) var<storage, read_write> psi: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<uniform> p: FFTParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= p.nx * p.ny) { return; }
  psi[k] = select(0.0, clamp(re[k] / f32(p.nx), -50.0, 50.0), mask[k] != 0u);
}
