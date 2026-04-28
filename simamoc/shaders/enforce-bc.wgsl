// {{PARAMS}} — replaced at load time with shared Params struct

@group(0) @binding(0) var<storage, read_write> psi: array<f32>;
@group(0) @binding(1) var<storage, read_write> zeta: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let k = id.x;
  if (k >= params.nx * params.ny) { return; }
  let j = k / params.nx;
  if (mask[k] == 0u || j == 0u || j == params.ny - 1u) {
    psi[k] = 0.0;
    zeta[k] = 0.0;
  }
}
