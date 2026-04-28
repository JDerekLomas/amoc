struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };
@group(0) @binding(0) var<storage, read_write> re: array<f32>;
@group(0) @binding(1) var<storage, read_write> im: array<f32>;
@group(0) @binding(2) var<uniform> p: FFTParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x / p.nx;
  let i = id.x % p.nx;
  if (row >= p.ny) { return; }

  // Compute bit-reversed index for log2(nx) bits
  var rev = 0u;
  var val = i;
  var bits = 0u;
  var tmp = p.nx >> 1u;
  while (tmp > 0u) { bits++; tmp >>= 1u; }
  for (var b = 0u; b < bits; b++) {
    rev = (rev << 1u) | (val & 1u);
    val >>= 1u;
  }

  if (i < rev) {
    let base = row * p.nx;
    let tR = re[base + i]; let tI = im[base + i];
    re[base + i] = re[base + rev]; im[base + i] = im[base + rev];
    re[base + rev] = tR; im[base + rev] = tI;
  }
}
