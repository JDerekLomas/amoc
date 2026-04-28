struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };
@group(0) @binding(0) var<storage, read_write> re: array<f32>;
@group(0) @binding(1) var<storage, read_write> im: array<f32>;
@group(0) @binding(2) var<uniform> p: FFTParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let row = id.x / (p.nx / 2u);
  let halfIdx = id.x % (p.nx / 2u);
  if (row >= p.ny) { return; }

  let stride = p.passStride;
  let halfLen = stride;
  let fullLen = stride * 2u;

  // Which butterfly group and position within it
  let group = halfIdx / halfLen;
  let j = halfIdx % halfLen;
  let base = row * p.nx + group * fullLen;
  let i0 = base + j;
  let i1 = base + j + halfLen;

  // Twiddle factor: exp(direction * -2*pi*i * j / fullLen)
  let ang = p.direction * 2.0 * 3.14159265358979 * f32(j) / f32(fullLen);
  let wR = cos(ang);
  let wI = sin(ang);

  let uR = re[i0]; let uI = im[i0];
  let vR = re[i1] * wR - im[i1] * wI;
  let vI = re[i1] * wI + im[i1] * wR;
  re[i0] = uR + vR; im[i0] = uI + vI;
  re[i1] = uR - vR; im[i1] = uI - vI;
}
