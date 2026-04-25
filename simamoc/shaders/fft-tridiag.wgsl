struct FFTParams { nx: u32, ny: u32, passStride: u32, direction: f32 };
@group(0) @binding(0) var<storage, read_write> reIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> imIn: array<f32>;
@group(0) @binding(2) var<storage, read_write> reOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> imOut: array<f32>;
@group(0) @binding(4) var<uniform> p: FFTParams;
@group(0) @binding(5) var<storage, read> cosLatArr: array<f32>;

@compute @workgroup_size(1)  // one workgroup per mode (sequential Thomas)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let m = id.x;
  if (m >= p.nx) { return; }
  let ny = p.ny;
  let dx = 1.0 / f32(p.nx - 1u);
  let dy = 1.0 / f32(ny - 1u);
  let invDy2 = 1.0 / (dy * dy);
  let invDx2 = 1.0 / (dx * dx);

  // Eigenvalue for mode m: km2 = invDx2 * 2 * (cos(2*pi*m/NX) - 1)
  let km2 = invDx2 * 2.0 * (cos(2.0 * 3.14159265358979 * f32(m) / f32(p.nx)) - 1.0);

  // Thomas algorithm — forward elimination
  // Interior rows only: a*psi_{j-1} + b_j*psi_j + c*psi_{j+1} = zeta_hat_j
  // a = c = invDy2, b_j = km2 - 2*invDy2
  // Boundary: psi(0) = psi(NY-1) = 0
  var b_prev = 1.0;
  var dR_prev = 0.0;
  var dI_prev = 0.0;

  reOut[m * ny] = 1.0;
  imOut[m * ny] = 0.0;

  for (var j = 1u; j < ny - 1u; j++) {
    let _cl = cosLatArr[j]; // keep binding alive for WGSL validation
    var b_j = km2 - 2.0 * invDy2;
    var rhs_r = reIn[m * ny + j];
    var rhs_i = imIn[m * ny + j];
    let a = invDy2;
    let cp = select(0.0, invDy2, j > 1u);
    let w = a / b_prev;
    b_j -= w * cp;
    rhs_r -= w * dR_prev;
    rhs_i -= w * dI_prev;
    reOut[m * ny + j] = b_j;
    b_prev = b_j;
    dR_prev = rhs_r;
    dI_prev = rhs_i;
    reIn[m * ny + j] = rhs_r;
    imIn[m * ny + j] = rhs_i;
  }

  // Back substitution — Dirichlet BCs: psi = 0 at boundaries
  let last = ny - 1u;
  reOut[m * ny + last] = 0.0;
  imOut[m * ny + last] = 0.0;
  for (var jj = 1u; jj < ny - 1u; jj++) {
    let j = last - jj;
    let b_j = reOut[m * ny + j];
    reOut[m * ny + j] = (reIn[m * ny + j] - invDy2 * reOut[m * ny + j + 1u]) / b_j;
    imOut[m * ny + j] = (imIn[m * ny + j] - invDy2 * imOut[m * ny + j + 1u]) / b_j;
  }
  reOut[m * ny] = 0.0;
  imOut[m * ny] = 0.0;
}
