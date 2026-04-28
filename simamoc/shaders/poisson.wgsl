// {{PARAMS}} — replaced at load time with shared Params struct

@group(0) @binding(0) var<storage, read_write> psi: array<f32>;
@group(0) @binding(1) var<storage, read> zeta: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn idx(i: u32, j: u32) -> u32 { return j * params.nx + i; }

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let j = id.y;
  let nx = params.nx;
  if (i >= nx || j < 1u || j >= params.ny - 1u) { return; }

  // Red-black SOR: _padS0 holds color (0=red, 1=black)
  let color = params._padS0;
  if ((i + j) % 2u != color) { return; }

  let k = idx(i, j);
  if (mask[k] == 0u) { return; }

  // Periodic wrapping in x
  let ip1 = select(i + 1u, 0u, i == nx - 1u);
  let im1 = select(i - 1u, nx - 1u, i == 0u);

  // Poisson uses computational (grid) Laplacian — zeta = nabla^2_grid psi
  // cos(lat) correction is in the physics operators, not the Poisson inversion
  let invDx2 = 1.0 / (params.dx * params.dx);
  let invDy2 = 1.0 / (params.dy * params.dy);
  let cx = invDx2;
  let cy = invDy2;
  let cc = -2.0 * (cx + cy);

  let rhs = zeta[k];
  let neighbor_sum = cx * (psi[idx(ip1, j)] + psi[idx(im1, j)])
                   + cy * (psi[idx(i, j + 1u)] + psi[idx(i, j - 1u)]);
  let psiNew = (rhs - neighbor_sum) / cc;
  let omega = 1.92;
  psi[k] = psi[k] + omega * (psiNew - psi[k]);
}
