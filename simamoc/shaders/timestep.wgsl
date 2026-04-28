// {{PARAMS}} — replaced at load time with shared Params struct

@group(0) @binding(0) var<storage, read> psi: array<f32>;
@group(0) @binding(1) var<storage, read> zeta: array<f32>;
@group(0) @binding(2) var<storage, read_write> zetaNew: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> tempIn: array<f32>;
@group(0) @binding(6) var<storage, read> deepPsiIn: array<f32>;
@group(0) @binding(7) var<storage, read> windCurlField: array<f32>;

fn idx(i: u32, j: u32) -> u32 { return j * params.nx + i; }

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let j = id.y;
  let nx = params.nx;
  let ny = params.ny;
  if (i >= nx || j < 1u || j >= ny - 1u) { return; }
  let k = idx(i, j);
  if (mask[k] == 0u) { zetaNew[k] = 0.0; return; }

  // Periodic wrapping in x (longitude)
  let ip1 = select(i + 1u, 0u, i == nx - 1u);
  let im1 = select(i - 1u, nx - 1u, i == 0u);

  // Check cardinal neighbors are ocean
  let ke = idx(ip1, j); let kw = idx(im1, j);
  let kn = idx(i, j + 1u); let ks = idx(i, j - 1u);
  if (mask[ke] == 0u || mask[kw] == 0u || mask[kn] == 0u || mask[ks] == 0u) {
    zetaNew[k] = zeta[k] * 0.9;
    return;
  }
  // Check diagonal neighbors for Arakawa Jacobian
  let kne = idx(ip1, j + 1u); let knw = idx(im1, j + 1u);
  let kse = idx(ip1, j - 1u); let ksw = idx(im1, j - 1u);
  if (mask[kne] == 0u || mask[knw] == 0u || mask[kse] == 0u || mask[ksw] == 0u) {
    zetaNew[k] = zeta[k] * 0.95;
    return;
  }

  // Latitude for this cell — needed for metric correction
  let lat = -79.5 + f32(j) / f32(ny - 1u) * 159.0;
  let latRad = lat * 3.14159265 / 180.0;
  let cosLat = max(cos(latRad), 0.087); // clamp at ~5° to avoid singularity

  // Grid derivatives with cos(lat) metric correction
  let invDxRaw = 1.0 / params.dx;
  let invDx = invDxRaw / cosLat;  // zonal: physical = grid / cos(lat)
  let invDy = 1.0 / params.dy;     // meridional: unchanged
  let invDx2 = invDx * invDx;
  let invDy2 = invDy * invDy;

  // Arakawa Jacobian J(psi, zeta) with metric correction
  let mDx = params.dx * cosLat;  // physical dx
  let mDy = params.dy;
  let J1 = (psi[ke] - psi[kw]) * (zeta[kn] - zeta[ks])
         - (psi[kn] - psi[ks]) * (zeta[ke] - zeta[kw]);
  let J2 = psi[ke] * (zeta[kne] - zeta[kse])
         - psi[kw] * (zeta[knw] - zeta[ksw])
         - psi[kn] * (zeta[kne] - zeta[knw])
         + psi[ks] * (zeta[kse] - zeta[ksw]);
  let J3 = zeta[ke] * (psi[kne] - psi[kse])
         - zeta[kw] * (psi[knw] - psi[ksw])
         - zeta[kn] * (psi[kne] - psi[knw])
         + zeta[ks] * (psi[kse] - psi[ksw]);
  let jac = (J1 + J2 + J3) / (12.0 * mDx * mDy);

  // Beta term: varies with latitude (beta ~ cos(lat) in real ocean)
  let betaLocal = params.beta * cos(latRad);
  let betaV = betaLocal * (psi[ke] - psi[kw]) * 0.5 * invDx;

  // Wind forcing from pre-scaled field (observed NCEP or analytical fallback)
  let F = params.windStrength * windCurlField[k];

  // Friction
  let fric = -params.r * zeta[k];

  // Viscosity: A * laplacian(zeta) with metric correction
  let lapZeta = invDx2 * (zeta[ke] + zeta[kw] - 2.0 * zeta[k])
             + invDy2 * (zeta[kn] + zeta[ks] - 2.0 * zeta[k]);
  let visc = params.A * lapZeta;

  // Buoyancy coupling: density gradient from T AND S
  let N_off = params.nx * params.ny;
  let dRhodx = -params.alphaT * (tempIn[ke] - tempIn[kw]) + params.betaS * (tempIn[ke + N_off] - tempIn[kw + N_off]);
  let buoyancy = -dRhodx * 0.5 * invDx;

  // Interfacial coupling to deep layer
  let coupling = params.fCoupleS * (deepPsiIn[k] - psi[k]);

  zetaNew[k] = clamp(zeta[k] + params.dt * (-jac - betaV + F + fric + visc + buoyancy + coupling), -500.0, 500.0);
}
