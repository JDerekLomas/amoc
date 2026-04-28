// {{PARAMS}} — replaced at load time with shared Params struct

@group(0) @binding(0) var<storage, read> deepPsi: array<f32>;
@group(0) @binding(1) var<storage, read> deepZeta: array<f32>;
@group(0) @binding(2) var<storage, read_write> deepZetaNew: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read> surfacePsi: array<f32>;
@group(0) @binding(6) var<storage, read> deepTempIn: array<f32>;

fn idx(i: u32, j: u32) -> u32 { return j * params.nx + i; }

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  let j = id.y;
  let nx = params.nx;
  let ny = params.ny;
  if (i >= nx || j < 1u || j >= ny - 1u) { return; }
  let k = idx(i, j);
  if (mask[k] == 0u) { deepZetaNew[k] = 0.0; return; }

  let ip1 = select(i + 1u, 0u, i == nx - 1u);
  let im1 = select(i - 1u, nx - 1u, i == 0u);
  let ke = idx(ip1, j); let kw = idx(im1, j);
  let kn = idx(i, j + 1u); let ks = idx(i, j - 1u);

  // Coastal damping
  if (mask[ke] == 0u || mask[kw] == 0u || mask[kn] == 0u || mask[ks] == 0u) {
    deepZetaNew[k] = deepZeta[k] * 0.9;
    return;
  }

  // Latitude and metric correction
  let lat = -79.5 + f32(j) / f32(ny - 1u) * 159.0;
  let latRad = lat * 3.14159265 / 180.0;
  let cosLat = max(cos(latRad), 0.087);
  let invDx = 1.0 / (params.dx * cosLat);
  let invDy = 1.0 / params.dy;
  let invDx2 = invDx * invDx;
  let invDy2 = invDy * invDy;

  // Simplified Jacobian J(deepPsi, deepZeta)
  let dPdx = (deepPsi[ke] - deepPsi[kw]) * 0.5 * invDx;
  let dPdy = (deepPsi[kn] - deepPsi[ks]) * 0.5 * invDy;
  let dZdx = (deepZeta[ke] - deepZeta[kw]) * 0.5 * invDx;
  let dZdy = (deepZeta[kn] - deepZeta[ks]) * 0.5 * invDy;
  let jac = dPdx * dZdy - dPdy * dZdx;

  // Beta term: varies with latitude
  let betaV = params.beta * cos(latRad) * (deepPsi[ke] - deepPsi[kw]) * 0.5 * invDx;

  // Bottom friction (stronger than surface)
  let fric = -params.rDeep * deepZeta[k];

  // Viscosity
  let lapZeta = invDx2 * (deepZeta[ke] + deepZeta[kw] - 2.0 * deepZeta[k])
             + invDy2 * (deepZeta[kn] + deepZeta[ks] - 2.0 * deepZeta[k]);
  let visc = params.A * lapZeta;

  // Interfacial coupling: deep layer pulled toward surface flow
  let coupling = params.fCoupleD * (surfacePsi[k] - deepPsi[k]);

  // Deep buoyancy forcing: density-driven overturning from deep temperature gradients
  let N_doff = params.nx * params.ny;
  let dRhodxDeep = -params.alphaT * (deepTempIn[ke] - deepTempIn[kw]) + params.betaS * (deepTempIn[ke + N_doff] - deepTempIn[kw + N_doff]);
  let deepBuoyancy = dRhodxDeep * 0.5 * invDx;

  // Meridional overturning: density gradient drives deep equatorward flow
  let deepTN = select(deepTempIn[k], deepTempIn[kn], mask[kn] != 0u);
  let deepTS = select(deepTempIn[k], deepTempIn[ks], mask[ks] != 0u);
  let dTdyDeep = (deepTN - deepTS) * 0.5 * invDy;
  let motTendency = 0.05 * dTdyDeep;

  deepZetaNew[k] = clamp(deepZeta[k] + params.dt * (-jac - betaV + fric + visc + coupling + deepBuoyancy + motTendency), -500.0, 500.0);
}
