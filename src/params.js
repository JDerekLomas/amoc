// Physical constants and tunable parameters
// All SI units unless noted

export const EARTH = {
  radius: 6.371e6,       // m
  omega: 7.292e-5,       // rad/s
  g: 9.81,               // m/s²
};

export const OCEAN = {
  rho: 1025,             // kg/m³
  cp: 4000,              // J/(kg·K)
  mixedLayerDepth: 50,   // m (uniform for now)
  viscosity: 5e4,        // m²/s (horizontal eddy viscosity, Munk-scale)
  diffusivity: 2e3,      // m²/s (horizontal thermal diffusivity)
  dragCoeff: 1.2e-5,      // 1/s (linear bottom drag — tuned for ~0.3 m/s max)
};

export const ATMOSPHERE = {
  solarConstant: 1361,   // W/m²
  albedoOcean: 0.06,
  albedoLand: 0.30,
  albedoIce: 0.60,
  // Linearized OLR: OLR = A + B * T(°C)
  olrA: 210,             // W/m²
  olrB: 2.0,             // W/(m²·°C)
  // Cloud effects
  cloudAlbedoEffect: 0.25,   // additional albedo from clouds
  cloudGreenhouseEffect: 30, // W/m² OLR reduction from clouds
};

export const SIMULATION = {
  nx: 360,
  ny: 160,
  latMin: -80,
  latMax: 80,
  dt: 1800,              // s (30 min ocean timestep)
  stepsPerFrame: 1,      // 1 step per render frame (SOR is expensive)
  sorOmega: 1.7,         // SOR relaxation parameter
  sorIterations: 30,     // Poisson solver iterations per step (uses prev psi as initial guess)
};
