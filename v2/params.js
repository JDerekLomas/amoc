/**
 * SimAMOC v2 — Physics Parameters
 *
 * All tunable constants in one place. Geometry and forcing are in state.
 * Parameters are organized by component.
 */

export function defaultParams() {
  return {
    // --- Radiation ---
    S_solar: 7.0,         // Solar amplitude (dimensionless, ~240 W/m² peak)
    A_olr: 2.0,           // OLR constant
    B_olr: 0.1,           // OLR feedback (fast restoring for interactive use)
    // Equilibrium: T_eq = (S_solar * cos(lat) - A_olr) / B_olr

    // --- Ocean dynamics ---
    beta: 1.4e-3,         // Planetary vorticity gradient (d f/dy, dimensionless)
    r_drag: 0.04,         // Bottom friction coefficient
    A_visc: 2.0e-4,       // Lateral viscosity
    kappa_diff: 2.5e-4,   // Horizontal thermal diffusion
    kappa_sal: 1.5e-4,    // Horizontal salinity diffusion
    omega_sor: 1.85,      // SOR relaxation parameter
    sor_iters: 30,        // Poisson solver iterations per step

    // --- Deep ocean ---
    r_deep: 0.02,         // Deep layer drag
    kappa_deep: 3e-5,     // Deep thermal diffusion
    kappa_deep_sal: 2e-5, // Deep salinity diffusion
    h_surface: 200,       // Surface layer depth (m)
    h_deep: 3800,         // Deep layer depth (m)
    f_couple_s: 0.005,    // Surface→deep vorticity coupling
    f_couple_d: 0.001,    // Deep→surface vorticity coupling

    // --- Thermohaline ---
    alpha_T: 2e-4,        // Thermal expansion (1/°C)
    beta_S: 7.5e-4,       // Haline contraction (1/PSU)
    gamma_mix: 1e-4,      // Vertical mixing rate
    gamma_deep_form: 0.5, // Deep water formation strength
    sal_restoring: 5e-5,  // Salinity restoring rate to observations

    // --- Atmosphere ---
    land_heat_k: 0.02,    // Land↔atmosphere heat exchange
    moisture_e0: 3.75e-3, // Clausius-Clapeyron reference (kg/kg)
    moisture_Lv: 2.5e6,   // Latent heat of vaporization (J/kg)
    moisture_scale: 0.067,// CC exponent scaling (1/°C)
    precip_threshold: 0.8,// RH threshold for precipitation
    precip_rate: 0.01,    // Precipitation efficiency

    // --- Ice ---
    ice_freeze: -1.8,     // Freezing temperature (°C)
    ice_melt: 0.0,        // Melting temperature (°C)
    ice_albedo: 0.7,      // Ice albedo
    ocean_albedo: 0.06,   // Open ocean albedo

    // --- Wind ---
    wind_strength: 1.0,   // Wind stress multiplier

    // --- Timestepping ---
    dt: 0.004,            // Timestep (dimensionless)
    T_year: 2 * Math.PI,  // One year in sim time (for seasonal cycle)

    // --- Freshwater (P-E) ---
    pe_sal_flux: 0.5,     // P-E to salinity conversion strength
  };
}
