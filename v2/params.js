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
    beta: 1.0,            // Planetary vorticity gradient (normalized, from v4)
    r_drag: 0.04,         // Bottom friction (Rayleigh)
    A_visc: 2.0e-4,       // Lateral viscosity
    kappa_diff: 2.5e-4,   // Horizontal thermal diffusion
    kappa_sal: 2.5e-4,    // Horizontal salinity diffusion (match v4)
    omega_sor: 1.97,      // SOR relaxation (optimized for this grid)
    sor_iters: 60,        // More iters at coarse resolution

    // --- Deep ocean ---
    r_deep: 0.1,          // Deep layer drag (from v4, higher than surface)
    kappa_deep: 2e-5,     // Deep thermal diffusion (from v4)
    kappa_deep_sal: 2e-5, // Deep salinity diffusion
    h_surface: 100,       // Surface layer depth (m, from v4)
    h_deep: 3900,         // Deep layer depth (m)
    f_couple_s: 0.5,      // Surface→deep coupling (from v4)
    f_couple_d: 0.0125,   // Deep→surface = (H_surf/H_deep) * f_couple_s

    // --- Thermohaline ---
    alpha_T: 0.15,        // Thermal expansion (from v4, tuned for density ratio)
    beta_S: 0.55,         // Haline contraction (α/β ≈ 0.27, matches seawater)
    gamma_mix: 0.001,     // Vertical mixing rate (from v4)
    gamma_deep_form: 0.05,// Deep water formation at high lat (from v4)
    sal_restoring: 0.005, // Salinity restoring rate (from v4)

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
    dt: 2.0e-4,           // Timestep (from v4, CFL: |u|*dt/dx < 1)
    T_year: 2 * Math.PI,  // One year in sim time (for seasonal cycle)

    // --- Freshwater (P-E) ---
    pe_sal_flux: 0.5,     // P-E to salinity conversion strength
  };
}
