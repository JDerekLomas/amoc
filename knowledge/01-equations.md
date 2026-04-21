# Physics Equations

## Barotropic Vorticity Equation (Surface Layer)

```
dq/dt + J(psi, q) = curl(tau) - r*zeta + A*laplacian(zeta) - alpha_T*dT/dx + F1*(psi_d - psi)
```

| Term | Meaning | Controls |
|------|---------|----------|
| `J(psi, q)` | Arakawa Jacobian advection | Conserves energy + enstrophy |
| `curl(tau)` | Wind stress curl | Drives gyres (Sverdrup balance) |
| `-r*zeta` | Bottom friction | Stommel boundary layer width ~ r/beta |
| `A*laplacian(zeta)` | Lateral viscosity | Munk boundary layer width ~ (A/beta)^(1/3) |
| `-alpha_T*dT/dx` | Buoyancy coupling | Links temperature to circulation |
| `F1*(psi_d - psi)` | Interfacial coupling | Surface feels deep layer |

where `q = zeta + beta*y` (potential vorticity = relative + planetary)

## Deep Layer Vorticity

```
dq_d/dt + J(psi_d, q_d) = -r_d*zeta_d + A*laplacian(zeta_d) + F2*(psi - psi_d)
```

Same structure but no wind forcing, different friction. `F2 = (H_s/H_d) * F1` by momentum conservation.

## Surface Temperature

```
dT/dt = S*cos(theta_z)*alpha_ice - (A_olr + B_olr*T) + kappa*laplacian(T) - gamma*(Ts-Td)/H + Q_land
```

| Term | Meaning |
|------|---------|
| `S*cos(theta_z)*alpha_ice` | Solar heating (seasonal, latitude-dependent, ice-albedo) |
| `-(A_olr + B_olr*T)` | Outgoing longwave radiation (linearized Stefan-Boltzmann) |
| `kappa*laplacian(T)` | Thermal diffusion |
| `-gamma*(Ts-Td)/H` | Vertical exchange with deep layer |
| `Q_land` | Land-ocean heat flux at coastal cells |

### Equilibrium Temperature

At equilibrium (`dT/dt = 0`, ignoring diffusion and advection):
```
T_eq = (S * avg_cos_z - A_olr) / B_olr
```

| Latitude | avg_cos_z | T_eq (S=100, A=40, B=2) |
|----------|-----------|------------------------|
| Equator | ~0.9 | 25 C |
| 30 deg | ~0.7 | 15 C |
| 60 deg | ~0.35 | -2.5 C |
| Pole | ~0.2 | -10 C |

## Deep Temperature

```
dT_d/dt = gamma*(Ts - Td)/H_d + kappa_d*laplacian(T_d)
```

Deep layer has 40x more thermal inertia (H_d = 4000m vs H_s = 100m), responds slowly.

## Key Physical Relationships

- **Western intensification**: exists because beta effect + friction/viscosity → asymmetric boundary layers
- **AMOC**: driven by alpha_T coupling — temperature gradients drive vorticity, creating overturning
- **Ice-albedo feedback**: alpha_ice reduces solar at |lat| > 40 deg when T < 0 C — positive feedback
- **Seasonal cycle**: solar declination varies with `T_YEAR = 10` simulation time units per year

## Sources
- Pedlosky, *Geophysical Fluid Dynamics* (2nd ed.)
- Vallis, *Atmospheric and Oceanic Fluid Dynamics*
- [Barotropic dynamics — GFDL](https://www.gfdl.noaa.gov/wp-content/uploads/files/user_files/stg/ch_6.pdf)
