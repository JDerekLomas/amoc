# Parameter Reference

## Our Simulation Defaults vs Published Ranges

| Parameter | Our Default | Published Range | Units | Notes |
|-----------|-------------|-----------------|-------|-------|
| **S_solar** | 100 | — | non-dim | ~20x physical for fast restoring |
| **A_olr** | 40 | 202-210 | W/m^2 (physical) | Our value is non-dim equivalent |
| **B_olr** | 2.0 | 1.9-2.2 | W/m^2/K (physical) | Budyko: 2.0, Sellers: 2.17 |
| **r_friction** | 0.04 | — | non-dim | Physical: r ~ 4e-4 m/s |
| **A_visc** | 5e-4 | — | non-dim | Physical: A_H ~ 2e4-1e5 m^2/s |
| **kappa_diff** | 2.5e-4 | — | non-dim | Physical: kappa_H ~ 1e3-1e4 m^2/s |
| **alpha_T** | 0.05 | — | non-dim | Buoyancy-vorticity coupling |
| **gamma_mix** | 0.01 | — | non-dim | Base vertical mixing |
| **gamma_deep_form** | 0.5 | — | non-dim | Deep water formation rate |
| **kappa_deep** | 2e-5 | — | non-dim | Deep horizontal diffusion |
| **F_couple_s** | 0.5 | — | non-dim | Interfacial coupling (surface) |
| **F_couple_d** | 0.0125 | — | non-dim | = (H_s/H_d) * F_couple_s |
| **r_deep** | 0.1 | — | non-dim | Deep bottom friction |
| **windStrength** | 1.0 | — | multiplier | 1.0 = standard pattern |

## What Real GCMs Use (Dimensional)

### Lateral Viscosity [source: MOM6, NEMO, MITgcm docs]
| Resolution | Laplacian A_H | Biharmonic A_4 | Smagorinsky C_smag |
|-----------|---------------|----------------|---------------------|
| 2 deg | 4e4 m^2/s | — | — |
| 1 deg | 2e4-1e5 m^2/s | 1e11-5e12 m^4/s | 0.06-0.2 |
| 1/4 deg | — | -1.5e11 m^4/s | 0.06-0.15 |

### Constraint: Munk layer must be resolved
```
delta_M = (A_H / beta)^(1/3) > dx
A_H > beta * dx^3
At 1 deg (dx ~ 1e5 m): A_H > 2e4 m^2/s
```

### Bottom Drag [source: OM4, NEMO]
| Type | Value | Timescale |
|------|-------|-----------|
| Linear r | 4e-4 m/s | ~115 days |
| Quadratic C_D | 0.001-0.003 | — |
| OM4 default | C_D = 0.003 | — |
| NEMO default | C_D = 0.001 | — |

### Vertical Mixing [source: KPP, Van Roekel 2018]
| Parameter | Value | Units |
|-----------|-------|-------|
| Background K_v (interior) | 1e-5 | m^2/s |
| Background A_v | 1e-4 | m^2/s |
| Convective K_v | 1-10 | m^2/s |
| KPP Ri_cr | 0.3 | — |

### GM/Redi Eddy Parameterization [source: MITgcm, Gent 2011]
| Parameter | Range | Default | Units |
|-----------|-------|---------|-------|
| kappa_GM | 200-1400 | 600-1000 | m^2/s |
| kappa_Redi | 400-2400 | 600-1000 | m^2/s |
| Slope limit | 0.002-0.01 | 0.005 | — |

### OLR Linearization [source: Budyko 1969, North 1975]
| Source | A (W/m^2) | B (W/m^2/K) |
|--------|-----------|-------------|
| Budyko (1969) | 210 | 2.0 |
| Sellers (1969) | 204 | 2.17 |
| North (1975) | 202 | 1.9 |
| Planck (no feedbacks) | — | 3.3 |

## Tuning Rules

1. **Never change S_solar and A_olr in the same direction** — that's compensating errors
2. **F_couple_d must track F_couple_s**: F_couple_d = (H_s/H_d) * F_couple_s = 0.025 * F_couple_s
3. **Max 3 parameters per iteration, max 30% step toward bound**
4. **Check equilibrium T after thermal param changes**: T_eq = (S*cos_z - A) / B

## Sources
- [MOM6 docs](https://mom6.readthedocs.io/)
- [NEMO lateral mixing](https://www.nemo-ocean.eu/doc/node63.html)
- [MITgcm GM/Redi](https://mitgcm.readthedocs.io/en/latest/phys_pkgs/gmredi.html)
- [Gent 2011 — GM hindsight](https://staff.cgd.ucar.edu/gent/gm20.pdf)
- [Megann 2021 — Viscosity space](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020MS002263)
