# Diagnostics: What to Measure

## Tier 1: Conservation (must pass)

| Check | How | Threshold | Why |
|-------|-----|-----------|-----|
| Temperature range | min/max of all ocean cells | [-15, 40] C | Physical ocean range |
| Equator-pole gradient | tropical T - polar T | 10-50 C | Must exist |
| Hemispheric symmetry | |NH mean - SH mean| | < 8 C | Roughly symmetric forcing |
| Deep < surface | count(deep > surface + 2) | 0 bands | Stable stratification |
| AMOC positive | amocStrength variable | > 1e-5 (non-dim) | Overturning exists |

## Tier 2: Structure (35% of score)

| Check | How | Threshold | Why |
|-------|-----|-----------|-----|
| Western intensification | max speed Gulf Stream region / NA interior | ratio > 1.5x | Stommel/Munk dynamics |
| Subtropical gyre | psi range in NA subtropics (20-40N, 70-20W) | range > 0.005 | Wind-driven Sverdrup |
| ACC eastward flow | mean zonal velocity at 58S | |u| > 0.01 | Drake Passage open |
| Deep water formation | polar deep T < tropical deep T | polar < tropical | Thermohaline sinking |
| Poleward heat transport | mean v*T at 30N | > 0 | Fundamental energy balance |
| Atlantic > Pacific at 40N | Atlantic mean T > Pacific mean T | Atlantic warmer | AMOC signature |

### Grid Geography (LON0=-180, LON1=180, LAT0=-80, LAT1=80)
```
i = (lon + 180) / 360 * 359    j = (lat + 80) / 160 * 179

Gulf Stream region:     lon -80 to -60, lat 25-50  → i=100-120, j=118-146
NA interior:            lon -50 to -10, lat 25-50  → i=130-170, j=118-146
NA subtropical gyre:    lon -70 to -20, lat 20-40  → i=110-160, j=112-135
ACC:                    lat -58                     → j=25
Atlantic at 40N:        lon -70 to -10, lat 40     → i=110-170, j=135
Pacific at 40N:         lon 140 to -140 (wrap)     → i=320-360,0-40, j=135
```

## Tier 3: Sensitivity (final only)

| Test | Perturbation | Expected | Why |
|------|-------------|----------|-----|
| Freshwater | freshwaterForcing = 2.0 | AMOC weakens | Stommel bifurcation |
| Ice age | globalTempOffset = -8 | Global cooling | Radiative balance |

## Tier 4: Quantitative (35% of score)

| Metric | Target | Source |
|--------|--------|--------|
| Zonal mean SST by latitude | Within ~5 C of NOAA | NOAA OI SST v2 |
| Global mean SST | ~14.2 C | NOAA OI SST v2 |
| Deep temp range | -0.3 to 5.3 C | WOA23 at 1000m |

### NOAA Reference Zonal Means (annual)
```
-70: -1.5   -60: 1.3    -50: 6.8    -40: 15.1
-30: 20.9   -20: 24.7   -10: 26.9     0: 27.6
 10: 27.8    20: 26.1    30: 22.2    40: 15.6
 50:  9.0    60:  5.2    70:  1.0
```

## AMOC Units

The `amocStrength` variable is in non-dimensional psi-gradient units, NOT Sverdrups.
- Typical range: 1e-5 to 1e-2
- "WEAK" displayed when |amocStrength| < 0.005
- "STRONG" displayed when |amocStrength| > 0.05
- Real AMOC: 16.9 Sv mean at 26.5N (RAPID array, 2004-2020)

## Composite Scoring

```
T1 conservation: GATE (must pass, else score capped at 20%)
T2 structure:    35% (fraction of checks passing)
T4 quantitative: 35% (1 - RMSE/15)
AMOC in range:   15% (0.0005 < amocVal < 0.5)
T1 bonus:        15% (passes = 1.0)
T3 sensitivity:  20% of final score (run only on best params)
```

## Sources
- [RAPID AMOC array](https://rapid.ac.uk/)
- [NOAA OI SST v2](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html)
- [WOA23](https://www.ncei.noaa.gov/products/world-ocean-atlas)
