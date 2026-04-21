# Observational Reference Data

## Datasets We Use

### NOAA OI SST v2 (1991-2020 annual mean)
- **File**: `sst_global_1deg.json` (key: `sst`, 360x160 array)
- **Source**: [NOAA PSL](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html)
- **Resolution**: 1 deg, lat -79.5 to +79.5
- **Global mean**: 14.2 C
- **Reliability by latitude**:
  - Tropics/midlatitudes: HIGH (dense ship + Argo + satellite coverage)
  - High latitudes (>60): MEDIUM (sparse, sea ice contamination)
  - Polar (>70): LOW (seasonal ice, interpolation-heavy)

### WOA23 (annual mean, 1000m depth)
- **File**: `deep_temp_1deg.json` (key: `temp`, 360x160 array)
- **Source**: [NCEI WOA](https://www.ncei.noaa.gov/products/world-ocean-atlas)
- **Range**: -0.3 C (70N) to 5.3 C (30N)
- **Reliability**: Good in well-sampled basins, poor in deep Southern Ocean

## Datasets We Could Use

### RAPID AMOC Array
- **Location**: 26.5 N across Atlantic
- **Period**: April 2004 - present
- **Mean**: 16.9 Sv, range 13-20 Sv
- **URL**: [rapid.ac.uk](https://rapid.ac.uk/)
- **Note**: Our AMOC is non-dimensional — can't directly compare to Sv

### ERA5 Reanalysis (atmospheric forcing)
- Wind stress, heat flux, P-E for driving ocean models
- **URL**: [CDS](https://cds.climate.copernicus.eu/)

### ORAS5 (ocean reanalysis)
- 3D T, S, currents from NEMO + data assimilation
- **URL**: [CDS](https://cds.climate.copernicus.eu/datasets/reanalysis-oras5)

### Argo Program
- ~4000 autonomous profiling floats
- T/S profiles 0-2000m, global coverage since ~2005
- [argo.ucsd.edu](https://argo.ucsd.edu/)

### ECCO v4 (state estimate)
- MITgcm adjoint-optimized, dynamically consistent
- No data insertion — adjusts forcing to fit observations
- [ecco-group.org](https://ecco-group.org/)

## Data Limitations to Remember

1. **Annual mean hides seasonal cycle** — polar SST is bimodal (ice/open water)
2. **Sparse deep data** — WOA at 1000m is interpolation-heavy in Southern Ocean
3. **SST ≠ bulk temperature** — NOAA SST is skin/near-surface, not mixed-layer average
4. **Our sim has no salinity** — can't validate against T-S relationships
5. **Observational period (1991-2020) includes warming trend** — our sim targets equilibrium
