# Earth Observation Data for SimAMOC

Real-world satellite data from NASA Earth Observations (NEO) used to drive and validate the ocean circulation model.

## Data Sources

All data downloaded from https://neo.gsfc.nasa.gov/ on 2026-04-24.
Grid: 360x180 (1-degree), -90 to 90 latitude, -180 to 180 longitude.
Remapped to model grid: 360x160, -79.5 to 79.5 latitude.

| Dataset | NEO ID | Scene ID | Sensor | Period | Units |
|---------|--------|----------|--------|--------|-------|
| Precipitation | GPM_3IMERGM | 2046974 | IMERG (GPM) | March 2026 | mm/month |
| Surface Albedo | MCD43C3_M_BSA | 2047279 | MODIS Terra+Aqua | March 2026 | 0-1 |
| Cloud Fraction | MODAL2_M_CLD_FR | 2046969 | MODIS Terra | March 2026 | 0-1 |
| Vegetation (NDVI) | MOD_NDVI_M | 2047277 | MODIS Terra | March 2026 | -0.1 to 0.9 |
| Land Surface Temp | MOD_LSTD_M | 2047022 | MODIS Terra | March 2026 | degrees C |
| Snow Cover | MOD10C1_M_SNOW | 2046871 | MODIS Terra | March 2026 | percent |

## Files

### Raw CSV data (from NEO, 360x180)
- `precipitation.csv` - Monthly rainfall (mm/month)
- `albedo.csv` - White-sky broadband albedo (land only)
- `cloud_fraction.csv` - Cloud fraction (land + ocean)
- `ndvi.csv` - Normalized Difference Vegetation Index (land only)
- `land_surface_temp.csv` - Daytime land surface temperature (land only)
- `snow_cover.csv` - Snow cover percentage (land only)

### Reference images (720x360 PNG, for visual inspection)
- `images/precipitation.png`
- `images/albedo.png`
- `images/cloud_fraction.png`
- `images/ndvi.png`
- `images/land_surface_temp.png`
- `images/snow_cover.png`

### Model JSON files (360x160, in project root)
- `precipitation_1deg.json` - Annual precipitation estimate (mm/yr)
- `albedo_1deg.json` - Surface albedo (0-1, ocean filled with 0.06)
- `cloud_fraction_1deg.json` - Observed cloud fraction (for validation)

## How to update

To download fresh data, change the scene ID in the curl command:
```
curl -sS "https://neo.gsfc.nasa.gov/servlet/RenderData?si=SCENE_ID&cs=gs&format=CSV&width=360&height=180" -o dataset.csv
```

Scene IDs change monthly. Find current IDs at:
https://neo.gsfc.nasa.gov/view.php?datasetId=DATASET_ID

## Fill values
- CSV: `99999.0` = no data (ocean for land-only datasets, polar gaps)
- JSON: `null` replaced with sensible defaults (0.06 for ocean albedo, 0 for precipitation)
