# Known Model Biases

Common biases in ocean/climate models. Useful for interpreting our sim's errors — some are structural and can't be fixed by tuning.

## Double ITCZ
- **What**: Spurious second ITCZ south of equator in most coupled models
- **Cause**: Southern Ocean cloud biases → too much solar → SH warms → ITCZ drawn south
- **Our sim**: Not directly applicable (no atmospheric coupling) but could manifest as hemispheric SST bias
- [Hwang & Frierson 2013 (PNAS)](https://www.pnas.org/doi/10.1073/pnas.1213302110)

## Equatorial Cold Tongue
- **What**: Cold tongue extends too far west, 1-3K too cold
- **Cause**: Overly strong equatorial upwelling, excessive thermocline diffusion, Bjerknes feedback
- **Our sim**: Possible if wind forcing too strong at equator or kappa_diff too high
- [Li & Xie 2014](https://journals.ametsoc.org/view/journals/clim/27/4/jcli-d-13-00337.1.xml)

## Gulf Stream Separation
- **What**: Gulf Stream overshoots Cape Hatteras at coarse resolution
- **Cause**: Can't resolve inertial jet at 1 deg. Needs ~1/10 deg to fix
- **Our sim**: EXPECTED at 1 deg — don't try to fix with parameters. Structural limitation.

## AMOC Strength Spread
- **What**: CMIP6 models range 8-25 Sv (factor 3x)
- **Cause**: Freshwater transport biases, Nordic overflow representation, resolution effects
- **Our sim**: AMOC in non-dimensional units, can't directly compare to Sv
- [Nayak et al. 2024](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024GL109055)

## Southern Ocean Warming
- **What**: Models warm Southern Ocean too slowly vs observations
- **Cause**: Cloud biases, wind stress biases, sea ice errors
- **Our sim**: May see SH temperature biases related to simplified wind pattern

## Biases We Should Accept (Structural Limitations)

These cannot be fixed by parameter tuning in our 2-layer model:
1. **No mesoscale eddies** — we parameterize; real models at 1 deg have same issue
2. **No salinity** — only temperature-driven density; misses haline component of AMOC
3. **Fixed wind pattern** — no atmospheric feedback, ENSO, NAO
4. **No realistic mixed layer** — 2 layers only, can't resolve seasonal thermocline
5. **No overflow parameterization** — Nordic Sea dense water cascading absent
