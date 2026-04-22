# Next Session Research Plan

## Quick Start
```bash
cd /Users/dereklomas/lukebarrington/amoc
python3 -m http.server 8780 &
open http://localhost:8780/simamoc/
```

## Links
- **Live:** https://amoc-sim.vercel.app/simamoc/
- **GitHub:** https://github.com/JDerekLomas/amoc

---

## Priority 1: AMOC Validation + Hosing Experiments (2-3 hrs)

### Goal: Demonstrate realistic AMOC tipping behavior

**1a. AMOC Diagnostic Improvement**
- Current AMOC metric is a point velocity difference at one latitude
- Replace with zonally-integrated meridional transport across Atlantic at multiple latitudes
- Display as a proper overturning cross-section (latitude vs depth, 2 layers)
- Add AMOC timeseries plot to the sidebar (running graph of strength over time)

**1b. Hosing Experiment Protocol**
- Design a systematic freshwater forcing ramp: 0 → 0.5 → 1.0 → 1.5 → 2.0 over time
- Record AMOC strength, SST, salinity at each level
- Find the tipping point (where AMOC collapses and doesn't recover)
- Compare to van Westen et al. 2024 qualitatively
- Test hysteresis: does AMOC recover when freshwater is removed?

**1c. FovS Early Warning Signal**
- Compute the freshwater transport by the AMOC at the southern boundary of the Atlantic (~34°S)
- FovS = ∫ v(S - S_ref) dx / ∫ v dx
- van Westen showed FovS < 0 → bistable regime (current real ocean is in this regime)
- Track this as a diagnostic during hosing experiments

### Validation Targets
- AMOC shows clear weakening with freshwater forcing
- Tipping occurs between fw=1.0 and fw=2.0
- Post-collapse: Europe cools, SH warms (qualitative match to van Westen)
- FovS goes more negative before collapse (early warning)

---

## Priority 2: Temperature Field Accuracy (2-3 hrs)

### Goal: RMSE < 3°C

**2a. Systematic Basin Analysis**
- Break RMSE into: Atlantic, Pacific, Indian, Southern Ocean, Arctic
- Identify which basin has the largest bias
- Current issue: NH 50°N too warm (+14°C in some runs), SH poles variable

**2b. Wind Stress Tuning**
- Current cos(3φ) with SH 2x boost may be too simple
- Try: `tau_x = -0.4*cos(2φ) + 0.25*sin(φ) + 0.1*sin(3φ)` (research-informed multi-term)
- Validate against ERA5 wind stress climatology
- Check Sverdrup transport integral matches theory (~30 Sv subtropical gyre)

**2c. Thermal Diffusion Optimization**
- Run Wiggum loop focused on kappa_diff and kappa_deep
- Current RMSE 3.8°C → target 3.0°C
- May need latitude-dependent diffusion (stronger at high latitudes)

**2d. Seasonal Cycle Validation**
- Does the sim produce the right amplitude of seasonal SST variation?
- Tropics: small (1-2°C), midlatitudes: large (5-10°C)
- Compare to NOAA monthly climatology

---

## Priority 3: Numerical Optimization (2-3 hrs)

### Goal: Smooth 60fps with salinity

**3a. Profile GPU Pipeline**
- Measure time per compute pass: vorticity, Poisson, tracer, deep
- Identify actual bottleneck (suspected: Poisson SOR at 25×2 dispatches per step)
- Consider: fewer steps/frame, more Poisson per step (batch red-black)

**3b. Multigrid Poisson Solver**
- V-cycle multigrid on GPU: ~10x convergence vs SOR
- Levels: 360×180 → 180×90 → 90×45 → 45×22
- Restrict, smooth, prolong at each level
- Would allow 5-10 multigrid iterations instead of 25 SOR → major speedup

**3c. Resolution Doubling (720×360)**
- 0.5° resolution resolves mesoscale eddies better
- 4x more cells → need GPU compute optimization first
- Gulf Stream separation should improve
- Consider: adaptive resolution (high in western boundary, low in interior)

**3d. Workgroup Size Optimization**
- Current: 8×8 = 64 threads per workgroup
- Try: 16×16 = 256 (better GPU occupancy on modern hardware)
- Profile both configurations

---

## Priority 4: UI/UX Polish (1-2 hrs)

### Goal: Publication/demo-ready interface

**4a. AMOC Timeseries Panel**
- Running graph of AMOC strength over last N years
- Shows tipping events in real-time
- Add to sidebar below the stats

**4b. Heat Transport Profile**
- Meridional heat transport (v×T) by latitude
- Replace or augment the existing radiative balance profile
- Show poleward transport peak at ~30°N

**4c. Scenario Improvements**
- "Melt Greenland" should ramp freshwater gradually (not instant jump)
- Add "Double CO2" scenario (increase radiative forcing to +4°C)
- Add "Last Glacial Maximum" (ice age with lower sea level, expanded ice)
- Add "Pangaea" (single continent, one ocean basin)

**4d. Information Panels**
- Clicking a cell shows: SST, salinity, density, depth, current speed/direction
- Hover tooltip with coordinates and values
- Color legend for current view mode

**4e. Mobile/Touch Support**
- Responsive layout for tablets
- Touch-based painting tools
- Pinch-to-zoom

---

## Priority 5: Observational Data Integration (2-3 hrs)

### Goal: Initialize and validate with real data

**5a. WOA23 Salinity Data**
- Fetch surface and 1000m salinity from WOA23
- Use for salinity initialization (replace latitude formula)
- Better initial density field → faster AMOC spinup

**5b. ERA5 Wind Stress**
- Download monthly ERA5 tau_x, tau_y at 1° resolution
- Replace analytical wind with observed wind stress curl
- Seasonal wind variation (monsoons, etc.)

**5c. AMOC Comparison**
- Compare our AMOC diagnostic to RAPID array timeseries
- Need to calibrate non-dimensional → Sverdrup conversion
- Plot sim vs observed on same axes

**5d. SST Anomaly Mode**
- View that shows difference from NOAA climatology
- Red = too warm, blue = too cold
- Immediately shows where the model is wrong

---

## Priority 6: Scientific Experiments (ongoing)

### Goal: Answer real questions with the sim

**6a. Drake Passage Opening**
- How does ACC formation affect global temperature?
- Compare to paleoclimate proxy data
- Time evolution of temperature field after opening

**6b. Panama Closure**
- Effect on Gulf Stream and AMOC strength
- Compare 3 Ma paleoclimate record
- Atlantic-Pacific salinity contrast

**6c. AMOC Collapse Cascade**
- Full hosing experiment with monitoring
- What happens to European temperatures?
- Does the SH warm (bipolar seesaw)?
- How fast does it collapse? Can it recover?

**6d. Sensitivity Analysis**
- Systematic: vary each parameter ±30%, measure RMSE and AMOC response
- Build a Jacobian matrix of parameter sensitivities
- Identify which parameters matter most for which diagnostics

---

## Stretch Goals

- [ ] Couple with simple atmospheric energy balance model
- [ ] Sea ice model (thermodynamic growth/melt + transport)
- [ ] River runoff (Amazon, Congo, Mississippi freshwater sources)
- [ ] Train neural network emulator for instant scenario exploration
- [ ] Differentiable physics for gradient-based optimization
- [ ] Publish methodology paper on AI-assisted ocean model development
- [ ] Deploy as educational tool (classroom, museum kiosk)
