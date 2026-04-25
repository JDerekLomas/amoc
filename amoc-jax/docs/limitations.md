# Limitations

Companion to `docs/physics.md` and `docs/roadmap.md`. The physics doc says
*what* we compute. This doc says *what we're knowingly leaving out*, and
flags how confident we are in the regimes where the simplifications are OK.

The point isn't to apologize. The point is: when a user turns a knob and
something looks weird, they should be able to find out which simplification
is biting. And when we cite this simulator against literature, we should
know which results to trust and which to flag.

Severity ratings (rough): **A** = first-order, can change qualitative
behavior. **B** = quantitative effect ~10–30%. **C** = small but worth
knowing.

---

## Layer 1 — model formulation

### A1. Streamfunction-vorticity instead of primitive equations

We solve `dζ/dt = -J(ψ,ζ) - β∂_λψ + curl(τ) - rζ + A∇²ζ`. Real ocean models
(MOM6, NEMO, MITgcm) solve momentum equations directly with a free surface.

- *Effect:* gravity waves filtered → no surface tides, no Kelvin waves on
  coastlines, no fast equatorial waves. The slow vorticity dynamics is
  unchanged.
- *Where it bites:* equatorial dynamics (where vorticity-only formulation
  is known to perform poorly — cos(φ) → 0 makes the metric ill-conditioned).
  Look at the equator with skepticism.
- *Defensible because:* this is exactly Vallis Ch. 14, Pedlosky §3 — the
  standard pedagogical formulation for gyres.

### A2. Two layers instead of continuous stratification

`ψ_s, ψ_d` for surface (~100 m) and deep (~3900 m). Real ocean is
continuously stratified with thermocline structure.

- *Effect:* first baroclinic mode is captured, higher modes are missed.
  Mode water, intermediate water, AABW vs NADW separation — all collapsed
  into "deep."
- *Where it bites:* anything that depends on the depth structure of MOC
  (its return path, its ventilation timescale). The 26.5°N AMOC strength
  is approximately right; the depth profile is not.
- *Roadmap:* could move to N layers; v3+ work.

### B1. Grid Laplacian Poisson instead of spherical Laplacian

The bare 5-point stencil sets `ζ = ∇²_grid ψ`, with cos(φ) corrections in
the dynamics operators (Jacobian, β-term as derived). This is internally
consistent but it is *not* the textbook spherical operator.

- *Effect:* the "ψ" we plot is not exactly the spherical streamfunction.
  Magnitudes are off by an O(1) factor that we absorb into tuned
  parameters. Sverdrup transport in physical Sv requires a calibration
  step.
- *Where it bites:* anywhere we want to put numbers in physical units.
  Currently parameters are dimensionless and tuned for stability; v1c will
  need a calibration pass.
- *Roadmap:* `docs/physics.md` §3a flags this; defer to v1b/c choice.

### A3. Linear equation of state

`ρ = ρ_0 (1 − α_T (T − T_0) + α_S (S − S_0))`. Real seawater has cabbeling
(mixing two equal-density waters can produce denser water) and
thermobaricity (compressibility depends on T).

- *Effect:* in the real Arctic, cabbeling + thermobaricity contribute to
  deep convection in ways linear EOS misses. Klocker & McDougall (2010)
  estimate ~10% AMOC effect.
- *Roadmap:* swap to nonlinear (Wright EOS or polynomial) in v1c.

---

## Layer 2 — closures (the parameters that hide complexity)

### A4. Laplacian viscosity instead of resolved mesoscale eddies

`A∇²ζ` damps small scales. Real mesoscale eddies (~50 km) carry most of the
mid-latitude heat poleward. Our 256×128 grid is ~150 km/cell, far from
eddy-resolving.

- *Effect:* poleward heat transport is too low; gyre boundaries are
  smoother than reality.
- *Roadmap:* eddy-permitting (~0.1°, 3600×1800) is computationally heavy
  but the data fetcher already supports `--resolution 3600x1800`.

### A5. Linear bottom drag instead of energetically-consistent friction

`-rζ` represents bottom drag and unresolved processes. Real ocean has
quadratic drag, topographic Reynolds stresses, internal-wave breaking.

- *Effect:* `r` is a tuning knob, not a measurement. Setting it 2× higher
  or lower doesn't break the model — that's a sign it's not physically
  constrained.

### B2. No tides, no internal-wave-driven mixing

Tidal dissipation contributes ~1 TW globally to deep ocean mixing (Munk &
Wunsch 1998 estimate). Major modern parameterizations (St-Laurent &
Garrett 2002, Polzin 2009) put this geographically — Brazil Basin, central
Pacific.

- *Effect:* our diffusion is uniform; real diffusion has hot spots over
  rough topography. Bryan scaling Ψ ∝ κ^(2/3) means MOC strength depends
  on this.
- *Roadmap:* later. v1d works at the level of "is there a bifurcation,"
  not "what's the exact transport in Sv."

### B3. Surface forcing is annual mean

ERA5 wind, NOAA SST — all annual climatology. Real ocean responds to
seasonal cycle, ENSO, NAO. The data files include `*_monthly.json` 12-month
climatologies for v1c.

- *Effect:* ITCZ migration, monsoon-driven Arabian Sea overturning, winter
  deep convection — all averaged out.

---

## Layer 3 — boundary processes

### A6. No sea ice mechanics or brine rejection

Sea ice has mechanical strength, drift, and — critically for AMOC — *brine
rejection*: when ice forms, salt is left behind in the underlying water,
making it dense. Brine-driven dense water formation is a major source
term for North Atlantic Deep Water.

- *Effect:* deep convection in the model is purely thermally driven. In
  reality it's thermo-haline.
- *Roadmap:* v3+. Need ice mass balance + a brine source term in S.

### B4. No overflows

Denmark Strait, Faroe Bank Channel are O(10 km)-wide passages where dense
water cascades from the Nordic Seas to the Atlantic abyss. At our 0.35°
grid these channels are 1-2 cells wide and the cascading dynamics is wrong.
Coarse-resolution real models add explicit overflow parameterizations
(Beckmann-Döscher, Campin-Goosse).

- *Effect:* downstream water mass properties (NADW T/S) are off.
- *Roadmap:* eventual.

### B5. Static bathymetry

Bathymetry is loaded once, never modified. Sea-level changes, sediment
transport, isostatic rebound — all ignored. Fine for thousand-year
simulations; matters for paleo (LGM, Eocene).

### C1. No biogeochemistry, no carbon cycle

Pure physical ocean. No carbon dissolution, no biological pump, no
acidification. Climate-feedback-relevant but well beyond v1.

---

## Layer 4 — what the big models also miss

This is where we line up with CMIP6 / CESM / MOM6, not against them.

### A7. Surface salinity restoring damps the salt-advection feedback

Most CMIP6 ocean models restore SSS strongly to climatology to control
drift. This is a known issue (Liu, Hu et al. 2017 *Sci Adv*; Brunnabend &
Dijkstra 2017): it suppresses the very Stommel feedback we're trying to
capture. Models that restore less show more AMOC variability.

We *don't* restore S in v1d (that's the whole hosing experiment). So
ironically our toy model is more honest about this feedback than many
CMIP6 runs.

### A8. Greenland melt is under-prescribed

CMIP6 melt scenarios (~700 Gt/yr by 2100) are derived from coupled ice
models that under-predict observed loss. With realistic melt, AMOC
weakening accelerates significantly (Bakker et al. 2025 *Nat Climate
Change*).

For v1d this is a freedom: we set hosing flux directly, can run any
scenario.

### A9. Vertical mixing is the largest source of CMIP6 spread

Bryan (1987): Ψ ∝ κ^(2/3). Different mixing schemes (KPP, TKE, GLS) give
different mean AMOC. There's no convergence in the literature on the
correct value, only on the scaling.

### A10. Tipping cascades are not captured

AMOC ↔ Amazon dieback ↔ Greenland melt ↔ permafrost methane are coupled
through atmospheric circulation. Wunderling et al. 2024 (*Nat Climate
Change*) catalogues this gap. CMIP6 typically runs components
independently or coupled only through global temperature.

This is a place we could *add value* — wire in toy models for cascade
elements and run experiments.

### B6. Atmosphere-ocean feedback under hosing is uncertain

When AMOC weakens, NH cools, ITCZ shifts south, winds change, evaporation
patterns change. The sign of these atmospheric feedbacks (do they damp or
amplify AMOC weakening?) varies between coupled models.

For v1d we will prescribe atmospheric forcing — a deliberate simplification
since we are not yet coupled.

---

## Current honest status (2026-04-25)

Where we are vs. where we'd like to be:

| Component | Current state | Honest assessment |
|---|---|---|
| Grid, FFT Poisson, Arakawa | Working, tested | Solid. Same kernels real models use. |
| Wind-driven gyres (v1a) | Producing patterns | **Patterns are visible but parameters are tuned for stability, not calibrated to physical units.** The "gyres" we see emerge from real wind curl, but the magnitude (Sv) is meaningless. |
| Two-layer baroclinic infrastructure (v1b) | Coupling code in place | **Currently dialing parameters. Buoyancy term is correctly placed but its strength relative to wind is hand-set. The MOC magnitude in the diagnostic plot is therefore not interpretable as Sv.** |
| Frictional inter-layer coupling | Just-fixed (was broken — was on ψ instead of ζ) | Better, but still parametric. |
| Land mask handling | Working — multiplies RHS, leaves Poisson on full rectangle | Defensible per architecture note. |
| Data orientation | Just-fixed (north-first hires data was being treated as south-first) | Caught a real bug; more places might need similar audits. |
| Tests | 26 passing | Property tests are solid. **Physical-correctness tests don't exist yet** — Munk solution, Sverdrup transport, conservation. That layer would have caught both the cos(φ) and the orientation bugs without manual inspection. |

If a user asked us today "is the simulator honest?" the right answer would
be: *"v1a produces qualitatively-correct gyre patterns from real wind
forcing; the magnitudes are tuned, not calibrated. v1b is still being
parameterized — treat plots as illustrative, not quantitative. Physical-
correctness regression tests are the highest-priority item before
extending further."*

---

## What we should add to the user-facing UX

Per the conversation that produced this doc:

- A *regime panel* in the simulator UI that updates as parameters change,
  flagging when:
  - Mesoscale eddy effects become important (wind ≫ friction-resolved)
  - Cabbeling/thermobaricity matters (high latitudes, large T gradient)
  - Resolution is inadequate (Munk layer δ_M < Δx)
  - Parameter regime is outside what literature has explored
- Each interactive in `interactives/` should mention what it doesn't capture.

---

## References

- Bakker, P. et al. (2025). *Nat Climate Change* (forthcoming).
- Beckmann, A. & Döscher, R. (1997). *J. Phys. Oceanogr.* 27:581.
- Brunnabend, S. E. & Dijkstra, H. A. (2017). *Geosci. Model Dev.* 10:3023.
- Bryan, F. (1987). *J. Phys. Oceanogr.* 17:970.
- Klocker, A. & McDougall, T. J. (2010). *J. Phys. Oceanogr.* 40:1690.
- Kuhlbrodt, T. & Dijkstra, H. A. et al. (2025). *Earth Syst. Dynam.* 16:2063.
- Liu, W., Xie, S.-P., Liu, Z. & Zhu, J. (2017). *Sci Adv* 3:e1601666.
- Munk, W. & Wunsch, C. (1998). *Deep Sea Res. I* 45:1977.
- Polzin, K. (2009). *J. Phys. Oceanogr.* 39:1556.
- St. Laurent, L. & Garrett, C. (2002). *J. Phys. Oceanogr.* 32:2882.
- van Westen, R. M., Kliphuis, M. & Dijkstra, H. A. (2024). *Sci Adv* 10:eadk1189.
- van Westen, R. M. & Dijkstra, H. A. (2025). *Geophys. Res. Lett.* 52:e2024GL114532.
- Wunderling, N. et al. (2024). *Nat Climate Change* 14:1132.
