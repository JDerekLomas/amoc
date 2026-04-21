# Root Cause Analysis — Wiggum Loop Stall at 20%

Found via Claude code review of v4-physics/index.html after 15 Gemini iterations failed to improve beyond 20%.

## Bug 1: No Buoyancy Forcing in Deep Vorticity (CRITICAL)

**Location:** Deep vorticity shader (GPU line ~632-653, CPU line ~1860-1879)

The surface layer has a buoyancy term: `-alpha_T * dT/dx` (line 477).
The deep layer has NO equivalent. Deep circulation is only driven by weak passive coupling `F_couple_d * (psi_surface - psi_deep)`.

**Impact:** Without deep buoyancy forcing, thermohaline overturning CANNOT emerge. The AMOC requires cold dense water at high latitudes to create a pressure gradient driving southward deep flow. This code has no such mechanism. No amount of parameter tuning can fix this.

**Fix:** Add `+alpha_T_deep * (deepTemp[ke] - deepTemp[kw]) * 0.5 * invDx` to deep vorticity equation.

## Bug 2: Ice-Albedo Death Spiral (CRITICAL)

**Location:** Temperature shader (GPU lines 735-739, CPU lines 1811-1814)

Ice-albedo activates at |lat| > 40° with 85% solar reduction at full ice (temp < -3°C).
Once a cell cools below 2°C → ice forms → solar drops → colder → more ice → crashes to -10°C clamp.

**Impact:** All cells poleward of ~50° lock into permanent deep freeze. This is why the 50-70° band shows -10°C when observations are 1-9°C.

**Fix:**
- Raise latitude threshold from 40° to 55-60°
- Reduce max albedo effect from 85% to ~50-60% reduction
- Widen transition range from [-3, 2]°C to [-2, 5]°C

## Bug 3: Deep Water Formation Without Stratification Check (HIGH)

**Location:** Temperature shader (GPU line 777, CPU line 1840)

`gamma_deep_form` activates when `temp < 5 && |lat| > 45` — even when surface is ALREADY colder than deep ocean. This extracts heat from a -10°C surface to a 1°C deep layer, making the surface even colder.

**Impact:** Creates inverted stratification (deep warmer than surface) — fails T1 check.

**Fix:** Add condition: `temp[k] < deepTemp[k]` — only form deep water when surface is actually denser.

## Bug 4: Poisson Solver Under-converged (HIGH)

**Location:** GPU Poisson shader, 60 Jacobi iterations

Jacobi convergence on 360×180 grid has spectral radius ~0.9999. Need O(N) iterations = hundreds. 60 iterations produces essentially noise for basin-scale modes.

**Impact:** Streamfunction is noisy → velocity field is noisy → western boundary currents can't form → no Gulf Stream.

**Fix:** Increase to 200+ iterations, or switch to red-black SOR or multigrid.

## Bug 5: OLR Rebalancing Needed (MEDIUM)

After fixing ice-albedo, the equilibrium temperature curve shifts. OLR constants (A_olr=40, B_olr=2) will need adjustment to maintain realistic tropical-polar gradient.

## Why Parameter Tuning Failed

The Gemini agents correctly identified the symptoms (cold poles, no AMOC, weak WBC) but could only propose parameter changes. The actual issues are:
1. A missing term in the equations (deep buoyancy)
2. An overly aggressive feedback (ice-albedo)
3. A missing condition (stratification check)
4. An insufficient numerical solver (Poisson iterations)

None of these can be fixed by changing S_solar, A_visc, gamma_deep_form, etc. This is exactly the scenario the Claude escalation was designed for.
