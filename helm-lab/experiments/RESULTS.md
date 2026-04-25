# Experimental results

Captured findings from helm-lab runs. Each entry references a script and
the artifacts it produced. `helm-lab/runs/` is gitignored; rerun the
script to regenerate.

## R1 — Long monitoring trajectory (April 2026)

**Script:** ad-hoc trajectory via `cli.mjs trajectory --steps 600000
--interval 20000` (caught the cli.mjs single-RPC-timeout bug; 26 of 30
samples completed before fetch died — fix landed in same commit).

**Findings — engine has a multi-stage spinup transient:**

| step      | sim yr | KE       | AMOC     | comment                  |
|-----------|--------|----------|----------|--------------------------|
| 0         | 0.03   | 2.75e+1  | +0.0005  | initial                  |
| 100,000   | 0.55   | 6.72e+3  | +0.0003  | rising                   |
| 200,000   | 1.07   | 1.14e+4  | -0.0002  | KE peak; AMOC flipping   |
| 220,000   | 1.17   | 1.16e+4  | -0.0011  | overshoot apex           |
| 280,000   | 1.49   | 5.51e+3  | -0.0020  | KE collapsed, AMOC settling   |
| 380,000   | 2.01   | 5.92e+3  | -0.0027  | quasi-steady negative    |
| 500,000   | 2.64   | 5.18e+3  | -0.0016  | seasonal drift           |

**Implication:** the AMOC undergoes a sign flip at year ~1.2 that has
nothing to do with the imposed forcing — it is the model's own spinup.
Any hysteresis-style experiment must run AT LEAST 2 model years of pure
spinup before perturbing, or it picks up the transient instead of the
forcing response.

## R2 — AMOC bifurcation v1 (short spinup)

**Script:** `helm-lab/experiments/amoc-bifurcation.mjs`
**Parameters:** spinup 60k, dwell 12k per F, F = 0…2 in 0.2 steps both
directions.

**Findings — small-magnitude hysteresis dominated by spinup transient.**
At every shared F the forward branch sat 1–2e-3 above the backward
branch, peaking at F=1.4 (gap 2.2e-3). Magnitudes were 1-2 orders of
magnitude below the engine's "STRONG" threshold (0.05) because we
caught the system mid-transient.

Useful as a methodology check: the harness produced 42 rendered frames,
a JSONL of diagnostics, and a composed bifurcation diagram in 3 minutes.

## R3 — AMOC bifurcation v2 (informed spinup)  ★

**Script:** `helm-lab/experiments/amoc-bifurcation-v2.mjs`
**Parameters:** spinup 400k (~2 model years), dwell 60k per F, F at 0,
0.4, 0.8, 1.2, 1.6, 2.0 forward + reverse.
**Wall clock:** 11.5 min on Apple Metal GPU.

**Findings — qualitatively textbook Stommel hysteresis.**

Spinup trace itself was revealing:
- step 200k: AMOC=+0.0004 (post-initial-rise plateau)
- step 250k: AMOC=-0.0023 (sudden flip to negative branch)
- step 350k: AMOC=-0.0020 (settled negative)
- step 400k: AMOC=**+0.0092** (flipped BACK to a strong positive branch)

The model is genuinely bistable. The 400k spinup happened to land on the
strong-positive branch.

Forward branch (starting from baseline AMOC ≈ +0.0092):

| F   | AMOC      | drop from baseline |
|-----|-----------|--------------------|
| 0.0 | +0.00457  | -50%               |
| 0.4 | +0.00164  | -82%               |
| 0.8 | +0.00276  |                    |
| 1.2 | +0.00281  |                    |
| 1.6 | +0.00198  |                    |
| 2.0 | +0.00162  |                    |

The big step is F=0 → F=0.4 — AMOC drops by another factor of three.
This is the collapse region.

Backward branch (returning from F=2.0):

| F   | AMOC      | vs forward at same F |
|-----|-----------|----------------------|
| 1.6 | +0.00143  | -28% vs forward      |
| 1.2 | +0.00142  | -49% vs forward      |
| 0.8 | +0.00113  | -59% vs forward      |
| 0.4 | +0.00112  | -32% vs forward      |
| 0.0 | +0.00055  | -88% vs forward      |

At F=0, the backward branch sits at 0.00055 vs forward 0.00457 — same
forcing, AMOC differs by a factor of ~8. **Recovery to baseline is 6%.**
Once collapsed, the conveyor does not come back when the freshwater is
removed, at least not within the dwell time we gave it. That is the
core Stommel finding.

**Spatial hysteresis is also visible** — see `hysteresis_2x2.png`:
the North Atlantic is colder and ice-broader on the backward F=0 frame
than on the forward F=0 frame, despite identical forcing.

**Caveats:**
- The within-branch numbers wobble (forward F=0.4 lower than F=0.8) —
  signal that 60k dwell is not yet equilibrium, just response.
- The system is bistable; the "baseline" depends on which branch the
  spinup happened to land on. A statistical study would need multiple
  spinups from different initial conditions.
- The "STRONG" UI threshold (0.05) is calibrated for a regime this run
  doesn't reach; everything here is "WEAK" or "COLLAPSED" in the UI's
  language even when the physics shows clear bifurcation.

## R5 — AMOC collapse-window scan  ★

**Script:** `helm-lab/experiments/amoc-collapse-window.mjs`
**Parameters:** spinup 400k, dwell 60k, F dense in [0.0, 0.4] (Δ=0.05),
sparse in [0.5, 2.0]. Forward + backward.
**Wall clock:** 18.6 min on Apple Metal GPU.

**The number we wanted: F\* ≈ 0.125** — between forward F=0.10 (AMOC
+0.00353, healthy) and forward F=0.15 (AMOC +0.00166, collapsed). That's
a 53% drop in a forcing window of 0.05 — the saddle-node signature.

Forward branch (the order parameter as F increases):

| F     | AMOC      | regime                  |
|-------|-----------|-------------------------|
| 0.00  | +0.00514  | healthy positive        |
| 0.05  | +0.00473  | healthy                 |
| 0.10  | +0.00353  | last healthy            |
| **0.15** | **+0.00166** | **collapsed (F\* crossed)** |
| 0.20  | +0.00166  | collapsed               |
| 0.25  | +0.00209  | collapsed (wobble)      |
| 0.30  | +0.00313  | collapsed (wobble)      |
| 0.40  | +0.00134  | collapsed               |
| 0.50  | -0.00040  | sign-flipped negative   |
| 0.70  | -0.00014  | reversed                |
| 1.50  | -0.00094  | reversed                |
| 2.00  | -0.00064  | reversed                |

Backward branch (returning from F=2.0):

| F    | AMOC      | recovery vs forward      |
|------|-----------|--------------------------|
| 1.50 | +0.00010  | (forward was -0.00094)   |
| 1.00 | +0.00143  | (forward was -0.00023)   |
| 0.50 | +0.00113  | (forward was -0.00040)   |
| 0.20 | +0.00177  | (forward was +0.00166)   |
| 0.10 | +0.00116  | (forward was +0.00353)   |
| 0.00 | +0.00145  | **only 28% of forward F=0** |

**The two key results:**
1. **F\* ≈ 0.125.** Sharp collapse threshold; the drop is monotone and
   confined to a 0.05-wide forcing window.
2. **Recovery is 28%.** Returning F → 0 after the conveyor has flipped
   gives back about a third of the original AMOC. Most of the
   circulation does not come back at this dwell time — that is the
   Stommel hysteresis, made quantitative.

Caveats unchanged from R3: the forward branch wobbles between F=0.25
and F=0.40 (60k dwell is response-time, not equilibrium-time), and the
"baseline" AMOC depends on which side of the bistability the spinup
landed on. The 400k spinup landed on the strong-positive branch this
run (baseline 0.0072) as it did in R3 (0.0092) — but R3's V2 ramp from
that baseline only reached F=2.0 endpoint of +0.00162; R5's denser
scan caught it earlier and at a finer resolution.

Spatial signature in `four_states.png`: forward F=0 (healthy NH) vs
backward F=0 (cooler NH, broader polar ice extent) at identical
forcing — the climate state is visibly different.

## R6 — Greenhouse sweep  ★

**Script:** `helm-lab/experiments/greenhouse-sweep.mjs`
**Parameters:** spinup 400k, dwell 80k per forcing, ΔT ∈ [−8, −4, −2,
0, +2, +4, +8] °C of equivalent radiative forcing.
**Wall clock:** 9.6 min.

**Findings:**

| ΔT  | T_glob  | T_trop  | T_pol  | AMOC      | sea-ice cells |
|-----|---------|---------|--------|-----------|---------------|
| −8  | 11.14°C | 13.31°C | 3.36°C | −0.0004   | **3320**      |
| −4  | 11.12°C | 13.28°C | 3.31°C | +0.0003   | 3310          |
| −2  | 11.65°C | 14.51°C | 4.14°C | −0.0007   | 3305          |
|  0  | 11.97°C | 16.23°C | 2.38°C | −0.0026   | 3287          |
| +2  | 12.72°C | 16.57°C | 4.12°C | −0.0012   | 2616          |
| +4  | 14.60°C | 17.84°C | 6.47°C | −0.0021   | 3083          |
| +8  | 14.95°C | 19.00°C | 6.88°C | −0.0005   | **2336**      |

**Climate sensitivity slope ≈ 0.28** (linear-fit of ΔT_global per ΔT_forcing).
A heavily damped response — at 80k-step dwell, the ocean has felt the
forcing change but is far from its new equilibrium. With a much longer
dwell the slope would approach unity (or higher with feedbacks). The
0.28 here is the *transient* sensitivity, not the *equilibrium*
sensitivity.

**Ice retreat is clear:** 3320 → 2336 cells across the 16°C forcing
range (−30%). Visible in the seven-panel tile (`sevenstack.png`):
North Atlantic ice tongue contracts; tropical band intensifies and
broadens northward.

**Tropics warm faster than poles** — at +8°C: T_trop +5.7°C vs T_pol
+3.5°C above the −8°C baseline. This is *inverted* from observed
Earth's polar amplification — the model lacks the cloud / sea-ice
albedo feedbacks that drive Arctic amplification, OR the dwell wasn't
long enough for them to express. Worth investigating with a longer-
dwell run.

**AMOC is uncorrelated with thermal forcing** — values bounce between
−0.0026 and +0.0003 with no clear trend. Expected: in this model,
freshwater (R5) is the AMOC knob; raising the global temperature does
not directly suppress the conveyor. (In real Earth, warming drives
ice-sheet melt → freshwater → AMOC effect — but that coupling is not
in this engine.)

**Spinup landed differently this run.** Baseline AMOC at end of spinup
was small/negative (the negative branch), so all values here are in
that regime. Comparison with R5's positive-branch baseline is
apples/oranges; a study of climate sensitivity should average across
multiple spinups.

## R7 — Snowball Earth threshold  ★

**Script:** `helm-lab/experiments/snowball-search.mjs`
**Parameters:** spinup 400k at S_solar = 5.0 (modern), then sweep S
downward: 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5. Dwell 80k
per S.
**Wall clock:** 12.3 min.

**The runaway is real and sharp.**

| S    | T_glob   | T_pol    | ice cells | ice frac | regime          |
|------|----------|----------|-----------|----------|-----------------|
| 5.0  | +15.20°C | +6.84°C  |   3,161   |  7%      | modern, ice-free |
| 4.5  | +14.17°C | +5.69°C  |   2,961   |  7%      | warm             |
| 4.0  | +11.68°C | +4.04°C  |   3,112   |  7%      | cooler           |
| 3.5  |  +9.07°C | +2.23°C  |   3,097   |  7%      | mid              |
| 3.0  |  +9.06°C | +2.23°C  |   3,101   |  7%      | mid (plateau)    |
| 2.5  |  +6.75°C | +1.02°C  |   3,141   |  7%      | cooling          |
| 2.0  |  +3.11°C | −1.63°C  |   3,151   |  7%      | freezing point   |
| 1.5  |  +0.07°C | −3.50°C  |   4,339   | 10%      | last ice-free    |
| **1.0** | **−3.24°C** | **−5.97°C** | **37,695** | **86%** | **SNOWBALL — runaway crossed** |
| 0.5  |  −6.50°C | −7.91°C  |  43,950   | 100%     | full snowball    |

**Snowball threshold S\* ≈ 1.25** (engine units; midpoint of S=1.5 and
S=1.0). Ice cover jumps from 4,339 cells (10%) to 37,695 cells (86%)
in a single 0.5-step solar reduction — a +33,356 cell jump, or +76%
of the ocean iced over at the threshold. Global mean SST plunges
3.3°C in that same step.

This is the textbook **ice-albedo runaway** (Budyko, 1969 / Sellers,
1969 / Hoffman & Schrag's Cryogenian work). New ice appears, reflects
more sunlight, cools the surface, makes more ice. Below S\* the
feedback wins; above it, balance.

**Note on irreversibility:** this run was a one-way ramp. The reverse
question — at what S\* does a snowball planet *unfreeze*? — is the
canonical second result, with a much higher threshold (the Hoffman
"snowball escape" problem). That experiment would warm a frozen
planet back up; expected S\* for de-glaciation is far above 5.0
because dropping back through the runaway region first requires
breaking the ice albedo's hold.

**Visual progression:** see `snowball_progression.png` — the
transition between S=1.5 and S=1.0 is unmistakable.

## R8 — Snowball escape (Hoffman thaw threshold)  ★

**Script:** `helm-lab/experiments/snowball-escape.mjs`
**Parameters:** freeze first at S=0.3 for 500k steps (lock in snowball);
then ramp S up through 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0,
10.0, 14.0, 20.0. Dwell 80k each.
**Wall clock:** 18.1 min.

**Escape threshold S\*_escape ≈ 4.5** (between S=4.0 and S=5.0).

| S    | T_glob   | ice cells | ice frac | regime           |
|------|----------|-----------|----------|------------------|
| 0.5  | −10.00°C |  43,950   | 100%     | frozen           |
| 1.0  | −10.00°C |  43,950   | 100%     | frozen (would have melted on first pass — see R7)|
| 1.5  |  −9.41°C |  43,950   | 100%     | frozen           |
| 2.0  |  −8.19°C |  43,947   | 100%     | frozen           |
| 2.5  |  −6.47°C |  43,842   | 100%     | frozen           |
| 3.0  |  −6.42°C |  43,862   | 100%     | frozen (plateau) |
| 4.0  |  −3.99°C |  40,089   | 91%      | frozen (cracking)|
| **5.0** | **+3.68°C** | **4,306** | **10%** | **ESCAPED** |
| 7.0  | +12.06°C |   3,291   | 7%       | warm             |
| 10.0 | +24.31°C |   1,984   | 5%       | hothouse         |
| 14.0 | +24.59°C |   1,990   | 5%       | hothouse plateau |
| 20.0 | +38.57°C |       0   | 0%       | extreme hothouse |

**Hysteresis window: ΔS = 3.25.**
- Going down (R7): freezes at S\* ≈ 1.25
- Going up (R8): thaws at S\* ≈ 4.5
- Bistability region: 1.25 ≤ S ≤ 4.5

The thaw threshold is **3.6× higher** than the freeze threshold —
exactly the Cryogenian "hard problem" Hoffman & Schrag named: once
the planet is iced over, you need to overdrive the system far past
where it originally crossed in order to break out. In the simulator
that's a factor of 3.6×; in real Earth's rock record it's hundreds
of ppm of CO₂ buildup over millions of years.

**Side observation: a second plateau between S=10 and S=14.** Both
points have T_glob ≈ 24.4°C and ice ≈ 1990 cells. Then S=20 jumps
to T_glob = 38.6°C, ice = 0. There may be another regime boundary
hiding in [14, 20] worth a future targeted scan — possibly a
secondary feedback (cloud, evaporation, OLR saturation) kicking in.

**Visual:** see `escape_progression.png` for the six-frame transition
sequence (S=0.5, 2.0, 4.0 frozen → S=5.0, 10.0, 20.0 thawed).

## R4 — Earth-over-time movie (April 2026)

**Script:** `helm-lab/experiments/earth-movie.mjs`
**Parameters:** spinup 20k, 120 frames every 1500 steps, view=temp, 24fps.
**Wall clock:** 3 min.

**Output:** 5-second MP4 (1.2 MB H.264) of the temperature field
evolving over 1.34 model years. Shows the spinup transient
in the visual record — gyres organizing, polar caps forming, North
Atlantic ice fluctuating with seasons.

Tile of every 10th frame: see `timelapse_grid.png`.
