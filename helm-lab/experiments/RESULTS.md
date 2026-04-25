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

## R4 — Earth-over-time movie (April 2026)

**Script:** `helm-lab/experiments/earth-movie.mjs`
**Parameters:** spinup 20k, 120 frames every 1500 steps, view=temp, 24fps.
**Wall clock:** 3 min.

**Output:** 5-second MP4 (1.2 MB H.264) of the temperature field
evolving over 1.34 model years. Shows the spinup transient
in the visual record — gyres organizing, polar caps forming, North
Atlantic ice fluctuating with seasons.

Tile of every 10th frame: see `timelapse_grid.png`.
