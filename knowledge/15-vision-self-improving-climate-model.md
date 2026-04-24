# Vision: Self-Improving Climate Model

## One-liner
An ocean circulation model that continuously discovers and improves its own physics by running AI agents against observational data.

## Why this matters
Climate model development takes years of PhD-level hand-tuning. We demonstrated in one session that AI agents can diagnose structural physics bugs, propose code fixes, and drop RMSE 37% — with zero human physics input. The polar OLR boost and brine rejection fixes were real scientific insights, not curve fitting. What if this process ran continuously?

## Architecture

```
                    ┌─────────────────────────┐
                    │   Observation Database   │
                    │  NOAA SST, Argo, RAPID,  │
                    │  ERA5 winds, sea ice     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Evaluation Engine     │
                    │  T1: Conservation        │
                    │  T2: Structure           │
                    │  T3: Sensitivity         │
                    │  T4: Quantitative (RMSE) │
                    │  T5: Tipping behavior    │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼────────┐ ┌──────▼──────┐ ┌─────────▼────────┐
    │  Physicist Agent  │ │ Numerics    │ │ Literature Agent │
    │  Diagnoses errors │ │ Agent       │ │ Checks published │
    │  from screenshots │ │ CFL, grid   │ │ model values     │
    │  + error profiles │ │ artifacts   │ │                  │
    └─────────┬────────┘ └──────┬──────┘ └─────────┬────────┘
              │                  │                   │
              └──────────────────┼───────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     Code Mutator        │
                    │  Proposes physics code   │
                    │  changes as git branches │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼──────┐ ┌────────▼───────┐ ┌────────▼───────┐
    │  Branch A      │ │  Branch B      │ │  Branch C      │
    │  "Add Ekman    │ │  "Fix wind     │ │  "Latitude-dep │
    │   heat trans"  │ │   stress curl" │ │   diffusion"   │
    └─────────┬──────┘ └────────┬───────┘ └────────┬───────┘
              │                  │                   │
              └──────────────────┼───────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Tournament Selection  │
                    │  Run all branches        │
                    │  Evaluate against obs    │
                    │  Merge winners to main   │
                    │  Prune losers            │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Improved Model (main) │
                    │   + git history of       │
                    │   physics discoveries    │
                    └──────────────────────────┘
```

## Three Layers

### Layer 1: Parameter Optimization (BUILT)
The current Wiggum loop. 13 tunable parameters, 3-agent dialectic, 4-tier evaluation.
- Status: Working, RMSE 7.6 → 4.7 in 10 iterations
- Limitation: Can't fix structural physics errors

### Layer 2: Physics Code Mutation (NEXT)
Agents propose modifications to the WGSL shaders and JS physics code.
- Each proposal runs in an isolated git worktree
- Evaluated against the same 4-tier framework
- Proposals that improve multiple independent metrics get merged
- Proposals that improve one metric but degrade others are suspicious (compensating errors)

Examples of mutations already discovered:
- Polar OLR boost (latitude-dependent outgoing longwave)
- Brine rejection (sea ice salt concentration for AABW formation)
- Ice-albedo threshold shift (-2/10°C → -5/7°C window)

### Layer 3: Differentiable Physics (FUTURE)
Port the model to JAX/WebGPU autodiff. Compute gradients of RMSE w.r.t. all parameters simultaneously. Combine gradient-based optimization with agent-based structural search.

## Evaluation Hierarchy

Not all metrics are equal. A model that fits SST perfectly but has wrong AMOC dynamics is worse than one with 1°C higher RMSE but correct tipping behavior.

1. **Conservation** (binary gate) — energy balance must close
2. **Structural** — right features for right reasons (WBCs, ACC, AMOC cell)
3. **Sensitivity** — correct response to perturbations (hosing → collapse)
4. **Quantitative** — SST RMSE, AMOC strength in Sverdrups
5. **Tipping** — bistability, hysteresis, FovS early warning

A mutation that breaks T1-T2 is rejected even if it improves T4.

## Observational Targets

| Dataset | What it constrains | Source |
|---------|-------------------|--------|
| NOAA OI SST v2 | Surface temperature | 1° monthly, 1991-2020 |
| WOA23 | Deep temperature + salinity | 1° annual, all depths |
| ERA5 | Wind stress, heat fluxes | 0.25° monthly |
| RAPID array | AMOC strength at 26°N | Daily, 2004-present |
| NSIDC | Sea ice extent | Daily, 1979-present |
| Argo | Deep ocean profiles | Global, 2000-present |

## What Makes This Different

Most ML-for-climate work falls into two camps:
1. **Emulators** — train NN to mimic an existing GCM (faster but no new physics)
2. **Parameterization** — learn subgrid closures from high-res simulations

We're doing neither. We're using AI to **discover physics** — to find the missing terms in the equations, propose code implementations, and validate them against observations. The AI is the scientist, not the model.

The git history becomes a record of scientific discoveries, each with a clear hypothesis, implementation, and observational validation. That's publishable.

## Immediate Roadmap

### Phase 1: Automated Branch Tournament (this week)
- Script that spawns N agents, each proposes a physics modification
- Each runs in an isolated worktree via Playwright
- Evaluated against T1-T4
- Best branch merged, results logged

### Phase 2: Observation Integration (next week)
- Ingest WOA23 salinity for initialization
- ERA5 wind stress to replace analytical formula
- RAPID AMOC comparison for T5 evaluation

### Phase 3: Continuous Improvement Loop
- GitHub Action or cron job runs tournament daily
- Each day: spawn agents, evaluate branches, merge winner, deploy
- Track RMSE over time — should monotonically decrease

### Phase 4: Publication
- Paper: "AI-Discovered Physics Improvements in a Real-Time Ocean Model"
- Show the git history of discoveries
- Compare to hand-tuned model of equivalent complexity
- Open-source everything
