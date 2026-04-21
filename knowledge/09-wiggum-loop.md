# Wiggum Loop Architecture

## Overview

Ralph Wiggum loop: iterative AI agent loop that runs the simulation, evaluates against physics + observations, and proposes improvements until convergence.

Model: Gemini 3.1 Flash Lite ($0.25/1M in, $1.50/1M out)
Escalation: Claude for structural code changes

## Core Agents (every iteration)

### 1. Physicist
- **Sees**: 4 screenshots (temp, psi, speed, deep temp) + scorecard + zonal errors
- **Does**: Generates 2-3 competing hypotheses for dominant error
- **Output**: Ranked hypotheses with predicted fixes and side effects
- **Key**: Must identify PHYSICAL MECHANISM, not just "change param X"

### 2. Tuner
- **Sees**: Physicist's winning hypothesis + current params + bounds
- **Does**: Translates hypothesis to specific parameter changes
- **Rules**: Max 3 params, max 30% step, must check equilibrium T
- **Output**: Parameter changes with predicted effect

### 3. Validator
- **Sees**: Tuner's proposal + Physicist's reasoning
- **Does**: Checks physical consistency (NOT whether it reduces RMSE)
- **Checks**: Equilibrium sanity, boundary layer width, coupling consistency, compensating errors, causal coherence
- **Output**: APPROVE / MODIFY / REJECT with explanation

## On-Demand Agents (triggered by conditions)

### 4. Numerical Analyst
- **Triggers**: T1 conservation fails
- **Sees**: Screenshots + params
- **Checks**: Checkerboard, CFL, Poisson convergence, coastal ringing, blow-up
- **If CRITICAL**: Applies numerical fix, skips physics tuning that round

### 5. Skeptic
- **Triggers**: Every 4th iteration
- **Checks**: Parameter drift from defaults, compensating errors, trajectory coherence, score gaming
- **If COMPROMISED**: Rolls back all params to defaults

### 6. Observational Scientist
- **Triggers**: Worst errors at polar latitudes (once per loop)
- **Checks**: Reference data quality at problematic latitudes
- **Output**: Uncertainty estimates, whether to downweight certain observations

### 7. Literature Agent
- **Triggers**: Any parameter hits its bound (once per param)
- **Checks**: What published ocean models use for that parameter
- **Can**: Widen parameter bounds if literature supports it

### 8. Claude (Escalation)
- **Triggers**: 3 consecutive iterations with <5% improvement, or 2+ rejections
- **Does**: Full code review of v4-physics/index.html
- **Looks for**: Missing physics terms, boundary condition bugs, numerical issues
- **Output**: Structural code changes (not just parameters)

## Evaluation Tiers

See `03-diagnostics.md` for full details.

```
T1 Conservation → GATE (binary)
T2 Structure    → 35% (7 checks)
T4 Quantitative → 35% (RMSE vs NOAA)
AMOC            → 15%
T1 bonus        → 15%
T3 Sensitivity  → 20% of final (run once on best params)
```

## Key Lessons from First Runs

1. **Reference data format**: SST uses `sst` key, deep temp uses `temp` key (not `data`)
2. **Parameter injection**: Must use `eval()` for `let`-scoped variables (not `window[key]`)
3. **AMOC is non-dimensional**: ~0.0001-0.01 range, not Sverdrups
4. **Grid geography**: i=0 → lon -180, j=0 → lat -80. Verify basin boundaries!
5. **Gemini agents hammer A_visc**: They fixate on viscosity because it's the most intuitive lever. Need to force diversity.
6. **Validator correctly rejects category errors**: Caught alpha_T proposed for western intensification (should be A_visc/r_friction)
7. **Numerical Analyst triggers usefully**: Caught Poisson convergence and coastal ringing issues
