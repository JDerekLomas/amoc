# AMOC Simulation Knowledge Bank

Organized reference for building and tuning the WebGPU ocean simulation.
Each file is self-contained and cross-referenced.

## Structure

```
knowledge/
  README.md              ← You are here
  01-equations.md        ← Physics equations used in our sim
  02-parameters.md       ← Parameter values with published ranges
  03-diagnostics.md      ← What to measure, how, and why
  04-ocean-models.md     ← Survey of MOM6, NEMO, MITgcm, etc.
  05-known-biases.md     ← Common model biases and their causes
  06-observations.md     ← Reference datasets (NOAA SST, WOA, RAPID)
  07-numerical.md        ← Grid, stability, solvers, time stepping
  08-emerging-ml.md      ← Neural operators, differentiable models, emulators
  09-wiggum-loop.md      ← Agent architecture and evaluation tiers
```

## How to Use

- **Tuning the sim?** Start with `02-parameters.md` for defensible ranges, cross-ref `04-ocean-models.md` for what real GCMs use.
- **Something looks wrong?** Check `03-diagnostics.md` for what to measure, `05-known-biases.md` for whether it's a known issue.
- **Comparing to observations?** See `06-observations.md` for dataset details and limitations.
- **Numerical artifacts?** See `07-numerical.md` for stability criteria and solver details.
- **Wiggum loop context?** See `09-wiggum-loop.md` for the agent team and evaluation framework.

## Annotation Convention

Each file uses this format for claims that need sources:
```
> **Claim** [source: short-ref]
```

Links are at the bottom of each file under `## Sources`.
