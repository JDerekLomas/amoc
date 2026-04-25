# amoc-jax

Compute-first ocean circulation simulator. JAX core (Mac CPU/Metal, cloud GPU
via CUDA), with the existing browser viewer as a downstream consumer of saved
state.

This is a from-scratch rewrite of SimAMOC's physics core. The browser version
got tangled in WebGPU/WGSL and a stuck GPU FFT Poisson solver; this rewrite
makes the simulator a pure compute kernel that runs anywhere JAX runs and
dumps zarr/PNG that any frontend can read.

## Status

**v1a shipped.** Barotropic vorticity equation on a global lat-lon grid, run
to wind-driven steady state under ERA5 wind-curl forcing. Subpolar cyclonic
+ subtropical anticyclonic gyres are visible, with western boundary
intensification and an ACC-like band in the Southern Ocean. Stable on Mac
CPU at 256×128 (~480 steps/sec).

Phased plan toward AMOC:

- **v1a — barotropic gyres** (✅) wind-driven Sverdrup/Munk circulation.
- **v1b — two-layer baroclinic.** Add a deep layer; couple via thermal-wind
  buoyancy from a prescribed surface buoyancy gradient. First diagnostic of
  meridional overturning ψ(y).
- **v1c — temperature + salinity.** Prognostic T and S; convective
  adjustment for deep-water formation; restoring-flux SST/SSS surface BCs.
- **v1d — hosing.** Add a freshwater flux knob in the North Atlantic. Reproduce
  Rahmstorf-style hysteresis: ramp the flux up, AMOC collapses; ramp down,
  recovery delayed. The AMOC science showpiece.

## Quickstart

```bash
cd amoc-jax
uv venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/pytest                                      # 21 tests should pass

# A first-run smoke test (~25 seconds on Mac CPU):
.venv/bin/python scripts/run.py \
    --nx 256 --ny 128 --steps 10000 --save-every 1000 \
    --wind 0.02 --beta 2.0 --dt 0.01 --A 0.005 --r 0.04
```

Output goes to `output/` — `00_wind_curl.png` shows the forcing,
`00_ocean_mask.png` shows the land mask, `99_final.png` shows the steady
state, `snap_NNNNNN.png` shows the spin-up sequence.

## Code layout

```
amoc-jax/
  src/amoc/
    grid.py        # spherical lat-lon mesh, cos(lat) metric, Coriolis
    data.py        # load existing JSON fields + hex-packed mask
    poisson.py     # FFT-x + DST-I-y solver for psi from zeta
    physics.py     # Arakawa Jacobian + masked barotropic vorticity RHS
    step.py        # JIT'd RK2 + jax.lax.scan
    state.py       # NamedTuple State, Params, Forcing pytrees
    render.py      # matplotlib 3-panel diagnostic plots
  tests/                   # 21 property + numerical tests
  scripts/
    run.py                 # driver: load forcing -> integrate -> save PNG/npz
    data_status.py         # check every catalogued data field is present + sane
  data_manifest.yaml       # single source of truth for the data layer
  docs/
    physics.md             # equations, derivations, discretization choices
    roadmap.md             # v1a/b/c/d phases
    data.md                # data layer reference (sources, roles, schemas)
```

## Data layer

The simulator depends on observational data for forcing, initial conditions,
restoring targets, geometry (mask, bathymetry), and validation. Every field
the project depends on is catalogued in [`data_manifest.yaml`](data_manifest.yaml)
with its role, source, fetch script, expected shape, and sanity bounds.

```bash
python scripts/data_status.py             # check everything
python scripts/data_status.py --by-role   # group by role
python scripts/data_status.py --sha       # add file hashes for provenance
```

The hires (1024×512) tier in `../data/` is preferred. Fall back to 1° JSONs
at the repo root if the hires version isn't present. To refetch a field:

```bash
python ../fetch-data-hires.py --field wind        # 1024x512 default
python ../fetch-data-hires.py --field wind --resolution 2048x1024  # higher
```

See [docs/data.md](docs/data.md) for the full reference, including the
bathymetry sign-convention gotcha (it's unsigned magnitudes — use
`ocean_mask`, not `depth < 0`).

## Formulation

See `docs/physics.md` for the derivation and the open formulation choices.
The short version: barotropic vorticity, periodic-x / Dirichlet-y, FFT/DST
Poisson solver, Arakawa-conserving advection, RK2 timestep. All terms in
dimensionless grid units; physical scaling is calibrated in v1b/c.

**One non-obvious choice:** the β·v term has no cos(φ) factor, even though
this differs from the existing browser model's `betaV = beta*cos(lat)*dpsi/dx`.
On a sphere, v itself contains a 1/cos(φ) and β contains a cos(φ); they
cancel exactly. Putting cos(φ) back in suppresses β at high latitudes where
it should be strongest, which was why gyres weren't appearing. See
`docs/physics.md` §2 for the derivation.

## Reading list

The project draws from:

- **Stommel (1948)**, **Munk (1950)**, **Sverdrup (1947)** — wind-driven gyres
- **Arakawa (1966)** — energy/enstrophy-conserving Jacobian
- **Stommel (1961)** — 2-box thermohaline bistability
- **Wright & Stocker (1991)** — zonally-averaged thermohaline
- **Cessi (1994)** — stochastic Stommel + analytical solutions
- **Vallis 2017** *Atmospheric and Oceanic Fluid Dynamics*, Ch. 5/14/19/21 — the
  textbook reference
- **Rahmstorf (1996, 2002)** — hosing experiments and AMOC thresholds

Recent (2020–2026):

- **Ditlevsen & Ditlevsen (2023, Nat Comms)** — early-warning AMOC collapse paper
- **van Westen, Kliphuis, Dijkstra (2024, Sci Adv)** + **van Westen & Dijkstra
  (2025, GRL)** — physics-based indicators; tipping in eddying GCMs
- **Kuhlbrodt, Dijkstra et al. (2025, ESD)** — Stommel-style bifurcation in CESM
- **Castellana et al. (2024)** — optimal transition paths (collapse fast,
  recovery slow)
- **van Westen et al. (2025, JGR Oceans)** — operational F_ovS indicator
- **Volkov et al. (2024)** RAPID 2004–2023 update — observed weakening trend

Closest siblings in code:

- **JAXSW** (https://github.com/jejjohnson/jaxsw) — differentiable QG/SW in JAX
- **Veros + Veros-Autodiff** — full primitive equations, JAX backend; correctness
  oracle for v1c+
- **Dinosaur** (NeuralGCM) — JAX spectral atmospheric dycore
- **CLIMBER-X** — modern EMIC reference for v1d

## License

TBD. Sources lifted in primitive form from Apache-2.0 jax-cfd (FFT Poisson
patterns) — license accordingly.
