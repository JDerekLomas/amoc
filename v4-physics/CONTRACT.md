# `window.lab` — the simulator's API contract

The Barotropic Observatory simulator (`v4-physics/console.html`) exposes a
single global object, `window.lab`, that is the contract between the
engine and any headless consumer (helm-lab, MCP servers, custom drivers).
Anything that goes through this surface is supported. Anything that pokes
at `gpuPsiBuf` or other internal globals is not.

This document is the authoritative reference. If a method here changes
shape, that is a breaking change.

## Lifecycle

The engine sets `window.lab._version = "0.1"` once everything is ready.
Wait for that before calling any other method:

```js
await new Promise(r => {
  if (window.lab && window.lab._version) return r();
  const t = setInterval(() => {
    if (window.lab && window.lab._version) { clearInterval(t); r(); }
  }, 50);
});
```

`getParams()` is also safe to poll — it returns a snapshot synchronously,
including `useGPU` and `NX` (both populated only after engine init).

## Methods

All methods are sync **or** return a `Promise`. Treat them all as awaitable.

### `getParams() → object`

A snapshot of every tunable parameter and a few read-only state fields.
Use this for status, for round-tripping config, and as the source of truth
for what knobs exist.

```ts
{
  // dynamics
  beta, r, A, windStrength, dt, doubleGyre,
  // thermodynamics
  S_solar, A_olr, B_olr, kappa_diff, alpha_T,
  // two-layer / thermohaline
  H_surface, H_deep, gamma_mix, gamma_deep_form, kappa_deep,
  F_couple_s, F_couple_d, r_deep,
  // forcings
  yearSpeed, freshwaterForcing, globalTempOffset, T_YEAR,
  // numerics
  stepsPerFrame, POISSON_ITERS, DEEP_POISSON_ITERS,
  // state (read-only)
  totalSteps, simTime, paused, showField, NX, NY, useGPU
}
```

### `setParams(p) → object`

Set any subset of the tunable fields above. Returns the post-update
`getParams()` so callers can confirm. Mirrors changes to the matching
DOM sliders so the UI stays consistent.

```js
await lab.setParams({ windStrength: 1.5, freshwaterForcing: 0.5 });
```

### `step(n) → { step, simTime, simYears }`

Run exactly `n` solver steps deterministically, off the rAF loop. Pauses
the rAF tick during the call and restores its prior pause state. Chunked
internally so the GPU driver doesn't time out.

`n` of 1 to ~5,000,000 is fine. One step is `dt = 5e-5` of model time;
`T_YEAR = 10` of those, so 200,000 steps ≈ 1 model year at default rate.

### `diag(opts?) → object`

A single-snapshot diagnostic readout from the current fields. No stepping.

```ts
diag({ profiles: false }) → {
  step, simTime, simYears, seasonFrac,
  oceanCells, maxVel, KE,
  globalSST, tropicalSST, polarSST, nhPolarSST, shPolarSST,
  amoc,        // Atlantic-band meridional transport at ~45N
  accU,        // zonal-mean u at ~60S (ACC proxy)
  iceArea,     // wet ocean cells with T < -1.5
  gyreMaxPsi, gyreMinPsi, gyreRangePsi,
}
```

`opts.profiles = true` adds `zonalMeanT`, `zonalMeanPsi`, `zonalMeanU`,
`latitudes` — each a length-NY array.

### `fields(names?) → object`

Pull raw field arrays back to the caller. Heavy — each is `NX*NY` floats.
Default names: `['psi', 'temp', 'deepTemp', 'deepPsi']`.

```ts
fields(['psi', 'temp']) → {
  NX, NY, LAT0, LAT1, LON0, LON1,
  psi: Float32Array, temp: Float32Array
}
```

### `view(name) → name`

Switch the rendered view. One of:
`temp`, `psi`, `vort`, `speed`, `deeptemp`, `deepflow`, `depth`.

The change takes effect on the next render frame. `view('psi')` does not
itself trigger a render; you need to wait at least two `requestAnimation
Frame`s for the GPU pipeline to settle.

### `reset() → { step }`

Re-initialize fields (psi, zeta, temp, deepTemp, deepPsi, deepZeta) to
their default initial state. Counter is zeroed; mask is preserved.

### `pause() / resume() / isPaused()`

Toggle the rAF stepping loop. `step(n)` works regardless of pause state —
the pause flag only affects the visual rAF loop. Use these only when you
want the bridge UI to stop updating but keep the engine programmable.

### `scenario(name) → name`

Trigger one of the paleoclimate scenarios. Equivalent to clicking the
scenario card in the UI; modifies the mask and resets state.

Names: `present`, `drake-open`, `drake-close`, `panama-open`, `greenland`,
`iceage`.

### `sweep(knob, values, opts?) → array`

Server-side parameter sweep. For each `value`:
1. (optional) reset
2. `setParams({ [knob]: value })`
3. `step(opts.settleSteps)` if given
4. `step(opts.stepsPerPoint)` (default 50000)
5. snapshot diagnostics

```ts
sweep('windStrength', [0.5, 1, 1.5, 2], {
  stepsPerPoint: 50000,
  settleSteps:    0,
  resetBetween:  false,
}) → [{ knob, value, step, simYears, KE, amoc, ... }, ...]
```

Pure convenience — equivalent to `setParams + step + diag` in a loop.

### `timeSeries(totalSteps, opts?) → array`

Server-side trajectory: step in chunks, snapshot each chunk.

```ts
timeSeries(500000, { interval: 25000 })
  → [{ step, simYears, KE, amoc, ... }, ...]
```

## Things outside the contract

- The page also renders a 2D overlay onto `<canvas id="sim">`. That canvas
  is for the bridge UI, not for callers. Capture from `gpu-render-canvas`
  (or the compositor via `page.screenshot`) instead.
- `window.lab` does not stream events. Poll `getParams` or `diag` if you
  need a clock. The cached daemon healthz in helm-lab is built on this.
- There is no support for multiple simultaneous callers per page. If you
  need parallelism, run multiple browser pages, each with its own engine.
- CPU fallback exists in the engine but the lab API requires WebGPU. If
  WebGPU is unavailable, lab methods throw `lab: GPU not initialized`.

## Versioning

`_version` is a string. We'll bump the second digit for additive changes
and the first digit for any breaking change. Today: `"0.1"`.

If you depend on this surface from outside the repo, check `_version` at
runtime and refuse to run on an unrecognized major.
