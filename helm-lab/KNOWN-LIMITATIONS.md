# Known limitations

A candid log of what's broken, fragile, or surprising in the
simulator + harness as of today. Update as we find more.

## The simulator (v4-physics & simamoc)

### `file://` does not work
**This is the most common source of "the rendering is broken."**

Both `v4-physics/console.html` and `simamoc/index.html` fetch dataset
JSON files (mask, coastlines, SST, bathymetry, salinity, …) at load
time. Browsers block `fetch()` of arbitrary local files when the page
is opened over `file://` — same-origin policy. The fetches silently
fail; the engine falls back to a featureless rectangular ocean with
no continents and no boundary currents. AMOC reads `+0.0e+0 weak`,
and you see clean zonal temperature bands instead of gyres.

**Fix:** always serve over HTTP. Either:

```bash
# from project root
python3 -m http.server 8000
open http://localhost:8000/v4-physics/console.html
# or
open http://localhost:8000/simamoc/

# or just use helm-lab's daemon, which runs its own static server:
node helm-lab/cli.mjs serve
```

The deployed Vercel site works fine because it's served over HTTPS.

### The model has bistability
The same 400k-step spinup at default parameters can land on **either**
the strong-positive AMOC branch (~+0.0072 to +0.0092) **or** the
weak/negative branch (~-0.001 to -0.003), depending on the random walk
through the spinup transient. Verified across R3 (positive), R5
(positive), R6 (negative).

**Implication:** any quantitative claim about climate sensitivity or
collapse threshold is conditioned on which branch the spinup landed
on. A proper study averages over many cold starts. The current scripts
don't do that — they bail with a warning if R5 lands on the wrong
branch (`POSITIVE_BRANCH_THRESHOLD = 0.003`).

### Spinup has a multi-stage transient
KE rises, peaks at sim year ~1.17 (1.16e+4), then **collapses** to
~6000 quasi-steady. AMOC starts weakly positive, sign-flips around
year 1.2, settles to ~−0.002 by year ~2 — UNLESS another flip kicks
it back to the positive branch (which is what produces the bistable
end-state). This isn't an artifact; it's the model finding its fixed
points. R1 documents the trajectory.

**Implication:** experiments need ≥ 2 model years (~400,000 steps) of
pure baseline spinup before any forcing perturbation, or they pick
up the spinup transient instead of the forcing response. R2 (60k
spinup) caught the system mid-transient and got noisy results.

### Dwell time is response-time, not equilibrium-time
The bifurcation experiments use 60–80k step dwells (~0.3–0.4 model
years per parameter value). That's enough to see the **shape** of
the response but not enough for the system to settle into a true
new equilibrium. R3 noted this directly: forward F=0.4 reading
+0.00164 is lower than F=0.8's +0.00276 — the within-collapse wobble
is the signal that we're seeing transient, not steady-state values.

**Implication:** thresholds are real and reproducible (R5's
F\*=0.125, R7's S\*=1.25, R8's S\*=4.5). Within-branch numerical
values are approximate.

### `lab.step()` is GPU-only
The harness's `lab.step(n)` deterministic stepping path requires
WebGPU. If WebGPU isn't available (some headless browsers, some
remote Chromium variants), it throws `lab: GPU not initialized` after
a 20-second wait. CPU fallback exists in the engine but the lab API
doesn't exercise it.

**Implication:** the harness only works on machines with a real
WebGPU implementation (macOS Apple Metal, modern Chromium with
hardware acceleration). Headless servers without a GPU will fall
through to SwiftShader (slow) or fail entirely.

### Renders of `psi`, `vort` are normalized off stale state
`gpuRenderField()` computes `absMax` for normalization off the
JS-side `psi` Float32Array. After `lab.step(N)` runs many GPU steps
without triggering a readback, that JS array can be stale. The
streamfunction colormap looks "off" because the GPU buffer is fresh
but the normalization is from an older state.

**Mitigation:** `lab.render()` forces a readback before screenshotting,
which mostly fixes this. But running the simulator from the bridge UI
during a render call can introduce inconsistency.

## helm-lab harness

### Single-RPC trajectory was timing out
Original `cli.mjs trajectory --steps N` issued one `lab.trajectory()`
RPC for the whole run. For N > ~300k steps, Node's `fetch` would
time out before the daemon finished. Fixed in `c5852d6`: client-side
orchestration in daemon mode (small RPCs in a loop, streaming
progress).

### Daemon shutdown sometimes leaves port 8810 bound
The static file server in `lab.mjs` is on 8810. On daemon shutdown,
`server.closeAllConnections()` is supposed to drain it, but
occasionally a Playwright-Chromium connection lingers and blocks the
port for several seconds. The next `cli.mjs serve` then fails with
`EADDRINUSE`.

**Mitigation:** wait a few seconds, or `lsof -ti:8810,8830 | xargs kill -9`.

### `composeBifurcation` is reverted
There used to be in-page Canvas2D composers for trajectory contact
sheets and bifurcation diagrams. `composer.js`'s `composeBifurcation`
was removed in a later edit. The experiment scripts that reference
it now produce a warning `(skipped bifurcation.png: composeBifurcation:
unknown method)` and continue. The numerical data in `samples.json` /
`*.jsonl` is the canonical result; the diagram was a convenience.

### `q` script depends on `jq`
For pretty output, `helm-lab/q` shells through `jq`. If `jq` isn't
installed, it falls back to raw curl output (still correct, just
unformatted). macOS ships without jq by default; install with
`brew install jq`.

### MCP server uses line-delimited JSON
This is the standard MCP stdio transport, but some clients expect
Content-Length-framed messages (LSP-style). If you connect a
non-Claude MCP client and it doesn't see responses, that's why.

### Image returns are inline base64
`helm_render`'s response includes the PNG as base64 in the MCP image
content block. For 1024x512 PNGs that's ~250 KB of base64 per call.
This is fine for one-off views but expensive if a long conversation
calls `helm_render` dozens of times — context budget burn.

## Engine code I've patched (not user-revertable)

Three engine fixes landed alongside the harness; they're in the
`v4-physics/sim-engine.js` source as of the helm-lab v0 commit:

1. **Stub `let` declarations** for 9 dataset load promises that were
   referenced in `init()`'s `Promise.all` but never declared
   (`salinityLoadPromise`, `windLoadPromise`, …). Without these, init
   threw `ReferenceError` and the engine never started.

2. **`COPY_DST` flag** added to `gpuPsiBuf`, `gpuZetaBuf`,
   `gpuZetaNewBuf`. Without it, the engine's `gpuReset()` triggered
   a WebGPU validation error on the first `writeBuffer()`, losing
   the GPU device.

3. **`window.lab` API** itself is the harness's contract surface
   added late in the engine. simamoc has its own equivalent.

Subsequent commits to `composer.js`, `lab.mjs`, `server.mjs`
documenting the bifurcation composer were reverted by the user — the
experiment scripts gracefully tolerate the missing method.

## What I'd fix next

In rough priority order:

1. **Multi-spinup statistics.** R5's bistability finding means every
   single-run threshold is one sample of a distribution. Repeat each
   bifurcation experiment from N=10 cold starts and report mean ± std.

2. **CPU fallback in `lab.step()`.** Currently throws after 20s if
   WebGPU isn't up. Could fall back to `cpuTimestep()` for headless
   environments without GPU.

3. **A reusable bifurcation composer.** Currently each experiment
   ad-hocs its own ffmpeg tile. A canvas-based composer that takes
   a JSONL path and produces a clean phase diagram would replace
   four near-duplicate code paths.

4. **simamoc target validation.** I claimed simamoc works as a target
   based on reading `main.js`. A smoketest that actually runs a
   trajectory against simamoc is overdue.

5. **Snapshot/restore.** `q snapshot` would dump the GPU buffers to
   disk; `q restore` would warm-start from there. Lets bifurcation
   sweeps avoid re-doing the 400k-step spinup each time.

6. **A real workspace split** (Option B from the earlier note)
   when there's a second helm-lab consumer to justify it.
