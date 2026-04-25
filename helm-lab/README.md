# helm-lab

Headless harness for the v4-physics ocean simulator. Drives the same
WebGPU engine the browser uses — same physics, no UI — and exposes it
over a tiny HTTP RPC so experiments can be scripted, screenshotted,
and replayed without ever opening a browser.

## Quick start

```bash
# 1. start the daemon (foreground; backgrounds easily with &)
node helm-lab/cli.mjs serve

# 2. in another shell, drive it
helm-lab/q getParams
helm-lab/q step 5000
helm-lab/q render '"helm-lab/runs/x.png"' '{"view":"temp"}'
helm-lab/q diag '{}'

# 3. when done
helm-lab/q --shutdown
```

That's the whole interface. WebGPU stays warm in the daemon's Chromium
between calls, so each call is 10–1000 ms (mostly the actual work)
instead of the 3–5 s of cold-start every time.

## Why a daemon

| call shape | cold one-shot | hot daemon |
|---|---|---|
| `diag` | ~5 s (Chromium boot + tiny work) | **~30 ms** |
| `step(1000)` | ~6 s | **~900 ms** |
| `render('temp')` | ~6 s | **~700 ms** |
| `setParams` | ~5 s | **~10 ms** |

The cold-start tax is paid once when the daemon starts. After that the
ratio you care about is *work / work + ~30 ms*. For tight iterative
loops (set → step → render → set → step → render…) this is the
difference between feeling responsive and feeling stuck.

## Three clients

Pick the one that fits.

### 1. `helm-lab/q` — shell, fastest

```bash
helm-lab/q step 1000
helm-lab/q setParams '{"windStrength":1.5,"r":0.06}'
helm-lab/q render '"/tmp/x.png"' '{"view":"temp"}'
helm-lab/q diag '{"profiles":true}' | jq '.maxVel'
helm-lab/q --health
helm-lab/q --shutdown
```

No Node startup; just `curl`. ~30–50 ms per call beyond the work itself.
Pretty-prints with `jq` if installed.

### 2. `helm-lab/cli.mjs` — Node CLI, full UX

```bash
node helm-lab/cli.mjs render --view temp --steps 5000 --out f.png
node helm-lab/cli.mjs trajectory --steps 100000 --interval 5000 --views temp,psi
node helm-lab/cli.mjs sweep --param windStrength --values 0.5,1,1.5,2 \
                             --spinup 10000 --post 30000 --views temp,psi
node helm-lab/cli.mjs diag --profiles
node helm-lab/cli.mjs run plan.json runs/exp_001
```

Auto-detects the daemon and uses it. Falls back to one-shot if no daemon
is up (slower, but zero setup). Adds ~1.5 s of Node startup per call on
macOS — fine for one-off commands, slow for tight loops.

### 3. Direct HTTP RPC

```bash
curl -X POST http://127.0.0.1:8830/rpc \
  -H 'content-type: application/json' \
  -d '{"method":"step","args":[5000]}'
```

Wire format: `{ method, args: [...] }` → `{ ok, result, ms }` or
`{ ok:false, error, stack }`. Methods are anything on the `HelmLab`
class: `reset`, `setParams`, `step`, `diag`, `fields`, `render`,
`setView`, `scenario`, `pause`, `resume`, `getParams`, `trajectory`,
`sweep`.

## What the harness produces

Every experiment writes:

- `frames/t<step>_<view>.png` — clean 1024×512 captures of each view at
  each sample (no HUD, no overlays — just the field).
- `diag.jsonl` — one line per sample, all key diagnostics
  (KE, AMOC, max velocity, global/tropical/polar SST, ACC, ice area,
  gyre psi range, season, simYears).
- `samples.json` — index of every sample with its frame paths and
  diagnostic snapshot.
- `contact-sheet.png` — **one image** showing all sample times × views
  in a tiled grid, with each cell annotated with timestamp + key
  diagnostics. The single most useful output.
- `summary.png` — sparklines of KE / AMOC / global SST / tropical SST /
  polar SST / ice area over the whole run, in the bridge aesthetic.
- `sweep-sheet.png` (sweeps) — analogous, one column per parameter
  value.

## Commands

### Render one frame

```bash
helm-lab/q render '"out.png"' '{"view":"temp"}'
# or via CLI:
node helm-lab/cli.mjs render --view temp --steps 5000 --out out.png
```

Views: `temp`, `psi`, `vort`, `speed`, `deeptemp`, `deepflow`, `depth`,
`sal`, `density`.

### Trajectory: sample a run over time

```bash
node helm-lab/cli.mjs trajectory \
  --steps 100000 \
  --interval 5000 \
  --views temp,psi \
  --spinup 20000 \
  --out helm-lab/runs/long_run
```

Produces 21 samples, one contact-sheet PNG, one summary PNG, full diag
JSONL.

### Parameter sweep

```bash
node helm-lab/cli.mjs sweep \
  --param windStrength \
  --values 0.5,1,1.5,2 \
  --spinup 10000 \
  --post 30000 \
  --views temp,psi
```

`--param` is anything in `HelmLab.setParams`: `windStrength`, `r`, `A`,
`freshwaterForcing`, `globalTempOffset`, `S_solar`, `kappa_diff`,
`alpha_T`, etc.

### Plan files

```json
{
  "scenario": "drake-open",
  "params": { "windStrength": 1.0, "r": 0.04 },
  "spinupSteps": 50000,
  "trajectory": {
    "totalSteps": 200000,
    "sampleEvery": 10000,
    "views": ["temp", "psi", "deepflow"]
  }
}
```

```bash
node helm-lab/cli.mjs run myplan.json helm-lab/runs/exp_001
```

### Benchmark the daemon

```bash
node helm-lab/bench.mjs
```

Useful for confirming the daemon is healthy and seeing per-op latencies.

## Architecture

```
                       browser users
                            │
                            ▼
           ┌──────────────────────────┐
           │   v4-physics/console.html │  ← user-facing bridge UI
           │   v4-physics/index.html   │
           └─────────────┬────────────┘
                         │  loads
                         ▼
           ┌──────────────────────────┐
           │   v4-physics/sim-engine.js │  ← single source of truth
           │   (WebGPU + WGSL physics)   │
           └─────────────┬────────────┘
                         │  same engine
                         ▼
           ┌──────────────────────────┐
           │   helm-lab/server.mjs     │  ← daemon: Playwright + RPC
           │   ┌──────────────┐        │
           │   │ headless     │        │
           │   │ Chromium     │        │
           │   │ (--headless=new       │
           │   │  + Apple Metal GPU)   │
           │   └──────────────┘        │
           └─────────────┬────────────┘
                         │  HTTP /rpc
                         ▼
           ┌─────────────────────────────────┐
           │   q  ·  cli.mjs  ·  bench.mjs   │  ← clients
           └─────────────────────────────────┘
```

The browser version and the harness load the **same** `sim-engine.js`.
There is no second engine to maintain.

## Engine fixes that landed alongside

Three bugs blocked headless operation:

1. `salinityLoadPromise` and 8 sibling load-promise variables were
   referenced in `init()`'s `Promise.all` but never declared — so init
   threw `ReferenceError` and the engine never started. Stubbed as
   `Promise.resolve()`.
2. `gpuPsiBuf` / `gpuZetaBuf` / `gpuZetaNewBuf` were created without the
   `COPY_DST` usage flag, but the engine's reset path calls
   `writeBuffer` on them. WebGPU validation killed the device on the
   first reset. Added the flag.
3. WebGPU canvases don't expose pixels to 2D contexts (`drawImage`
   returns blank), so the original render path produced empty PNGs.
   Switched to `page.screenshot({clip})` after temporarily hiding the
   HUD/overlays — the Chromium compositor reads WebGPU canvases
   correctly.

All three are in `v4-physics/sim-engine.js`; the browser benefits from
the same fixes.

## Limitations / caveats

- **Single page, single state.** The daemon runs one Playwright page;
  experiments are sequential. Parallelism would need multiple daemons on
  different ports.
- **CPU fallback not supported by the harness.** If WebGPU isn't
  available, the engine's CPU path runs in the browser tab but the
  `lab.step()` API throws. This isn't relevant on macOS with WebGPU; on
  other systems we'd need to extend `ensureReady`.
- **`q` is bash + curl + jq.** Designed for macOS / Linux dev
  machines; not portable to plain Windows shells.
- **`render`'s "psi" view normalization is sometimes coarse.** The
  engine computes `absMax` from a JS-side readback that isn't
  guaranteed fresh after off-loop ticks. The image is correct in shape
  but the colormap range can be off when stepping from a paused state.

## Next

- **Daemon resume across PID changes.** Currently the daemon's state
  vanishes when it's restarted. A `q snapshot` / `q restore` pair
  serializing the GPU buffers to disk would make sweeps resilient to
  Chromium hiccups.
- **MCP server wrapper.** The daemon already has the right shape for
  this; ~150 lines of MCP boilerplate would expose it as Claude tools.
  Worth doing once the use case justifies the install friction.
