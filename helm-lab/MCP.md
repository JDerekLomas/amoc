# helm-lab as a Claude Code MCP server

`helm-lab/mcp-server.mjs` exposes the simulator as a set of native
Claude Code tools so you (or I) can call `helm_step` / `helm_render`
in conversation without shelling out to Bash.

The MCP server is a thin proxy over the running daemon. If no daemon
is up when the MCP server starts, it spawns one automatically.

## Install (Claude Code)

Add this to your `~/.claude.json` under `"mcpServers"`:

```jsonc
{
  "mcpServers": {
    "helm-lab": {
      "type": "stdio",
      "command": "node",
      "args": ["/Users/dereklomas/lukebarrington/amoc/helm-lab/mcp-server.mjs"]
    }
  }
}
```

(Replace the absolute path with wherever this repo lives on your
machine.)

Restart Claude Code. The tools appear under `mcp__helm-lab__*`.

## Tools

| name              | what it does                                              |
|-------------------|-----------------------------------------------------------|
| `helm_health`     | daemon status (instant; cached)                           |
| `helm_get_params` | full parameter snapshot                                   |
| `helm_set_params` | set any subset of params                                  |
| `helm_step`       | run N solver steps                                        |
| `helm_diag`       | diagnostic snapshot (KE, AMOC, SST, ice, …)               |
| `helm_render`     | PNG of any view, returned **inline as image content**     |
| `helm_set_view`   | switch view without rendering                             |
| `helm_reset`      | reset fields to defaults                                  |
| `helm_scenario`   | load paleoclimate scenario                                |
| `helm_pause` / `helm_resume` | toggle the rAF loop                            |
| `helm_sweep`      | server-side parameter sweep                               |

`helm_render` returns the rendered image as an MCP `image` content
block, so I see the field directly without an extra `Read` call.

## Example session

Once installed, a typical loop looks like:

```
[me] mcp__helm-lab__helm_set_params({ params: { freshwaterForcing: 0.3 } })
       → { ok, ... }
[me] mcp__helm-lab__helm_step({ n: 30000 })
       → { step, simYears, simTime }
[me] mcp__helm-lab__helm_render({ view: 'temp' })
       → <inline 1024x512 PNG>
[me] mcp__helm-lab__helm_diag({})
       → { KE, amoc, globalSST, … }
```

Each call is ~30–1000 ms (the GPU work itself dominates; the MCP layer
is microseconds). No browser, no Bash, no waiting on Chromium boot.

## Compared to the other clients

| client          | per-call overhead         | best for                       |
|-----------------|---------------------------|--------------------------------|
| `q` (bash)      | ~30–50 ms                 | shell scripts, CI              |
| `cli.mjs`       | ~1.5 s (Node startup)     | one-off humans-on-keyboard     |
| MCP             | ~5 ms (in-process)        | **interactive sessions**       |
| direct `/rpc`   | ~20 ms                    | other tools, custom clients    |

## Lifecycle

- The MCP server's lifetime is tied to Claude Code's session.
- The daemon's lifetime is independent — it stays up across MCP
  restarts (if you launched it via `cli.mjs serve`), or it dies when
  the MCP server it spawned exits.
- To keep the daemon warm across many sessions, run
  `node helm-lab/cli.mjs serve &` once and forget about it. The MCP
  server will detect and reuse it.

## Smoke test

```bash
node helm-lab/mcp-smoketest.mjs
```

Walks `initialize → tools/list → helm_health → helm_render` against
the MCP server in a child process. Validates protocol + daemon
proxy + image return. Run it after any change to `mcp-server.mjs`.
