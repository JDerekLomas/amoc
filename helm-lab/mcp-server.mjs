#!/usr/bin/env node
// helm-lab/mcp-server.mjs
// ---------------------------------------------------------------
// Model Context Protocol (MCP) server that exposes the helm-lab
// daemon as a set of native Claude Code tools. The simulator becomes
// callable like Bash or Read — no shell wrapping needed.
//
// Wire shape: line-delimited JSON-RPC 2.0 over stdio (the standard
// MCP stdio transport). No SDK dependency; this is ~200 lines of
// hand-rolled protocol.
//
// Behavior:
//   - On startup, this process attempts to talk to a running helm-lab
//     daemon on $HELM_LAB_PORT (default 8830). If none is running it
//     auto-spawns `node helm-lab/cli.mjs serve` and waits up to 30s
//     for it to come up.
//   - All tools proxy through to the daemon's /rpc endpoint.
//   - helm_render returns an inline image content block so I see the
//     PNG without an extra Read call.
//
// To install:
//   Add to ~/.claude.json under "mcpServers":
//
//     "helm-lab": {
//       "type": "stdio",
//       "command": "node",
//       "args": ["/abs/path/to/amoc/helm-lab/mcp-server.mjs"]
//     }
//
//   Then restart Claude Code. Tools appear as mcp__helm-lab__helm_step etc.

import { spawn } from 'node:child_process';
import { readFile, mkdir } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { setTimeout as wait } from 'node:timers/promises';

const HERE = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.HELM_LAB_PORT || 8830);
const URL  = `http://127.0.0.1:${PORT}`;

// ===== logging — must go to stderr so it doesn't corrupt the JSON-RPC channel
function log(...a) { console.error('[helm-lab mcp]', ...a); }

// ===== daemon health/spawn

async function pingDaemon(timeoutMs = 1500) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(`${URL}/healthz`, { signal: ctl.signal, keepalive: false });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; } finally { clearTimeout(t); }
}

async function rpc(method, args = []) {
  const r = await fetch(`${URL}/rpc`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ method, args }),
  });
  const j = await r.json();
  if (!j.ok) {
    const e = new Error(j.error || 'rpc failed');
    if (j.stack) e.stack = j.stack;
    throw e;
  }
  return j.result;
}

let daemonChild = null;

async function ensureDaemon() {
  let h = await pingDaemon();
  if (h) { log(`daemon already up (pid ${h.pid}, useGPU=${h.useGPU})`); return h; }

  log('no daemon — spawning one...');
  daemonChild = spawn(process.execPath, [join(HERE, 'server.mjs')], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, HELM_LAB_PORT: String(PORT) },
    detached: false,
  });
  daemonChild.stderr.on('data', d => log('[daemon]', d.toString().trim()));
  daemonChild.stdout.on('data', d => log('[daemon]', d.toString().trim()));
  daemonChild.on('exit', (code) => { log(`daemon exited code=${code}`); daemonChild = null; });

  for (let i = 0; i < 60; i++) {
    await wait(500);
    h = await pingDaemon();
    if (h) { log(`daemon up after ${(i + 1) * 0.5}s (pid ${h.pid}, useGPU=${h.useGPU})`); return h; }
  }
  throw new Error('daemon failed to start within 30s');
}

// ===== JSON-RPC framing — line-delimited JSON over stdin/stdout

function send(msg) {
  process.stdout.write(JSON.stringify(msg) + '\n');
}
function ok(id, result)  { send({ jsonrpc: '2.0', id, result }); }
function err(id, code, message) { send({ jsonrpc: '2.0', id, error: { code, message } }); }

// ===== tool catalog

const TOOLS = [
  {
    name: 'helm_health',
    description: 'Get the helm-lab daemon health snapshot (engine status, current step, simYears, paused, etc.). Cheap; cached on the server, returns instantly even mid-step.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    fn: async () => await pingDaemon(2000),
  },
  {
    name: 'helm_get_params',
    description: 'Snapshot every tunable parameter and read-only state field on the simulator (beta, r, A, windStrength, freshwater, solar, totalSteps, simTime, useGPU, NX, NY, …). Source of truth for what knobs exist.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    fn: async () => await rpc('getParams'),
  },
  {
    name: 'helm_set_params',
    description: 'Set any subset of the simulator\'s tunable parameters. Returns the post-update params.',
    inputSchema: {
      type: 'object',
      properties: {
        params: {
          type: 'object',
          description: 'Object of {paramName: value}. Examples: {"windStrength": 1.5}, {"freshwaterForcing": 0.4, "r": 0.06}, {"globalTempOffset": 4}. See helm_get_params for the full list.',
          additionalProperties: true,
        },
      },
      required: ['params'],
      additionalProperties: false,
    },
    fn: async ({ params }) => await rpc('setParams', [params]),
  },
  {
    name: 'helm_step',
    description: 'Run exactly N solver steps deterministically. One step is dt=5e-5 model time; T_YEAR=10, so ~200,000 steps ≈ 1 model year. n in [1, 5000000].',
    inputSchema: {
      type: 'object',
      properties: { n: { type: 'integer', minimum: 1, maximum: 5000000 } },
      required: ['n'], additionalProperties: false,
    },
    fn: async ({ n }) => await rpc('step', [n]),
  },
  {
    name: 'helm_diag',
    description: 'Diagnostic snapshot from current fields. Returns step, simYears, KE, maxVel, AMOC, ACC, global/tropical/polar SST, sea-ice cells, gyre psi range, etc. Optionally include zonal-mean profiles by lat.',
    inputSchema: {
      type: 'object',
      properties: { profiles: { type: 'boolean', default: false } },
      additionalProperties: false,
    },
    fn: async ({ profiles = false } = {}) => await rpc('diag', [{ profiles }]),
  },
  {
    name: 'helm_render',
    description: 'Render the current state as a clean 1024x512 PNG of the chosen view. Returns an inline image so I can see it directly. Views: temp, psi, vort, speed, deeptemp, deepflow, depth, sal, density.',
    inputSchema: {
      type: 'object',
      properties: {
        view: {
          type: 'string',
          enum: ['temp', 'psi', 'vort', 'speed', 'deeptemp', 'deepflow', 'depth', 'sal', 'density'],
          default: 'temp',
        },
        outPath: {
          type: 'string',
          description: 'Optional file path. If omitted, writes to a temp file and still returns the image inline.',
        },
        includeOverlay: {
          type: 'boolean', default: false,
          description: 'If true, includes the bridge HUD (corner brackets, lat/lon ticks, compass) in the capture.',
        },
      },
      additionalProperties: false,
    },
    fn: async ({ view = 'temp', outPath, includeOverlay = false } = {}) => {
      const path = outPath || join(tmpdir(), `helm-render-${Date.now()}-${view}.png`);
      await mkdir(dirname(path), { recursive: true });
      await rpc('render', [path, { view, includeOverlay }]);
      return { path, view, includeOverlay };
    },
    // Custom result handler — return image content block.
    resultToContent: async (r) => {
      const buf = await readFile(r.path);
      return [
        { type: 'image', data: buf.toString('base64'), mimeType: 'image/png' },
        { type: 'text', text: `view=${r.view} path=${r.path}` },
      ];
    },
  },
  {
    name: 'helm_set_view',
    description: 'Switch the rendered view (does not itself trigger a redraw — call helm_render after).',
    inputSchema: {
      type: 'object',
      properties: { view: { type: 'string', enum: ['temp', 'psi', 'vort', 'speed', 'deeptemp', 'deepflow', 'depth', 'sal', 'density'] } },
      required: ['view'], additionalProperties: false,
    },
    fn: async ({ view }) => await rpc('setView', [view]),
  },
  {
    name: 'helm_reset',
    description: 'Re-initialize the simulator state (psi, zeta, temp, deepTemp, deepPsi, deepZeta) to defaults. Mask is preserved. totalSteps is zeroed.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    fn: async () => await rpc('reset'),
  },
  {
    name: 'helm_scenario',
    description: 'Load a paleoclimate scenario (modifies the mask + resets state). Names: present, drake-open, drake-close, panama-open, greenland, iceage.',
    inputSchema: {
      type: 'object',
      properties: { name: { type: 'string', enum: ['present', 'drake-open', 'drake-close', 'panama-open', 'greenland', 'iceage'] } },
      required: ['name'], additionalProperties: false,
    },
    fn: async ({ name }) => await rpc('scenario', [name]),
  },
  {
    name: 'helm_pause',
    description: 'Pause the simulator\'s rAF loop. Does not affect helm_step which always works.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    fn: async () => await rpc('pause'),
  },
  {
    name: 'helm_resume',
    description: 'Resume the simulator\'s rAF loop.',
    inputSchema: { type: 'object', properties: {}, additionalProperties: false },
    fn: async () => await rpc('resume'),
  },
  {
    name: 'helm_sweep',
    description: 'Server-side parameter sweep: for each value, set the param, optionally settle, then step and snapshot diag. Useful for quick what-if scans.',
    inputSchema: {
      type: 'object',
      properties: {
        knob: { type: 'string', description: 'param name (e.g. windStrength, r, freshwaterForcing)' },
        values: { type: 'array', items: { type: 'number' } },
        stepsPerPoint: { type: 'integer', default: 50000 },
        settleSteps: { type: 'integer', default: 0 },
        resetBetween: { type: 'boolean', default: false },
      },
      required: ['knob', 'values'], additionalProperties: false,
    },
    fn: async (opts) => await rpc('sweep', [opts.knob, opts.values, {
      stepsPerPoint: opts.stepsPerPoint, settleSteps: opts.settleSteps, resetBetween: opts.resetBetween,
    }]),
  },
];

// ===== JSON-RPC dispatch

async function handle(req) {
  const { id, method, params } = req;
  // Notifications (no id) — log and ignore
  if (id == null) { log('notification:', method); return; }

  if (method === 'initialize') {
    return ok(id, {
      protocolVersion: '2024-11-05',
      capabilities: { tools: {} },
      serverInfo: { name: 'helm-lab', version: '0.1' },
    });
  }
  if (method === 'tools/list') {
    return ok(id, {
      tools: TOOLS.map(t => ({ name: t.name, description: t.description, inputSchema: t.inputSchema })),
    });
  }
  if (method === 'tools/call') {
    const { name, arguments: args = {} } = params || {};
    const tool = TOOLS.find(t => t.name === name);
    if (!tool) return err(id, -32601, `unknown tool: ${name}`);
    try {
      const raw = await tool.fn(args);
      const content = tool.resultToContent
        ? await tool.resultToContent(raw)
        : [{ type: 'text', text: JSON.stringify(raw, null, 2) }];
      return ok(id, { content, isError: false });
    } catch (e) {
      log('tool error:', tool.name, e.message);
      return ok(id, {
        content: [{ type: 'text', text: `Error: ${e.message}\n${e.stack ? e.stack.slice(0, 1000) : ''}` }],
        isError: true,
      });
    }
  }
  err(id, -32601, `method not found: ${method}`);
}

// ===== stdin loop

let buf = '';
process.stdin.on('data', chunk => {
  buf += chunk.toString('utf8');
  let nl;
  while ((nl = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (!line) continue;
    let req;
    try { req = JSON.parse(line); } catch (e) { log('parse error:', e.message, 'line:', line.slice(0, 200)); continue; }
    handle(req).catch(e => log('handle threw:', e.message));
  }
});
process.stdin.on('end', () => process.exit(0));
process.on('SIGINT',  () => process.exit(0));
process.on('SIGTERM', () => process.exit(0));

// ===== main

ensureDaemon()
  .then(h => log(`ready · port ${h?.port} · pid ${h?.pid} · useGPU=${h?.useGPU}`))
  .catch(e => { log('FATAL:', e.message); process.exit(1); });
