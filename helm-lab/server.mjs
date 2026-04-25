// helm-lab daemon — long-lived HelmLab wrapped in a minimal HTTP RPC server.
//
// Why: the bottleneck in iterative experiments is the 3-5s Chromium boot.
// A daemon keeps Chromium + the simulator state warm; clients call into it
// over loopback HTTP and pay only ~50ms per call.
//
// Wire protocol (intentionally tiny):
//   GET  /healthz      → { ok, useGPU, totalSteps, simYears, NX, port, pid }
//   POST /rpc          → { method, args } → { ok, result } or { ok:false, error }
//   POST /shutdown     → { ok:true } and the daemon exits
//
// Bind is 127.0.0.1 only — no auth, no remote access by design.

import { createServer } from 'node:http';
import { HelmLab } from './lab.mjs';

const PORT = Number(process.env.HELM_LAB_PORT || 8830);
const VERBOSE = process.env.HELM_LAB_VERBOSE === '1';

console.log(`[helm-lab daemon] starting (pid ${process.pid}, port ${PORT})`);
const lab = new HelmLab({ verbose: VERBOSE });
await lab.start();
await lab.installComposers();
await lab.setParams({ stepsPerFrame: 50, yearSpeed: 1 });
console.log(`[helm-lab daemon] engine ready — useGPU=${(await lab.getParams()).useGPU}`);

// Serialize all RPC calls — single browser page, single sim state.
const serialize = (() => {
  let chain = Promise.resolve();
  return fn => {
    const next = chain.then(() => fn());
    chain = next.catch(() => {});
    return next;
  };
})();

// Cached params snapshot, refreshed after each RPC. /healthz reads this
// without going through page.evaluate, so it's instantaneous even when the
// daemon is mid-step. Refreshed in the background after every RPC.
let lastParams = await lab.getParams();
let lastRefresh = Date.now();
async function refreshParams() {
  try { lastParams = await lab.getParams(); lastRefresh = Date.now(); } catch {}
}

// Bound RPC method table. Anything that takes arrays on the wire passes
// through unchanged thanks to JSON.
const methods = {
  reset:      ()       => lab.reset(),
  setParams:  (p)      => lab.setParams(p),
  step:       (n)      => lab.step(n),
  diag:       (o)      => lab.diag(o || {}),
  fields:     (n)      => lab.fields(n),
  render:     (out, o) => lab.render(out, o || {}),
  setView:    (v)      => lab.setView(v),
  scenario:   (n)      => lab.scenario(n),
  pause:      ()       => lab.pause(),
  resume:     ()       => lab.resume(),
  getParams:  ()       => lab.getParams(),
  trajectory: (a)      => lab.trajectory(a),
  sweep:      (a)      => lab.sweep(a),
};

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  if (!chunks.length) return null;
  const text = Buffer.concat(chunks).toString('utf8');
  try { return JSON.parse(text); } catch { return text; }
}

function send(res, code, obj) {
  res.statusCode = code;
  res.setHeader('content-type', 'application/json');
  res.end(JSON.stringify(obj));
}

let stopping = false;

const server = createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  try {
    if (url.pathname === '/healthz' && req.method === 'GET') {
      // Return cached params without page.evaluate — instant even when the
      // daemon is mid-step. Snapshot age tells the client how stale it is.
      const p = lastParams;
      return send(res, 200, {
        ok: true, version: lab._version || '0.1', port: PORT, pid: process.pid,
        useGPU: p.useGPU, NX: p.NX, NY: p.NY,
        totalSteps: p.totalSteps, simYears: p.simTime / p.T_YEAR,
        showField: p.showField, paused: p.paused,
        snapshotAgeMs: Date.now() - lastRefresh, busy: stopping ? 'stopping' : false,
      });
    }
    if (url.pathname === '/rpc' && req.method === 'POST') {
      const body = await readBody(req);
      if (!body || typeof body.method !== 'string') return send(res, 400, { ok: false, error: 'expected { method, args }' });
      const fn = methods[body.method];
      if (!fn) return send(res, 404, { ok: false, error: 'unknown method: ' + body.method });
      const args = Array.isArray(body.args) ? body.args : [];
      const t0 = Date.now();
      try {
        const result = await serialize(() => fn(...args));
        const ms = Date.now() - t0;
        if (VERBOSE) console.log(`[helm-lab daemon] ${body.method} (${ms}ms)`);
        // Refresh cached params after any state-changing call (background).
        if (body.method !== 'getParams') refreshParams();
        return send(res, 200, { ok: true, result, ms });
      } catch (e) {
        console.error(`[helm-lab daemon] ${body.method} failed:`, e.message);
        return send(res, 500, { ok: false, error: e.message, stack: e.stack });
      }
    }
    if (url.pathname === '/shutdown' && req.method === 'POST') {
      send(res, 200, { ok: true });
      stopping = true;
      console.log('[helm-lab daemon] shutdown requested');
      setTimeout(async () => { await lab.stop(); server.close(); process.exit(0); }, 100);
      return;
    }
    if (url.pathname === '/') {
      res.setHeader('content-type', 'text/html');
      const p = await lab.getParams();
      return res.end(`<!doctype html><meta charset=utf-8><title>helm-lab daemon</title>
<style>body{font:14px/1.5 monospace;background:#03070d;color:#d4dee9;padding:24px}
h1{color:#a4dcf2;font-weight:300;letter-spacing:.04em}
.k{color:#876e3d}.v{color:#5ce0c0}</style>
<h1>helm-lab daemon · pid ${process.pid} · port ${PORT}</h1>
<p><span class=k>useGPU</span> <span class=v>${p.useGPU}</span></p>
<p><span class=k>NX × NY</span> <span class=v>${p.NX} × ${p.NY}</span></p>
<p><span class=k>totalSteps</span> <span class=v>${p.totalSteps}</span></p>
<p><span class=k>simYears</span> <span class=v>${(p.simTime / p.T_YEAR).toFixed(3)}</span></p>
<p><span class=k>showField</span> <span class=v>${p.showField}</span></p>
<p><span class=k>paused</span> <span class=v>${p.paused}</span></p>
<p><a href=/healthz style=color:#d8b06a>/healthz</a> · POST /rpc · POST /shutdown</p>`);
    }
    send(res, 404, { ok: false, error: 'not found' });
  } catch (e) {
    send(res, 500, { ok: false, error: e.message });
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[helm-lab daemon] listening on http://127.0.0.1:${PORT}`);
  console.log(`[helm-lab daemon] try: curl http://127.0.0.1:${PORT}/healthz`);
});

process.on('SIGINT', async () => { console.log('\n[helm-lab daemon] SIGINT'); await lab.stop(); process.exit(0); });
process.on('SIGTERM', async () => { console.log('\n[helm-lab daemon] SIGTERM'); await lab.stop(); process.exit(0); });
