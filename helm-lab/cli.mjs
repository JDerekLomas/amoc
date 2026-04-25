#!/usr/bin/env node
// helm-lab CLI — drive the v4-physics ocean simulator headlessly.
//
// Two modes, transparently:
//
//   1. Daemon mode (preferred). Start once: `helm-lab serve`. Subsequent
//      commands connect over loopback HTTP to the warm Chromium and
//      complete in ~50ms instead of paying a 3-5s boot tax per call.
//   2. One-shot fallback. If no daemon is detected, the CLI spawns a
//      private Chromium for the duration of the command. Slower but
//      requires zero setup.
//
// Daemon management:  helm-lab serve [--port N]   foreground daemon
//                     helm-lab status              query daemon
//                     helm-lab stop                graceful exit
//
// Experiments:        helm-lab render --view temp --steps 5000 --out f.png
//                     helm-lab trajectory --steps 100000 --interval 5000
//                     helm-lab sweep --param windStrength --values 0.5,1,1.5,2
//                     helm-lab diag [--profiles]
//                     helm-lab reset
//                     helm-lab set --params '{"windStrength":1.5}'
//                     helm-lab run plan.json [outDir]
//                     helm-lab eval --js 'lab.getParams().simTime'

import { HelmLab } from './lab.mjs';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const HERE = dirname(fileURLToPath(import.meta.url));
const DEFAULT_PORT = Number(process.env.HELM_LAB_PORT || 8830);

// ---- arg parsing ----
function parseArgs(argv) {
  const out = { _: [] }; let i = 0;
  while (i < argv.length) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const k = a.slice(2);
      const v = (i + 1 < argv.length && !argv[i + 1].startsWith('--')) ? argv[++i] : true;
      out[k] = v;
    } else if (a.startsWith('-') && a.length === 2) {
      out[a.slice(1)] = true;
    } else out._.push(a);
    i++;
  }
  return out;
}
const csv = (v, parse = String) => v == null || v === true ? [] : v.split(',').map(s => parse(s.trim()));

// ---- daemon detection / RPC client ----
async function pingDaemon(port = DEFAULT_PORT, timeoutMs = 1500) {
  // /healthz is served without page.evaluate, so it's instant even when the
  // daemon is mid-step. We still set a generous timeout in case the host is
  // under load. fetch keep-alive is disabled so the CLI process can exit.
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const r = await fetch(`http://127.0.0.1:${port}/healthz`, { signal: ctl.signal, keepalive: false });
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; } finally { clearTimeout(t); }
}

async function rpc(method, args = [], port = DEFAULT_PORT) {
  const r = await fetch(`http://127.0.0.1:${port}/rpc`, {
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

// A daemon-or-spawned facade: same surface as HelmLab but routes through
// HTTP if a daemon is up, else spins up a private one for the duration.
class Conn {
  constructor() { this.kind = null; this.lab = null; }
  async open(port = DEFAULT_PORT, opts = {}) {
    const h = await pingDaemon(port);
    if (h) {
      this.kind = 'daemon'; this.port = port; this.health = h;
      if (!opts.quiet) console.error(`[helm-lab] using daemon (pid ${h.pid}, useGPU=${h.useGPU}, step=${h.totalSteps})`);
      return this;
    }
    this.kind = 'oneshot';
    if (!opts.quiet) console.error('[helm-lab] no daemon — running one-shot (start `helm-lab serve` to keep state warm)');
    this.lab = new HelmLab({ verbose: !!opts.verbose });
    await this.lab.start();
    await this.lab.installComposers();
    await this.lab.setParams({ stepsPerFrame: 50, yearSpeed: 1 });
    return this;
  }
  async close() { if (this.kind === 'oneshot' && this.lab) await this.lab.stop(); }
  async call(m, args = []) {
    if (this.kind === 'daemon') return rpc(m, args, this.port);
    return this.lab[m](...args);
  }
  // Convenience wrappers, same names as HelmLab.
  reset()                    { return this.call('reset'); }
  setParams(p)               { return this.call('setParams', [p]); }
  step(n)                    { return this.call('step', [n]); }
  diag(o = {})               { return this.call('diag', [o]); }
  fields(n)                  { return this.call('fields', [n]); }
  render(out, o = {})        { return this.call('render', [out, o]); }
  setView(v)                 { return this.call('setView', [v]); }
  scenario(n)                { return this.call('scenario', [n]); }
  pause()                    { return this.call('pause'); }
  resume()                   { return this.call('resume'); }
  getParams()                { return this.call('getParams'); }
  trajectory(a)              { return this.call('trajectory', [a]); }
  sweep(a)                   { return this.call('sweep', [a]); }
}

// ---- commands ----
async function cmdServe(args) {
  // Replace this process with the daemon so Ctrl-C / signals work cleanly.
  const env = { ...process.env };
  if (args.port) env.HELM_LAB_PORT = String(args.port);
  if (args.verbose || args.v) env.HELM_LAB_VERBOSE = '1';
  const child = spawn(process.execPath, [resolve(HERE, 'server.mjs')], { stdio: 'inherit', env });
  child.on('exit', code => process.exit(code ?? 0));
  process.on('SIGINT', () => child.kill('SIGINT'));
  process.on('SIGTERM', () => child.kill('SIGTERM'));
}

async function cmdStop(args) {
  const port = Number(args.port || DEFAULT_PORT);
  try {
    const r = await fetch(`http://127.0.0.1:${port}/shutdown`, { method: 'POST' });
    const j = await r.json();
    console.log(j.ok ? '[helm-lab] daemon stopped' : '[helm-lab] daemon: ' + JSON.stringify(j));
  } catch (e) {
    console.error('[helm-lab] no daemon at :' + port);
    process.exit(1);
  }
}

async function cmdStatus(args) {
  const port = Number(args.port || DEFAULT_PORT);
  const h = await pingDaemon(port, 1000);
  if (!h) { console.error('[helm-lab] no daemon at :' + port); process.exit(1); }
  console.log(JSON.stringify(h, null, 2));
}

async function cmdRender(conn, args) {
  const view = args.view || 'temp';
  const out = resolve(args.out || `./helm-lab/runs/frame_${view}_${Date.now()}.png`);
  const steps = parseInt(args.steps || '0', 10);
  if (steps > 0) await conn.step(steps);
  await conn.render(out, { view, includeOverlay: !!args.overlay });
  console.log(out);
}

async function cmdTrajectory(conn, args) {
  const outDir = resolve(args.out || `./helm-lab/runs/traj_${Date.now()}`);
  const totalSteps = parseInt(args.steps || '50000', 10);
  const sampleEvery = parseInt(args.interval || String(Math.max(1000, Math.floor(totalSteps / 20))), 10);
  const views = csv(args.views || 'temp,psi');
  const spinup = parseInt(args.spinup || '0', 10);
  if (args.scenario) await conn.scenario(args.scenario);
  if (args.params) await conn.setParams(JSON.parse(args.params));
  if (spinup) { console.error(`[helm-lab] spin-up ${spinup}`); await conn.step(spinup); }
  console.error(`[helm-lab] trajectory steps=${totalSteps} interval=${sampleEvery} views=${views}`);
  await conn.trajectory({ totalSteps, sampleEvery, views, outDir });
  console.log(outDir);
}

async function cmdSweep(conn, args) {
  if (!args.param) throw new Error('--param required');
  const values = csv(args.values, parseFloat);
  const outDir = resolve(args.out || `./helm-lab/runs/sweep_${args.param}_${Date.now()}`);
  const views = csv(args.views || 'temp,psi');
  const spinup = parseInt(args.spinup || '20000', 10);
  const post = parseInt(args.post || '50000', 10);
  console.error(`[helm-lab] sweep ${args.param}=[${values.join(', ')}] spinup=${spinup} post=${post} views=${views}`);
  await conn.sweep({ param: args.param, values, spinupSteps: spinup, postSteps: post, views, outDir, resetEach: args.resetEach !== 'false' });
  console.log(outDir);
}

async function cmdDiag(conn, args) {
  const d = await conn.diag({ profiles: !!args.profiles });
  const json = JSON.stringify(d, null, 2);
  if (args.out) await writeFile(resolve(args.out), json);
  else console.log(json);
}

async function cmdRun(conn, args) {
  const planPath = args._[0];
  if (!planPath) throw new Error('usage: run <plan.json> [outDir]');
  const outDir = resolve(args._[1] || './helm-lab/runs/run_' + Date.now());
  const plan = JSON.parse(await readFile(planPath, 'utf8'));
  await mkdir(outDir, { recursive: true });
  await writeFile(resolve(outDir, 'plan.json'), JSON.stringify(plan, null, 2));
  if (plan.scenario) await conn.scenario(plan.scenario);
  if (plan.reset) await conn.reset();
  if (plan.params) await conn.setParams(plan.params);
  if (plan.spinupSteps) { console.error(`[helm-lab] spinup ${plan.spinupSteps}`); await conn.step(plan.spinupSteps); }
  if (plan.trajectory) {
    const t = plan.trajectory;
    await conn.trajectory({ totalSteps: t.totalSteps, sampleEvery: t.sampleEvery, views: t.views || ['temp'], outDir });
  }
  if (plan.sweep) {
    const s = plan.sweep;
    await conn.sweep({ param: s.param, values: s.values, spinupSteps: s.spinupSteps ?? 20000, postSteps: s.postSteps ?? 50000, views: s.views || ['temp', 'psi'], outDir, resetEach: s.resetEach !== false });
  }
  console.log(outDir);
}

function help() {
  console.error(`helm-lab — headless ocean simulator harness

Daemon:
  serve [--port N] [-v]            run long-lived daemon (foreground)
  status [--port N]                show daemon health JSON
  stop [--port N]                  shut down daemon

Experiments (use daemon if running, else one-shot):
  render --view <v> [--steps N] [--out PATH] [--overlay]
  trajectory --steps N [--interval N] [--views v,v] [--spinup N] [--scenario name] [--params JSON] [--out DIR]
  sweep --param NAME --values v,v,v [--spinup N] [--post N] [--views v,v] [--out DIR]
  diag [--profiles] [--out PATH]
  reset
  set --params '{"windStrength":1.5,"r":0.06}'
  run <plan.json> [outDir]

Views:  temp · psi · vort · speed · deeptemp · deepflow · depth · sal · density
Scenarios: present · drake-open · drake-close · panama-open · greenland · iceage
`);
}

async function main() {
  const cmd = process.argv[2];
  const args = parseArgs(process.argv.slice(3));

  if (!cmd || cmd === 'help' || cmd === '--help' || cmd === '-h') return help();
  if (cmd === 'serve')  return cmdServe(args);
  if (cmd === 'stop')   return cmdStop(args);
  if (cmd === 'status') return cmdStatus(args);

  const conn = new Conn();
  await conn.open(DEFAULT_PORT, { verbose: !!(args.verbose || args.v) });
  try {
    if (cmd === 'render') await cmdRender(conn, args);
    else if (cmd === 'trajectory') await cmdTrajectory(conn, args);
    else if (cmd === 'sweep') await cmdSweep(conn, args);
    else if (cmd === 'diag') await cmdDiag(conn, args);
    else if (cmd === 'reset') { await conn.reset(); console.error('[helm-lab] reset'); }
    else if (cmd === 'set') {
      if (!args.params) throw new Error('--params JSON required');
      const p = JSON.parse(args.params); await conn.setParams(p); console.log(JSON.stringify(p));
    }
    else if (cmd === 'run') await cmdRun(conn, args);
    else { help(); process.exit(2); }
  } finally {
    await conn.close();
  }
}

main()
  .then(() => process.exit(0))   // node fetch keep-alive can delay exit by ~60s otherwise
  .catch(e => { console.error('[helm-lab] error:', e.message); if (e.stack && process.env.HELM_LAB_VERBOSE) console.error(e.stack); process.exit(1); });
