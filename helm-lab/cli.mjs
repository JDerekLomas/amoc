#!/usr/bin/env node
// helm-lab CLI — drive the v4-physics ocean simulator headlessly.
//
// Usage:
//   node helm-lab/cli.mjs run <plan.json> <outDir>
//   node helm-lab/cli.mjs trajectory --steps 50000 --interval 5000 --views temp,psi --out runs/test
//   node helm-lab/cli.mjs sweep --param windStrength --values 0.5,1,1.5,2 --out runs/wind
//   node helm-lab/cli.mjs render --view temp --out frame.png --steps 10000
//   node helm-lab/cli.mjs diag

import { HelmLab } from './lab.mjs';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

function parseArgs(argv) {
  const out = { _: [], }; let i = 0;
  while (i < argv.length) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const k = a.slice(2);
      const v = (i + 1 < argv.length && !argv[i + 1].startsWith('--')) ? argv[++i] : true;
      out[k] = v;
    } else out._.push(a);
    i++;
  }
  return out;
}

function csv(v, parse = String) {
  if (v == null || v === true) return [];
  return v.split(',').map(s => parse(s.trim()));
}

async function main() {
  const cmd = process.argv[2];
  const args = parseArgs(process.argv.slice(3));
  const verbose = !!args.verbose || !!args.v;

  const lab = new HelmLab({ verbose });
  await lab.start();
  await lab.installComposers();

  // Apply default speed-up so we don't run on the rAF loop forever.
  await lab.setParams({ stepsPerFrame: 50, yearSpeed: 1 });

  try {
    if (cmd === 'run') {
      const planPath = args._[0];
      const outDir = resolve(args._[1] || './helm-lab/runs/run_' + Date.now());
      const plan = JSON.parse(await readFile(planPath, 'utf8'));
      console.log(`[helm-lab] plan: ${planPath} → ${outDir}`);
      await runPlan(lab, plan, outDir);
      console.log(`[helm-lab] done. open ${outDir}`);
    }
    else if (cmd === 'trajectory') {
      const outDir = resolve(args.out || `./helm-lab/runs/traj_${Date.now()}`);
      const totalSteps = parseInt(args.steps || '50000', 10);
      const sampleEvery = parseInt(args.interval || String(Math.max(1000, Math.floor(totalSteps / 20))), 10);
      const views = csv(args.views || 'temp,psi');
      const spinup = parseInt(args.spinup || '0', 10);
      if (args.scenario) await lab.scenario(args.scenario);
      if (args.params) {
        await lab.setParams(JSON.parse(args.params));
      }
      if (spinup) { console.log(`[helm-lab] spin-up ${spinup} steps`); await lab.step(spinup); }
      console.log(`[helm-lab] trajectory: ${totalSteps} steps, sample every ${sampleEvery}, views=${views}`);
      await lab.trajectory({ totalSteps, sampleEvery, views, outDir });
      console.log(`[helm-lab] done. open ${outDir}`);
    }
    else if (cmd === 'sweep') {
      const param = args.param;
      if (!param) throw new Error('--param required');
      const values = csv(args.values, parseFloat);
      const outDir = resolve(args.out || `./helm-lab/runs/sweep_${param}_${Date.now()}`);
      const views = csv(args.views || 'temp,psi');
      const spinup = parseInt(args.spinup || '20000', 10);
      const post = parseInt(args.post || '50000', 10);
      const resetEach = args.resetEach !== 'false';
      console.log(`[helm-lab] sweep: ${param} = [${values.join(', ')}], spinup=${spinup}, post=${post}, views=${views}`);
      await lab.sweep({ param, values, spinupSteps: spinup, postSteps: post, views, outDir, resetEach });
      console.log(`[helm-lab] done. open ${outDir}`);
    }
    else if (cmd === 'render') {
      const view = args.view || 'temp';
      const out = resolve(args.out || `./helm-lab/runs/frame_${view}_${Date.now()}.png`);
      const steps = parseInt(args.steps || '0', 10);
      if (steps > 0) await lab.step(steps);
      await lab.render(out, { view, includeOverlay: !!args.overlay });
      console.log(`[helm-lab] wrote ${out}`);
    }
    else if (cmd === 'diag') {
      const d = await lab.diag({ profiles: !!args.profiles });
      const out = args.out ? resolve(args.out) : null;
      const json = JSON.stringify(d, null, 2);
      if (out) await writeFile(out, json); else console.log(json);
    }
    else if (cmd === 'reset') {
      await lab.reset(); console.log('[helm-lab] reset');
    }
    else if (cmd === 'set') {
      const params = JSON.parse(args.params || '{}');
      await lab.setParams(params);
      console.log('[helm-lab] params set:', params);
    }
    else {
      console.log(`helm-lab — usage:
  cli.mjs run <plan.json> <outDir>
  cli.mjs trajectory [--steps N] [--interval N] [--views temp,psi] [--spinup N] [--scenario name] [--params '{"k":v}'] [--out dir]
  cli.mjs sweep --param NAME --values v1,v2,v3 [--spinup N] [--post N] [--views temp,psi] [--out dir]
  cli.mjs render [--view NAME] [--steps N] [--out file.png] [--overlay]
  cli.mjs diag [--profiles] [--out file.json]
  cli.mjs reset
  cli.mjs set --params '{"windStrength":1.5,"r":0.06}'

Common flags:  -v / --verbose
`);
    }
  } finally {
    await lab.stop();
  }
}

async function runPlan(lab, plan, outDir) {
  await mkdir(outDir, { recursive: true });
  await writeFile(resolve(outDir, 'plan.json'), JSON.stringify(plan, null, 2));
  if (plan.scenario) await lab.scenario(plan.scenario);
  if (plan.reset) await lab.reset();
  if (plan.params) await lab.setParams(plan.params);
  if (plan.spinupSteps) { console.log(`[helm-lab] spin-up ${plan.spinupSteps}`); await lab.step(plan.spinupSteps); }
  if (plan.trajectory) {
    const t = plan.trajectory;
    await lab.trajectory({
      totalSteps: t.totalSteps,
      sampleEvery: t.sampleEvery,
      views: t.views || ['temp'],
      outDir,
    });
  }
  if (plan.sweep) {
    const s = plan.sweep;
    await lab.sweep({
      param: s.param, values: s.values,
      spinupSteps: s.spinupSteps ?? 20000,
      postSteps: s.postSteps ?? 50000,
      views: s.views || ['temp', 'psi'],
      outDir, resetEach: s.resetEach !== false,
    });
  }
}

main().catch(e => { console.error(e); process.exit(1); });
