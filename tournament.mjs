#!/usr/bin/env node
/**
 * Physics Tournament: Self-Improving Climate Model
 *
 * Spawns N physics mutation proposals as isolated git worktrees,
 * evaluates each against observations, merges the winner.
 *
 * Each mutation is a concrete code change to simamoc/model.js (WGSL shaders + JS physics),
 * proposed by an AI agent and tested against the 4-tier framework.
 *
 * Usage:
 *   node tournament.mjs [--branches 4] [--spinup 150]
 *
 * Requires: claude (CLI) in PATH for agent spawning
 */

import { execSync, spawn } from 'child_process';
import { mkdirSync, writeFileSync, readFileSync, existsSync } from 'fs';
import { resolve } from 'path';

const ROOT = '/Users/dereklomas/lukebarrington/amoc';
const OUT = './screenshots/tournament';
const N_BRANCHES = parseInt(process.argv.find((_, i, a) => a[i - 1] === '--branches') || '4');
const SPINUP = parseInt(process.argv.find((_, i, a) => a[i - 1] === '--spinup') || '150');

mkdirSync(OUT, { recursive: true });

// ---------------------------------------------------------------------------
// Physics mutation proposals — each is a self-contained hypothesis
// ---------------------------------------------------------------------------
const MUTATIONS = [
  {
    id: 'ekman-heat',
    name: 'Add Ekman heat transport from wind stress',
    hypothesis: 'Wind-driven Ekman transport moves warm water poleward in subtropical gyres. Without it, mid-latitudes are too cold and tropics are too warm.',
    prompt: `You are modifying a WebGPU ocean circulation model to add Ekman heat transport.

The physics code is split across two files:
- simamoc/model.js — contains GPU shader strings (temperatureShaderCode) and CPU fallback (cpuTimestep function)
- simamoc/index.html — contains rendering and UI only

The temperature shader in model.js (var temperatureShaderCode) currently has advection via J(psi, T) but no explicit Ekman transport.

In the real ocean, wind stress drives a surface Ekman layer that transports heat. The Ekman transport is perpendicular to the wind: tau_x drives meridional transport, tau_y drives zonal transport.

Add an Ekman heat transport term to the temperature equation in BOTH the GPU shader (temperatureShaderCode in model.js) AND CPU fallback (cpuTimestep in model.js):
- Compute Ekman velocity: v_ek = tau_x / (rho * f), where f = 2*omega*sin(lat), omega = 7.27e-5
- Add the Ekman heat advection: -v_ek * dT/dy
- Scale appropriately for the non-dimensional model
- Only apply where |lat| > 5° (avoid equatorial singularity in f)

Make the changes to simamoc/model.js. Keep changes minimal and focused.`,
  },
  {
    id: 'wind-curl-realistic',
    name: 'Multi-term wind stress for realistic gyres',
    hypothesis: 'The current cos(3*lat) wind pattern is too simple. A multi-term Fourier expansion matching ERA5 would produce correct Sverdrup transport and gyre structure.',
    prompt: `You are modifying a WebGPU ocean circulation model to improve the wind stress pattern.

The physics code is in simamoc/model.js. Find the wind stress curl computation in the timestep shader (var timestepShaderCode) and the CPU fallback (function cpuWindCurl).

The current formula uses a simple cos(3*phi) pattern. Replace it with a more realistic multi-term expansion:
  tau_x = windStrength * (-0.4*cos(2*lat_rad) + 0.25*sin(lat_rad) + 0.1*sin(3*lat_rad))

This produces:
- Easterlies in the tropics (trade winds)
- Westerlies at 40-60° (roaring forties)
- Polar easterlies
- Stronger Southern Hemisphere westerlies (factor of ~1.5x)

Apply the change in BOTH the GPU shader AND CPU fallback in simamoc/model.js. Keep the windStrength parameter as a multiplier.`,
  },
  {
    id: 'latitude-diffusion',
    name: 'Latitude-dependent thermal diffusion',
    hypothesis: 'Uniform kappa_diff causes too much tropical heat to leak poleward. In reality, diffusion is stronger at high latitudes (mesoscale eddies) and weaker in the tropics.',
    prompt: `You are modifying a WebGPU ocean circulation model to add latitude-dependent thermal diffusion.

The physics code is in simamoc/model.js. Find the thermal diffusion computation in the temperature shader (var temperatureShaderCode, look for "let diff = params.kappaDiff * lapT") and CPU fallback (function cpuTimestep, look for "var diff = kappa_diff * lapT").

Currently: diff = kappa_diff * laplacian(T) — uniform everywhere.

Change to: kappa_effective = kappa_diff * (0.3 + 0.7 * abs(lat) / 80.0)

This makes diffusion:
- 30% of base value at the equator (preserves sharp tropical gradients)
- 100% at 80° latitude (allows polar mixing)
- Linearly interpolated between

Apply in BOTH GPU shader AND CPU fallback in simamoc/model.js. The lat variable is already computed in both paths.`,
  },
  {
    id: 'mixed-layer-depth',
    name: 'Variable mixed layer depth by latitude',
    hypothesis: 'The fixed H_surface=100m is wrong. Real mixed layer is ~20m in tropics, ~200m at high latitudes. Thin tropical mixed layer heats up faster (closer to observed 27°C), thick polar layer has more thermal inertia.',
    prompt: `You are modifying a WebGPU ocean circulation model to add variable mixed layer depth.

The physics code is in simamoc/model.js. Find where H_surface/hSurface is used in the temperature shader (var temperatureShaderCode, look for "let hSurf = min(params.hSurface, localDepth)") and CPU fallback (function cpuTimestep, look for "var hSurf = Math.min(H_surface, localDepth)").

Currently hSurf = min(H_surface, localDepth) where H_surface is a constant 100m.

Change to compute hSurf from latitude:
  hMLD = 20.0 + 180.0 * pow(abs(lat) / 80.0, 1.5)
  hSurf = min(hMLD, localDepth)

This gives:
- 20m at equator (fast heating, realistic tropical SST)
- 60m at 30° (subtropical)
- 130m at 60° (subpolar, more thermal inertia)
- 200m at 80° (polar, resists seasonal extremes)

Apply in BOTH GPU shader AND CPU fallback in simamoc/model.js. The lat variable is already available.`,
  },
  {
    id: 'meridional-overturning',
    name: 'Explicit meridional overturning cell',
    hypothesis: 'The AMOC is driven by density gradients but the current coupling only works through interfacial friction. Adding an explicit overturning stream that transports deep water equatorward when surface water flows poleward would strengthen the thermohaline circulation.',
    prompt: `You are modifying a WebGPU ocean circulation model to add an explicit meridional overturning tendency.

The physics code is in simamoc/model.js. Find the deep layer vorticity shader (var deepTimestepShaderCode). Currently the deep layer is driven only by interfacial coupling to the surface (F_couple terms) and bottom friction. Also find the corresponding CPU code in function cpuTimestep (the "Deep layer vorticity" section).

Add a meridional overturning tendency: when there is strong deep water formation (density-driven sinking at high latitudes), add a vorticity source that drives equatorward deep flow:

  // In the deep vorticity tendency, after the coupling term:
  // Meridional overturning: density gradient drives equatorward deep flow
  let dRhoDy = (rhoN - rhoS) * 0.5 * invDy;  // meridional density gradient
  let motTendency = 0.1 * dRhoDy;  // overturning vorticity source

Where rhoN, rhoS are density at neighboring latitude points computed from deep temp + salinity.

This should help establish the AMOC cell by directly forcing deep equatorward flow when there's a density gradient. Apply in BOTH GPU and CPU paths in simamoc/model.js.`,
  },
];

// ---------------------------------------------------------------------------
// Tournament runner
// ---------------------------------------------------------------------------
async function main() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║  PHYSICS TOURNAMENT: Self-Improving Climate Model      ║');
  console.log('║  Testing ' + N_BRANCHES + ' mutations against observations              ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');

  // Select mutations to test this round
  const selected = MUTATIONS.slice(0, N_BRANCHES);

  console.log('Mutations to test:');
  for (const m of selected) {
    console.log(`  [${m.id}] ${m.name}`);
    console.log(`    Hypothesis: ${m.hypothesis}\n`);
  }

  // Get baseline score first
  console.log('Running baseline (current main)...');
  const baselineReport = await runCapture('baseline', {});
  console.log(`  Baseline RMSE: ${baselineReport?.rmse || 'FAILED'}°C\n`);

  const results = [{ id: 'baseline', rmse: baselineReport?.rmse || 999, report: baselineReport }];

  // Run each mutation
  for (const mutation of selected) {
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`Testing: [${mutation.id}] ${mutation.name}`);
    console.log(`${'─'.repeat(60)}`);

    // Create worktree branch
    const branch = `tournament/${mutation.id}`;
    try {
      execSync(`git branch -D ${branch} 2>/dev/null || true`, { cwd: ROOT });
      execSync(`git worktree add -b ${branch} /tmp/tournament-${mutation.id} HEAD`, { cwd: ROOT, stdio: 'pipe' });
    } catch (e) {
      console.log(`  Failed to create worktree: ${e.message}`);
      results.push({ id: mutation.id, rmse: 999, error: e.message });
      continue;
    }

    const worktree = `/tmp/tournament-${mutation.id}`;

    // Apply mutation using Claude
    console.log('  Applying mutation via Claude agent...');
    try {
      const claudeCmd = `claude --print -p "${mutation.prompt.replace(/"/g, '\\"')}" --allowedTools Edit,Read,Grep,Glob`;
      execSync(claudeCmd, { cwd: worktree, timeout: 120000, stdio: 'pipe' });
    } catch (e) {
      console.log(`  Claude mutation failed: ${e.message.slice(0, 200)}`);
      cleanup(mutation.id, branch);
      results.push({ id: mutation.id, rmse: 999, error: 'mutation failed' });
      continue;
    }

    // Verify syntax (check both model.js and inline scripts in index.html)
    try {
      const checkResult = execSync(`node -e "
        const fs=require('fs');
        // Check model.js parses
        new Function(fs.readFileSync('simamoc/model.js','utf8'));
        // Check inline scripts in index.html parse
        const h=fs.readFileSync('simamoc/index.html','utf8');
        const scripts=[];const re=/<script>([\\s\\S]*?)<\\/script>/gi;let m;
        while((m=re.exec(h))!==null)scripts.push(m[1]);
        scripts.sort((a,b)=>b.length-a.length);
        new Function(scripts[0]);
        console.log('OK');
      "`, { cwd: worktree, stdio: 'pipe' }).toString().trim();
      if (checkResult !== 'OK') throw new Error('syntax check failed');
    } catch (e) {
      console.log(`  Syntax check failed — skipping`);
      cleanup(mutation.id, branch);
      results.push({ id: mutation.id, rmse: 999, error: 'syntax error' });
      continue;
    }

    // Run simulation from worktree
    console.log('  Running simulation...');
    const report = await runCaptureFromDir(worktree, mutation.id);
    const rmse = report?.rmse || 999;
    console.log(`  RMSE: ${rmse}°C (baseline: ${baselineReport?.rmse}°C)`);

    results.push({ id: mutation.id, name: mutation.name, rmse, report });
    cleanup(mutation.id, branch);
  }

  // --- TOURNAMENT RESULTS ---
  console.log(`\n${'═'.repeat(60)}`);
  console.log('  TOURNAMENT RESULTS');
  console.log(`${'═'.repeat(60)}\n`);

  results.sort((a, b) => a.rmse - b.rmse);
  for (let i = 0; i < results.length; i++) {
    const r = results[i];
    const medal = i === 0 ? '  *** WINNER ***' : '';
    const delta = r.id === 'baseline' ? '' : ` (${r.rmse < baselineReport.rmse ? '' : '+'}${(r.rmse - baselineReport.rmse).toFixed(2)})`;
    console.log(`  ${i + 1}. [${r.id}] RMSE: ${r.rmse}°C${delta}${medal}`);
    if (r.error) console.log(`     Error: ${r.error}`);
  }

  const winner = results[0];
  if (winner.id !== 'baseline') {
    console.log(`\n  Winner: ${winner.name || winner.id}`);
    console.log(`  RMSE improvement: ${baselineReport.rmse}°C → ${winner.rmse}°C`);
    console.log(`\n  To merge this mutation into main:`);
    console.log(`    git merge tournament/${winner.id}`);
  } else {
    console.log(`\n  Baseline won — no mutations improved the model this round.`);
  }

  // Save results
  const reportPath = `${OUT}/tournament-${new Date().toISOString().slice(0, 10)}.json`;
  writeFileSync(reportPath, JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2));
  console.log(`\n  Full results: ${reportPath}`);
}

function cleanup(id, branch) {
  try {
    execSync(`git worktree remove /tmp/tournament-${id} --force 2>/dev/null || true`, { cwd: ROOT });
    execSync(`git branch -D ${branch} 2>/dev/null || true`, { cwd: ROOT });
  } catch {}
}

async function runCapture(id, params) {
  return runCaptureFromDir(ROOT, id, params);
}

async function runCaptureFromDir(dir, id, params = {}) {
  try {
    // Copy capture script to worktree if not there
    const captureScript = resolve(ROOT, 'wiggum-capture.mjs');
    const paramsJson = JSON.stringify(params).replace(/"/g, '\\"');
    const cmd = `node "${captureScript}" --iter "${id}" --spinup ${SPINUP} --params "${paramsJson}"`;
    const output = execSync(cmd, { cwd: dir, timeout: SPINUP * 1000 + 60000, stdio: 'pipe' }).toString();

    // Parse RMSE from output
    const rmseMatch = output.match(/RMSE:\s+([\d.]+)/);
    const reportPath = `${OUT}/iter-${id}-report.json`;
    if (existsSync(resolve(ROOT, `screenshots/wiggum-claude/iter-${id}-report.json`))) {
      const report = JSON.parse(readFileSync(resolve(ROOT, `screenshots/wiggum-claude/iter-${id}-report.json`)));
      return report;
    }
    return { rmse: rmseMatch ? parseFloat(rmseMatch[1]) : 999 };
  } catch (e) {
    console.log(`  Capture failed: ${e.message.slice(0, 200)}`);
    return { rmse: 999, error: e.message.slice(0, 200) };
  }
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
