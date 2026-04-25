// experiments/run-all.mjs
// ----------------------------------------------------------------------
// Run every experiment in sequence. Useful for validating that a fresh
// helm-lab installation produces the documented R-numbers, or for
// regenerating the artifact set after an engine change.
//
// Total wall-clock at default sizes: ~75 minutes on Apple Metal.
// Results land in helm-lab/runs/<experiment>/.
//
// Usage:
//   node helm-lab/cli.mjs serve &
//   node helm-lab/experiments/run-all.mjs
//
// Or with --skip to skip experiments by name:
//   node helm-lab/experiments/run-all.mjs --skip earth-movie,greenhouse

import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

const EXPERIMENTS = [
  // ordered by typical runtime, fastest first so failures surface early
  { name: 'earth-movie',         script: 'earth-movie.mjs',         eta: '~3 min'  },
  { name: 'amoc-bifurcation-v2', script: 'amoc-bifurcation-v2.mjs', eta: '~12 min' },
  { name: 'amoc-collapse-window',script: 'amoc-collapse-window.mjs',eta: '~18 min' },
  { name: 'greenhouse-sweep',    script: 'greenhouse-sweep.mjs',    eta: '~10 min' },
  { name: 'snowball-search',     script: 'snowball-search.mjs',     eta: '~12 min' },
  { name: 'snowball-escape',     script: 'snowball-escape.mjs',     eta: '~16 min' },
];

const args = new Set();
const skip = new Set();
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a === '--skip' && process.argv[i + 1]) {
    process.argv[++i].split(',').forEach(s => skip.add(s.trim()));
  } else if (a === '--only' && process.argv[i + 1]) {
    process.argv[++i].split(',').forEach(s => args.add(s.trim()));
  }
}

function shouldRun(name) {
  if (skip.has(name)) return false;
  if (args.size && !args.has(name)) return false;
  return true;
}

const results = [];
const start = Date.now();
console.log('=== run-all ===');
console.log(`scheduled: ${EXPERIMENTS.filter(e => shouldRun(e.name)).map(e => e.name).join(', ')}`);
console.log(`eta:       ~75 min total at full set\n`);

for (const exp of EXPERIMENTS) {
  if (!shouldRun(exp.name)) {
    console.log(`SKIP ${exp.name}`);
    results.push({ ...exp, status: 'skipped' });
    continue;
  }
  const expStart = Date.now();
  console.log(`\n┌─ ${exp.name}  (${exp.eta})`);
  await new Promise((res, rej) => {
    const child = spawn(process.execPath, [resolve(HERE, exp.script)], { stdio: 'inherit' });
    child.on('exit', code => code === 0 ? res() : rej(new Error(`${exp.name} exited ${code}`)));
  }).then(() => {
    const dt = ((Date.now() - expStart) / 60000).toFixed(2);
    console.log(`└─ ${exp.name} ✓  (${dt} min)`);
    results.push({ ...exp, status: 'ok', minutes: dt });
  }).catch(e => {
    console.log(`└─ ${exp.name} ✗  ${e.message}`);
    results.push({ ...exp, status: 'failed', error: e.message });
  });
}

const totalMin = ((Date.now() - start) / 60000).toFixed(2);
console.log(`\n=== run-all done in ${totalMin} min ===`);
for (const r of results) {
  const mark = r.status === 'ok' ? '✓' : r.status === 'skipped' ? '·' : '✗';
  console.log(`  ${mark} ${r.name.padEnd(24)}  ${r.status}${r.minutes ? '  ' + r.minutes + ' min' : ''}${r.error ? '  ' + r.error : ''}`);
}
const failed = results.filter(r => r.status === 'failed');
process.exit(failed.length ? 1 : 0);
