// experiments/earth-movie.mjs
// ----------------------------------------------------------------------
// Capture a continuous video of the Earth's ocean evolving over time.
//
// Method:
//   1. Reset the engine, set baseline params.
//   2. (Optional) spin up briefly so the gyres have begun to form.
//   3. Loop: step N, render → frame_NNNN.png, repeat M times.
//   4. ffmpeg the frames into an MP4 (and a small GIF preview).
//   5. Also build a contact sheet of every K-th frame so the run is
//      visible at a glance (since I can't actually watch video).
//
// Flags via env or CLI:
//   FRAMES   number of frames to capture       (default 180)
//   STRIDE   sim steps between frames          (default 1500)
//   FPS      mp4 framerate                     (default 24)
//   SPINUP   pre-roll steps                    (default 30000)
//   VIEW     view name                         (default temp)
//   OUT      output dir                        (default helm-lab/runs/movie_<view>_<ts>)
//
// Usage:
//   node helm-lab/cli.mjs serve &     # daemon must be running
//   node helm-lab/experiments/earth-movie.mjs

import { spawn } from 'node:child_process';
import { writeFile, mkdir, readFile, readdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';

const FRAMES = Number(process.env.FRAMES || 180);
const STRIDE = Number(process.env.STRIDE || 1500);
const FPS    = Number(process.env.FPS    || 24);
const SPINUP = Number(process.env.SPINUP || 30000);
const VIEW   = process.env.VIEW || 'temp';
const PORT   = Number(process.env.HELM_LAB_PORT || 8830);
const URL    = `http://127.0.0.1:${PORT}`;
const OUT    = resolve(process.env.OUT || `helm-lab/runs/movie_${VIEW}_${Date.now()}`);

async function rpc(method, args = []) {
  const r = await fetch(`${URL}/rpc`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ method, args }),
  });
  const j = await r.json();
  if (!j.ok) throw new Error(`${method}: ${j.error}`);
  return j.result;
}

await mkdir(join(OUT, 'frames'), { recursive: true });
console.log(`=== earth-movie ===
out:    ${OUT}
view:   ${VIEW}
spinup: ${SPINUP.toLocaleString()} steps
frames: ${FRAMES} (every ${STRIDE.toLocaleString()} steps)
fps:    ${FPS} (= ${(FRAMES / FPS).toFixed(1)}s of video)
total:  ${(SPINUP + FRAMES * STRIDE).toLocaleString()} sim steps
`);

const tStart = Date.now();

// Reset & spin up
console.log('[1/3] reset + spin-up...');
await rpc('reset');
await rpc('setParams', [{
  windStrength: 1.0, r: 0.04, A: 2e-4,
  freshwaterForcing: 0.0, yearSpeed: 1.0, stepsPerFrame: 50,
}]);
if (SPINUP > 0) await rpc('step', [SPINUP]);
const initDiag = await rpc('diag', [{}]);
console.log(`     simYears=${initDiag.simYears.toFixed(2)} KE=${initDiag.KE.toExponential(2)} (${((Date.now() - tStart) / 1000).toFixed(1)}s)`);

// Capture loop
console.log(`\n[2/3] capturing ${FRAMES} frames...`);
const series = [];
for (let i = 0; i < FRAMES; i++) {
  await rpc('step', [STRIDE]);
  const fp = join(OUT, 'frames', `frame_${String(i).padStart(5, '0')}.png`);
  await rpc('render', [fp, { view: VIEW }]);
  // Light-touch diagnostics every 10 frames so we can label in the contact sheet
  if (i % 10 === 0 || i === FRAMES - 1) {
    const d = await rpc('diag', [{}]);
    series.push({ i, simYears: d.simYears, KE: d.KE, amoc: d.amoc, globalSST: d.globalSST, polarSST: d.polarSST, iceArea: d.iceArea });
    process.stdout.write(`\r     frame ${i + 1}/${FRAMES}  simYr=${d.simYears.toFixed(2)}  KE=${d.KE.toExponential(2)}  ice=${d.iceArea.toString().padStart(5)}    `);
  }
}
process.stdout.write('\n');
const tCapture = Date.now();
console.log(`     captured in ${((tCapture - tStart) / 1000).toFixed(1)}s`);

await writeFile(join(OUT, 'series.jsonl'), series.map(s => JSON.stringify(s)).join('\n') + '\n');

// Stitch with ffmpeg
console.log('\n[3/3] stitching MP4 + GIF preview...');
const mp4 = join(OUT, 'earth.mp4');
const gif = join(OUT, 'earth.gif');
const palette = join(OUT, 'palette.png');

function run(cmd, args) {
  return new Promise((res, rej) => {
    const p = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let err = '';
    p.stderr.on('data', d => err += d.toString());
    p.on('exit', code => code === 0 ? res() : rej(new Error(`${cmd} exited ${code}: ${err.split('\n').slice(-5).join('\n')}`)));
  });
}

// MP4 — H.264, decent quality, web-compatible
await run('ffmpeg', [
  '-y', '-loglevel', 'error',
  '-framerate', String(FPS),
  '-i', join(OUT, 'frames', 'frame_%05d.png'),
  '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
  '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
  '-crf', '20',
  mp4,
]);

// GIF — small preview that some viewers can scrub through
await run('ffmpeg', [
  '-y', '-loglevel', 'error',
  '-i', join(OUT, 'frames', 'frame_%05d.png'),
  '-vf', `fps=${Math.min(FPS, 12)},scale=512:-1:flags=lanczos,palettegen`,
  palette,
]);
await run('ffmpeg', [
  '-y', '-loglevel', 'error',
  '-framerate', String(FPS),
  '-i', join(OUT, 'frames', 'frame_%05d.png'),
  '-i', palette,
  '-filter_complex', `fps=${Math.min(FPS, 12)},scale=512:-1:flags=lanczos[x];[x][1:v]paletteuse`,
  gif,
]);

const tDone = Date.now();

// Build a tiled "contact sheet" of every Kth frame so the run is visible
// at-a-glance — useful for me, since I can't watch video. We pick ~16 frames.
const allFrames = (await readdir(join(OUT, 'frames'))).filter(f => f.startsWith('frame_')).sort();
const N = allFrames.length;
const PICK = 16;
const stride = Math.max(1, Math.floor(N / PICK));
const picked = [];
for (let i = 0; i < N; i += stride) picked.push(allFrames[i]);
if (picked[picked.length - 1] !== allFrames[N - 1]) picked.push(allFrames[N - 1]);

// Read frames as data URLs and call composer (if available) or fall back
// to a hand-built tile in pure Node via PNG concatenation isn't trivial,
// so use the daemon's page to lay them out.
try {
  const data = [];
  for (let i = 0; i < picked.length; i++) {
    const f = picked[i];
    const idx = parseInt(f.match(/frame_(\d+)/)[1], 10);
    const found = series.find(s => s.i >= idx) || series[series.length - 1];
    data.push({
      t: idx, simYears: found?.simYears ?? 0,
      frames: { [VIEW]: 'data:image/png;base64,' + (await readFile(join(OUT, 'frames', f))).toString('base64') },
      diag: found || { KE: 0, maxVel: 0, amoc: 0, globalSST: 0, tropicalSST: 0, polarSST: 0, iceArea: 0 },
    });
  }
  const png = await rpc('composeBifurcation', [
    data.map((d, i) => ({ branch: i % 2 === 0 ? 'forward' : 'forward', F: d.simYears, amoc: d.diag.amoc, KE: d.diag.KE, globalSST: d.diag.globalSST, polarSST: d.diag.polarSST, frames: d.frames })),
    join(OUT, 'overview.png'),
    { title: `Earth over ${(series[series.length - 1]?.simYears ?? 0).toFixed(1)} model years (${VIEW})`, subtitle: `${N} frames · stride ${STRIDE.toLocaleString()} steps · ${FPS} fps` },
  ]).catch(() => null);
} catch (e) {
  console.warn('     (skipped overview PNG: ' + e.message + ')');
}

console.log(`\n=== done in ${((tDone - tStart) / 1000).toFixed(1)}s ===
${mp4}                    <- main result (open in QuickTime / browser)
${gif}                    <- 512px GIF preview
${OUT}/frames/               (${N} PNGs)
${OUT}/series.jsonl
${OUT}/overview.png          <- tiled snapshot (if composer installed)
`);
