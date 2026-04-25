// helm-lab/bench.mjs — measure per-RPC latency against the daemon.
// Usage: node helm-lab/bench.mjs [--port 8830]

const PORT = Number(process.env.HELM_LAB_PORT || 8830);
const URL = `http://127.0.0.1:${PORT}`;

async function rpc(method, args = []) {
  const r = await fetch(`${URL}/rpc`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ method, args }),
  });
  const j = await r.json();
  if (!j.ok) throw new Error(j.error);
  return { result: j.result, ms: j.ms };
}

const sequence = [
  ['getParams', []],
  ['diag',      [{}]],
  ['diag',      [{}]],
  ['setParams', [{ windStrength: 1.5 }]],
  ['step',      [1000]],
  ['render',    ['/tmp/_bench_temp.png', { view: 'temp' }]],
  ['render',    ['/tmp/_bench_psi.png',  { view: 'psi'  }]],
  ['diag',      [{ profiles: false }]],
  ['setParams', [{ windStrength: 1.0 }]],
  ['step',      [5000]],
  ['render',    ['/tmp/_bench_temp2.png', { view: 'temp' }]],
];

console.log(`server     local`);
console.log(`reported   round-trip   call`);
console.log(`---------- ------------ -------------------------------------`);
for (const [m, args] of sequence) {
  const t0 = performance.now();
  const { ms } = await rpc(m, args);
  const round = Math.round(performance.now() - t0);
  const summary = `${m}(${args.map(a => typeof a === 'object' ? JSON.stringify(a) : JSON.stringify(a)).join(', ')})`;
  console.log(`${String(ms).padStart(6)}ms     ${String(round).padStart(5)}ms      ${summary}`);
}
