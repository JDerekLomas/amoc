// helm-lab/mcp-smoketest.mjs
// ---------------------------------------------------------------
// Spawns the MCP server as a child and walks it through:
//   initialize → tools/list → helm_health → helm_render → exit
//
// Validates the JSON-RPC protocol shape and the daemon proxy +
// image-return path. Run after any change to mcp-server.mjs.
//
// Will reuse a running daemon (won't spawn a competing one).

import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));

const child = spawn(process.execPath, [resolve(HERE, 'mcp-server.mjs')], {
  stdio: ['pipe', 'pipe', 'inherit'],
});

let buf = '';
const pending = new Map();
let nextId = 1;

child.stdout.on('data', d => {
  buf += d.toString('utf8');
  let nl;
  while ((nl = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, nl).trim();
    buf = buf.slice(nl + 1);
    if (!line) continue;
    let msg;
    try { msg = JSON.parse(line); } catch { continue; }
    if (msg.id != null && pending.has(msg.id)) {
      const { res, rej } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) rej(new Error(`${msg.error.code}: ${msg.error.message}`));
      else res(msg.result);
    }
  }
});

function call(method, params) {
  const id = nextId++;
  return new Promise((res, rej) => {
    pending.set(id, { res, rej });
    child.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
    setTimeout(() => { if (pending.has(id)) { pending.delete(id); rej(new Error(`timeout: ${method}`)); } }, 60000);
  });
}

function dot(label, ok = true) { console.log(`  ${ok ? '✓' : '✗'} ${label}`); }

async function main() {
  console.log('=== mcp-smoketest ===');
  console.log('1. initialize');
  const init = await call('initialize', { protocolVersion: '2024-11-05', clientInfo: { name: 'smoketest', version: '0.1' }, capabilities: {} });
  dot(`server: ${init.serverInfo.name} v${init.serverInfo.version}`);

  console.log('2. tools/list');
  const tl = await call('tools/list', {});
  dot(`got ${tl.tools.length} tools: ${tl.tools.map(t => t.name).join(', ')}`);

  console.log('3. helm_health');
  const hh = await call('tools/call', { name: 'helm_health', arguments: {} });
  if (hh.isError) { dot('helm_health errored', false); console.error(hh.content); process.exit(1); }
  const health = JSON.parse(hh.content[0].text);
  dot(`useGPU=${health.useGPU} step=${health.totalSteps} simYears=${health.simYears.toFixed(2)}`);

  console.log('4. helm_render (view=temp)');
  const hr = await call('tools/call', { name: 'helm_render', arguments: { view: 'temp' } });
  if (hr.isError) { dot('helm_render errored', false); console.error(hr.content); process.exit(1); }
  const img = hr.content.find(c => c.type === 'image');
  if (!img) { dot('no image content returned', false); process.exit(1); }
  dot(`returned image: ${img.mimeType}, ${img.data.length} base64 chars (~${Math.round(img.data.length * 0.75 / 1024)} KB)`);
  const text = hr.content.find(c => c.type === 'text');
  dot(`metadata: ${text?.text || '<none>'}`);

  console.log('\nall checks passed');
  child.stdin.end();
  child.kill();
  process.exit(0);
}

main().catch(e => { console.error('FAILED:', e.message); child.kill(); process.exit(1); });
