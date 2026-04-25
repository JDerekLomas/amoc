// SimAMOC Headless API Server
// Runs the ocean simulation in a Playwright browser, exposes HTTP API for control.
// Usage: node sim-api.js
// Then: curl localhost:9876/screenshot -o /tmp/sim.png
//       curl localhost:9876/params
//       curl -X POST -H 'Content-Type: application/json' -d '{"dt":1e-4}' localhost:9876/params
//       curl -X POST localhost:9876/step?n=5000
//       curl -X POST localhost:9876/reset
//       curl -X POST localhost:9876/view?field=speed

const http = require('http');
const { chromium } = require('playwright');
const { execSync } = require('child_process');

const SIM_PORT = 8765;
const API_PORT = 9876;
let page = null;
let simServer = null;

async function startSimServer() {
  return new Promise((resolve) => {
    const handler = require('http').createServer((req, res) => {
      const fs = require('fs');
      const path = require('path');
      let urlPath = req.url.split('?')[0];  // strip query params
      let filePath = path.join(__dirname, urlPath === '/' ? '/simamoc/index.html' : urlPath);
      if (filePath.endsWith('/')) filePath += 'index.html';
      const ext = path.extname(filePath);
      const mimeTypes = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream', '.csv': 'text/csv', '.png': 'image/png' };
      fs.readFile(filePath, (err, data) => {
        if (err) { res.writeHead(404); res.end('Not found: ' + req.url); return; }
        res.writeHead(200, { 'Content-Type': mimeTypes[ext] || 'application/octet-stream' });
        res.end(data);
      });
    });
    handler.listen(SIM_PORT, () => {
      console.log(`Sim server: http://localhost:${SIM_PORT}`);
      resolve(handler);
    });
  });
}

async function startBrowser() {
  const browser = await chromium.launch({ headless: true });
  page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  // Collect console output
  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('error') || text.includes('Error') || text.includes('NaN') ||
        text.includes('blow') || text.includes('physics') || text.includes('CPU') ||
        text.includes('GPU') || text.includes('FFT') || text.includes('RMSE') ||
        text.includes('init') || text.includes('loaded')) {
      console.log(`[browser] ${text}`);
    }
  });

  // Use smaller grid for headless (256x128 is ~16x faster than 1024x512)
  const nx = parseInt(process.env.SIM_NX || '256');
  const ny = parseInt(process.env.SIM_NY || '128');
  console.log(`Loading simulation at ${nx}x${ny}...`);

  // Inject pause-on-load to prevent requestAnimationFrame loop from crashing headless
  await page.addInitScript('window._HEADLESS_MODE = true;');
  await page.goto(`http://localhost:${SIM_PORT}/simamoc/index.html?nx=${nx}&ny=${ny}`, { waitUntil: 'networkidle', timeout: 30000 });
  // Wait a bit for async init, then ensure paused
  await page.waitForTimeout(3000);
  await page.evaluate(() => { paused = true; });
  console.log('Simulation ready.');
  return browser;
}

function parseBody(req) {
  return new Promise((resolve) => {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try { resolve(JSON.parse(body)); } catch { resolve({}); }
    });
  });
}

async function handleRequest(req, res) {
  const url = new URL(req.url, `http://localhost:${API_PORT}`);
  const path = url.pathname;

  try {
    if (path === '/screenshot') {
      const buf = await page.screenshot({ type: 'png' });
      // Also save to /tmp for easy reading
      require('fs').writeFileSync('/tmp/sim-screenshot.png', buf);
      res.writeHead(200, { 'Content-Type': 'image/png' });
      res.end(buf);
    }

    else if (path === '/params' && req.method === 'GET') {
      const params = await page.evaluate(() => window.lab.getParams());
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(params, null, 2));
    }

    else if (path === '/params' && req.method === 'POST') {
      const body = await parseBody(req);
      const result = await page.evaluate((p) => { window.lab.setParams(p); return window.lab.getParams(); }, body);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result, null, 2));
    }

    else if (path === '/step') {
      const n = parseInt(url.searchParams.get('n') || '100');
      // Use small batches to avoid browser timeout
      const batchSize = 50;
      let remaining = n;
      while (remaining > 0) {
        const batch = Math.min(batchSize, remaining);
        await page.evaluate((steps) => {
          for (var i = 0; i < steps; i++) cpuTimestep();
          totalSteps += steps;
          simTime += steps * dt * yearSpeed;
        }, batch);
        remaining -= batch;
      }
      // Redraw after stepping
      await page.evaluate(() => { draw(); updateStats(); drawProfile(); drawRadProfile(); pushAmocSample(); drawAmocChart(); });
      const params = await page.evaluate(() => window.lab.getParams());
      const buf = await page.screenshot({ type: 'png' });
      require('fs').writeFileSync('/tmp/sim-screenshot.png', buf);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ stepped: n, totalSteps: params.totalSteps, simTime: params.simTime }));
    }

    else if (path === '/reset') {
      await page.evaluate(() => window.lab.reset());
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'reset' }));
    }

    else if (path === '/view') {
      const field = url.searchParams.get('field') || 'temp';
      await page.evaluate((f) => window.lab.view(f), field);
      await page.waitForTimeout(200);
      const buf = await page.screenshot({ type: 'png' });
      require('fs').writeFileSync('/tmp/sim-screenshot.png', buf);
      res.writeHead(200, { 'Content-Type': 'image/png' });
      res.end(buf);
    }

    else if (path === '/diagnostics') {
      const diag = await page.evaluate(() => window.lab.diagnostics());
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(diag, null, 2));
    }

    else if (path === '/eval') {
      // Run arbitrary JS in the sim context (for debugging)
      const body = await parseBody(req);
      const result = await page.evaluate((code) => {
        try { return eval(code); } catch (e) { return { error: e.message }; }
      }, body.code || '');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    }

    else if (path === '/run' || path === '/go') {
      // Run N steps using lab.step() which handles all physics + rendering
      const n = parseInt(url.searchParams.get('n') || '500');
      const result = await page.evaluate(async (steps) => {
        var t0 = performance.now();
        await window.lab.step(steps);
        var p = window.lab.getParams();
        return {
          stepped: steps,
          totalSteps: p.totalSteps,
          simTime: p.simTime,
          wallMs: Math.round(performance.now() - t0)
        };
      }, n);
      const buf = await page.screenshot({ type: 'png' });
      require('fs').writeFileSync('/tmp/sim-screenshot.png', buf);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result));
    }

    else if (path === '/pause') {
      await page.evaluate(() => window.lab.pause());
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ paused: true }));
    }

    else if (path === '/resume') {
      await page.evaluate(() => window.lab.resume());
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ paused: false }));
    }

    else {
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.end([
        'SimAMOC API',
        '',
        'GET  /screenshot        - PNG of current state (also saves /tmp/sim-screenshot.png)',
        'GET  /params            - current parameters as JSON',
        'POST /params            - set parameters (JSON body)',
        'POST /step?n=1000       - advance N steps, returns state + auto-screenshots',
        'POST /reset             - reset simulation',
        'GET  /view?field=speed  - switch view and screenshot (temp/speed/psi/vorticity/salinity/...)',
        'GET  /diagnostics       - RMSE, max velocity, etc.',
        'POST /pause             - pause simulation',
        'POST /resume            - resume simulation',
        'POST /eval              - run JS in sim context (body: {"code": "..."})',
      ].join('\n'));
    }
  } catch (e) {
    console.error('API error:', e.message);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: e.message }));
  }
}

async function main() {
  simServer = await startSimServer();
  const browser = await startBrowser();

  const apiServer = http.createServer(handleRequest);
  apiServer.listen(API_PORT, () => {
    console.log(`\nAPI server: http://localhost:${API_PORT}`);
    console.log('Ready for commands.');
  });

  process.on('SIGINT', async () => {
    console.log('\nShutting down...');
    await browser.close();
    simServer.close();
    apiServer.close();
    process.exit(0);
  });
}

main().catch(e => { console.error(e); process.exit(1); });
