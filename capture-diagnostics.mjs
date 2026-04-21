import { chromium } from 'playwright';
import { createServer } from 'http';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const root = '/Users/dereklomas/lukebarrington/amoc';
const server = createServer((req, res) => {
  const urlPath = new URL(req.url, 'http://localhost').pathname;
  const filePath = resolve(root, urlPath.replace(/^\//, ''));
  try {
    const data = readFileSync(filePath);
    const ext = filePath.split('.').pop();
    const mime = { html: 'text/html', json: 'application/json' }[ext] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  } catch { res.writeHead(404); res.end('Not found'); }
});
await new Promise(r => server.listen(8766, r));

const browser = await chromium.launch({
  headless: true,
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
});

// Capture each panel separately for readability
const panels = ['obs', 'solar', 'olr', 'qnet', 'land', 'landflux'];

for (const month of [0, 6]) {
  const page = await browser.newPage();
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.goto(`http://localhost:8766/reference-sst.html?month=${month}`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(3000);

  const name = month === 0 ? 'jan' : 'jul';

  for (const panel of panels) {
    // Scroll panel into view and screenshot just that region
    const el = await page.$(`#${panel}`);
    if (el) {
      await el.scrollIntoViewIfNeeded();
      await page.waitForTimeout(300);
      const box = await el.boundingBox();
      // Capture canvas + some context above
      await page.screenshot({
        path: `${root}/screenshots/diag-${name}-${panel}.png`,
        clip: { x: 0, y: Math.max(0, box.y - 30), width: 1400, height: box.height + 70 }
      });
    }
  }
  console.log(`Captured ${name} panels`);
  await page.close();
}

await browser.close();
server.close();
