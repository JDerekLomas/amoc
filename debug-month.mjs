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
await new Promise(r => server.listen(8767, r));

const browser = await chromium.launch({
  headless: true,
  executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
});

for (const month of [0, 6]) {
  const page = await browser.newPage();
  page.on('console', msg => console.log(`  [page] ${msg.text()}`));
  await page.goto(`http://localhost:8767/reference-sst.html?month=${month}`, { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(3000);
  
  const html = await page.evaluate(() => document.body.innerText.substring(0, 500));
  console.log(`Month=${month} URL param, page text:\n${html}\n---`);
  await page.close();
}

await browser.close();
server.close();
