#!/usr/bin/env node
/**
 * Fetch land elevation from ETOPO1 and add to bathymetry file
 * Queries only land cells (saves API calls)
 */

import { writeFileSync, readFileSync } from 'fs';

const NX = 360, NY = 160;
const API = 'https://api.opentopodata.org/v1/etopo1';
const BATCH_SIZE = 100;
const DELAY_MS = 1000;

async function fetchBatch(locations) {
  const locStr = locations.map(([lat, lon]) => `${lat},${lon}`).join('|');
  const resp = await fetch(`${API}?locations=${locStr}`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  const data = await resp.json();
  if (data.status !== 'OK') throw new Error(`API status: ${data.status}`);
  return data.results.map(r => r.elevation);
}

async function main() {
  // Load existing bathymetry to know which cells are land (depth == 0)
  const bathy = JSON.parse(readFileSync('bathymetry_1deg.json', 'utf8'));

  // Find land cells that need elevation
  const landCells = [];
  for (let j = 0; j < NY; j++) {
    const lat = -79.5 + j;
    for (let i = 0; i < NX; i++) {
      const k = j * NX + i;
      if (bathy.depth[k] === 0) {
        landCells.push({ k, lat, lon: -179.5 + i });
      }
    }
  }

  console.log(`Land cells to query: ${landCells.length}`);
  console.log(`Batches: ${Math.ceil(landCells.length / BATCH_SIZE)}`);

  const elevation = new Array(NX * NY).fill(0);
  // Copy ocean depth as negative elevation
  for (let k = 0; k < NX * NY; k++) {
    elevation[k] = bathy.depth[k] > 0 ? -bathy.depth[k] : 0;
  }

  let fetched = 0;
  for (let b = 0; b < landCells.length; b += BATCH_SIZE) {
    const batch = landCells.slice(b, b + BATCH_SIZE);
    const locs = batch.map(c => [c.lat, c.lon]);
    try {
      const results = await fetchBatch(locs);
      for (let i = 0; i < results.length; i++) {
        elevation[batch[i].k] = Math.max(0, results[i]); // land elevation >= 0
      }
      fetched += batch.length;
      if (fetched % 2000 === 0 || fetched >= landCells.length) {
        console.log(`  ${fetched}/${landCells.length} (${(100*fetched/landCells.length).toFixed(1)}%)`);
      }
    } catch (err) {
      console.log(`  Batch ${b} failed: ${err.message}`);
      for (let i = 0; i < batch.length; i++) elevation[batch[i].k] = 100; // default
      fetched += batch.length;
    }
    if (b + BATCH_SIZE < landCells.length) await new Promise(r => setTimeout(r, DELAY_MS));
  }

  // Save combined: depth (ocean, positive) + elevation (land, positive)
  const output = {
    ...bathy,
    source: 'ETOPO1 via Open Topo Data API, 1-degree (ocean depth + land elevation)',
    elevation: elevation.map(e => Math.max(0, e)),  // land elevation (m above sea level)
  };

  // Stats
  let maxElev = 0, landCount = 0;
  for (const e of output.elevation) {
    if (e > 0) { landCount++; maxElev = Math.max(maxElev, e); }
  }
  console.log(`\nLand cells with elevation: ${landCount}`);
  console.log(`Max elevation: ${maxElev}m`);

  writeFileSync('bathymetry_1deg.json', JSON.stringify(output));
  console.log(`Updated bathymetry_1deg.json`);
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
