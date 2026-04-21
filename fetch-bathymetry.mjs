#!/usr/bin/env node
/**
 * Fetch real ocean bathymetry from ETOPO1 via Open Topo Data API
 * Downsampled to 360x180 (1-degree) grid matching our simulation
 * Output: bathymetry_1deg.json
 */

import { writeFileSync } from 'fs';

const NX = 360, NY = 160; // Match SST grid: lat -79.5 to +79.5
const API = 'https://api.opentopodata.org/v1/etopo1';
const BATCH_SIZE = 100; // API limit per request
const DELAY_MS = 1000; // rate limit courtesy

async function fetchBatch(locations) {
  const locStr = locations.map(([lat, lon]) => `${lat},${lon}`).join('|');
  const url = `${API}?locations=${locStr}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  const data = await resp.json();
  if (data.status !== 'OK') throw new Error(`API status: ${data.status}`);
  return data.results.map(r => r.elevation);
}

async function main() {
  console.log(`Fetching ETOPO1 bathymetry at ${NX}x${NY} (1-degree)...`);

  // Build all query points
  const points = [];
  for (let j = 0; j < NY; j++) {
    const lat = -79.5 + j; // -79.5 to +79.5
    for (let i = 0; i < NX; i++) {
      const lon = -179.5 + i; // -179.5 to +179.5
      points.push([lat, lon]);
    }
  }

  console.log(`Total points: ${points.length}`);
  console.log(`Batches: ${Math.ceil(points.length / BATCH_SIZE)}`);

  const elevations = new Array(points.length);
  let fetched = 0;

  for (let b = 0; b < points.length; b += BATCH_SIZE) {
    const batch = points.slice(b, b + BATCH_SIZE);
    try {
      const results = await fetchBatch(batch);
      for (let i = 0; i < results.length; i++) {
        elevations[b + i] = results[i];
      }
      fetched += batch.length;
      if (fetched % 5000 === 0 || fetched === points.length) {
        console.log(`  ${fetched}/${points.length} (${(100*fetched/points.length).toFixed(1)}%)`);
      }
    } catch (err) {
      console.log(`  Batch ${b} failed: ${err.message}, retrying...`);
      await new Promise(r => setTimeout(r, 5000));
      try {
        const results = await fetchBatch(batch);
        for (let i = 0; i < results.length; i++) {
          elevations[b + i] = results[i];
        }
        fetched += batch.length;
      } catch (err2) {
        console.log(`  Retry failed: ${err2.message}, filling with -4000`);
        for (let i = 0; i < batch.length; i++) {
          elevations[b + i] = -4000;
        }
        fetched += batch.length;
      }
    }
    // Rate limiting
    if (b + BATCH_SIZE < points.length) {
      await new Promise(r => setTimeout(r, DELAY_MS));
    }
  }

  // Convert to ocean depth (positive values = depth below sea level)
  // ETOPO returns negative values for ocean, positive for land
  const depth = elevations.map(e => e < 0 ? -e : 0);

  // Stats
  let oceanCount = 0, maxDepth = 0, sumDepth = 0;
  for (const d of depth) {
    if (d > 0) { oceanCount++; sumDepth += d; maxDepth = Math.max(maxDepth, d); }
  }
  console.log(`\nOcean cells: ${oceanCount}/${depth.length}`);
  console.log(`Max depth: ${maxDepth}m`);
  console.log(`Mean depth: ${(sumDepth/oceanCount).toFixed(0)}m`);

  // Save
  const output = {
    nx: NX, ny: NY,
    lat0: -79.5, lat1: 79.5,
    lon0: -179.5, lon1: 179.5,
    source: 'ETOPO1 via Open Topo Data API, 1-degree',
    depth: Array.from(depth),
  };
  writeFileSync('bathymetry_1deg.json', JSON.stringify(output));
  console.log(`Saved bathymetry_1deg.json (${(JSON.stringify(output).length/1024).toFixed(0)} KB)`);
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
