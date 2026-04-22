#!/usr/bin/env node
// Generate a 360x180 global ocean mask from Natural Earth 110m TopoJSON
// Output: mask.json with {nx, ny, hex} and coastlines.json

var fs = require('fs');

var NX = 360, NY = 180;
var LON0 = -180, LON1 = 180, LAT0 = -80, LAT1 = 80;

// --- Load and decode TopoJSON ---
var topo = JSON.parse(fs.readFileSync('/tmp/land-110m.json', 'utf8'));
var transform = topo.transform;
var arcs = topo.arcs;

function decodeArc(arcIdx) {
  var arc = arcs[arcIdx < 0 ? ~arcIdx : arcIdx];
  var coords = [];
  var x = 0, y = 0;
  for (var i = 0; i < arc.length; i++) {
    x += arc[i][0];
    y += arc[i][1];
    coords.push([x * transform.scale[0] + transform.translate[0], y * transform.scale[1] + transform.translate[1]]);
  }
  if (arcIdx < 0) coords.reverse();
  return coords;
}

function ringToCoords(ring) {
  var coords = [];
  for (var i = 0; i < ring.length; i++) {
    var arcCoords = decodeArc(ring[i]);
    for (var j = (i === 0 ? 0 : 1); j < arcCoords.length; j++) {
      coords.push(arcCoords[j]);
    }
  }
  return coords;
}

var geom = topo.objects.land.geometries[0];

// Split a ring at antimeridian jumps into multiple closed sub-rings
// When a ring has edges that jump from ~-180 to ~+180, we split at those jumps
// and close each sub-ring by tracing along the antimeridian boundary.
function splitRingAtAntimeridian(ring) {
  // Find jump points (edges where lon changes by > 180 degrees)
  var jumps = [];
  for (var i = 1; i < ring.length; i++) {
    if (Math.abs(ring[i][0] - ring[i-1][0]) > 180) {
      jumps.push(i);
    }
  }
  if (jumps.length === 0) return [ring]; // no jumps, return as-is

  // Split ring at jump points into segments
  var segments = [];
  var prev = 0;
  for (var j = 0; j < jumps.length; j++) {
    segments.push(ring.slice(prev, jumps[j]));
    prev = jumps[j];
  }
  segments.push(ring.slice(prev));

  // For each segment, close it by adding boundary points
  var subRings = [];
  for (var s = 0; s < segments.length; s++) {
    var seg = segments[s];
    if (seg.length < 2) continue;

    // Determine which side of the antimeridian this segment is on
    var avgLon = 0;
    for (var i = 0; i < seg.length; i++) avgLon += seg[i][0];
    avgLon /= seg.length;

    // Close the segment: add points along the antimeridian boundary
    var closed = seg.slice();
    var lastPt = seg[seg.length - 1];
    var firstPt = seg[0];

    // The boundary is at +/-180. Add points tracing along the boundary.
    var boundaryLon = avgLon > 0 ? 180 : -180;

    // Add closing edges: last point -> boundary -> first point
    closed.push([boundaryLon, lastPt[1]]);
    // If the segment spans a large latitude range, add intermediate boundary points
    var latStep = lastPt[1] > firstPt[1] ? -5 : 5;
    var lat = lastPt[1] + latStep;
    while ((latStep > 0 && lat < firstPt[1]) || (latStep < 0 && lat > firstPt[1])) {
      closed.push([boundaryLon, lat]);
      lat += latStep;
    }
    closed.push([boundaryLon, firstPt[1]]);
    closed.push(firstPt.slice()); // close the ring

    subRings.push(closed);
  }

  return subRings;
}

// Extract all polygons, splitting antimeridian-crossing ones
var polygons = [];

for (var p = 0; p < geom.arcs.length; p++) {
  var polyArcs = geom.arcs[p];
  var outer = ringToCoords(polyArcs[0]);
  var holes = [];
  for (var h = 1; h < polyArcs.length; h++) {
    holes.push(ringToCoords(polyArcs[h]));
  }

  // Check if this polygon crosses the antimeridian
  var minLon = Infinity, maxLon = -Infinity;
  for (var i = 0; i < outer.length; i++) {
    if (outer[i][0] < minLon) minLon = outer[i][0];
    if (outer[i][0] > maxLon) maxLon = outer[i][0];
  }

  if (maxLon - minLon > 350) {
    console.log('  Splitting polygon ' + p + ' at antimeridian (lon ' + minLon.toFixed(1) + ' to ' + maxLon.toFixed(1) + ')');
    var subRings = splitRingAtAntimeridian(outer);
    for (var sr = 0; sr < subRings.length; sr++) {
      // Holes would also need splitting, but Natural Earth 110m antimeridian
      // polygons typically don't have holes crossing the boundary
      polygons.push({ outer: subRings[sr], holes: [] });
    }
  } else {
    polygons.push({ outer: outer, holes: holes });
  }
}

console.log('Total polygons after splitting: ' + polygons.length);

// --- Point-in-polygon test (ray casting) ---
function pointInRing(lon, lat, ring) {
  var inside = false;
  var n = ring.length;
  for (var i = 0, j = n - 1; i < n; j = i++) {
    var xi = ring[i][0], yi = ring[i][1];
    var xj = ring[j][0], yj = ring[j][1];
    if ((yi > lat) !== (yj > lat) &&
        lon < (xj - xi) * (lat - yi) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function pointInPolygon(lon, lat, poly) {
  if (!pointInRing(lon, lat, poly.outer)) return false;
  for (var h = 0; h < poly.holes.length; h++) {
    if (pointInRing(lon, lat, poly.holes[h])) return false;
  }
  return true;
}

// Compute bounding boxes
for (var p = 0; p < polygons.length; p++) {
  var poly = polygons[p];
  var minLon = Infinity, maxLon = -Infinity;
  var minLat = Infinity, maxLat = -Infinity;
  for (var i = 0; i < poly.outer.length; i++) {
    if (poly.outer[i][0] < minLon) minLon = poly.outer[i][0];
    if (poly.outer[i][0] > maxLon) maxLon = poly.outer[i][0];
    if (poly.outer[i][1] < minLat) minLat = poly.outer[i][1];
    if (poly.outer[i][1] > maxLat) maxLat = poly.outer[i][1];
  }
  poly.minLon = minLon;
  poly.maxLon = maxLon;
  poly.minLat = minLat;
  poly.maxLat = maxLat;
}

// --- Build raw land mask ---
var grid = new Uint8Array(NX * NY);

console.log('Testing ' + NX + 'x' + NY + ' grid cells...');

for (var j = 0; j < NY; j++) {
  var lat = LAT0 + (j + 0.5) * (LAT1 - LAT0) / NY;
  for (var i = 0; i < NX; i++) {
    var lon = LON0 + (i + 0.5) * (LON1 - LON0) / NX;
    var isLand = false;
    for (var p = 0; p < polygons.length; p++) {
      var poly = polygons[p];
      if (lat < poly.minLat || lat > poly.maxLat) continue;
      if (lon < poly.minLon || lon > poly.maxLon) continue;
      if (pointInPolygon(lon, lat, poly)) {
        isLand = true;
        break;
      }
    }
    grid[j * NX + i] = isLand ? 1 : 0;
  }
  if (j % 20 === 0) process.stdout.write('  row ' + j + '/' + NY + '\r');
}
console.log('');

// Invert: 1=ocean, 0=land
for (var k = 0; k < NX * NY; k++) {
  grid[k] = grid[k] ? 0 : 1;
}

// --- Flood fill from Gulf of Guinea (lon=0, lat=0) ---
var seedI = Math.floor((0 - LON0) / (LON1 - LON0) * NX);
var seedJ = Math.floor((0 - LAT0) / (LAT1 - LAT0) * NY);
console.log('Flood fill seed: i=' + seedI + ', j=' + seedJ);
console.log('  Seed cell value: ' + grid[seedJ * NX + seedI]);

var connected = new Uint8Array(NX * NY);
var queue = [[seedI, seedJ]];
connected[seedJ * NX + seedI] = 1;
var fillCount = 0;

while (queue.length > 0) {
  var cell = queue.shift();
  var ci = cell[0], cj = cell[1];
  fillCount++;

  var neighbors = [
    [(ci + 1) % NX, cj],
    [(ci - 1 + NX) % NX, cj],
    [ci, cj + 1],
    [ci, cj - 1]
  ];

  for (var n = 0; n < neighbors.length; n++) {
    var ni = neighbors[n][0], nj = neighbors[n][1];
    if (nj < 0 || nj >= NY) continue;
    var nk = nj * NX + ni;
    if (grid[nk] && !connected[nk]) {
      connected[nk] = 1;
      queue.push([ni, nj]);
    }
  }
}

console.log('Connected ocean cells: ' + fillCount + ' / ' + (NX * NY) + ' (' + (100 * fillCount / (NX * NY)).toFixed(1) + '%)');

var finalMask = connected;

// --- Manual fixes for narrow straits blocked at 1-degree resolution ---
// Open Gibraltar (lon ~ -5.5, lat ~ 36)
function openStrait(lon, lat, name) {
  var gi = Math.floor((lon - LON0) / (LON1 - LON0) * NX);
  var gj = Math.floor((lat - LAT0) / (LAT1 - LAT0) * NY);
  if (gi >= 0 && gi < NX && gj >= 0 && gj < NY) {
    var k = gj * NX + gi;
    if (!finalMask[k] && grid[k]) {
      // Cell is ocean in raw grid but not connected — force open
      finalMask[k] = 1;
      console.log('  Opened ' + name + ' at i=' + gi + ', j=' + gj);
    } else if (!grid[k]) {
      // Cell is land in raw grid — force to ocean
      grid[k] = 1;
      finalMask[k] = 1;
      console.log('  Forced ' + name + ' open at i=' + gi + ', j=' + gj);
    }
  }
}

// Gibraltar strait
for (var glon = -6; glon <= -4; glon++) {
  for (var glat = 35; glat <= 37; glat++) {
    openStrait(glon, glat, 'Gibraltar');
  }
}

// Now re-flood-fill from any connected ocean cell to pick up Mediterranean etc.
// Use a multi-seed approach: start from all currently-connected cells
console.log('Re-flood-filling to connect Mediterranean...');
var queue2 = [];
for (var k = 0; k < NX * NY; k++) {
  if (finalMask[k] && grid[k]) {
    // Already connected ocean — check if neighbors in raw grid are unconnected
    var ci = k % NX, cj = Math.floor(k / NX);
    var nbrs = [[(ci+1)%NX, cj], [(ci-1+NX)%NX, cj], [ci, cj+1], [ci, cj-1]];
    for (var n = 0; n < nbrs.length; n++) {
      var ni = nbrs[n][0], nj = nbrs[n][1];
      if (nj < 0 || nj >= NY) continue;
      var nk = nj * NX + ni;
      if (grid[nk] && !finalMask[nk]) {
        finalMask[nk] = 1;
        queue2.push([ni, nj]);
      }
    }
  }
}
// Also add the forced-open strait cells as seeds
for (var glon = -6; glon <= -4; glon++) {
  for (var glat = 35; glat <= 37; glat++) {
    var gi2 = Math.floor((glon - LON0) / (LON1 - LON0) * NX);
    var gj2 = Math.floor((glat - LAT0) / (LAT1 - LAT0) * NY);
    if (finalMask[gj2 * NX + gi2]) queue2.push([gi2, gj2]);
  }
}

var extraFill = 0;
while (queue2.length > 0) {
  var cell2 = queue2.shift();
  var ci2 = cell2[0], cj2 = cell2[1];
  extraFill++;
  var nbrs2 = [[(ci2+1)%NX, cj2], [(ci2-1+NX)%NX, cj2], [ci2, cj2+1], [ci2, cj2-1]];
  for (var n2 = 0; n2 < nbrs2.length; n2++) {
    var ni2 = nbrs2[n2][0], nj2 = nbrs2[n2][1];
    if (nj2 < 0 || nj2 >= NY) continue;
    var nk2 = nj2 * NX + ni2;
    if (grid[nk2] && !finalMask[nk2]) {
      finalMask[nk2] = 1;
      queue2.push([ni2, nj2]);
    }
  }
}
console.log('Re-flood-fill connected ' + extraFill + ' additional cells');

// Polar walls
for (var i = 0; i < NX; i++) {
  finalMask[i] = 0;
  finalMask[(NY - 1) * NX + i] = 0;
}

// --- Verify ---
function checkPoint(name, lon, lat, expected) {
  var gi = Math.max(0, Math.min(NX - 1, Math.floor((lon - LON0) / (LON1 - LON0) * NX)));
  var gj = Math.max(0, Math.min(NY - 1, Math.floor((lat - LAT0) / (LAT1 - LAT0) * NY)));
  var val = finalMask[gj * NX + gi];
  var status = (val === expected) ? 'OK' : 'FAIL';
  console.log('  ' + status + ': ' + name + ' = ' + (val ? 'ocean' : 'land'));
  return val === expected;
}

console.log('\nVerification:');
checkPoint('Mid-Pacific (-170, 0)', -170, 0, 1);
checkPoint('Mid-Atlantic (-30, 30)', -30, 30, 1);
checkPoint('Indian Ocean (70, -20)', 70, -20, 1);
checkPoint('Mediterranean (15, 35)', 15, 35, 1);
checkPoint('Drake Passage W (-65, -60)', -65, -60, 1);
checkPoint('Drake Passage C (-60, -58)', -60, -58, 1);
checkPoint('Drake Passage E (-55, -56)', -55, -56, 1);
checkPoint('Southern Ocean (0, -65)', 0, -65, 1);
checkPoint('N Pacific (-150, 40)', -150, 40, 1);
checkPoint('Amazon (-60, -5)', -60, -5, 0);
checkPoint('Sahara (10, 25)', 10, 25, 0);
checkPoint('Australia (135, -25)', 135, -25, 0);
checkPoint('Greenland (-45, 70)', -45, 70, 0);
checkPoint('Antarctica (0, -78)', 0, -78, 0);
checkPoint('Russia (80, 60)', 80, 60, 0);
checkPoint('Kamchatka (158, 56)', 158, 56, 0);

var oceanCount = 0;
for (var k = 0; k < NX * NY; k++) if (finalMask[k]) oceanCount++;
console.log('\nTotal ocean: ' + oceanCount + ' / ' + (NX * NY) + ' (' + (100 * oceanCount / (NX * NY)).toFixed(1) + '%)');

// Drake Passage detail
console.log('\nDrake Passage (. = ocean, # = land):');
for (var lat = -68; lat <= -50; lat += 2) {
  var row = 'lat=' + (lat > -10 ? ' ' : '') + lat + ': ';
  for (var lon = -75; lon <= -50; lon += 2) {
    var gi = Math.floor((lon - LON0) / (LON1 - LON0) * NX);
    var gj = Math.floor((lat - LAT0) / (LAT1 - LAT0) * NY);
    row += finalMask[gj * NX + gi] ? '.' : '#';
  }
  console.log('  ' + row);
}

// --- Encode as hex ---
var hexStr = '';
for (var k = 0; k < NX * NY; k += 4) {
  var nibble = ((finalMask[k] || 0) << 3) | ((finalMask[k+1] || 0) << 2) | ((finalMask[k+2] || 0) << 1) | (finalMask[k+3] || 0);
  hexStr += nibble.toString(16);
}

fs.writeFileSync('/Users/dereklomas/lukebarrington/amoc/v4-physics/mask.json', JSON.stringify({ nx: NX, ny: NY, hex: hexStr }));
console.log('\nWrote mask.json (' + hexStr.length + ' hex chars)');

// --- Simplified coastlines ---
var origPolygons = [];
for (var p = 0; p < geom.arcs.length; p++) {
  origPolygons.push(ringToCoords(geom.arcs[p][0]));
}
var coastlines = [];
for (var p = 0; p < origPolygons.length; p++) {
  var outer = origPolygons[p];
  var simplified = [];
  var step = outer.length > 100 ? 3 : 1;
  for (var i = 0; i < outer.length; i += step) {
    simplified.push([Math.round(outer[i][0] * 100) / 100, Math.round(outer[i][1] * 100) / 100]);
  }
  if (simplified.length > 2) coastlines.push(simplified);
}
fs.writeFileSync('/Users/dereklomas/lukebarrington/amoc/v4-physics/coastlines.json', JSON.stringify(coastlines));
console.log('Wrote coastlines.json (' + coastlines.length + ' polygons)');
