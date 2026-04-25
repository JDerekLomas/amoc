// ============================================================
// UI CONTROLS, PAINT TOOL, SCENARIOS
// ============================================================
// Extracted from index.html. Wires DOM elements to model.js globals.
// Depends on main.js (resetSim, init), renderer.js (drawMapUnderlay, simCanvas),
// gpu-solver.js (updateGPUBuffersAfterPaint, readbackFrameCounter).

// CONTROLS
// ============================================================
document.getElementById('wind-slider').oninput = function(e) { windStrength = +e.target.value; document.getElementById('wind-val').textContent = windStrength.toFixed(2); };
document.getElementById('r-slider').oninput = function(e) { r_friction = +e.target.value; document.getElementById('r-val').textContent = r_friction.toFixed(3); };
document.getElementById('a-slider').oninput = function(e) { A_visc = +e.target.value; document.getElementById('a-val').textContent = A_visc.toExponential(1); };
document.getElementById('speed-slider').oninput = function(e) { stepsPerFrame = +e.target.value; document.getElementById('speed-val').textContent = stepsPerFrame; };
document.getElementById('year-speed-slider').oninput = function(e) { yearSpeed = +e.target.value; document.getElementById('year-speed-val').textContent = yearSpeed.toFixed(2); };
document.getElementById('fw-slider').oninput = function(e) { freshwaterForcing = +e.target.value; document.getElementById('fw-val').textContent = freshwaterForcing.toFixed(2); };
document.getElementById('co2-slider').oninput = function(e) { co2_ppm = +e.target.value; document.getElementById('co2-val').textContent = Math.round(co2_ppm); };
document.getElementById('dwf-slider').oninput = function(e) { gamma_deep_form = +e.target.value; document.getElementById('dwf-val').textContent = gamma_deep_form.toFixed(2); };
document.getElementById('solar-slider').oninput = function(e) { S_solar = +e.target.value; document.getElementById('solar-val').textContent = S_solar.toFixed(0); };
document.getElementById('btn-reset').onclick = resetSim;
document.getElementById('btn-pause').onclick = function() { paused = !paused; this.textContent = paused ? 'Resume' : 'Pause'; this.classList.toggle('active', paused); };
document.getElementById('btn-doublegyre').onclick = function() { doubleGyre = true; this.classList.add('active'); document.getElementById('btn-singlegyre').classList.remove('active'); resetSim(); };
document.getElementById('btn-singlegyre').onclick = function() { doubleGyre = false; this.classList.add('active'); document.getElementById('btn-doublegyre').classList.remove('active'); resetSim(); };
var allViewFields = [
  // Surface ocean
  'temp','psi','speed','sal','density','seaice',
  // Deep ocean
  'deeptemp','deepflow','deepsal',
  // Atmosphere
  'airtemp','moisture','precip','clouds','windcurl','ekman',
  // Land
  'elevation','snow','albedo','landtemp',
  // Observations
  'obssst','obsclouds','obsprecip','obsevap','obssalinity',
  // Diagnostics
  'sstdiff','vort','depth'
];
var viewBtnSelector = allViewFields.map(function(v) { return '#btn-' + v; }).join(',');
allViewFields.forEach(function(v) {
  var el = document.getElementById('btn-' + v);
  if (el) el.onclick = function() { showField = v; document.querySelectorAll(viewBtnSelector).forEach(function(b) { b.classList.remove('active'); }); this.classList.add('active'); };
});
document.getElementById('btn-particles').onclick = function() { showParticles = !showParticles; this.classList.toggle('active', showParticles); };

// ============================================================
// PAINT TOOL
// ============================================================
var paintMode = 'none', brushSize = 3;
document.getElementById('brush-slider').oninput = function(e) { brushSize = +e.target.value; document.getElementById('brush-val').textContent = brushSize; };
document.querySelectorAll('.ptile').forEach(function(btn) {
  btn.addEventListener('click', function() { document.querySelectorAll('.ptile').forEach(function(b) { b.classList.remove('active'); });
    document.getElementById('brush-popup').classList.toggle('show', this.dataset.mode !== 'none');
    this.classList.add('active'); paintMode = this.dataset.mode; simCanvas.style.cursor = paintMode === 'none' ? 'default' : 'crosshair'; }); });
document.querySelector('.ptile[data-mode="none"]').classList.add('active');
function canvasToGrid(cx, cy) { var rect = simCanvas.getBoundingClientRect(); return [Math.floor((cx - rect.left) / rect.width * NX), Math.floor((1 - (cy - rect.top) / rect.height) * NY)]; }
function applyBrush(cx, cy) {
  if (paintMode === 'none') return; var coords = canvasToGrid(cx, cy), ci = coords[0], cj = coords[1], r2 = brushSize * brushSize;
  for (var dj = -brushSize; dj <= brushSize; dj++) for (var di = -brushSize; di <= brushSize; di++) {
    if (di * di + dj * dj > r2) continue; var i = ((ci + di) % NX + NX) % NX, j = cj + dj;
    if (j < 1 || j >= NY - 1) continue; var k = j * NX + i;
    if (paintMode === 'land') { mask[k] = 0; psi[k] = 0; zeta[k] = 0; temp[k] = 0; }
    else if (paintMode === 'ocean') { mask[k] = 1; if (temp[k] === 0) { temp[k] = 28.0 - 0.5 * Math.abs(LAT0 + (j / (NY - 1)) * (LAT1 - LAT0)); } }
    else if (paintMode === 'heat') { if (mask[k]) { zeta[k] -= 0.5; temp[k] += 3; } }
    else if (paintMode === 'cold') { if (mask[k]) { zeta[k] += 0.5; temp[k] -= 3; } }
    else if (paintMode === 'ice') { mask[k] = 1; temp[k] = -5; if (deepTemp) deepTemp[k] = -1; zeta[k] = 0; psi[k] = 0; }
    else if (paintMode === 'wind-cw') { if (mask[k]) zeta[k] -= 0.3; }
    else if (paintMode === 'wind-ccw') { if (mask[k]) zeta[k] += 0.3; }
  }
  if (typeof updateGPUBuffersAfterPaint === 'function') updateGPUBuffersAfterPaint();
  drawMapUnderlay(); readbackFrameCounter = 0;
}
var painting = false;
simCanvas.addEventListener('mousedown', function(e) { if (paintMode === 'none') return; painting = true; applyBrush(e.clientX, e.clientY); });
simCanvas.addEventListener('mousemove', function(e) { if (painting) applyBrush(e.clientX, e.clientY); });
window.addEventListener('mouseup', function() { painting = false; });

// Mobile: long-press to paint, short touch / drag = scroll
var touchPaintTimer = null, touchStartX = 0, touchStartY = 0, touchPaintConfirmed = false;
var PAINT_HOLD_MS = 200, PAINT_MOVE_THRESHOLD = 10;
simCanvas.addEventListener('touchstart', function(e) {
  if (paintMode === 'none' || e.touches.length > 1) return;
  touchStartX = e.touches[0].clientX; touchStartY = e.touches[0].clientY;
  touchPaintConfirmed = false;
  touchPaintTimer = setTimeout(function() {
    touchPaintConfirmed = true; painting = true;
    applyBrush(touchStartX, touchStartY);
  }, PAINT_HOLD_MS);
}, { passive: true });
simCanvas.addEventListener('touchmove', function(e) {
  if (paintMode === 'none') return;
  if (!touchPaintConfirmed && touchPaintTimer) {
    // Check if moved too far — cancel paint, allow scroll
    var dx = e.touches[0].clientX - touchStartX, dy = e.touches[0].clientY - touchStartY;
    if (dx * dx + dy * dy > PAINT_MOVE_THRESHOLD * PAINT_MOVE_THRESHOLD) {
      clearTimeout(touchPaintTimer); touchPaintTimer = null; return;
    }
  }
  if (touchPaintConfirmed && painting) {
    e.preventDefault(); applyBrush(e.touches[0].clientX, e.touches[0].clientY);
  }
}, { passive: false });
simCanvas.addEventListener('touchend', function() {
  if (touchPaintTimer) { clearTimeout(touchPaintTimer); touchPaintTimer = null; }
  painting = false; touchPaintConfirmed = false;
});

// SCENARIOS
// ============================================================
var originalMask = null;
function showScenarioExplanation(text) { var el = document.getElementById('scenario-explanation'), t = document.getElementById('scenario-text'); if (!text) { el.classList.remove('show'); return; } t.innerHTML = text; el.classList.add('show'); }
function lonLatToGrid(lon, lat) { return [Math.round((lon - LON0) / (LON1 - LON0) * (NX - 1)), Math.round((lat - LAT0) / (LAT1 - LAT0) * (NY - 1))]; }
function setMaskRect(lon0, lon1, lat0, lat1, value) {
  var g0 = lonLatToGrid(lon0, lat0), g1 = lonLatToGrid(lon1, lat1);
  for (var j = Math.max(1, Math.min(g0[1], g1[1])); j <= Math.min(NY - 2, Math.max(g0[1], g1[1])); j++)
    for (var i = Math.min(g0[0], g1[0]); i <= Math.max(g0[0], g1[0]); i++) {
      var wi = ((i % NX) + NX) % NX, k = j * NX + wi; mask[k] = value;
      if (value === 0) { psi[k] = 0; zeta[k] = 0; temp[k] = 0; }
      else if (value === 1 && temp[k] === 0) { temp[k] = 28.0 - 0.5 * Math.abs(LAT0 + (j / (NY - 1)) * (LAT1 - LAT0)); } } }
function applyMaskChange() { drawMapUnderlay(); updateGPUBuffersAfterPaint(); initParticles(); }
document.getElementById('sc-drake').addEventListener('click', function() { setMaskRect(-70,-55,-70,-55,1); applyMaskChange(); showScenarioExplanation('<strong>Drake Passage Opened</strong> &mdash; 35 Mya. Antarctica becomes thermally isolated.'); });
document.getElementById('sc-close-drake').addEventListener('click', function() { setMaskRect(-70,-55,-70,-55,0); applyMaskChange(); showScenarioExplanation('<strong>Drake Passage Closed</strong> &mdash; Heat can reach Antarctica from the tropics.'); });
var panamaOpen = false;
document.getElementById('sc-panama').addEventListener('click', function() {
  if (!panamaOpen) { setMaskRect(-85,-75,5,18,1); applyMaskChange(); showScenarioExplanation('<strong>Panama Seaway Open</strong> &mdash; Atlantic-Pacific connected. Gulf Stream weakened.'); panamaOpen = true; this.querySelector('.sc-title').textContent = 'Close Panama Seaway'; }
  else { setMaskRect(-85,-75,5,18,0); applyMaskChange(); showScenarioExplanation('<strong>Isthmus of Panama</strong> &mdash; Gulf Stream intensifies, warming Europe.'); panamaOpen = false; this.querySelector('.sc-title').textContent = 'Open Panama Seaway'; } });
document.getElementById('sc-greenland').addEventListener('click', function() { freshwaterForcing = 2.0; document.getElementById('fw-slider').value = 2.0; document.getElementById('fw-val').textContent = '2.00';
  showScenarioExplanation('<strong>Greenland Melting</strong> &mdash; Fresh water caps the North Atlantic. Can you collapse the AMOC?'); });
document.getElementById('sc-iceage').addEventListener('click', function() { co2_ppm = 180; freshwaterForcing = 0;
  document.getElementById('co2-slider').value = 180; document.getElementById('co2-val').textContent = '180';
  document.getElementById('fw-slider').value = 0; document.getElementById('fw-val').textContent = '0.00';
  showScenarioExplanation('<strong>Ice Age</strong> &mdash; CO\u2082 at 180 ppm. Watch the ice advance and AMOC weaken.'); });
document.getElementById('sc-reset').addEventListener('click', function() {
  windStrength = 1.0; document.getElementById('wind-slider').value = 1; document.getElementById('wind-val').textContent = '1.00';
  r_friction = 0.04; document.getElementById('r-slider').value = 0.04; document.getElementById('r-val').textContent = '0.040';
  A_visc = 2e-4; document.getElementById('a-slider').value = 0.0002; document.getElementById('a-val').textContent = '2.0e-4';
  freshwaterForcing = 0; document.getElementById('fw-slider').value = 0; document.getElementById('fw-val').textContent = '0.00';
  co2_ppm = 420; document.getElementById('co2-slider').value = 420; document.getElementById('co2-val').textContent = '420';
  yearSpeed = 1.0; document.getElementById('year-speed-slider').value = 1; document.getElementById('year-speed-val').textContent = '1.00';
  if (originalMask) { for (var k = 0; k < NX * NY; k++) mask[k] = originalMask[k]; }
  panamaOpen = false; document.getElementById('sc-panama').querySelector('.sc-title').textContent = 'Open Panama Seaway';
  resetSim(); drawMapUnderlay(); updateGPUBuffersAfterPaint(); showScenarioExplanation(''); });
var _origInit = init; init = async function() { await _origInit(); originalMask = new Uint8Array(NX * NY); for (var k = 0; k < NX * NY; k++) originalMask[k] = mask[k]; }; init();

