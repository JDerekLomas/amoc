// ============================================================
// OVERLAY UI — reparent sidebar into drawers, drawer logic, FAB, speed presets
// ============================================================
// Depends on: all overlay HTML elements (m-viewbar, m-drawer-*, m-toolbar, etc.)
// Must load after ui.js (needs controls wired) and after overlay HTML is in DOM.

(function() {
  var paintDr = document.getElementById('m-drawer-paint');
  var ctrlDr = document.getElementById('m-drawer-controls');
  var scDr = document.getElementById('m-drawer-scenarios');
  var infoDr = document.getElementById('m-drawer-info');

  // View buttons → top layer bar
  document.getElementById('m-viewbar').appendChild(document.getElementById('grp-views'));

  // Palette + brush → paint drawer
  paintDr.appendChild(document.getElementById('palette'));
  var brush = document.getElementById('brush-popup');
  brush.classList.add('show');
  paintDr.appendChild(brush);

  // Sliders → controls drawer
  function addH3(parent, text) { var h = document.createElement('h3'); h.textContent = text; parent.appendChild(h); }
  addH3(ctrlDr, 'Climate');  ctrlDr.appendChild(document.getElementById('grp-climate'));
  addH3(ctrlDr, 'Ocean');    ctrlDr.appendChild(document.getElementById('grp-ocean'));
  addH3(ctrlDr, 'Speed');    ctrlDr.appendChild(document.getElementById('grp-speed'));
  ctrlDr.appendChild(document.getElementById('grp-actions'));

  // Scenarios → scenarios drawer
  scDr.appendChild(document.getElementById('panel-scenarios'));

  // Physics + charts → info drawer
  var helpDiv = infoDr.querySelector('.m-help');
  infoDr.insertBefore(document.getElementById('panel-physics'), helpDiv);
  infoDr.insertBefore(document.getElementById('panel-profile'), helpDiv);
  infoDr.insertBefore(document.getElementById('panel-rad'), helpDiv);

  // --- Drawer open/close ---
  var tbBtns = document.querySelectorAll('.m-tb-btn');
  var drawers = document.querySelectorAll('.m-drawer');
  var scrim = document.getElementById('m-scrim');
  var openId = null;
  function closeAll() { drawers.forEach(function(d){d.classList.remove('open');}); tbBtns.forEach(function(b){b.classList.remove('active');}); scrim.classList.remove('open'); openId = null; }
  tbBtns.forEach(function(btn) { btn.addEventListener('click', function(e) { e.stopPropagation(); var id = btn.getAttribute('data-drawer');
    if (openId === id) { closeAll(); return; } closeAll(); document.getElementById(id).classList.add('open'); btn.classList.add('active'); scrim.classList.add('open'); openId = id; }); });
  scrim.addEventListener('click', closeAll);
  drawers.forEach(function(d) { var y0 = 0;
    d.addEventListener('touchstart', function(e) { y0 = e.touches[0].clientY; }, {passive:true});
    d.addEventListener('touchmove', function(e) { if (e.touches[0].clientY - y0 > 60 && d.scrollTop <= 0) closeAll(); }, {passive:true}); });

  // --- Pause FAB ---
  document.getElementById('m-fab-pause').addEventListener('click', function() {
    document.getElementById('btn-pause').click(); this.classList.toggle('playing'); this.classList.toggle('paused'); });

  // --- Speed presets ---
  var spdBtns = document.querySelectorAll('#m-speed button');
  spdBtns.forEach(function(btn) {
    btn.addEventListener('click', function() {
      spdBtns.forEach(function(b) { b.classList.remove('active'); });
      btn.classList.add('active');
      var spd = +btn.getAttribute('data-spd');
      var yr = +btn.getAttribute('data-yr');
      stepsPerFrame = spd; yearSpeed = yr;
      document.getElementById('speed-slider').value = spd;
      document.getElementById('speed-val').textContent = spd;
      document.getElementById('year-speed-slider').value = yr;
      document.getElementById('year-speed-val').textContent = yr.toFixed(2);
    });
  });
})();
