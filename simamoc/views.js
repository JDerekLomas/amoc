// ============================================================
// VIEWS — Layer × Variable × Mode registry
// ============================================================
// One source of truth for every viewable map. Each variable belongs
// to a layer (shallow ocean / deep ocean / atmosphere / land / static)
// and exposes one or more modes: 'sim', 'obs', 'diff'.
//
// showField encodes both variable and mode:
//   - 'temp'         → sim mode of variable 'temp'
//   - 'temp.obs'     → obs mode
//   - 'temp.diff'    → sim − obs anomaly
// Legacy aliases (e.g. 'obsclouds' === 'clouds.obs') are mapped at the bottom.
//
// Depends on: renderer.js colormap helpers (tempToRGB, salToRGB, ...)
//             model.js obs arrays (obsSSTData, obsSalinityData, ...)

// ---------- Bipolar diff colormap ----------
function diffToRGB(v, halfRange) {
  if (!isFinite(v)) return [40, 40, 50];
  var t = Math.max(-1, Math.min(1, v / halfRange));
  if (t > 0) {
    var r = 240, g = Math.floor(240 - 200 * t), b = Math.floor(240 - 220 * t);
    return [r, g, b];
  } else {
    var s = -t;
    return [Math.floor(240 - 220 * s), Math.floor(240 - 130 * s), 250];
  }
}

// ---------- Scalar 0..1 helpers ----------
function albedoToRGB(a) {
  var t = Math.max(0, Math.min(1, a));
  var v = Math.floor(40 + 215 * t);
  return [v, v, Math.min(255, v + 5)];
}
function iceFracToRGB(a) {
  var t = Math.max(0, Math.min(1, a));
  return [Math.floor(180 + 75 * t), Math.floor(210 + 45 * t), 255];
}
function snowToRGB(a) { return iceFracToRGB(a); }
function evapToRGB(e) {
  // e ~ kg/m^2/s  (~1e-5)
  var t = Math.max(0, Math.min(1, e / 2e-5));
  return [Math.floor(60 + 90 * t), Math.floor(120 + 100 * t), Math.floor(180 + 60 * t)];
}
function windCurlToRGB(c) {
  var s = Math.max(-1, Math.min(1, c / 5e-7));
  if (s > 0) return [Math.floor(220 - 80 * s), Math.floor(150 - 60 * s), 80];
  return [80, Math.floor(150 - 60 * (-s)), Math.floor(220 - 80 * (-s))];
}
function windMagToRGB(m) {
  var t = Math.max(0, Math.min(1, m / 0.3));
  return [Math.floor(40 + 200 * t), Math.floor(60 + 150 * t), Math.floor(120 - 80 * t)];
}
function curMagToRGB(m) {
  var t = Math.max(0, Math.min(1, m / 0.5));
  return [Math.floor(30 + 220 * t), Math.floor(60 + 180 * t), Math.floor(120 + 60 * t)];
}

// ---------- FIELDS registry ----------
// modes: which {sim, obs, diff} each variable supports
// layer: shallow | deep | atm | land | static
// label: short UI label
// diffScale: half-range (degrees, units) for diff colormap
var FIELDS = {
  // Shallow ocean
  temp:     { layer: 'shallow', label: 'Temp',      modes: ['sim','obs','diff'], diffScale: 3,    legend: 'sst' },
  sal:      { layer: 'shallow', label: 'Salinity',  modes: ['sim','obs','diff'], diffScale: 1.5,  legend: 'sal' },
  psi:      { layer: 'shallow', label: 'ψ',          modes: ['sim'],              legend: 'psi' },
  speed:    { layer: 'shallow', label: 'Speed',     modes: ['sim','obs'],         legend: 'speed' },
  density:  { layer: 'shallow', label: 'Density',   modes: ['sim'],              legend: 'density' },
  vort:     { layer: 'shallow', label: 'Vorticity', modes: ['sim'],              legend: 'vort' },

  // Deep ocean
  deeptemp: { layer: 'deep',    label: 'Temp',      modes: ['sim','obs','diff'], diffScale: 1.5,  legend: 'sst' },
  deepflow: { layer: 'deep',    label: 'ψ',          modes: ['sim'],              legend: 'psi' },

  // Atmosphere
  airtemp:  { layer: 'atm',     label: 'Air Temp',  modes: ['sim','obs','diff'], diffScale: 3,    legend: 'sst' },
  moisture: { layer: 'atm',     label: 'Humidity',  modes: ['sim','obs'],        legend: 'moisture' },
  clouds:   { layer: 'atm',     label: 'Clouds',    modes: ['sim','obs','diff'], diffScale: 0.3,  legend: 'clouds' },
  precip:   { layer: 'atm',     label: 'Precip',    modes: ['sim','obs','diff'], diffScale: 0.001,legend: 'precip' },
  wind:     { layer: 'atm',     label: 'Wind',      modes: ['obs'],              legend: 'wind' },

  // Land / surface forcing
  albedo:   { layer: 'land',    label: 'Albedo',    modes: ['obs'],              legend: 'albedo' },
  lst:      { layer: 'land',    label: 'Land Temp', modes: ['obs'],              legend: 'sst' },
  seaice:   { layer: 'land',    label: 'Sea Ice',   modes: ['obs'],              legend: 'ice' },
  snow:     { layer: 'land',    label: 'Snow',      modes: ['obs'],              legend: 'ice' },
  evap:     { layer: 'land',    label: 'Evap',      modes: ['obs'],              legend: 'evap' },
  currents: { layer: 'land',    label: 'Currents',  modes: ['obs'],              legend: 'speed' },

  // Static
  depth:    { layer: 'static',  label: 'Depth',     modes: ['sim'],              legend: 'depth' }
};

var LAYERS = [
  { id: 'shallow', label: 'Surface ocean' },
  { id: 'deep',    label: 'Deep ocean' },
  { id: 'atm',     label: 'Atmosphere' },
  { id: 'land',    label: 'Land / surface' },
  { id: 'static',  label: 'Static' }
];

// ---------- Field id parsing ----------
function parseField(showField) {
  // Legacy aliases
  if (showField === 'obsclouds') return { variable: 'clouds', mode: 'obs' };
  var dot = showField.indexOf('.');
  if (dot < 0) return { variable: showField, mode: 'sim' };
  return { variable: showField.slice(0, dot), mode: showField.slice(dot + 1) };
}

function makeField(variable, mode) {
  var spec = FIELDS[variable];
  if (!spec) return variable; // unknown — pass through
  if (mode === 'sim') return variable;
  return variable + '.' + mode;
}

// ---------- Sim value accessors at cell k ----------
// Return the simulated value for variable at index k, or null if unavailable.
function simValue(variable, k) {
  switch (variable) {
    case 'temp':     return typeof temp !== 'undefined' && temp ? temp[k] : null;
    case 'sal':      return typeof sal !== 'undefined' && sal ? sal[k] : null;
    case 'deeptemp': return typeof deepTemp !== 'undefined' && deepTemp ? deepTemp[k] : null;
    case 'airtemp':  return typeof airTemp !== 'undefined' && airTemp ? airTemp[k] : null;
    case 'moisture': return typeof moisture !== 'undefined' && moisture ? moisture[k] : null;
    case 'precip':   return typeof precipField !== 'undefined' && precipField ? precipField[k] : null;
    case 'clouds':   return typeof cloudField !== 'undefined' && cloudField ? cloudField[k] : null;
    case 'psi':      return typeof psi !== 'undefined' && psi ? psi[k] : null;
    case 'deepflow': return typeof deepPsi !== 'undefined' && deepPsi ? deepPsi[k] : null;
    case 'vort':     return typeof zeta !== 'undefined' && zeta ? zeta[k] : null;
    case 'density':  {
      if (typeof temp === 'undefined' || !temp) return null;
      var s = (typeof sal !== 'undefined' && sal) ? sal[k] : 35;
      return temp[k] - 0.5 * (s - 35);
    }
    case 'depth':    return typeof depth !== 'undefined' && depth ? depth[k] : null;
  }
  return null;
}

// ---------- Obs value accessors ----------
function obsValue(variable, k) {
  switch (variable) {
    case 'temp':
      if (typeof obsSSTData !== 'undefined' && obsSSTData && obsSSTData.sst) return obsSSTData.sst[k];
      return null;
    case 'sal':
      if (typeof obsSalinityData !== 'undefined' && obsSalinityData && obsSalinityData.salinity) {
        var s = obsSalinityData.salinity[k]; return s > 1 ? s : null;
      }
      return null;
    case 'deeptemp':
      if (typeof obsDeepData !== 'undefined' && obsDeepData && obsDeepData.temp) return obsDeepData.temp[k];
      return null;
    case 'airtemp':
      if (typeof obsAirTempData !== 'undefined' && obsAirTempData && obsAirTempData.air_temp) return obsAirTempData.air_temp[k];
      return null;
    case 'moisture':
      // Use column water vapor as a proxy if available
      if (typeof obsEvapData !== 'undefined' && obsEvapData && obsEvapData.evaporation) return obsEvapData.evaporation[k];
      return null;
    case 'clouds':
      if (typeof obsCloudField !== 'undefined' && obsCloudField) return obsCloudField[k];
      return null;
    case 'precip':
      if (typeof remappedPrecip !== 'undefined' && remappedPrecip) return remappedPrecip[k];
      if (typeof obsPrecipData !== 'undefined' && obsPrecipData && obsPrecipData.precipitation) return obsPrecipData.precipitation[k];
      return null;
    case 'wind':
      if (typeof obsWindData !== 'undefined' && obsWindData && obsWindData.tau_x && obsWindData.tau_y) {
        var tx = obsWindData.tau_x[k], ty = obsWindData.tau_y[k];
        return Math.sqrt(tx*tx + ty*ty);
      }
      return null;
    case 'speed':
    case 'currents':
      if (typeof obsCurrentsData !== 'undefined' && obsCurrentsData && obsCurrentsData.u && obsCurrentsData.v) {
        var u = obsCurrentsData.u[k], v = obsCurrentsData.v[k];
        return Math.sqrt(u*u + v*v);
      }
      return null;
    case 'albedo':
      if (typeof remappedAlbedo !== 'undefined' && remappedAlbedo) return remappedAlbedo[k];
      return null;
    case 'lst':
      if (typeof remappedLST !== 'undefined' && remappedLST) return remappedLST[k];
      return null;
    case 'seaice':
      if (typeof remappedSeaIce !== 'undefined' && remappedSeaIce) return remappedSeaIce[k];
      return null;
    case 'snow':
      if (typeof obsSnowData !== 'undefined' && obsSnowData && obsSnowData.snow_cover) return obsSnowData.snow_cover[k];
      return null;
    case 'evap':
      if (typeof remappedEvap !== 'undefined' && remappedEvap) return remappedEvap[k];
      return null;
  }
  return null;
}

// ---------- Cell colormap dispatch ----------
// Returns [r,g,b] or null if value unavailable / out of bounds for this cell.
function getViewRGB(showField, k) {
  var p = parseField(showField);
  var spec = FIELDS[p.variable];
  if (!spec) return null;

  if (p.mode === 'diff') {
    var s = simValue(p.variable, k);
    var o = obsValue(p.variable, k);
    if (s == null || o == null) return null;
    return diffToRGB(s - o, spec.diffScale || 1);
  }

  var v = (p.mode === 'obs') ? obsValue(p.variable, k) : simValue(p.variable, k);
  if (v == null) return null;

  // Variable-specific colormap
  switch (p.variable) {
    case 'temp':
    case 'deeptemp':
    case 'airtemp':
    case 'lst':       return tempToRGB(v);
    case 'sal':       return salToRGB(v);
    case 'density':   return densityToRGB(temp[k], sal ? sal[k] : 35);
    case 'depth':     return depthToRGB(v);
    case 'clouds':    return cloudFracToRGB(v);
    case 'moisture':  return moistureToRGB(v);
    case 'precip':    return precipToRGB(v);
    case 'psi':
    case 'deepflow':  return psiToRGB(v, _psiAbsMax || 1);
    case 'vort':      return vortToRGB(v, _vortAbsMax || 1);
    case 'speed':     return curMagToRGB(v);
    case 'currents':  return curMagToRGB(v);
    case 'wind':      return windMagToRGB(v);
    case 'albedo':    return albedoToRGB(v);
    case 'seaice':    return iceFracToRGB(v);
    case 'snow':      return snowToRGB(v);
    case 'evap':      return evapToRGB(v);
  }
  return null;
}

// Globals for ψ/vort normalization (set per-frame by renderer).
var _psiAbsMax = 1, _vortAbsMax = 1;
function setViewNormalization(absMax, vortAbs) {
  _psiAbsMax = absMax || 1;
  _vortAbsMax = vortAbs || absMax || 1;
}

// Whether this view shows on land (vs. only ocean cells).
function viewShowsLand(showField) {
  var p = parseField(showField);
  return p.variable === 'airtemp' || p.variable === 'moisture' || p.variable === 'precip' ||
         p.variable === 'clouds'  || p.variable === 'albedo'   || p.variable === 'lst' ||
         p.variable === 'snow'    || p.variable === 'evap';
}
