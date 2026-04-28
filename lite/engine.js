// SimAMOC Lite — CPU-only physics engine at 360x180
// Uses native 1-degree JSON data, SOR Poisson solver

(function () {
  'use strict';

  // Grid: 360x180 (1-degree, matches working original versions)
  var NX = 360, NY = 180;
  var LON0 = -180, LON1 = 180, LAT0 = -80, LAT1 = 80;
  var dx = 1 / (NX - 1), dy = 1 / (NY - 1);
  var invDx = 1 / dx, invDy = 1 / dy;
  var invDx2 = invDx * invDx, invDy2 = invDy * invDy;

  // Physics parameters
  var beta = 1.0;
  var r_friction = 0.04;
  var A_visc = 2e-4;
  var windStrength = 1.0;
  var stepsPerFrame = 30;
  var paused = false;
  var dt = 5e-5, dtBase = 5e-5;
  var totalSteps = 0, simTime = 0;
  var T_YEAR = 10.0, yearSpeed = 1.0;
  var S_solar = 6.2, A_olr = 1.8, B_olr = 0.13;
  var kappa_diff = 2.5e-4;
  var alpha_T = 0.05, beta_S = 0.8;
  var gamma_mix = 0.001, gamma_deep_form = 0.05;
  var kappa_deep = 2e-5;
  var F_couple_s = 0.5, F_couple_d = 0.0125;
  var r_deep = 0.1;
  var kappa_sal = 2.5e-4, kappa_deep_sal = 2e-5;
  var salRestoringRate = 0.005;
  var freshwaterForcing = 0.0;
  var globalTempOffset = 0.0;
  var co2_ppm = 420;
  var kappa_atm = 3e-3;
  var gamma_oa = 0.005, gamma_ao = 0.001, gamma_la = 0.01;
  var E0 = 0.003, greenhouse_q = 0.4, q_ref = 0.015;
  var freshwaterScale_pe = 0.5;

  // Fields
  var mask, psi, zeta, zetaNew;
  var temp, tempNew, deepTemp, deepTempNew;
  var sal, salNew, deepSal, deepSalNew;
  var deepPsi, deepZeta, deepZetaNew;
  var airTemp, moisture, precipField;
  var depth, windCurlField, ekmanField;
  var elevation;
  var salClimatology;
  var landTempField;

  // Observational data (raw JSON objects)
  var obsSSTData, obsDeepData, obsBathyData, obsSalinityData, obsWindData;
  var obsAlbedoData, obsPrecipData;

  // SOR solver
  var rhoGS, omegaSOR;

  // Particles
  var NP = 2000;
  var px = new Float64Array(NP), py = new Float64Array(NP), page = new Float64Array(NP);
  var MAX_AGE = 400;

  // ── DATA LOADING (native 1-degree JSON) ──

  function loadMask() {
    return fetch('mask.json')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var bits = [];
        for (var c = 0; c < d.hex.length; c++) {
          var v = parseInt(d.hex[c], 16);
          bits.push((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1);
        }
        var srcNX = d.nx || 360, srcNY = d.ny || 180;
        mask = new Uint8Array(NX * NY);
        for (var j = 0; j < NY; j++) {
          var sj = Math.min(Math.floor(j * srcNY / NY), srcNY - 1);
          for (var i = 0; i < NX; i++) {
            var si = Math.min(Math.floor(i * srcNX / NX), srcNX - 1);
            mask[j * NX + i] = bits[sj * srcNX + si] || 0;
          }
        }
        for (var i = 0; i < NX; i++) { mask[i] = 0; mask[(NY - 1) * NX + i] = 0; }
      })
      .catch(function () {});
  }

  // Remap obs data (360x160, lat -79.5..79.5) to sim grid (360x180, lat -80..80)
  function obsIndex(j) {
    var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
    var obsJ = Math.round((lat - (-79.5)) / 159 * 159);
    return (obsJ >= 0 && obsJ < 160) ? obsJ : -1;
  }

  function remapObs(obsArray, obsNX, obsNY) {
    if (!obsArray) return null;
    var out = new Float32Array(NX * NY);
    for (var j = 0; j < NY; j++) {
      var lat = LAT0 + (j / (NY - 1)) * (LAT1 - LAT0);
      var oj = Math.round((lat - (-79.5)) / 159 * (obsNY - 1));
      if (oj < 0 || oj >= obsNY) continue;
      for (var i = 0; i < NX; i++) {
        var oi = Math.min(Math.floor(i * obsNX / NX), obsNX - 1);
        out[j * NX + i] = obsArray[oj * obsNX + oi] || 0;
      }
    }
    return out;
  }

  // ── FIELD INITIALIZATION ──

  function lat(j) { return LAT0 + (j / (NY - 1)) * (LAT1 - LAT0); }
  function cosLat(j) { return Math.max(Math.cos(lat(j) * Math.PI / 180), 0.087); }
  function qSat(T) { return 3.75e-3 * Math.exp(0.067 * T); }

  function initFields() {
    psi = new Float64Array(NX * NY);
    zeta = new Float64Array(NX * NY);
    zetaNew = new Float64Array(NX * NY);
    temp = new Float64Array(NX * NY);
    tempNew = new Float64Array(NX * NY);
    deepTemp = new Float64Array(NX * NY);
    deepTempNew = new Float64Array(NX * NY);
    sal = new Float64Array(NX * NY);
    salNew = new Float64Array(NX * NY);
    deepSal = new Float64Array(NX * NY);
    deepSalNew = new Float64Array(NX * NY);
    deepPsi = new Float64Array(NX * NY);
    deepZeta = new Float64Array(NX * NY);
    deepZetaNew = new Float64Array(NX * NY);
    airTemp = new Float64Array(NX * NY);
    moisture = new Float64Array(NX * NY);
    precipField = new Float64Array(NX * NY);

    // Depth from bathymetry
    depth = new Float32Array(NX * NY);
    elevation = null;
    if (obsBathyData) {
      var obsNX = obsBathyData.nx || 360, obsNY = obsBathyData.ny || 160;
      elevation = new Float32Array(NX * NY);
      for (var j = 0; j < NY; j++) {
        var oj = obsIndex(j);
        for (var i = 0; i < NX; i++) {
          var k = j * NX + i;
          if (oj >= 0 && oj < obsNY) {
            var ok = oj * obsNX + Math.min(i, obsNX - 1);
            depth[k] = Math.max(10, obsBathyData.depth ? (obsBathyData.depth[ok] || 100) : 100);
            elevation[k] = obsBathyData.elevation ? (obsBathyData.elevation[ok] || 0) : 0;
          } else {
            depth[k] = 4000;
          }
        }
      }
    } else {
      for (var k = 0; k < NX * NY; k++) depth[k] = 4000;
    }

    // Wind curl
    windCurlField = new Float32Array(NX * NY);
    if (obsWindData && obsWindData.wind_curl) {
      var wc = remapObs(obsWindData.wind_curl, obsWindData.nx || 360, obsWindData.ny || 160);
      if (wc) {
        var rms = 0, n = 0;
        for (var k = 0; k < NX * NY; k++) { if (mask[k]) { rms += wc[k] * wc[k]; n++; } }
        rms = n > 0 ? Math.sqrt(rms / n) : 1;
        var scale = 0.15 / Math.max(rms, 1e-10);
        for (var k = 0; k < NX * NY; k++) windCurlField[k] = wc[k] * scale;
      }
    } else {
      for (var j = 0; j < NY; j++) {
        var lt = lat(j), ltR = lt * Math.PI / 180;
        for (var i = 0; i < NX; i++)
          windCurlField[j * NX + i] = -Math.sin(2 * Math.PI * j / (NY - 1)) * 0.15 * Math.cos(ltR);
      }
    }

    // Ekman velocities
    ekmanField = new Float32Array(NX * NY * 2);
    for (var j = 1; j < NY - 1; j++) {
      var ltR = lat(j) * Math.PI / 180;
      var sinLat = Math.sin(ltR);
      if (Math.abs(sinLat) < Math.sin(5 * Math.PI / 180))
        sinLat = Math.sin(5 * Math.PI / 180) * (lat(j) >= 0 ? 1 : -1);
      var ve = Math.cos(3 * ltR) / sinLat;
      for (var i = 0; i < NX; i++) {
        ekmanField[j * NX + i] = 0;
        ekmanField[j * NX + i + NX * NY] = ve * 0.15;
      }
    }

    // Salinity climatology
    if (obsSalinityData && obsSalinityData.salinity) {
      salClimatology = remapObs(obsSalinityData.salinity, obsSalinityData.nx || 360, obsSalinityData.ny || 160);
    }

    // Temperature initialization
    var sstNX = obsSSTData ? (obsSSTData.nx || 360) : 0;
    var sstNY = obsSSTData ? (obsSSTData.ny || 160) : 0;
    var deepNX = obsDeepData ? (obsDeepData.nx || 360) : 0;
    var deepNY = obsDeepData ? (obsDeepData.ny || 160) : 0;

    for (var j = 0; j < NY; j++) {
      var lt = lat(j), ltR = lt * Math.PI / 180;
      var oj = obsIndex(j);
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        if (!mask[k]) { temp[k] = 0; deepTemp[k] = 0; sal[k] = 0; deepSal[k] = 0; continue; }

        // SST
        var gotSST = false;
        if (obsSSTData && obsSSTData.sst && oj >= 0 && oj < sstNY) {
          var sv = obsSSTData.sst[oj * sstNX + Math.min(i, sstNX - 1)];
          if (sv > -90) { temp[k] = sv; gotSST = true; }
        }
        if (!gotSST) temp[k] = Math.max(-2, 28 - 0.55 * Math.abs(lt) - 3e-4 * lt * lt);

        // Deep temp
        var gotDeep = false;
        if (obsDeepData && obsDeepData.temp && oj >= 0 && oj < deepNY) {
          var dv = obsDeepData.temp[oj * deepNX + Math.min(i, deepNX - 1)];
          if (dv > -90) { deepTemp[k] = dv; gotDeep = true; }
        }
        if (!gotDeep) deepTemp[k] = 0.5 + 3.0 * j / (NY - 1);

        // Salinity
        if (salClimatology && salClimatology[k] > 1) {
          sal[k] = salClimatology[k];
        } else {
          sal[k] = 34 + 2 * Math.cos(2 * ltR) - 0.5 * Math.cos(4 * ltR);
        }
        deepSal[k] = 34.7 + 0.2 * Math.cos(2 * ltR);
      }
    }

    // Air temperature
    for (var k = 0; k < NX * NY; k++) {
      if (mask[k]) {
        airTemp[k] = temp[k];
      } else {
        var j = Math.floor(k / NX);
        airTemp[k] = 28 - 0.55 * Math.abs(lat(j));
      }
      moisture[k] = 0.80 * qSat(airTemp[k]);
    }

    // SOR init
    rhoGS = Math.cos(Math.PI / NX) + Math.cos(Math.PI / NY);
    omegaSOR = 2 / (1 + Math.sqrt(1 - rhoGS * rhoGS / 4));

    initParticles();
  }

  // ── SOR POISSON SOLVER ──

  function solveSOR(psiArr, zetaArr, nIter) {
    var cx = invDx2, cy = invDy2, cc = -2 * (cx + cy), invCC = 1 / cc;
    for (var iter = 0; iter < nIter; iter++) {
      for (var j = 1; j < NY - 1; j++) {
        for (var i = 0; i < NX; i++) {
          var k = j * NX + i;
          if (!mask[k]) continue;
          var ip = (i + 1) % NX, im = (i - 1 + NX) % NX;
          var res = cx * (psiArr[im + j * NX] + psiArr[ip + j * NX])
                  + cy * (psiArr[i + (j + 1) * NX] + psiArr[i + (j - 1) * NX])
                  + cc * psiArr[k] - zetaArr[k];
          psiArr[k] -= omegaSOR * res * invCC;
        }
      }
    }
  }

  // ── PHYSICS TIMESTEP ──

  function step() {
    var N = NX * NY;
    for (var j = 1; j < NY - 1; j++) {
      var cl = cosLat(j), lt = lat(j), ltR = lt * Math.PI / 180;
      var invDxP = invDx / cl;
      var betaLocal = beta * Math.cos(ltR);
      for (var i = 0; i < NX; i++) {
        var ip = (i + 1) % NX, im = (i - 1 + NX) % NX;
        var k = j * NX + i;
        if (!mask[k]) { zetaNew[k] = 0; tempNew[k] = 0; deepTempNew[k] = 0; continue; }

        var ke = j * NX + ip, kw = j * NX + im;
        var kn = (j + 1) * NX + i, ks = (j - 1) * NX + i;

        // Vorticity
        if (!mask[ke] || !mask[kw] || !mask[kn] || !mask[ks]) {
          zetaNew[k] = zeta[k] * 0.9;
        } else {
          var kne = (j + 1) * NX + ip, knw = (j + 1) * NX + im;
          var kse = (j - 1) * NX + ip, ksw = (j - 1) * NX + im;
          if (!mask[kne] || !mask[knw] || !mask[kse] || !mask[ksw]) {
            zetaNew[k] = zeta[k] * 0.95;
          } else {
            var mDx = dx * cl, mDy = dy;
            var J1 = (psi[ke] - psi[kw]) * (zeta[kn] - zeta[ks]) - (psi[kn] - psi[ks]) * (zeta[ke] - zeta[kw]);
            var J2 = psi[ke] * (zeta[kne] - zeta[kse]) - psi[kw] * (zeta[knw] - zeta[ksw]) - psi[kn] * (zeta[kne] - zeta[knw]) + psi[ks] * (zeta[kse] - zeta[ksw]);
            var J3 = zeta[ke] * (psi[kne] - psi[kse]) - zeta[kw] * (psi[knw] - psi[ksw]) - zeta[kn] * (psi[kne] - psi[knw]) + zeta[ks] * (psi[kse] - psi[ksw]);
            var jac = (J1 + J2 + J3) / (12 * mDx * mDy);
            var betaV = betaLocal * (psi[ke] - psi[kw]) * 0.5 * invDxP;
            var F = windStrength * windCurlField[k];
            var fric = -r_friction * zeta[k];
            var lapZ = invDx2 / (cl * cl) * (zeta[ke] + zeta[kw] - 2 * zeta[k]) + invDy2 * (zeta[kn] + zeta[ks] - 2 * zeta[k]);
            var visc = A_visc * lapZ;
            var dRhodx = -alpha_T * (temp[ke] - temp[kw]) + beta_S * (sal[ke] - sal[kw]);
            var buoyancy = -dRhodx * 0.5 * invDxP;
            var coupling = F_couple_s * (deepPsi[k] - psi[k]);
            zetaNew[k] = Math.max(-500, Math.min(500, zeta[k] + dt * (-jac - betaV + F + fric + visc + buoyancy + coupling)));
          }
        }

        // Temperature
        var tE = mask[ke] ? temp[ke] : temp[k], tW = mask[kw] ? temp[kw] : temp[k];
        var tN = mask[kn] ? temp[kn] : temp[k], tS = mask[ks] ? temp[ks] : temp[k];
        var pE = mask[ke] ? psi[ke] : psi[k], pW = mask[kw] ? psi[kw] : psi[k];
        var pN = mask[kn] ? psi[kn] : psi[k], pS = mask[ks] ? psi[ks] : psi[k];
        var dPdx = (pE - pW) * 0.5 * invDxP, dPdy = (pN - pS) * 0.5 * invDy;
        var dTdx = (tE - tW) * 0.5 * invDxP, dTdy = (tN - tS) * 0.5 * invDy;
        var advec = dPdx * dTdy - dPdy * dTdx;
        var u_ek = ekmanField ? ekmanField[k] * windStrength : 0;
        var v_ek = ekmanField ? ekmanField[k + N] * windStrength : 0;
        advec += u_ek * dTdx + v_ek * dTdy;

        // Solar + OLR + clouds
        var yearPhase = 2 * Math.PI * simTime / T_YEAR;
        var decl = 23.44 * Math.sin(yearPhase) * Math.PI / 180;
        var cosZ = Math.cos(ltR) * Math.cos(decl) + Math.sin(ltR) * Math.sin(decl);
        var qSolar = S_solar * Math.max(0, cosZ);
        var absLat = Math.abs(lt);
        if (absLat > 45) {
          var iceT = Math.max(0, Math.min(1, (temp[k] + 2) / 10));
          var iceFrac = 1 - iceT * iceT * (3 - 2 * iceT);
          qSolar *= 1 - 0.5 * iceFrac * Math.max(0, Math.min(1, (absLat - 45) / 20));
        }
        // Cloud model (regime-based)
        var humidity = Math.max(0, Math.min(1, (temp[k] - 5) / 25));
        var lts = Math.max(0, Math.min(1, ((28 - 0.55 * absLat) - temp[k]) / 15));
        var itczLat = 5 * Math.sin(yearPhase);
        var convCloud = 0.30 * Math.exp(-Math.pow((lt - itczLat) / 10, 2)) * humidity;
        var warmPool = 0.20 * Math.max(0, Math.min(1, (temp[k] - 26) / 4));
        var tradeCu = 0.25 * Math.exp(-Math.pow((absLat - 22) / 12, 2)) * humidity;
        var subsidence = 0.15 * Math.exp(-Math.pow((absLat - 25) / 10, 2));
        if (lt < 0 && lt > -50) subsidence += 0.08 * Math.exp(-Math.pow((lt + 32) / 10, 2));
        var stratocu = Math.max(0, Math.min(0.20, 0.30 * lts * Math.max(0, Math.min(1, (35 - absLat) / 20))));
        var nhStorm = lt > 0 ? 0.22 * Math.max(0, Math.min(1, (absLat - 35) / 10)) * Math.max(0, Math.min(1, (58 - absLat) / 12)) : 0;
        var stormTrack = 0.30 * Math.max(0, Math.min(1, (absLat - 35) / 10)) * Math.max(0, Math.min(1, (80 - absLat) / 15)) + nhStorm;
        var soCloud = lt < 0 ? (0.70 * Math.exp(-Math.pow((absLat - 62) / 7, 2)) + 0.18 * Math.max(0, Math.min(1, (absLat - 53) / 8))) : 0;
        var polarCloud = 0.10 * Math.max(0, Math.min(1, (absLat - 60) / 10));
        var nhCloud = lt > 0 ? 0.35 * Math.max(0, Math.min(1, (absLat - 40) / 12)) * Math.max(0, Math.min(1, (70 - absLat) / 10)) : 0;
        var cirrus = 0.15 * humidity * Math.max(0.1, Math.min(1, 1 - absLat / 80));
        var thickConv = convCloud + warmPool;
        var lowCloud = stratocu + tradeCu + stormTrack + soCloud + nhCloud + polarCloud;
        var cloudFrac = Math.max(0.05, Math.min(0.90, cirrus + thickConv + lowCloud - subsidence * (1 - humidity)));
        var w_total = Math.max(cirrus + thickConv + lowCloud, 0.01);
        var w_ci = cirrus / w_total, w_cv = thickConv / w_total, w_lo = lowCloud / w_total;
        qSolar *= 1 - cloudFrac * (w_ci * 0.05 + w_cv * 0.30 + w_lo * 0.35);

        var olr = A_olr - B_olr * globalTempOffset + B_olr * temp[k];
        if (lt < -53) olr *= 1 + 0.55 * Math.max(0, Math.min(1, (absLat - 58) / 8));
        var cloudGH = cloudFrac * (w_ci * 0.20 + w_cv * 0.10 + w_lo * 0.03);
        var vaporGH = moisture ? greenhouse_q * Math.min(1, moisture[k] / q_ref) : 0;
        var co2GH = 5.35 * Math.log(co2_ppm / 280) / 240;
        var qNet = qSolar - olr * (1 - cloudGH) * (1 - vaporGH) * (1 - co2GH);
        var lapT = invDx2 / (cl * cl) * (tE + tW - 2 * temp[k]) + invDy2 * (tN + tS - 2 * temp[k]);
        var diff = kappa_diff * lapT;

        // Land heat flux
        var landFlux = 0;
        var nLand = (!mask[ke] ? 1 : 0) + (!mask[kw] ? 1 : 0) + (!mask[kn] ? 1 : 0) + (!mask[ks] ? 1 : 0);
        if (nLand > 0 && nLand < 4) {
          var landT = 50 * Math.max(0, cosZ) - 20;
          landFlux = Math.max(-0.5, Math.min(0.5, 0.02 * (landT - temp[k]) * ((4 - nLand) / 4)));
        }

        tempNew[k] = temp[k] + dt * (-advec + qNet + diff + landFlux);
        var dT = tempNew[k] - temp[k];
        if (dT > 0.5) tempNew[k] = temp[k] + 0.5;
        else if (dT < -0.5) tempNew[k] = temp[k] - 0.5;
        tempNew[k] = Math.max(-2.5, Math.min(38, tempNew[k]));

        // Salinity
        var sE = mask[ke] ? sal[ke] : sal[k], sW = mask[kw] ? sal[kw] : sal[k];
        var sN = mask[kn] ? sal[kn] : sal[k], sS = mask[ks] ? sal[ks] : sal[k];
        var salAdvec = dPdx * ((sN - sS) * 0.5 * invDy) - dPdy * ((sE - sW) * 0.5 * invDxP) + u_ek * ((sE - sW) * 0.5 * invDxP) + v_ek * ((sN - sS) * 0.5 * invDy);
        var lapS = invDx2 / (cl * cl) * (sE + sW - 2 * sal[k]) + invDy2 * (sN + sS - 2 * sal[k]);
        var salClim = (salClimatology && salClimatology[k] > 1) ? salClimatology[k] : (34 + 2 * Math.cos(2 * ltR) - 0.5 * Math.cos(4 * ltR));
        var salRestore = salRestoringRate * (salClim - sal[k]);
        var y = j / (NY - 1);
        var fwSal = y > 0.75 ? -freshwaterForcing * 3 * (y - 0.75) * 4 : 0;
        salNew[k] = sal[k] + dt * (-salAdvec + kappa_sal * lapS + salRestore + fwSal);

        // Vertical exchange
        var mldBase = 30 + 70 * Math.pow(absLat / 80, 1.5);
        var mldACC = (lt < -35 && lt > -65) ? 250 * Math.exp(-Math.pow((lt + 50) / 12, 2)) : 0;
        var mldSub = (lt > 50 && lt < 75) ? 150 * Math.exp(-Math.pow((lt - 62) / 8, 2)) : 0;
        var mld = mldBase + mldACC + mldSub;
        var localD = depth[k] || 4000;
        var hS = Math.min(mld, localD), hD = Math.max(1, localD - mld);
        var hasDeep = localD > mld ? 1 : 0;
        var rhoS = -alpha_T * temp[k] + beta_S * sal[k];
        var rhoD = -alpha_T * deepTemp[k] + beta_S * deepSal[k];
        var gam = gamma_mix;
        if ((lt > 40 && rhoS > rhoD) || (lt < -62 && rhoS > rhoD)) gam = gamma_deep_form;
        var vtT = gam * (temp[k] - deepTemp[k]) * hasDeep;
        var vtS = gam * (sal[k] - deepSal[k]) * hasDeep;
        tempNew[k] -= dt * vtT / hS;
        salNew[k] -= dt * vtS / hS;

        // Deep layer T/S
        var dE = mask[ke] ? deepTemp[ke] : deepTemp[k], dW = mask[kw] ? deepTemp[kw] : deepTemp[k];
        var dN = mask[kn] ? deepTemp[kn] : deepTemp[k], dS = mask[ks] ? deepTemp[ks] : deepTemp[k];
        deepTempNew[k] = deepTemp[k] + dt * (vtT / hD + kappa_deep * (invDx2 * (dE + dW - 2 * deepTemp[k]) + invDy2 * (dN + dS - 2 * deepTemp[k]))) * hasDeep;
        var dsE = mask[ke] ? deepSal[ke] : deepSal[k], dsW = mask[kw] ? deepSal[kw] : deepSal[k];
        var dsN = mask[kn] ? deepSal[kn] : deepSal[k], dsS = mask[ks] ? deepSal[ks] : deepSal[k];
        deepSalNew[k] = deepSal[k] + dt * (vtS / hD + kappa_deep_sal * (invDx2 * (dsE + dsW - 2 * deepSal[k]) + invDy2 * (dsN + dsS - 2 * deepSal[k]))) * hasDeep;
      }
    }

    // Swap buffers
    var t;
    t = zeta; zeta = zetaNew; zetaNew = t;
    t = temp; temp = tempNew; tempNew = t;
    t = deepTemp; deepTemp = deepTempNew; deepTempNew = t;
    t = sal; sal = salNew; salNew = t;
    t = deepSal; deepSal = deepSalNew; deepSalNew = t;

    for (var k = 0; k < N; k++) {
      if (!mask[k]) { zeta[k] = 0; temp[k] = 0; deepTemp[k] = 0; sal[k] = 0; deepSal[k] = 0; deepZeta[k] = 0; }
    }

    // Atmosphere
    if (airTemp && moisture) {
      var airNew = new Float64Array(N), qNew = new Float64Array(N);
      for (var aj = 1; aj < NY - 1; aj++) {
        for (var ai = 0; ai < NX; ai++) {
          var ak = aj * NX + ai;
          var aip = (ai + 1) % NX, aim = (ai - 1 + NX) % NX;
          var aE = airTemp[aj * NX + aip], aW = airTemp[aj * NX + aim];
          var aN = airTemp[(aj + 1) * NX + ai], aS = airTemp[(aj - 1) * NX + ai];
          var lapA = invDx2 * (aE + aW - 2 * airTemp[ak]) + invDy2 * (aN + aS - 2 * airTemp[ak]);
          var qE = moisture[aj * NX + aip], qW = moisture[aj * NX + aim];
          var qN = moisture[(aj + 1) * NX + ai], qS = moisture[(aj - 1) * NX + ai];
          var lapQ = invDx2 * (qE + qW - 2 * moisture[ak]) + invDy2 * (qN + qS - 2 * moisture[ak]);
          var surfT = mask[ak] ? temp[ak] : (28 - 0.55 * Math.abs(lat(aj)));
          var gam = mask[ak] ? gamma_oa : gamma_la;
          var exchange = gam * (surfT - airTemp[ak]);
          var evap = 0;
          if (mask[ak]) evap = E0 * Math.max(0, qSat(surfT) - moisture[ak]);
          qNew[ak] = moisture[ak] + dt * kappa_atm * lapQ + evap;
          var qs_air = qSat(airTemp[ak]);
          var precip = 0;
          if (qNew[ak] > qs_air) { precip = qNew[ak] - qs_air; qNew[ak] = qs_air; }
          qNew[ak] = Math.max(1e-5, qNew[ak]);
          precipField[ak] = precip;
          airNew[ak] = airTemp[ak] + dt * (kappa_atm * lapA + exchange) + 800 * precip;
        }
      }
      for (var ai = 0; ai < NX; ai++) {
        airNew[ai] = airNew[NX + ai]; airNew[(NY - 1) * NX + ai] = airNew[(NY - 2) * NX + ai];
        qNew[ai] = qNew[NX + ai]; qNew[(NY - 1) * NX + ai] = qNew[(NY - 2) * NX + ai];
      }
      for (var ak = 0; ak < N; ak++) { airTemp[ak] = airNew[ak]; moisture[ak] = qNew[ak]; }
      for (var ak = 0; ak < N; ak++) {
        if (mask[ak]) {
          temp[ak] += dt * gamma_ao * (airTemp[ak] - temp[ak]);
          var qs = qSat(temp[ak]);
          var deficit = Math.max(0, qs - moisture[ak]);
          temp[ak] -= dt * E0 * deficit * 400;
          sal[ak] -= dt * freshwaterScale_pe * (precipField[ak] - E0 * deficit);
        }
      }
    }

    solveSOR(psi, zeta, 40);

    // Deep vorticity
    for (var j = 1; j < NY - 1; j++) {
      for (var i = 0; i < NX; i++) {
        var ip = (i + 1) % NX, im = (i - 1 + NX) % NX;
        var k = j * NX + i;
        if (!mask[k]) { deepZetaNew[k] = 0; continue; }
        var ke = j * NX + ip, kw = j * NX + im, kn = (j + 1) * NX + i, ks = (j - 1) * NX + i;
        if (!mask[ke] || !mask[kw] || !mask[kn] || !mask[ks]) { deepZetaNew[k] = deepZeta[k] * 0.9; continue; }
        var dPdx = (deepPsi[ke] - deepPsi[kw]) * 0.5 * invDx;
        var dPdy = (deepPsi[kn] - deepPsi[ks]) * 0.5 * invDy;
        var dZdx = (deepZeta[ke] - deepZeta[kw]) * 0.5 * invDx;
        var dZdy = (deepZeta[kn] - deepZeta[ks]) * 0.5 * invDy;
        var jac = dPdx * dZdy - dPdy * dZdx;
        var betaV = beta * Math.cos(lat(j) * Math.PI / 180) * (deepPsi[ke] - deepPsi[kw]) * 0.5 * invDx;
        var fric = -r_deep * deepZeta[k];
        var lapZ = invDx2 * (deepZeta[ke] + deepZeta[kw] - 2 * deepZeta[k]) + invDy2 * (deepZeta[kn] + deepZeta[ks] - 2 * deepZeta[k]);
        var coupling = F_couple_d * (psi[k] - deepPsi[k]);
        var dRdx = -alpha_T * (deepTemp[ke] - deepTemp[kw]) + beta_S * (deepSal[ke] - deepSal[kw]);
        var dTdy = ((mask[kn] ? deepTemp[kn] : deepTemp[k]) - (mask[ks] ? deepTemp[ks] : deepTemp[k])) * 0.5 * invDy;
        deepZetaNew[k] = deepZeta[k] + dt * (-jac - betaV + fric + A_visc * lapZ + coupling + dRdx * 0.5 * invDx + 0.05 * dTdy);
      }
    }
    t = deepZeta; deepZeta = deepZetaNew; deepZetaNew = t;
    for (var k = 0; k < N; k++) { if (!mask[k]) deepZeta[k] = 0; }
    solveSOR(deepPsi, deepZeta, 20);

    totalSteps++;
    simTime += dt * yearSpeed;
  }

  // ── STABILITY ──

  function stabilityCheck() {
    var N = NX * NY, maxZ = 0;
    for (var k = 0; k < N; k++) {
      if (!mask[k]) continue;
      var az = Math.abs(zeta[k]);
      if (az > maxZ) maxZ = az;
      if (az > 500) zeta[k] = zeta[k] > 0 ? 500 : -500;
      if (Math.abs(psi[k]) > 50) psi[k] = psi[k] > 0 ? 50 : -50;
      if (temp[k] > 40) temp[k] = 40; else if (temp[k] < -10) temp[k] = -10;
      if (deepTemp[k] > 30) deepTemp[k] = 30; else if (deepTemp[k] < -2) deepTemp[k] = -2;
      if (deepPsi && Math.abs(deepPsi[k]) > 50) deepPsi[k] = deepPsi[k] > 0 ? 50 : -50;
      if (deepZeta && Math.abs(deepZeta[k]) > 500) deepZeta[k] = deepZeta[k] > 0 ? 500 : -500;
      if (zeta[k] !== zeta[k] || psi[k] !== psi[k] || temp[k] !== temp[k]) {
        zeta[k] = 0; psi[k] = 0; temp[k] = 0; deepTemp[k] = 0;
        if (deepPsi) deepPsi[k] = 0; if (deepZeta) deepZeta[k] = 0;
      }
    }
    if (maxZ > 200) {
      var damp = 200 / maxZ;
      for (var k = 0; k < N; k++) { if (mask[k]) zeta[k] *= damp; }
    }
    var maxV = 0;
    for (var j = 1; j < NY - 1; j += 2) {
      for (var i = 1; i < NX - 1; i += 4) {
        if (!mask[j * NX + i]) continue;
        var vel = getVel(i, j);
        var s2 = vel[0] * vel[0] + vel[1] * vel[1];
        if (s2 > maxV) maxV = s2;
      }
    }
    maxV = Math.sqrt(maxV);
    var dtTarget = maxV > 0 ? Math.min(dtBase, 0.3 * dx / maxV) : dtBase;
    dt = 0.7 * dt + 0.3 * dtTarget;
  }

  // ── VELOCITY / PARTICLES ──

  function getVel(fi, fj) {
    var i = Math.floor(fi), j = Math.min(Math.max(Math.floor(fj), 1), NY - 2);
    i = ((i % NX) + NX) % NX;
    var ip = (i + 1) % NX, im = (i - 1 + NX) % NX;
    var cl = cosLat(j);
    return [
      -(psi[(j + 1) * NX + i] - psi[(j - 1) * NX + i]) * 0.5 * invDy,
      (psi[j * NX + ip] - psi[j * NX + im]) * 0.5 * invDx / cl
    ];
  }

  function spawnInOcean() {
    var x, y, tries = 0;
    do { x = Math.random() * NX; y = 2 + Math.random() * (NY - 4); tries++; }
    while (!mask[Math.floor(y) * NX + (Math.floor(x) % NX)] && tries < 100);
    return [x, y];
  }

  function initParticles() {
    for (var p = 0; p < NP; p++) {
      var pos = spawnInOcean();
      px[p] = pos[0]; py[p] = pos[1]; page[p] = Math.floor(Math.random() * MAX_AGE);
    }
  }

  function advectParticles() {
    var dtA = dt * stepsPerFrame;
    for (var p = 0; p < NP; p++) {
      var vel = getVel(px[p], py[p]);
      px[p] += vel[0] * dtA * invDx;
      py[p] += vel[1] * dtA * invDy;
      if (px[p] >= NX) px[p] -= NX;
      if (px[p] < 0) px[p] += NX;
      page[p]++;
      var gi = ((Math.floor(px[p]) % NX) + NX) % NX, gj = Math.floor(py[p]);
      if (gj < 1 || gj >= NY - 1 || !mask[gj * NX + gi] || page[p] > MAX_AGE) {
        var pos = spawnInOcean(); px[p] = pos[0]; py[p] = pos[1]; page[p] = 0;
      }
    }
  }

  // ── DIAGNOSTICS ──

  function computeAMOC() {
    var sum = 0, cnt = 0;
    var jA = Math.floor(NY * 0.65);
    var iW = Math.floor(0.28 * NX), iE = Math.floor(0.5 * NX);
    for (var i = iW; i < iE; i++) {
      var k = jA * NX + i;
      if (!mask[k]) continue;
      var ip = (i + 1) % NX, im = (i - 1 + NX) % NX;
      sum += (psi[jA * NX + ip] - psi[jA * NX + im]) * 0.5 * invDx;
      cnt++;
    }
    return cnt > 0 ? sum / cnt : 0;
  }

  function computeRMSE() {
    if (!obsSSTData || !obsSSTData.sst) return NaN;
    var se = 0, n = 0;
    var sstNX = obsSSTData.nx || 360, sstNY = obsSSTData.ny || 160;
    for (var j = 0; j < NY; j++) {
      var oj = obsIndex(j);
      if (oj < 0 || oj >= sstNY) continue;
      for (var i = 0; i < NX; i++) {
        var k = j * NX + i;
        if (!mask[k]) continue;
        var obsV = obsSSTData.sst[oj * sstNX + Math.min(i, sstNX - 1)];
        if (obsV <= -90) continue;
        var err = temp[k] - obsV;
        se += err * err; n++;
      }
    }
    return n > 0 ? Math.sqrt(se / n) : NaN;
  }

  function getSeason() {
    var phase = (simTime / T_YEAR % 1 + 1) % 1;
    var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    return months[Math.floor(phase * 12) % 12];
  }

  function reset() { totalSteps = 0; simTime = 0; initFields(); }

  // ── INITIALIZATION ──

  function loadJSON(url, name) {
    console.log('[lite] loading ' + name + '...');
    return fetch(url).then(function(r) {
      if (!r.ok) throw new Error(name + ': HTTP ' + r.status);
      return r.json();
    }).then(function(d) {
      console.log('[lite] ' + name + ' loaded (' + (d.nx||'?') + 'x' + (d.ny||'?') + ')');
      return d;
    }).catch(function(e) {
      console.warn('[lite] ' + name + ' failed:', e.message);
      return null;
    });
  }

  function init() {
    return Promise.all([
      loadMask(),
      loadJSON('../sst_global_1deg.json', 'SST').then(function(d){obsSSTData=d;}),
      loadJSON('../deep_temp_1deg.json', 'deep temp').then(function(d){obsDeepData=d;}),
      loadJSON('../bathymetry_1deg.json', 'bathymetry').then(function(d){obsBathyData=d;}),
      loadJSON('../salinity_1deg.json', 'salinity').then(function(d){obsSalinityData=d;}),
      loadJSON('../wind_stress_1deg.json', 'wind stress').then(function(d){obsWindData=d;}),
    ]).then(function () {
      if (!mask) {
        mask = new Uint8Array(NX * NY);
        for (var j = 1; j < NY - 1; j++) for (var i = 0; i < NX; i++) mask[j * NX + i] = 1;
      }
      initFields();
      console.log('SimAMOC Lite initialized: ' + NX + 'x' + NY + ' (SOR Poisson)');
    });
  }

  // ── PUBLIC API ──

  window.sim = {
    init: init, step: step, stabilityCheck: stabilityCheck,
    advectParticles: advectParticles, reset: reset,
    getVel: getVel, computeAMOC: computeAMOC,
    computeRMSE: computeRMSE, getSeason: getSeason,
    get NX() { return NX; }, get NY() { return NY; },
    get LON0() { return LON0; }, get LON1() { return LON1; },
    get LAT0() { return LAT0; }, get LAT1() { return LAT1; },
    get mask() { return mask; },
    get psi() { return psi; }, get zeta() { return zeta; },
    get temp() { return temp; }, get deepTemp() { return deepTemp; },
    get sal() { return sal; }, get deepSal() { return deepSal; },
    get deepPsi() { return deepPsi; },
    get airTemp() { return airTemp; }, get moisture() { return moisture; },
    get precipField() { return precipField; },
    get depth() { return depth; }, get elevation() { return elevation; },
    get windCurlField() { return windCurlField; },
    get obsSSTData() { return obsSSTData; },
    get obsSalinityData() { return obsSalinityData; },
    get obsBathyData() { return obsBathyData; },
    get px() { return px; }, get py() { return py; }, get page() { return page; },
    get NP() { return NP; }, get MAX_AGE() { return MAX_AGE; },
    get totalSteps() { return totalSteps; }, get simTime() { return simTime; },
    get paused() { return paused; }, set paused(v) { paused = v; },
    get stepsPerFrame() { return stepsPerFrame; }, set stepsPerFrame(v) { stepsPerFrame = v; },
    get yearSpeed() { return yearSpeed; }, set yearSpeed(v) { yearSpeed = v; },
    get windStrength() { return windStrength; }, set windStrength(v) { windStrength = v; },
    get freshwaterForcing() { return freshwaterForcing; }, set freshwaterForcing(v) { freshwaterForcing = v; },
    get co2_ppm() { return co2_ppm; }, set co2_ppm(v) { co2_ppm = v; },
    get S_solar() { return S_solar; }, set S_solar(v) { S_solar = v; },
    obsIndex: obsIndex,
  };
})();
