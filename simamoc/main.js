// ============================================================
// MAIN LOOP, INIT, LAB API
// ============================================================
// Extracted from index.html. Depends on model.js, gpu-solver.js, renderer.js globals.

// ============================================================
// MAIN LOOP
// ============================================================
var frameCount = 0;
async function gpuTick() {
  if (!paused) { gpuRunSteps(stepsPerFrame); readbackFrameCounter++;
    var needReadback = gpuRenderEnabled ? ((readbackFrameCounter - 1) % READBACK_INTERVAL === 0) : true;
    if (needReadback) {
      await gpuReadback();
      var needReupload = stabilityCheck();
      if (needReupload) updateGPUBuffersAfterPaint();
      // Atmosphere update (CPU-side, two-way coupled, runs at readback rate)
      // Sub-step to avoid instability at high speed settings
      if (airTemp && temp) {
        var atmDt = dt * stepsPerFrame;
        var atmSubSteps = Math.max(1, Math.ceil(atmDt / 0.002));
        var atmSubDt = atmDt / atmSubSteps;
        if (!moisture) { moisture = new Float64Array(NX * NY); for (var mi = 0; mi < NX * NY; mi++) moisture[mi] = 0.80 * qSat(airTemp[mi]); }
        if (!precipField) precipField = new Float64Array(NX * NY);
        for (var asub = 0; asub < atmSubSteps; asub++) {
          var airNew = new Float32Array(NX * NY);
          var qNew = new Float64Array(NX * NY);
          for (var aj = 1; aj < NY - 1; aj++) for (var ai = 0; ai < NX; ai++) {
            var ak = aj * NX + ai;
            var aip = (ai+1)%NX, aim = (ai-1+NX)%NX;
            var lapAir = invDx2*(airTemp[aj*NX+aip]+airTemp[aj*NX+aim]-2*airTemp[ak]) + invDy2*(airTemp[(aj+1)*NX+ai]+airTemp[(aj-1)*NX+ai]-2*airTemp[ak]);
            var lapQ = invDx2*(moisture[aj*NX+aip]+moisture[aj*NX+aim]-2*moisture[ak]) + invDy2*(moisture[(aj+1)*NX+ai]+moisture[(aj-1)*NX+ai]-2*moisture[ak]);
            var surfT;
            if (mask[ak]) { surfT = temp[ak]; }
            else if (landTempField && landTempField[ak] !== 0) { surfT = landTempField[ak]; }
            else { surfT = 28-0.55*Math.abs(LAT0+(aj/(NY-1))*(LAT1-LAT0)); }
            var gam = mask[ak] ? gamma_oa : gamma_la;
            var evap = 0;
            if (mask[ak]) { evap = E0 * Math.max(0, qSat(surfT) - moisture[ak]); }
            qNew[ak] = moisture[ak] + atmSubDt * kappa_atm * lapQ + evap;
            var qs_air = qSat(airTemp[ak]);
            var precip = 0;
            if (qNew[ak] > qs_air) { precip = qNew[ak] - qs_air; qNew[ak] = qs_air; }
            qNew[ak] = Math.max(1e-5, qNew[ak]);
            precipField[ak] = precip;
            var latentHeat = 800 * precip;
            airNew[ak] = airTemp[ak] + atmSubDt*(kappa_atm*lapAir + gam*(surfT-airTemp[ak])) + latentHeat;
          }
          for (var ai=0;ai<NX;ai++){airNew[ai]=airNew[NX+ai];airNew[(NY-1)*NX+ai]=airNew[(NY-2)*NX+ai];qNew[ai]=qNew[NX+ai];qNew[(NY-1)*NX+ai]=qNew[(NY-2)*NX+ai];}
          airTemp = airNew;
          for (var mk = 0; mk < NX * NY; mk++) moisture[mk] = qNew[mk];
        }
        // Two-way feedback + evaporative cooling + P-E salinity
        for (var ak = 0; ak < NX * NY; ak++) {
          if (mask[ak]) {
            temp[ak] += atmDt * gamma_ao * (airTemp[ak] - temp[ak]);
            var deficit = Math.max(0, qSat(temp[ak]) - moisture[ak]);
            temp[ak] -= E0 * deficit * 400;
            sal[ak] -= atmDt * freshwaterScale_pe * (precipField[ak] - E0 * deficit);
          }
        }
        // Re-upload corrected temperature to GPU
        if (gpuDevice && gpuTempBuf) {
          var surfTr = new Float32Array(NX * NY * 2);
          for (var tk = 0; tk < NX * NY; tk++) { surfTr[tk] = temp[tk]; surfTr[tk + NX * NY] = sal[tk]; }
          gpuDevice.queue.writeBuffer(gpuTempBuf, 0, surfTr);
        }
      }
      // Update cloud fraction field from regime-based physics
      if (temp) {
        if (!cloudField) cloudField = new Float32Array(NX * NY);
        var cyearPhase = 2 * Math.PI * simTime / T_YEAR;
        var citczLat = 5 * Math.sin(cyearPhase);
        for (var cj = 0; cj < NY; cj++) {
          var clat = LAT0 + (cj / (NY - 1)) * (LAT1 - LAT0);
          var cabsLat = Math.abs(clat);
          for (var ci = 0; ci < NX; ci++) {
            var ck = cj * NX + ci;
            if (!mask[ck]) {
              // Land clouds from precipitation data + latitude
              if (remappedPrecip) {
                var cprecip = remappedPrecip[ck] || 0;
                // Precipitation → cloud fraction: 0mm=0.05, 1000mm=0.40, 2500mm=0.70
                var cpBase = Math.max(0.05, Math.min(0.70, cprecip / 3500));
                // Mid-latitude storm track boost
                var cstormL = 0.10 * Math.max(0, Math.min(1, (cabsLat - 35) / 10)) * Math.max(0, Math.min(1, (65 - cabsLat) / 10));
                // ITCZ convective boost over wet land
                var citczDL = (clat - citczLat) / 12;
                var cconvL = 0.15 * Math.exp(-citczDL * citczDL) * Math.min(1, cprecip / 1500);
                cloudField[ck] = Math.max(0.03, Math.min(0.80, cpBase + cstormL + cconvL));
              } else {
                cloudField[ck] = 0;
              }
              continue;
            }
            var chum = Math.max(0, Math.min(1, (temp[ck] - 5) / 25));
            var cairEst = 28 - 0.55 * cabsLat;
            var clts = Math.max(0, Math.min(1, (cairEst - temp[ck]) / 15));
            var citczD = (clat - citczLat) / 10;
            var cconv = 0.30 * Math.exp(-citczD * citczD) * chum;
            var cwp = 0.20 * Math.max(0, Math.min(1, (temp[ck] - 26) / 4));
            var csubD = (cabsLat - 25) / 10;
            var csub = 0.25 * Math.exp(-csubD * csubD);
            var cstrat = 0.30 * clts * Math.max(0, Math.min(1, (35 - cabsLat) / 20));
            var cstorm = 0.25 * Math.max(0, Math.min(1, (cabsLat - 35) / 10)) * Math.max(0, Math.min(1, (80 - cabsLat) / 15));
            var csoCloud = clat < 0 ? 0.35 * Math.max(0, Math.min(1, (cabsLat - 45) / 10)) : 0;
            var cpolar = 0.15 * Math.max(0, Math.min(1, (cabsLat - 55) / 15));
            cloudField[ck] = Math.max(0.05, Math.min(0.85, cconv + cwp + cstrat + cstorm + csoCloud + cpolar - csub * (1 - chum)));
          }
        }
      }
    }
    advectParticles(); }
  if (gpuRenderEnabled && showField !== 'deeptemp' && showField !== 'deepflow' && showField !== 'depth' && showField !== 'clouds' && showField !== 'obsclouds' && showField !== 'airtemp' && showField !== 'moisture' && showField !== 'precip') { gpuRenderField(); drawOverlay(); } else { draw(); }
  updateStats(); frameCount++;
  if (frameCount % 10 === 0) { drawProfile(); drawRadProfile(); pushAmocSample(); drawAmocChart(); drawMOCSection(); }
  requestAnimationFrame(gpuTick);
}
function cpuTick() {
  if (!paused) { stabilityCheck(); for (var i = 0; i < stepsPerFrame; i++) cpuTimestep(); advectParticles(); }
  draw(); updateStats(); frameCount++;
  if (frameCount % 5 === 0) { drawProfile(); drawRadProfile(); pushAmocSample(); drawAmocChart(); drawMOCSection(); }
  requestAnimationFrame(cpuTick);
}
var monthNames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
function updateStats() {
  var maxV = 0, ke = 0, sampleCount = 0;
  for (var j = 1; j < NY - 1; j += 2) for (var i = 1; i < NX - 1; i += 4) {
    if (!mask[j * NX + i]) continue; var vel = getVel(i, j); var s2 = vel[0] * vel[0] + vel[1] * vel[1];
    if (s2 > maxV * maxV) maxV = Math.sqrt(s2); ke += s2; sampleCount++; }
  var totalOcean = 0; for (var k = 0; k < NX * NY; k++) { if (mask[k]) totalOcean++; }
  if (sampleCount > 0) ke *= totalOcean / sampleCount;
  document.getElementById('stat-vel').textContent = maxV.toFixed(3);
  document.getElementById('stat-ke').textContent = ke.toExponential(2);
  document.getElementById('stat-step').textContent = totalSteps;
  var yearFrac = (simTime % T_YEAR) / T_YEAR; if (yearFrac < 0) yearFrac += 1;
  var seasonText = monthNames[Math.floor(yearFrac * 12) % 12];
  document.getElementById('stat-season').textContent = seasonText;
  var mhS = document.getElementById('mh-season'); if (mhS) mhS.textContent = seasonText;
  var amocSum = 0, amocCount = 0, jAmoc = Math.floor(NY * 0.65);
  var iAtlW = Math.floor(0.28 * NX), iAtlE = Math.floor(0.5 * NX);
  for (var i = iAtlW; i < iAtlE; i++) { var k = jAmoc * NX + i; if (mask[k]) {
    amocSum += (psi[k + 1] - psi[k - 1]) * 0.5 * invDx - (deepPsi ? (deepPsi[k + 1] - deepPsi[k - 1]) * 0.5 * invDx : 0); amocCount++; } }
  amocStrength = amocCount > 0 ? amocSum / amocCount : 0;
  var amocDisplay = Math.abs(amocStrength), amocEl = document.getElementById('stat-amoc');
  var amocSign = amocStrength >= 0 ? '+' : '-';
  if (amocDisplay < 0.001) { amocEl.textContent = amocSign + amocDisplay.toExponential(1) + ' weak'; amocEl.style.color = '#e06050'; }
  else if (amocDisplay > 0.05) { amocEl.textContent = amocSign + amocDisplay.toFixed(3) + ' strong'; amocEl.style.color = '#4aba70'; }
  else { amocEl.textContent = amocSign + amocDisplay.toFixed(3); amocEl.style.color = '#4a9ec8'; }
  var mhA = document.getElementById('mh-amoc'); if (mhA) mhA.textContent = amocEl.textContent;
}
function resetSim() { if (useGPU) gpuReset(); else cpuReset(); initParticles(); }

// INIT
// ============================================================
async function init() {
  await Promise.all([maskLoadPromise, coastLoadPromise, sstLoadPromise, deepLoadPromise, bathyLoadPromise, albedoLoadPromise, precipLoadPromise, salinityLoadPromise, windLoadPromise, cloudLoadPromise, currentsLoadPromise]);
  drawMapUnderlay();
  var gpuOk = false;
  try { gpuOk = await initWebGPU(); } catch (e) { console.warn('WebGPU init failed:', e); }
  if (gpuOk) {
    useGPU = true; document.getElementById('backend-badge').textContent = 'GPU';
    document.getElementById('backend-badge').className = 'gpu-badge gpu';
    console.log('WebGPU active: ' + NX + 'x' + NY);
    try { initGPURenderPipeline(); } catch (e) { gpuRenderEnabled = false; }
  } else {
    useGPU = false; document.getElementById('backend-badge').textContent = 'CPU+FFT';
    initCPU(); initSOR(); initCirculationFromObs(); console.log('CPU+FFT fallback: ' + NX + 'x' + NY);
  }
  drawMapUnderlay(); initFieldCanvas(); initParticles(); initAmocChart();
  if (useGPU) gpuTick(); else cpuTick();
}
(function() { var o = document.getElementById('onboarding-overlay'); if (!o) return;
  if (!localStorage.getItem('amoc-onboarded')) o.classList.remove('hidden');
  var btn = document.getElementById('btn-start-exploring');
  if (btn) btn.addEventListener('click', function() { o.classList.add('hidden'); localStorage.setItem('amoc-onboarded', '1'); }); })();

// LAB API (window.lab)
// ============================================================
window.lab = (function() {
  async function ensureReady() { var w = 0; while (!NX && w < 20000) { await new Promise(function(r) { setTimeout(r, 100); }); w += 100; }
    if (!NX) throw new Error('lab: not initialized'); }
  function getParams() { return { beta:beta, r:r_friction, A:A_visc, windStrength:windStrength, dt:dt, doubleGyre:doubleGyre,
    S_solar:S_solar, A_olr:A_olr, B_olr:B_olr, kappa_diff:kappa_diff, alpha_T:alpha_T,
    H_surface:H_surface, H_deep:H_deep, gamma_mix:gamma_mix, gamma_deep_form:gamma_deep_form, kappa_deep:kappa_deep,
    F_couple_s:F_couple_s, F_couple_d:F_couple_d, r_deep:r_deep, yearSpeed:yearSpeed, freshwaterForcing:freshwaterForcing,
    globalTempOffset:globalTempOffset, T_YEAR:T_YEAR, stepsPerFrame:stepsPerFrame,
    POISSON_ITERS:POISSON_ITERS, DEEP_POISSON_ITERS:DEEP_POISSON_ITERS,
    E0:E0, greenhouse_q:greenhouse_q, q_ref:q_ref, freshwaterScale_pe:freshwaterScale_pe,
    totalSteps:totalSteps, simTime:simTime, paused:paused, showField:showField, NX:NX, NY:NY, useGPU:useGPU }; }
  function setParams(p) {
    if ('beta' in p) beta=p.beta; if ('r' in p) r_friction=p.r; if ('A' in p) A_visc=p.A;
    if ('windStrength' in p) windStrength=p.windStrength; if ('dt' in p) dt=p.dt; if ('doubleGyre' in p) doubleGyre=p.doubleGyre;
    if ('S_solar' in p) S_solar=p.S_solar; if ('A_olr' in p) A_olr=p.A_olr; if ('B_olr' in p) B_olr=p.B_olr;
    if ('kappa_diff' in p) kappa_diff=p.kappa_diff; if ('alpha_T' in p) alpha_T=p.alpha_T;
    if ('H_surface' in p) H_surface=p.H_surface; if ('H_deep' in p) H_deep=p.H_deep;
    if ('gamma_mix' in p) gamma_mix=p.gamma_mix; if ('gamma_deep_form' in p) gamma_deep_form=p.gamma_deep_form;
    if ('kappa_deep' in p) kappa_deep=p.kappa_deep; if ('F_couple_s' in p) F_couple_s=p.F_couple_s;
    if ('F_couple_d' in p) F_couple_d=p.F_couple_d; if ('r_deep' in p) r_deep=p.r_deep;
    if ('yearSpeed' in p) yearSpeed=p.yearSpeed; if ('freshwaterForcing' in p) freshwaterForcing=p.freshwaterForcing;
    if ('globalTempOffset' in p) globalTempOffset=p.globalTempOffset; if ('T_YEAR' in p) T_YEAR=p.T_YEAR;
    if ('stepsPerFrame' in p) stepsPerFrame=p.stepsPerFrame; if ('POISSON_ITERS' in p) POISSON_ITERS=p.POISSON_ITERS;
    if ('DEEP_POISSON_ITERS' in p) DEEP_POISSON_ITERS=p.DEEP_POISSON_ITERS;
    if ('E0' in p) E0=p.E0; if ('greenhouse_q' in p) greenhouse_q=p.greenhouse_q;
    if ('q_ref' in p) q_ref=p.q_ref; if ('freshwaterScale_pe' in p) freshwaterScale_pe=p.freshwaterScale_pe;
    var sm = {windStrength:'wind-slider',r:'r-slider',A:'a-slider',stepsPerFrame:'speed-slider',yearSpeed:'year-speed-slider',freshwaterForcing:'fw-slider',globalTempOffset:'gt-slider'};
    for (var k in sm) { if (k in p) { var el = document.getElementById(sm[k]); if (el) el.value = p[k]; } } return getParams(); }
  async function step(n) { await ensureReady(); var wp=paused; paused=true;
    if (useGPU) {
      while (readbackPending) await new Promise(function(r){setTimeout(r,5);}); var C=500,d=0;
      while (d<n) { var k=Math.min(C,n-d); gpuRunSteps(k); d+=k; if (d%(C*10)===0) await gpuDevice.queue.onSubmittedWorkDone(); }
      await gpuDevice.queue.onSubmittedWorkDone(); await gpuReadback();
    } else { for (var i=0;i<n;i++) cpuTimestep(); }
    paused=wp; return {step:totalSteps,simTime:simTime,simYears:simTime/T_YEAR}; }
  function fields() { return {psi:psi?new Float32Array(psi):null,zeta:zeta?new Float32Array(zeta):null,temp:temp?new Float32Array(temp):null,
    deepTemp:deepTemp?new Float32Array(deepTemp):null,deepPsi:deepPsi?new Float32Array(deepPsi):null,mask:mask?new Uint8Array(mask):null,
    moisture:moisture?new Float64Array(moisture):null,precipField:precipField?new Float64Array(precipField):null,
    NX:NX,NY:NY,LAT0:LAT0,LAT1:LAT1,LON0:LON0,LON1:LON1}; }
  function _lat(j) { return LAT0 + (j / (NY - 1)) * (LAT1 - LAT0); }
  function diagnostics(opts) { opts=opts||{}; var ip=!!opts.profiles; if (!temp||!psi) return {error:'no fields yet'};
    var maxVel=0,KE=0,oc=0,zST=new Float64Array(NY),zSP=new Float64Array(NY),zSU=new Float64Array(NY),zN=new Int32Array(NY);
    var tS=0,tN=0,pS=0,pN=0,gS=0,gN=0,nhS=0,nhN=0,shS=0,shN=0,ice=0,iDx=invDx,iDy=invDy;
    for (var j=1;j<NY-1;j++){var lat=_lat(j),al=Math.abs(lat);for(var i=0;i<NX;i++){var k=j*NX+i;if(!mask[k])continue;oc++;
      var ip1=(i+1)%NX,im1=(i-1+NX)%NX;var u=-(psi[(j+1)*NX+i]-psi[(j-1)*NX+i])*.5*iDy,v=(psi[j*NX+ip1]-psi[j*NX+im1])*.5*iDx;
      var s2=u*u+v*v;if(s2>maxVel*maxVel)maxVel=Math.sqrt(s2);KE+=s2;var T=temp[k];zST[j]+=T;zSP[j]+=psi[k];zSU[j]+=u;zN[j]++;
      gS+=T;gN++;if(al<20){tS+=T;tN++;}if(al>60){pS+=T;pN++;if(lat>0){nhS+=T;nhN++;}else{shS+=T;shN++;}}if(T<-1.5)ice++;}}KE*=.5;
    var aS=0,aN=0,jA=Math.floor(NY*.65),iAW=Math.floor(.28*NX),iAE=Math.floor(.5*NX);
    for(var ia=iAW;ia<iAE;ia++){var ka=jA*NX+ia;if(!mask[ka])continue;aS+=(psi[ka+1]-psi[ka-1])*.5*iDx-(deepPsi?(deepPsi[ka+1]-deepPsi[ka-1])*.5*iDx:0);aN++;}
    var amoc=aN>0?aS/aN:0;var jACC=Math.round((-60-LAT0)/(LAT1-LAT0)*(NY-1)),accS=0,accN=0;
    for(var ic=0;ic<NX;ic++){var kc=jACC*NX+ic;if(!mask[kc])continue;accS+=-(psi[(jACC+1)*NX+ic]-psi[(jACC-1)*NX+ic])*.5*iDy;accN++;}
    var accU=accN>0?accS/accN:0;var mxP=-Infinity,mnP=Infinity,jA0=Math.floor(.5*NY),jA1=Math.floor(.75*NY);
    for(var jg=jA0;jg<jA1;jg++)for(var ig=iAW;ig<iAE;ig++){var kg=jg*NX+ig;if(!mask[kg])continue;if(psi[kg]>mxP)mxP=psi[kg];if(psi[kg]<mnP)mnP=psi[kg];}
    var out={step:totalSteps,simTime:simTime,simYears:simTime/T_YEAR,seasonFrac:((simTime%T_YEAR)/T_YEAR+1)%1,oceanCells:oc,maxVel:maxVel,KE:KE,
      globalSST:gN?gS/gN:NaN,tropicalSST:tN?tS/tN:NaN,polarSST:pN?pS/pN:NaN,nhPolarSST:nhN?nhS/nhN:NaN,shPolarSST:shN?shS/shN:NaN,
      amoc:amoc,accU:accU,iceArea:ice,gyreMaxPsi:mxP,gyreMinPsi:mnP,gyreRangePsi:mxP-mnP};
    // Cloud RMSE vs MODIS observations
    if(cloudField&&obsCloudField){var cSE=0,cN=0;for(var ck=0;ck<NX*NY;ck++){if(!mask[ck]||!obsCloudField[ck])continue;var cErr=cloudField[ck]-obsCloudField[ck];cSE+=cErr*cErr;cN++;}
      out.cloudRMSE=cN>0?Math.sqrt(cSE/cN):NaN;}
    if(ip){var zT=new Float32Array(NY),zP=new Float32Array(NY),zU=new Float32Array(NY);
      for(var jz=0;jz<NY;jz++){zT[jz]=zN[jz]>0?zST[jz]/zN[jz]:NaN;zP[jz]=zN[jz]>0?zSP[jz]/zN[jz]:NaN;zU[jz]=zN[jz]>0?zSU[jz]/zN[jz]:NaN;}
      var lats=new Float32Array(NY);for(var jl=0;jl<NY;jl++)lats[jl]=_lat(jl);
      out.zonalMeanT=Array.from(zT);out.zonalMeanPsi=Array.from(zP);out.zonalMeanU=Array.from(zU);out.latitudes=Array.from(lats);}return out;}
  async function reset(){await ensureReady();var wp=paused;paused=true;while(readbackPending)await new Promise(function(r){setTimeout(r,5);});if(typeof gpuReset==='function')gpuReset();await gpuReadback();paused=wp;return{step:totalSteps};}
  function view(n){var v=['psi','vort','speed','temp','deeptemp','deepflow','depth'];if(v.indexOf(n)<0)throw new Error('view must be one of '+v.join(','));showField=n;return showField;}
  function pause(){paused=true;return paused;} function resume(){paused=false;return paused;} function isPaused(){return paused;}
  async function sweep(knob,values,opts){opts=opts||{};var spp=opts.stepsPerPoint||50000,ss=opts.settleSteps||0,rb=!!opts.resetBetween,res=[];
    for(var i=0;i<values.length;i++){if(rb)await reset();setParams({[knob]:values[i]});if(ss)await step(ss);await step(spp);var d=diagnostics();d._sweep_knob=knob;d._sweep_value=values[i];res.push(d);}return res;}
  async function timeSeries(n,opts){opts=opts||{};var iv=opts.interval||10000,s=[],r=n;while(r>0){var k=Math.min(iv,r);await step(k);s.push(diagnostics());r-=k;}return s;}
  function scenario(name){var m={'drake-open':'sc-drake','drake-close':'sc-close-drake','panama-open':'sc-panama','greenland':'sc-greenland','iceage':'sc-iceage','present':'sc-reset'};
    if(!(name in m))throw new Error('scenario must be one of '+Object.keys(m).join(','));document.getElementById(m[name]).click();return name;}
  async function benchmark(){await ensureReady();var wp=paused;paused=true;while(readbackPending)await new Promise(function(r){setTimeout(r,5);});
    gpuRunSteps(200);await gpuDevice.queue.onSubmittedWorkDone();var t0=performance.now();gpuRunSteps(1000);await gpuDevice.queue.onSubmittedWorkDone();
    var cm=performance.now()-t0;paused=false;var ft=[],last=performance.now();
    await new Promise(function(res){var c=0;function tick(){var n=performance.now();ft.push(n-last);last=n;if(++c<60)requestAnimationFrame(tick);else res();}requestAnimationFrame(tick);});
    paused=wp;await gpuReadback();var bl=0,mZ=0,nn=0;for(var i=0;i<NX*NY;i++){if(!mask[i])continue;var az=Math.abs(zeta[i]);if(az>mZ)mZ=az;if(az>200)bl++;if(zeta[i]!==zeta[i])nn++;}
    var avg=ft.reduce(function(a,b){return a+b;},0)/ft.length;var sorted=ft.slice().sort(function(a,b){return a-b;});
    return{stepsPerSec:Math.round(1000/(cm/1000)),fps:+(1000/avg).toFixed(1),avgFrameMs:+avg.toFixed(1),p95FrameMs:+sorted[Math.floor(sorted.length*.95)].toFixed(1),
      jitterMs:+Math.sqrt(ft.reduce(function(a,b){return a+(b-avg)*(b-avg);},0)/ft.length).toFixed(1),stable:bl===0&&nn===0,maxVorticity:+mZ.toFixed(0)};}
  function poissonCheck(extraIters) {
    extraIters = extraIters || 0;
    var before = poissonResidual();
    if (extraIters > 0) { cpuSolveSOR(extraIters); }
    var after = poissonResidual();
    var psiMax = -Infinity, psiMin = Infinity, zetaMax = -Infinity, zetaMin = Infinity;
    for (var k = 0; k < NX * NY; k++) {
      if (!mask[k]) continue;
      if (psi[k] > psiMax) psiMax = psi[k]; if (psi[k] < psiMin) psiMin = psi[k];
      if (zeta[k] > zetaMax) zetaMax = zeta[k]; if (zeta[k] < zetaMin) zetaMin = zeta[k];
    }
    return { before: before, after: after, extraIters: extraIters,
      psiRange: [+psiMin.toFixed(6), +psiMax.toFixed(6)],
      zetaRange: [+zetaMin.toFixed(2), +zetaMax.toFixed(2)],
      omegaSOR: omegaSOR };
  }
  return{getParams:getParams,setParams:setParams,step:step,reset:reset,view:view,pause:pause,resume:resume,isPaused:isPaused,
    fields:fields,diagnostics:diagnostics,sweep:sweep,timeSeries:timeSeries,scenario:scenario,benchmark:benchmark,poissonCheck:poissonCheck,_version:'0.4'};
})();
console.log('[lab] API ready — try lab.benchmark(), lab.diagnostics()');
