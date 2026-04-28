// UI chrome: top bar, footer, diagnostics panel, layer switcher, perturbation controls
// Dark mode, minimal, scientific aesthetic

export class UI {
  constructor(container, renderer, coupler) {
    this.container = container;
    this.renderer = renderer;
    this.coupler = coupler;

    this.running = true;
    this.speedMultiplier = 1;
    this.speedSteps = [1, 4, 16, 64, 256];
    this.speedIndex = 0;

    // Perturbation state
    this.freshwaterHosing = 0;    // Sv into North Atlantic
    this.co2Multiplier = 1.0;     // multiplier on OLR reduction (1 = modern, 2 = doubled)
    this.globalTempOffset = 0;    // °C offset

    // Sparkline history
    this.history = {
      sst: [],
      amoc: [],
      energy: [],
    };
    this.maxHistory = 200;

    this._build();
    this._bindKeys();
  }

  _build() {
    this.container.innerHTML = '';
    this.container.style.cssText = `
      display: flex; flex-direction: column; width: 100vw; height: 100vh;
      background: #111; color: #ccc; font-family: Inter, system-ui, sans-serif;
      overflow: hidden; user-select: none;
    `;

    // Top bar
    this.topBar = document.createElement('div');
    this.topBar.style.cssText = `
      display: flex; align-items: center; justify-content: space-between;
      padding: 6px 16px; height: 32px; background: #1a1a1a;
      border-bottom: 1px solid #333; font-size: 13px; flex-shrink: 0;
    `;

    const title = document.createElement('div');
    title.style.cssText = 'font-weight: 600; letter-spacing: 0.5px;';
    title.textContent = 'SimAMOC';

    this.statusEl = document.createElement('div');
    this.statusEl.style.cssText = 'color: #888; font-size: 12px; font-variant-numeric: tabular-nums;';

    const panelToggle = document.createElement('button');
    panelToggle.style.cssText = 'background:none;border:1px solid #444;color:#888;padding:2px 8px;border-radius:3px;cursor:pointer;font-size:11px;font-family:inherit;';
    panelToggle.textContent = 'Panel';
    panelToggle.onclick = () => this._togglePanel();

    this.topBar.append(title, this.statusEl, panelToggle);

    // Main area (viewport + optional panel)
    const mainArea = document.createElement('div');
    mainArea.style.cssText = 'flex:1;display:flex;overflow:hidden;';

    // Viewport
    this.viewport = document.createElement('div');
    this.viewport.style.cssText = `
      flex: 1; display: flex; align-items: center; justify-content: center;
      background: #0a0a0a; position: relative; overflow: hidden;
    `;

    // Right panel (diagnostics + perturbations)
    this.panel = document.createElement('div');
    this.panel.style.cssText = `
      width: 240px; background: #151515; border-left: 1px solid #333;
      overflow-y: auto; padding: 12px; font-size: 11px; flex-shrink: 0;
      display: none;
    `;
    this._buildPanel();

    mainArea.append(this.viewport, this.panel);

    // Footer
    this.footer = document.createElement('div');
    this.footer.style.cssText = `
      display: flex; align-items: center; justify-content: space-between;
      padding: 4px 12px; height: 28px; background: #1a1a1a;
      border-top: 1px solid #333; font-size: 11px; flex-shrink: 0;
      font-variant-numeric: tabular-nums;
    `;

    // Play/pause + speed
    const controls = document.createElement('div');
    controls.style.cssText = 'display: flex; align-items: center; gap: 6px;';

    this.playBtn = document.createElement('button');
    this.playBtn.style.cssText = this._btnStyle();
    this.playBtn.textContent = 'II';
    this.playBtn.onclick = () => this.togglePlay();

    this.speedBtn = document.createElement('button');
    this.speedBtn.style.cssText = this._btnStyle();
    this.speedBtn.textContent = '1x';
    this.speedBtn.onclick = () => this.cycleSpeed();

    controls.append(this.playBtn, this.speedBtn);

    // Diagnostics bar
    this.diagEl = document.createElement('div');
    this.diagEl.style.cssText = 'display: flex; gap: 12px; color: #999; font-size: 11px;';

    // Layer switcher
    const layers = document.createElement('div');
    layers.style.cssText = 'display: flex; gap: 3px;';
    const layerNames = ['sst', 'sal', 'psi', 'speed', 'qnet', 'section'];
    const layerLabels = ['SST', 'Sal', 'Stream', 'Speed', 'Q', 'AMOC'];
    layerNames.forEach((name, i) => {
      const btn = document.createElement('button');
      btn.style.cssText = this._btnStyle(name === 'sst');
      btn.textContent = layerLabels[i];
      btn.onclick = () => {
        if (this.renderer) this.renderer.setLayer(name);
        layers.querySelectorAll('button').forEach(b => b.style.cssText = this._btnStyle(false));
        btn.style.cssText = this._btnStyle(true);
      };
      layers.append(btn);
    });

    this.footer.append(controls, this.diagEl, layers);
    this.container.append(this.topBar, mainArea, this.footer);
  }

  _buildPanel() {
    this.panel.innerHTML = '';

    // --- Diagnostics section ---
    const diagHeader = this._sectionHeader('Diagnostics');
    this.panel.append(diagHeader);

    // Sparkline canvases
    this.sstSparkline = this._sparklineRow('SST', 'sst', '°C');
    this.amocSparkline = this._sparklineRow('AMOC @ 26.5°N', 'amoc', 'Sv');
    this.energySparkline = this._sparklineRow('Energy Balance', 'energy', 'W/m²');

    this.panel.append(this.sstSparkline.el, this.amocSparkline.el, this.energySparkline.el);

    // --- Perturbations section ---
    const pertHeader = this._sectionHeader('Perturbations');
    this.panel.append(pertHeader);

    // Freshwater hosing
    this.hosingSlider = this._sliderRow('Greenland Melt', 0, 1.5, 0, 0.1, 'Sv', (v) => {
      this.freshwaterHosing = v;
    });
    this.panel.append(this.hosingSlider.el);

    // CO2 effect
    this.co2Slider = this._sliderRow('CO₂ Level', 0.5, 4, 1, 0.1, '×', (v) => {
      this.co2Multiplier = v;
    });
    this.panel.append(this.co2Slider.el);

    // Global temp offset
    this.tempSlider = this._sliderRow('Temp Offset', -5, 10, 0, 0.5, '°C', (v) => {
      this.globalTempOffset = v;
    });
    this.panel.append(this.tempSlider.el);

    // --- Model status ---
    const statusHeader = this._sectionHeader('Model Status');
    this.panel.append(statusHeader);

    this.modelStatusEl = document.createElement('div');
    this.modelStatusEl.style.cssText = 'color:#666;font-size:10px;line-height:1.6;';
    this.modelStatusEl.innerHTML = `
      <div style="color:#8a6;">Calibrated: SST patterns, wind-driven gyres</div>
      <div style="color:#a86;">Preliminary: AMOC strength, deep circulation</div>
      <div style="color:#a66;">Simplified: clouds, atmospheric dynamics</div>
    `;
    this.panel.append(this.modelStatusEl);
  }

  _sectionHeader(text) {
    const h = document.createElement('div');
    h.style.cssText = 'color:#666;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin:12px 0 6px;padding-bottom:4px;border-bottom:1px solid #2a2a2a;';
    h.textContent = text;
    return h;
  }

  _sparklineRow(label, key, unit) {
    const el = document.createElement('div');
    el.style.cssText = 'margin-bottom:8px;';

    const header = document.createElement('div');
    header.style.cssText = 'display:flex;justify-content:space-between;margin-bottom:2px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#888;';
    lbl.textContent = label;
    const val = document.createElement('span');
    val.style.cssText = 'color:#4a9ec8;font-weight:600;font-variant-numeric:tabular-nums;';
    header.append(lbl, val);

    const canvas = document.createElement('canvas');
    canvas.width = 216;
    canvas.height = 32;
    canvas.style.cssText = 'width:100%;height:32px;border-radius:3px;background:#0c0c0c;';

    el.append(header, canvas);
    return { el, val, canvas, key, unit };
  }

  _sliderRow(label, min, max, initial, step, unit, onChange) {
    const el = document.createElement('div');
    el.style.cssText = 'margin-bottom:10px;';

    const header = document.createElement('div');
    header.style.cssText = 'display:flex;justify-content:space-between;margin-bottom:2px;';
    const lbl = document.createElement('span');
    lbl.style.cssText = 'color:#888;';
    lbl.textContent = label;
    const val = document.createElement('span');
    val.style.cssText = 'color:#4a9ec8;font-weight:600;font-variant-numeric:tabular-nums;';
    val.textContent = `${initial} ${unit}`;
    header.append(lbl, val);

    const input = document.createElement('input');
    input.type = 'range';
    input.min = min;
    input.max = max;
    input.value = initial;
    input.step = step;
    input.style.cssText = 'width:100%;height:20px;accent-color:#4a9ec8;cursor:pointer;';
    input.oninput = () => {
      const v = parseFloat(input.value);
      val.textContent = `${v.toFixed(1)} ${unit}`;
      onChange(v);
    };

    el.append(header, input);
    return { el, val, input };
  }

  _togglePanel() {
    const showing = this.panel.style.display !== 'none';
    this.panel.style.display = showing ? 'none' : 'block';
    // Trigger renderer resize
    if (this.renderer) {
      setTimeout(() => this.renderer._resize(), 50);
    }
  }

  _btnStyle(active = false) {
    return `
      background: ${active ? '#333' : 'transparent'}; border: 1px solid #444;
      color: ${active ? '#fff' : '#888'}; padding: 2px 8px; border-radius: 3px;
      cursor: pointer; font-size: 11px; font-family: inherit;
    `;
  }

  _bindKeys() {
    document.addEventListener('keydown', (e) => {
      if (e.code === 'Space') { e.preventDefault(); this.togglePlay(); }
      if (e.code === 'ArrowRight') this.cycleSpeed();
      if (e.code === 'KeyP') this._togglePanel();
    });
  }

  togglePlay() {
    this.running = !this.running;
    this.playBtn.textContent = this.running ? 'II' : '>';
  }

  cycleSpeed() {
    this.speedIndex = (this.speedIndex + 1) % this.speedSteps.length;
    this.speedMultiplier = this.speedSteps[this.speedIndex];
    this.speedBtn.textContent = this.speedMultiplier + 'x';
  }

  update() {
    const d = this.coupler.diagnostics;
    const mode = this.running ? 'Live' : 'Paused';
    this.statusEl.textContent = `${mode}  ${this.coupler.getTimeString()}  ${this.speedMultiplier}x`;

    this.diagEl.innerHTML = [
      `SST: ${d.globalMeanSST.toFixed(1)}°C`,
      `AMOC: ${d.amocStrength.toFixed(1)} Sv`,
      `Q: ${d.energyBalance > 0 ? '+' : ''}${d.energyBalance.toFixed(0)} W/m²`,
      `v: ${(d.maxSpeed * 100).toFixed(0)} cm/s`,
    ].join('<span style="color:#333"> | </span>');

    // Update sparkline history (every 10 steps to avoid overhead)
    if (this.coupler.stepCount % 10 === 0) {
      this.history.sst.push(d.globalMeanSST);
      this.history.amoc.push(d.amocStrength);
      this.history.energy.push(d.energyBalance);
      for (const k of ['sst', 'amoc', 'energy']) {
        if (this.history[k].length > this.maxHistory) this.history[k].shift();
      }
    }

    // Update panel if visible
    if (this.panel.style.display !== 'none') {
      this._updateSparkline(this.sstSparkline, d.globalMeanSST);
      this._updateSparkline(this.amocSparkline, d.amocStrength);
      this._updateSparkline(this.energySparkline, d.energyBalance);
    }
  }

  _updateSparkline(spark, currentValue) {
    const data = this.history[spark.key];
    if (data.length < 2) return;

    spark.val.textContent = `${currentValue.toFixed(1)} ${spark.unit}`;

    const canvas = spark.canvas;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    let min = Infinity, max = -Infinity;
    for (const v of data) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;
    const pad = range * 0.1;
    min -= pad;
    max += pad;

    // Draw line
    ctx.strokeStyle = '#4a9ec8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((data[i] - min) / (max - min)) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Zero line for energy balance
    if (spark.key === 'energy' && min < 0 && max > 0) {
      const zeroY = h - ((0 - min) / (max - min)) * h;
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(0, zeroY);
      ctx.lineTo(w, zeroY);
      ctx.stroke();
    }
  }

  getViewport() {
    return this.viewport;
  }
}
