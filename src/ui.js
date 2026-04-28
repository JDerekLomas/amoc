// UI chrome: top bar, footer, diagnostics, layer switcher
// Dark mode, minimal, scientific aesthetic

export class UI {
  constructor(container, renderer, coupler) {
    this.container = container;
    this.renderer = renderer;
    this.coupler = coupler;

    // State
    this.running = true;
    this.speedMultiplier = 1;
    this.speedSteps = [1, 4, 16, 64, 256];
    this.speedIndex = 0;

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

    const actions = document.createElement('div');
    actions.style.cssText = 'display: flex; gap: 12px; font-size: 12px; color: #666;';
    actions.innerHTML = '<span title="Info">i</span>';

    this.topBar.append(title, this.statusEl, actions);

    // Viewport container
    this.viewport = document.createElement('div');
    this.viewport.style.cssText = `
      flex: 1; display: flex; align-items: center; justify-content: center;
      background: #0a0a0a; position: relative; overflow: hidden;
    `;

    // Footer
    this.footer = document.createElement('div');
    this.footer.style.cssText = `
      display: flex; align-items: center; justify-content: space-between;
      padding: 4px 16px; height: 28px; background: #1a1a1a;
      border-top: 1px solid #333; font-size: 12px; flex-shrink: 0;
      font-variant-numeric: tabular-nums;
    `;

    // Play/pause + speed controls
    const controls = document.createElement('div');
    controls.style.cssText = 'display: flex; align-items: center; gap: 8px;';

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
    this.diagEl.style.cssText = 'display: flex; gap: 16px; color: #999;';

    // Layer switcher
    const layers = document.createElement('div');
    layers.style.cssText = 'display: flex; gap: 4px;';
    const layerNames = ['sst', 'sal', 'psi', 'speed', 'qnet', 'section'];
    const layerLabels = ['SST', 'Sal', 'Stream', 'Speed', 'Q', 'AMOC'];
    layerNames.forEach((name, i) => {
      const btn = document.createElement('button');
      btn.style.cssText = this._btnStyle(name === 'sst');
      btn.textContent = layerLabels[i];
      btn.onclick = () => {
        this.renderer.setLayer(name);
        layers.querySelectorAll('button').forEach(b => {
          b.style.cssText = this._btnStyle(false);
        });
        btn.style.cssText = this._btnStyle(true);
      };
      layers.append(btn);
    });

    this.footer.append(controls, this.diagEl, layers);
    this.container.append(this.topBar, this.viewport, this.footer);
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
      `SST: ${d.globalMeanSST.toFixed(1)}C`,
      `AMOC: ${d.amocStrength.toFixed(1)} Sv`,
      `Q: ${d.energyBalance > 0 ? '+' : ''}${d.energyBalance.toFixed(1)} W/m2`,
      `v<sub>max</sub>: ${(d.maxSpeed * 100).toFixed(1)} cm/s`,
    ].join('<span style="color:#444"> | </span>');
  }

  getViewport() {
    return this.viewport;
  }
}
