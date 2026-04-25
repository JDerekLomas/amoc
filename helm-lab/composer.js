// Image composition helpers, run inside the headless browser page.
// Produces PNG dataURLs for: contact sheet (trajectory), sweep sheet
// (parameter sweep), and summary (sparkline of diagnostics over time).
//
// Style aligned with the Barotropic Observatory bridge — abyss black,
// brass accents, Cormorant SC + JetBrains Mono.

(function () {
  const COL = {
    bg: '#03070d', panel: '#0a141f', hairline: 'rgba(105,182,214,0.22)',
    brass: '#b18c4e', brass2: '#d8b06a', brass3: '#f0d693',
    mariner: '#69b6d6', marinerBright: '#a4dcf2',
    phosphor: '#5ce0c0', ember: '#ff8b4f', blood: '#d94650',
    ink: '#d4dee9', mute: '#738eaa', faint: '#3e556d',
  };
  const FONT_S = '11px "Cormorant SC", "Cormorant Garamond", serif';
  const FONT_LBL = "10px 'Cormorant SC', serif";
  const FONT_NUM = "11px 'JetBrains Mono', monospace";
  const FONT_TITLE = "20px 'Cormorant Garamond', serif";

  function loadImage(dataUrl) {
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = rej;
      img.src = dataUrl;
    });
  }

  function bracketBox(ctx, x, y, w, h, color = COL.brass2, len = 10) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.4;
    [
      [[x, y + len], [x, y], [x + len, y]],
      [[x + w - len, y], [x + w, y], [x + w, y + len]],
      [[x + w, y + h - len], [x + w, y + h], [x + w - len, y + h]],
      [[x + len, y + h], [x, y + h], [x, y + h - len]],
    ].forEach(seg => {
      ctx.beginPath();
      ctx.moveTo(seg[0][0], seg[0][1]); ctx.lineTo(seg[1][0], seg[1][1]); ctx.lineTo(seg[2][0], seg[2][1]);
      ctx.stroke();
    });
  }

  function smallCaps(ctx, text, x, y, color = COL.brass2) {
    ctx.fillStyle = color; ctx.font = FONT_LBL;
    ctx.fillText(text.toUpperCase(), x, y);
  }

  function num(ctx, text, x, y, color = COL.phosphor, font = FONT_NUM) {
    ctx.fillStyle = color; ctx.font = font;
    ctx.fillText(text, x, y);
  }

  function fmt(v, d = 2) { return (v == null || !isFinite(v)) ? '—' : Number(v).toFixed(d); }
  function fmtE(v) { return (v == null || !isFinite(v)) ? '—' : Number(v).toExponential(2); }

  // ====================================================================
  // CONTACT SHEET — trajectory in time, multiple sample points
  // ====================================================================
  window.composeContactSheetInPage = async function (samples) {
    if (!samples.length) return null;
    const views = Object.keys(samples[0].frames);
    const N = samples.length;
    const V = views.length;

    // Card sizing
    const cardW = 480, frameH = 240; // 2:1 ratio, downscaled from 1024x512
    const headerH = 28, footerH = 70;
    const cardH = headerH + V * frameH + footerH;
    const cols = Math.min(N, Math.floor(2200 / cardW)) || 1;
    const rows = Math.ceil(N / cols);
    const padX = 18, padY = 18, gap = 16;
    const titleH = 90;
    const W = padX * 2 + cols * cardW + (cols - 1) * gap;
    const H = padY * 2 + titleH + rows * cardH + (rows - 1) * gap + 30;

    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const ctx = c.getContext('2d');

    // Background
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#050a14'); grad.addColorStop(1, '#02050b');
    ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);

    // Title block
    ctx.fillStyle = COL.brass2; ctx.font = "11px 'Cormorant SC', serif";
    ctx.fillText('BAROTROPIC OBSERVATORY · TRAJECTORY CONTACT SHEET', padX, padY + 14);
    ctx.fillStyle = COL.marinerBright; ctx.font = "26px 'Cormorant Garamond', serif";
    const span = samples[N - 1].simYears - samples[0].simYears;
    ctx.fillText(`${N} sample${N > 1 ? 's' : ''} across ${span.toFixed(1)} model years`, padX, padY + 50);
    ctx.fillStyle = COL.mute; ctx.font = "italic 13px 'Cormorant Garamond', serif";
    ctx.fillText(`step ${samples[0].t} → ${samples[N - 1].t}  ·  views: ${views.join(', ')}`, padX, padY + 72);

    // Hairline rule under title
    ctx.strokeStyle = COL.hairline; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padX, padY + titleH); ctx.lineTo(W - padX, padY + titleH); ctx.stroke();

    // Pre-load all frames
    const imgs = await Promise.all(samples.map(async s => {
      const o = {};
      for (const v of views) o[v] = await loadImage(s.frames[v]);
      return o;
    }));

    for (let i = 0; i < N; i++) {
      const col = i % cols, row = Math.floor(i / cols);
      const x = padX + col * (cardW + gap);
      const y = padY + titleH + 8 + row * (cardH + gap);

      // Card surface
      ctx.fillStyle = '#0a141fcc';
      ctx.fillRect(x, y, cardW, cardH);
      bracketBox(ctx, x, y, cardW, cardH);

      // Header
      const s = samples[i];
      ctx.fillStyle = COL.brass2; ctx.font = "10px 'Cormorant SC', serif";
      ctx.fillText(`STEP ${s.t.toLocaleString()}`, x + 12, y + 18);
      ctx.fillStyle = COL.phosphor; ctx.font = "11px 'JetBrains Mono', monospace";
      const yrTxt = `t = ${s.simYears.toFixed(2)} yr`;
      ctx.textAlign = 'right'; ctx.fillText(yrTxt, x + cardW - 12, y + 18); ctx.textAlign = 'left';

      // Frames
      for (let j = 0; j < V; j++) {
        const fy = y + headerH + j * frameH;
        ctx.drawImage(imgs[i][views[j]], x + 8, fy + 4, cardW - 16, frameH - 6);
        // view label inside frame
        ctx.fillStyle = '#03070dcc'; ctx.fillRect(x + 14, fy + 8, 80, 16);
        ctx.fillStyle = COL.brass3; ctx.font = "9px 'Cormorant SC', serif";
        ctx.fillText(views[j].toUpperCase(), x + 18, fy + 19);
      }

      // Footer diagnostics
      const fy = y + headerH + V * frameH + 8;
      const cellW = (cardW - 24) / 4;
      const diag = s.diag;
      const cells = [
        ['KE', fmtE(diag.KE), COL.phosphor],
        ['v_max', fmt(diag.maxVel, 2), COL.phosphor],
        ['AMOC', fmt(diag.amoc, 4), Math.abs(diag.amoc) < 0.005 ? COL.blood : Math.abs(diag.amoc) < 0.02 ? COL.ember : COL.phosphor],
        ['T_glob', fmt(diag.globalSST, 1) + '°', COL.phosphor],
      ];
      cells.forEach((cell, k) => {
        const cx = x + 12 + k * cellW;
        ctx.fillStyle = COL.mute; ctx.font = "9px 'Cormorant SC', serif";
        ctx.fillText(cell[0].toUpperCase(), cx, fy + 12);
        ctx.fillStyle = cell[2]; ctx.font = "13px 'JetBrains Mono', monospace";
        ctx.fillText(cell[1], cx, fy + 28);
      });
      // Twin line: tropical / polar SST
      const fy2 = fy + 36;
      const tropTxt = `T_trop ${fmt(diag.tropicalSST, 1)}°  ·  T_pol ${fmt(diag.polarSST, 1)}°  ·  ice ${diag.iceArea?.toLocaleString() ?? '—'}`;
      ctx.fillStyle = COL.faint; ctx.font = "11px 'JetBrains Mono', monospace";
      ctx.fillText(tropTxt, x + 12, fy2);
    }

    // Footer
    ctx.fillStyle = COL.faint; ctx.font = "italic 11px 'Cormorant Garamond', serif";
    ctx.fillText('helm-lab · automated render', padX, H - 14);

    return c.toDataURL('image/png');
  };

  // ====================================================================
  // SWEEP SHEET — one column per parameter value, frames stacked
  // ====================================================================
  window.composeSweepInPage = async function (points, paramName) {
    const N = points.length;
    if (!N) return null;
    const views = Object.keys(points[0].frames);
    const V = views.length;
    const cardW = 360, frameH = 180;
    const headerH = 32, footerH = 80;
    const cardH = headerH + V * frameH + footerH;
    const padX = 24, padY = 22, gap = 18, titleH = 100;
    const cols = Math.min(N, Math.floor(2200 / (cardW + gap))) || 1;
    const rows = Math.ceil(N / cols);
    const W = padX * 2 + cols * cardW + (cols - 1) * gap;
    const H = padY * 2 + titleH + rows * cardH + (rows - 1) * gap + 40;

    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const ctx = c.getContext('2d');
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#050a14'); grad.addColorStop(1, '#02050b');
    ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);

    ctx.fillStyle = COL.brass2; ctx.font = "11px 'Cormorant SC', serif";
    ctx.fillText('BAROTROPIC OBSERVATORY · PARAMETER SWEEP', padX, padY + 14);
    ctx.fillStyle = COL.marinerBright; ctx.font = "28px 'Cormorant Garamond', serif";
    ctx.fillText(`Sweep over `, padX, padY + 58);
    const m = ctx.measureText('Sweep over ');
    ctx.fillStyle = COL.brass3; ctx.font = "italic 28px 'Cormorant Garamond', serif";
    ctx.fillText(paramName, padX + m.width, padY + 58);
    ctx.fillStyle = COL.mute; ctx.font = "italic 13px 'Cormorant Garamond', serif";
    ctx.fillText(`${N} points  ·  views: ${views.join(', ')}`, padX, padY + 80);

    ctx.strokeStyle = COL.hairline; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padX, padY + titleH); ctx.lineTo(W - padX, padY + titleH); ctx.stroke();

    const imgs = await Promise.all(points.map(async p => {
      const o = {}; for (const v of views) o[v] = await loadImage(p.frames[v]); return o;
    }));

    for (let i = 0; i < N; i++) {
      const col = i % cols, row = Math.floor(i / cols);
      const x = padX + col * (cardW + gap);
      const y = padY + titleH + 8 + row * (cardH + gap);
      ctx.fillStyle = '#0a141fcc'; ctx.fillRect(x, y, cardW, cardH);
      bracketBox(ctx, x, y, cardW, cardH);

      const p = points[i];
      ctx.fillStyle = COL.brass2; ctx.font = "10px 'Cormorant SC', serif";
      ctx.fillText(paramName.toUpperCase(), x + 12, y + 18);
      ctx.fillStyle = COL.phosphor; ctx.font = "13px 'JetBrains Mono', monospace";
      const valTxt = typeof p.value === 'number'
        ? (Math.abs(p.value) < 1e-3 || Math.abs(p.value) >= 1e4 ? p.value.toExponential(2) : p.value.toString())
        : String(p.value);
      ctx.textAlign = 'right'; ctx.fillText(valTxt, x + cardW - 12, y + 22); ctx.textAlign = 'left';

      for (let j = 0; j < V; j++) {
        const fy = y + headerH + j * frameH;
        ctx.drawImage(imgs[i][views[j]], x + 8, fy + 4, cardW - 16, frameH - 6);
        ctx.fillStyle = '#03070dcc'; ctx.fillRect(x + 14, fy + 8, 80, 16);
        ctx.fillStyle = COL.brass3; ctx.font = "9px 'Cormorant SC', serif";
        ctx.fillText(views[j].toUpperCase(), x + 18, fy + 19);
      }

      const fy = y + headerH + V * frameH + 12;
      const d = p.diag;
      const lines = [
        ['KE', fmtE(d.KE)], ['AMOC', fmt(d.amoc, 4)],
        ['T_global', fmt(d.globalSST, 1) + '°'], ['T_trop', fmt(d.tropicalSST, 1) + '°'],
        ['T_pole', fmt(d.polarSST, 1) + '°'],
      ];
      const colW = (cardW - 24) / 5;
      lines.forEach((cell, k) => {
        const cx = x + 12 + k * colW;
        ctx.fillStyle = COL.mute; ctx.font = "8px 'Cormorant SC', serif";
        ctx.fillText(cell[0].toUpperCase(), cx, fy + 8);
        ctx.fillStyle = COL.phosphor; ctx.font = "11px 'JetBrains Mono', monospace";
        ctx.fillText(cell[1], cx, fy + 24);
      });
    }
    ctx.fillStyle = COL.faint; ctx.font = "italic 11px 'Cormorant Garamond', serif";
    ctx.fillText('helm-lab · automated sweep', padX, H - 14);
    return c.toDataURL('image/png');
  };

  // ====================================================================
  // SUMMARY — sparkline of diagnostics over time, full run on one PNG
  // ====================================================================
  window.composeSummaryInPage = function (series) {
    const W = 1600, H = 900;
    const c = document.createElement('canvas');
    c.width = W; c.height = H;
    const ctx = c.getContext('2d');

    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, '#050a14'); grad.addColorStop(1, '#02050b');
    ctx.fillStyle = grad; ctx.fillRect(0, 0, W, H);

    // Title
    const padX = 36, padY = 28;
    ctx.fillStyle = COL.brass2; ctx.font = "11px 'Cormorant SC', serif";
    ctx.fillText('BAROTROPIC OBSERVATORY · DIAGNOSTIC SUMMARY', padX, padY + 14);
    ctx.fillStyle = COL.marinerBright; ctx.font = "32px 'Cormorant Garamond', serif";
    const span = series.length ? series[series.length - 1].simYears - series[0].simYears : 0;
    ctx.fillText(`${series.length} samples · ${span.toFixed(1)} model years`, padX, padY + 58);
    ctx.strokeStyle = COL.hairline; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padX, padY + 80); ctx.lineTo(W - padX, padY + 80); ctx.stroke();

    // Six panels in a 3x2 grid
    const panels = [
      { key: 'KE',          label: 'Kinetic Energy',     fmt: fmtE,  color: COL.phosphor, log: true },
      { key: 'amoc',        label: 'AMOC Strength',      fmt: v => fmt(v, 4), color: COL.mariner, zero: true },
      { key: 'globalSST',   label: 'Global Mean SST',    fmt: v => fmt(v, 1) + '°C', color: COL.ember },
      { key: 'tropicalSST', label: 'Tropical SST',       fmt: v => fmt(v, 1) + '°C', color: COL.ember },
      { key: 'polarSST',    label: 'Polar SST',          fmt: v => fmt(v, 1) + '°C', color: COL.marinerBright },
      { key: 'iceArea',     label: 'Sea-Ice Cells',      fmt: v => v?.toLocaleString() ?? '—', color: COL.marinerBright },
    ];
    const gx = padX, gy = padY + 100, gw = W - 2 * padX, gh = H - gy - padY;
    const cellW = gw / 3, cellH = gh / 2;
    panels.forEach((p, i) => {
      const cx = gx + (i % 3) * cellW, cy = gy + Math.floor(i / 3) * cellH;
      drawSparklinePanel(ctx, cx + 8, cy + 8, cellW - 16, cellH - 16, series, p);
    });

    ctx.fillStyle = COL.faint; ctx.font = "italic 11px 'Cormorant Garamond', serif";
    ctx.fillText('helm-lab · summary', padX, H - 12);
    return c.toDataURL('image/png');
  };

  function drawSparklinePanel(ctx, x, y, w, h, series, p) {
    // Panel surface
    ctx.fillStyle = '#0a141faa'; ctx.fillRect(x, y, w, h);
    bracketBox(ctx, x, y, w, h, COL.brass);

    // Header
    ctx.fillStyle = COL.brass2; ctx.font = "11px 'Cormorant SC', serif";
    ctx.fillText(p.label.toUpperCase(), x + 14, y + 22);

    // Big readout (last value)
    const last = series.length ? series[series.length - 1][p.key] : null;
    ctx.fillStyle = p.color; ctx.font = "32px 'JetBrains Mono', monospace";
    ctx.textAlign = 'right'; ctx.fillText(p.fmt(last), x + w - 14, y + 38); ctx.textAlign = 'left';

    // Plot area
    const px = x + 14, py = y + 60;
    const pw = w - 28, ph = h - 84;

    if (series.length < 2) return;
    let min = Infinity, max = -Infinity;
    const vals = series.map(s => s[p.key]).filter(v => isFinite(v));
    if (!vals.length) return;
    for (const v of vals) { if (v < min) min = v; if (v > max) max = v; }
    if (p.zero) min = Math.min(min, 0);
    if (max - min < 1e-12) max = min + 1;
    const pad = (max - min) * 0.08; min -= pad; max += pad;

    // axes
    ctx.strokeStyle = 'rgba(184,140,78,0.18)'; ctx.lineWidth = 0.6;
    ctx.beginPath();
    ctx.moveTo(px, py); ctx.lineTo(px, py + ph); ctx.lineTo(px + pw, py + ph); ctx.stroke();

    // gridlines
    ctx.strokeStyle = 'rgba(184,140,78,0.08)'; ctx.lineWidth = 0.4;
    for (let g = 1; g < 4; g++) {
      const yy = py + (g / 4) * ph;
      ctx.beginPath(); ctx.moveTo(px, yy); ctx.lineTo(px + pw, yy); ctx.stroke();
    }

    // zero baseline if applicable
    if (min < 0 && max > 0) {
      const zy = py + ph - (-min) / (max - min) * ph;
      ctx.strokeStyle = 'rgba(216,176,106,0.4)'; ctx.setLineDash([4, 4]); ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.moveTo(px, zy); ctx.lineTo(px + pw, zy); ctx.stroke(); ctx.setLineDash([]);
    }

    // gradient fill area
    const grad = ctx.createLinearGradient(0, py, 0, py + ph);
    const cc = p.color;
    grad.addColorStop(0, hexA(cc, 0.35)); grad.addColorStop(1, hexA(cc, 0));
    ctx.fillStyle = grad;
    ctx.beginPath();
    series.forEach((s, i) => {
      const xx = px + (i / (series.length - 1)) * pw;
      const yy = py + ph - (s[p.key] - min) / (max - min) * ph;
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    });
    ctx.lineTo(px + pw, py + ph); ctx.lineTo(px, py + ph); ctx.closePath(); ctx.fill();

    // line
    ctx.strokeStyle = cc; ctx.lineWidth = 1.4;
    ctx.shadowColor = cc; ctx.shadowBlur = 4;
    ctx.beginPath();
    series.forEach((s, i) => {
      const xx = px + (i / (series.length - 1)) * pw;
      const yy = py + ph - (s[p.key] - min) / (max - min) * ph;
      if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
    });
    ctx.stroke(); ctx.shadowBlur = 0;

    // last-point dot
    const lx = px + pw, ly = py + ph - (series[series.length - 1][p.key] - min) / (max - min) * ph;
    ctx.fillStyle = cc; ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fill();

    // x-axis tick (years span)
    ctx.fillStyle = COL.faint; ctx.font = "9px 'JetBrains Mono', monospace";
    const t0 = series[0].simYears, t1 = series[series.length - 1].simYears;
    ctx.fillText(`yr ${t0.toFixed(1)}`, px, py + ph + 12);
    ctx.textAlign = 'right'; ctx.fillText(`yr ${t1.toFixed(1)}`, px + pw, py + ph + 12); ctx.textAlign = 'left';

    // y range labels
    ctx.fillStyle = COL.faint; ctx.font = "9px 'JetBrains Mono', monospace";
    const fmtY = (v) => p.key === 'KE' ? Number(v).toExponential(1) : Number(v).toFixed(2);
    ctx.fillText(fmtY(max), px + pw - 50, py + 10);
    ctx.fillText(fmtY(min), px + pw - 50, py + ph - 4);
  }

  function hexA(hex, a) {
    const h = hex.replace('#', '');
    const r = parseInt(h.substr(0, 2), 16);
    const g = parseInt(h.substr(2, 2), 16);
    const b = parseInt(h.substr(4, 2), 16);
    return `rgba(${r},${g},${b},${a})`;
  }
})();
