/* Shared canvas drawing helpers + sample-input generators.

   All functions operate on a CanvasRenderingContext2D in "logical" pixel
   coords; callers are expected to have set up devicePixelRatio scaling.

   Color decisions:
   - Grayscale inputs (binary shapes): black-on-cream in light, cream-on-black in dark.
     We read CSS variables --ink and --bg at draw time.
   - Heatmaps (feature maps, dot-product products): diverging palette
     red (negative) → neutral → blue (positive) so positive matches read as the
     scene's "fit" color. We use the categorical palette tokens via a temp DOM
     probe so light/dark theming follows automatically.
   - Kernel grids: the same diverging palette, plus value labels rendered in mono. */
(function () {

  /* Read current theme tokens. Cached per-call since theme rarely toggles
     during a single render. */
  function tokens() {
    const root = document.documentElement;
    const cs = getComputedStyle(root);
    return {
      bg: cs.getPropertyValue('--bg').trim() || '#f9f7f1',
      ink: cs.getPropertyValue('--ink').trim() || '#1a1a1a',
      inkSecondary: cs.getPropertyValue('--ink-secondary').trim() || '#6b6b6b',
      rule: cs.getPropertyValue('--rule').trim() || '#d8d4ca',
      pos: cs.getPropertyValue('--cnn-pos').trim() || '#2f6cb1',     // blue
      neg: cs.getPropertyValue('--cnn-neg').trim() || '#b8323a',     // red
      neutral: cs.getPropertyValue('--cnn-neutral').trim() || '#f0ece2',
      gridLine: cs.getPropertyValue('--rule').trim() || '#d8d4ca',
    };
  }

  /* Linear interp between two hex colors, t in [0,1]. */
  function lerpHex(a, b, t) {
    const pa = parseInt(a.replace('#', ''), 16);
    const pb = parseInt(b.replace('#', ''), 16);
    const ar = (pa >> 16) & 0xff, ag = (pa >> 8) & 0xff, ab = pa & 0xff;
    const br = (pb >> 16) & 0xff, bg = (pb >> 8) & 0xff, bb = pb & 0xff;
    const r = Math.round(ar + (br - ar) * t);
    const g = Math.round(ag + (bg - ag) * t);
    const bl = Math.round(ab + (bb - ab) * t);
    return `rgb(${r},${g},${bl})`;
  }

  /* Diverging colormap: -1 -> neg, 0 -> neutral, +1 -> pos. v expected in [-1, 1]. */
  function divergingColor(v, t) {
    if (v >= 0) return lerpHex(t.neutral, t.pos, Math.min(1, v));
    return lerpHex(t.neutral, t.neg, Math.min(1, -v));
  }

  /* Draw a 2D grid of values onto the given canvas region.
     opts:
       cellSize       : px per cell (computed from canvas if omitted)
       valueRange     : [lo, hi] for normalization; defaults to symmetric around 0
       diverging      : true to use divergingColor; false for sequential ink-on-bg
       cellBorder     : true to draw thin grid lines between cells
       labels         : true to draw numeric labels in each cell
       labelDecimals  : digits after decimal point for labels (default 1)
   */
  function drawGrid(ctx, data, x, y, w, h, opts) {
    opts = opts || {};
    const t = tokens();
    const rows = data.length, cols = data[0].length;
    const cellW = (opts.cellSize != null) ? opts.cellSize : (w / cols);
    const cellH = (opts.cellSize != null) ? opts.cellSize : (h / rows);
    let lo, hi;
    if (opts.valueRange) {
      lo = opts.valueRange[0]; hi = opts.valueRange[1];
    } else {
      const r = ((window.UNET && window.UNET.range2D) || (window.CNN && window.CNN.range2D))(data);
      lo = r.lo; hi = r.hi;
    }

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = data[i][j];
        let fill;
        if (opts.diverging !== false) {
          const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
          fill = divergingColor(v / m, t);
        } else {
          // Sequential: ink at high, bg at low
          const norm = (hi !== lo) ? (v - lo) / (hi - lo) : 0;
          fill = lerpHex(t.bg, t.ink, Math.max(0, Math.min(1, norm)));
        }
        ctx.fillStyle = fill;
        ctx.fillRect(x + j * cellW, y + i * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    if (opts.cellBorder) {
      ctx.strokeStyle = t.gridLine;
      ctx.lineWidth = 1;
      for (let i = 0; i <= rows; i++) {
        ctx.beginPath();
        ctx.moveTo(x, y + i * cellH);
        ctx.lineTo(x + cols * cellW, y + i * cellH);
        ctx.stroke();
      }
      for (let j = 0; j <= cols; j++) {
        ctx.beginPath();
        ctx.moveTo(x + j * cellW, y);
        ctx.lineTo(x + j * cellW, y + rows * cellH);
        ctx.stroke();
      }
    }

    if (opts.labels) {
      ctx.fillStyle = t.ink;
      ctx.font = `${Math.max(9, Math.floor(cellH * 0.32))}px "SF Mono", Menlo, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const dec = opts.labelDecimals != null ? opts.labelDecimals : 1;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const v = data[i][j];
          const s = (Math.abs(v) < 1e-9) ? '0' : v.toFixed(dec);
          ctx.fillText(s, x + (j + 0.5) * cellW, y + (i + 0.5) * cellH);
        }
      }
    }
  }

  /* High-level canvas setup with devicePixelRatio scaling.
     Returns { canvas, ctx, w, h } with logical dimensions. */
  function setupCanvas(parent, w, h) {
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    parent.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { canvas, ctx, w, h };
  }

  /* ---- Sample-input generators (28×28 grayscale in [0,1]) ---- */

  function blank28() {
    return (window.UNET || window.CNN).zeros2D(28, 28);
  }

  function makeCross(size) {
    size = size || 28;
    const out = (window.UNET || window.CNN).zeros2D(size, size);
    const mid = Math.floor(size / 2);
    const thickness = Math.max(2, Math.floor(size / 12));
    const arm = Math.floor(size * 0.36);
    for (let i = mid - arm; i <= mid + arm; i++) {
      for (let d = -thickness + 1; d <= thickness; d++) {
        if (i >= 0 && i < size && mid + d >= 0 && mid + d < size) out[i][mid + d] = 1;
        if (mid + d >= 0 && mid + d < size && i >= 0 && i < size) out[mid + d][i] = 1;
      }
    }
    return out;
  }

  function makeL(size) {
    size = size || 28;
    const out = (window.UNET || window.CNN).zeros2D(size, size);
    const thick = Math.max(2, Math.floor(size / 14));
    const top = Math.floor(size * 0.18), bot = Math.floor(size * 0.82);
    const left = Math.floor(size * 0.30);
    const right = Math.floor(size * 0.74);
    for (let i = top; i < bot; i++) for (let d = 0; d < thick; d++) out[i][left + d] = 1;
    for (let j = left; j < right; j++) for (let d = 0; d < thick; d++) out[bot - 1 - d][j] = 1;
    return out;
  }

  function makeVerticalLine(size) {
    size = size || 28;
    const out = (window.UNET || window.CNN).zeros2D(size, size);
    const mid = Math.floor(size / 2);
    const thick = Math.max(2, Math.floor(size / 12));
    for (let i = Math.floor(size * 0.16); i < Math.floor(size * 0.84); i++) {
      for (let d = -thick + 1; d <= thick - 1; d++) out[i][mid + d] = 1;
    }
    return out;
  }

  function makeHorizontalLine(size) {
    size = size || 28;
    const out = (window.UNET || window.CNN).zeros2D(size, size);
    const mid = Math.floor(size / 2);
    const thick = Math.max(2, Math.floor(size / 12));
    for (let j = Math.floor(size * 0.16); j < Math.floor(size * 0.84); j++) {
      for (let d = -thick + 1; d <= thick - 1; d++) out[mid + d][j] = 1;
    }
    return out;
  }

  function makeCircle(size) {
    size = size || 28;
    const out = (window.UNET || window.CNN).zeros2D(size, size);
    const mid = (size - 1) / 2;
    const r = size * 0.32;
    const thick = Math.max(1.4, size / 14);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const d = Math.hypot(i - mid, j - mid);
        if (Math.abs(d - r) < thick) out[i][j] = 1;
      }
    }
    return out;
  }

  function makeTriangle(size) {
    size = size || 28;
    const out = (window.UNET || window.CNN).zeros2D(size, size);
    const top = { x: size / 2,            y: size * 0.20 };
    const bl  = { x: size * 0.22,         y: size * 0.80 };
    const br  = { x: size * 0.78,         y: size * 0.80 };
    function distToSeg(px, py, ax, ay, bx, by) {
      const dx = bx - ax, dy = by - ay;
      const tt = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)));
      const cx = ax + tt * dx, cy = ay + tt * dy;
      return Math.hypot(px - cx, py - cy);
    }
    const thick = Math.max(1.2, size / 16);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        const d = Math.min(
          distToSeg(j, i, top.x, top.y, bl.x, bl.y),
          distToSeg(j, i, top.x, top.y, br.x, br.y),
          distToSeg(j, i, bl.x, bl.y, br.x, br.y)
        );
        if (d < thick) out[i][j] = 1;
      }
    }
    return out;
  }

  /* Catalog of named samples, keyed by short labels. */
  const samples = {
    cross: makeCross,
    L: makeL,
    vertical: makeVerticalLine,
    horizontal: makeHorizontalLine,
    circle: makeCircle,
    triangle: makeTriangle,
  };

  function makeSample(name, size) {
    const fn = samples[name];
    if (!fn) throw new Error('Unknown sample: ' + name);
    return fn(size);
  }

  /* ---------------------------------------------------------------------
     Promoted painters (originally local to cnn-deepdive scene9.js).
     Multiple U-Net scenes need them, so they live here.
     ---------------------------------------------------------------------

     Conventions:
     - `host` is a DOM container; the painter clears and re-fills it.
     - `px` is the logical edge length in CSS pixels.
     - Colors come from CSS variables when applicable so theme switching
       follows automatically. The exception is `paintRGB`, which renders
       the RGB array faithfully (input pixels are not theme-affected). */

  /* Robust accessor for the 2D-range helper. UNET owns it now; CNN had it
     first. Either is fine; whichever loaded first wins. */
  function rangeFn() {
    if (window.UNET && typeof window.UNET.range2D === 'function') return window.UNET.range2D;
    if (window.CNN && typeof window.CNN.range2D === 'function') return window.CNN.range2D;
    // Last resort: inline scan.
    return function (x) {
      let lo = Infinity, hi = -Infinity;
      for (const row of x) for (const v of row) {
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      if (!isFinite(lo)) lo = 0;
      if (!isFinite(hi)) hi = 0;
      return { lo, hi };
    };
  }

  /* Read class colors live from CSS so theme switching follows automatically.
     Names match the segmentation taxonomy (sky, grass, sun, tree, person). */
  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];
  function readClassColors() {
    const cs = getComputedStyle(document.documentElement);
    return CLASS_NAMES.map(function (name) {
      return cs.getPropertyValue('--class-' + name).trim() || '#888';
    });
  }

  function parseHex(hex) {
    let s = (hex || '').trim().replace('#', '');
    if (s.length === 3) s = s.split('').map(function (c) { return c + c; }).join('');
    if (!/^[0-9a-fA-F]{6}$/.test(s)) return [136, 136, 136];
    const n = parseInt(s, 16);
    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
  }

  /* Paint an HxWx3 RGB array onto a canvas at logical size [px, px].
     Uses ImageData for speed. Colors are NOT theme-affected -- the input
     is the input. */
  function paintRGB(host, rgb, px) {
    host.innerHTML = '';
    const setup = setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const H = rgb.length, W = rgb[0].length;
    const off = document.createElement('canvas');
    off.width = W; off.height = H;
    const offCtx = off.getContext('2d');
    const id = offCtx.createImageData(W, H);
    let p = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const r = rgb[i][j][0], g = rgb[i][j][1], b = rgb[i][j][2];
        id.data[p++] = Math.max(0, Math.min(255, Math.round(r * 255)));
        id.data[p++] = Math.max(0, Math.min(255, Math.round(g * 255)));
        id.data[p++] = Math.max(0, Math.min(255, Math.round(b * 255)));
        id.data[p++] = 255;
      }
    }
    offCtx.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, px, px);
  }

  /* Paint an integer label map [H][W] of class indices using the per-class
     CSS colors. Optionally outline pixels where pred != label via opts.diffMask. */
  function paintLabelMap(host, lbl, px, opts) {
    host.innerHTML = '';
    opts = opts || {};
    const colors = opts.colors || readClassColors();
    const setup = setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const H = lbl.length, W = lbl[0].length;
    const off = document.createElement('canvas');
    off.width = W; off.height = H;
    const offCtx = off.getContext('2d');
    const id = offCtx.createImageData(W, H);
    let p = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const c = lbl[i][j] | 0;
        const hex = colors[c] || '#888';
        const rgb = parseHex(hex);
        id.data[p++] = rgb[0];
        id.data[p++] = rgb[1];
        id.data[p++] = rgb[2];
        id.data[p++] = 255;
      }
    }
    offCtx.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, px, px);

    // If a diff mask is supplied, outline disagreements.
    if (opts.diffMask) {
      const cw = px / W, ch = px / H;
      const t = tokens();
      ctx.strokeStyle = t.ink;
      ctx.lineWidth = 1.4;
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          if (opts.diffMask[i][j]) {
            ctx.strokeRect(j * cw + 0.5, i * ch + 0.5, cw - 1, ch - 1);
          }
        }
      }
    }
  }

  /* Paint a 4-channel feature stack [4][H][W] as a 2x2 grid of small
     thumbnails using the diverging colormap. `host` should be the card body. */
  function paintFeatureCard(host, stack4, px) {
    host.innerHTML = '';
    const setup = setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const t = tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, px, px);

    // Symmetric range across all 4 channels for fair contrast.
    const range = rangeFn();
    let m = 0;
    for (let c = 0; c < stack4.length; c++) {
      const r = range(stack4[c]);
      m = Math.max(m, Math.abs(r.lo), Math.abs(r.hi));
    }
    if (!m) m = 1;

    const half = px / 2;
    const gap = 2;
    const cellW = half - gap;
    const positions = [
      [0, 0], [half + gap, 0],
      [0, half + gap], [half + gap, half + gap],
    ];
    for (let c = 0; c < 4; c++) {
      const pos = positions[c];
      drawGrid(ctx, stack4[c], pos[0], pos[1], cellW, cellW, {
        diverging: true, valueRange: [-m, m],
      });
    }
    // Thin separator lines so the 2x2 reads as a grid of channels.
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(half, 0); ctx.lineTo(half, px);
    ctx.moveTo(0, half); ctx.lineTo(px, half);
    ctx.stroke();
  }

  function paintBlankCard(host, px) {
    host.innerHTML = '';
    const setup = setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const t = tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, px, px);
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.strokeRect(0.5, 0.5, px - 1, px - 1);
    ctx.beginPath();
    ctx.moveTo(0, px / 2); ctx.lineTo(px, px / 2);
    ctx.moveTo(px / 2, 0); ctx.lineTo(px / 2, px);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  window.Drawing = {
    tokens, divergingColor, lerpHex,
    drawGrid, setupCanvas,
    makeSample, samples,
    paintRGB, paintLabelMap, paintFeatureCard, paintBlankCard,
    parseHex, readClassColors,
  };
})();
