/* Demo 2 — Stack neurons, build curves.

   Foundation module. Builds the static plot, evaluates the bold fit and
   per-neuron kinks from precomputed parameters, and exposes a tiny state
   bus on `window.demo2` so add-on layers (ghost shape functions, neuron
   strip) can subscribe and react without touching this file. */
(function () {
  function init() {
    if (!window.DATA) {
      console.error('DATA missing -- did data/datasets.js load?');
      return;
    }

    const D = window.DATA;
    const svg = document.getElementById('plot');
    const slider = document.getElementById('width');
    const widthOut = document.getElementById('width-value');
    const caption = document.getElementById('caption');

    const VB_W = 720;
    const VB_H = 300;
    const M = { top: 12, right: 24, bottom: 32, left: 48 };
    const plotW = VB_W - M.left - M.right;
    const plotH = VB_H - M.top - M.bottom;

    // X grid for evaluating curves in the browser. The same length as the
    // precompute used; sum-of-neuron-contributions reconstructs the bold
    // fit to within machine epsilon (asserted at precompute time).
    const N_GRID = D.nGrid || 240;
    const xGrid = new Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      xGrid[i] = D.xMin + (D.xMax - D.xMin) * (i / (N_GRID - 1));
    }

    // ---------------- Y range (pad around all observed y) ----------------

    let yMin = Infinity, yMax = -Infinity;
    for (const [, py] of D.points) { if (py < yMin) yMin = py; if (py > yMax) yMax = py; }
    for (const [, py] of D.truthCurve) { if (py < yMin) yMin = py; if (py > yMax) yMax = py; }
    const yPad = 0.25 * (yMax - yMin);
    yMin -= yPad; yMax += yPad;

    function sx(x) { return M.left + (x - D.xMin) / (D.xMax - D.xMin) * plotW; }
    function sy(y) { return M.top + (yMax - y) / (yMax - yMin) * plotH; }

    // ---------------- Math utilities --------------------------------------

    function evalLinear(fit, xs) {
      const out = new Array(xs.length);
      for (let i = 0; i < xs.length; i++) out[i] = fit.a * xs[i] + fit.b;
      return out;
    }

    function evalNeuron(fit, idx, xs) {
      const w1 = fit.w1[idx], b = fit.bias[idx], v = fit.v[idx];
      const out = new Array(xs.length);
      for (let i = 0; i < xs.length; i++) {
        const pre = w1 * xs[i] + b;
        out[i] = v * (pre > 0 ? pre : 0);
      }
      return out;
    }

    function evalMlp(fit, xs) {
      const W = fit.w1.length;
      const out = new Array(xs.length).fill(fit.b2);
      for (let n = 0; n < W; n++) {
        const w1 = fit.w1[n], b = fit.bias[n], v = fit.v[n];
        for (let i = 0; i < xs.length; i++) {
          const pre = w1 * xs[i] + b;
          if (pre > 0) out[i] += v * pre;
        }
      }
      return out;
    }

    function evalFit(fit, xs) {
      return fit.kind === 'linear' ? evalLinear(fit, xs) : evalMlp(fit, xs);
    }

    // Kinks: per-neuron x = -bias/w1 with metadata (sign, strength, kept).
    // `inView` flag tells layers whether to render in the main plot region.
    function kinkData(fit) {
      if (fit.kind === 'linear') return [];
      const out = [];
      for (let i = 0; i < fit.w1.length; i++) {
        const w1 = fit.w1[i], b = fit.bias[i], v = fit.v[i];
        if (Math.abs(w1) < 1e-8) continue;
        const x = -b / w1;
        const inView = x >= D.xMin && x <= D.xMax;
        out.push({
          idx: i,
          x,
          w1, bias: b, v,
          inView,
          firesRight: w1 > 0,
          strength: Math.abs(w1 * v),
        });
      }
      return out;
    }

    // ---------------- Static layers (axes + truth + data) -----------------

    const NS = 'http://www.w3.org/2000/svg';
    function el(tag, attrs) {
      const node = document.createElementNS(NS, tag);
      if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
      return node;
    }

    // Grid (subtle, behind everything).
    const gGrid = el('g', { class: 'grid' });
    for (let yt = Math.ceil(yMin); yt <= Math.floor(yMax); yt += 1) {
      gGrid.appendChild(el('line', {
        x1: M.left, x2: M.left + plotW,
        y1: sy(yt), y2: sy(yt),
      }));
    }
    for (let xt = Math.ceil(D.xMin); xt <= Math.floor(D.xMax); xt += 1) {
      gGrid.appendChild(el('line', {
        x1: sx(xt), x2: sx(xt),
        y1: M.top, y2: M.top + plotH,
      }));
    }
    svg.appendChild(gGrid);

    // X axis.
    const gAx = el('g', { class: 'axis' });
    gAx.appendChild(el('line', {
      x1: M.left, x2: M.left + plotW,
      y1: M.top + plotH, y2: M.top + plotH,
    }));
    for (let xt = Math.ceil(D.xMin); xt <= Math.floor(D.xMax); xt += 1) {
      const x = sx(xt);
      gAx.appendChild(el('line', {
        x1: x, x2: x,
        y1: M.top + plotH, y2: M.top + plotH + 5,
      }));
      const t = el('text', { x, y: M.top + plotH + 18, 'text-anchor': 'middle' });
      t.textContent = xt.toString();
      gAx.appendChild(t);
    }
    const xLabel = el('text', {
      x: M.left + plotW / 2, y: M.top + plotH + 36, 'text-anchor': 'middle',
    });
    xLabel.textContent = 'x';
    gAx.appendChild(xLabel);

    // Y axis.
    gAx.appendChild(el('line', { x1: M.left, x2: M.left, y1: M.top, y2: M.top + plotH }));
    for (let yt = Math.ceil(yMin); yt <= Math.floor(yMax); yt += 1) {
      const y = sy(yt);
      gAx.appendChild(el('line', { x1: M.left - 5, x2: M.left, y1: y, y2: y }));
      const t = el('text', { x: M.left - 9, y: y + 4, 'text-anchor': 'end' });
      t.textContent = yt.toString();
      gAx.appendChild(t);
    }
    const yLabel = el('text', {
      x: 14, y: M.top + plotH / 2,
      'text-anchor': 'middle',
      transform: `rotate(-90 14 ${M.top + plotH / 2})`,
    });
    yLabel.textContent = 'y';
    gAx.appendChild(yLabel);
    svg.appendChild(gAx);

    // Ground truth curve.
    const truthPath = el('polyline', { class: 'truth-curve' });
    truthPath.setAttribute(
      'points',
      D.truthCurve.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' ')
    );
    svg.appendChild(truthPath);

    // Clip rectangle for the ghost layer: per-neuron contributions can blow
    // far past the plot's data range (one ReLU is unbounded above), so we
    // clip them to the plot's drawing region.
    const defs = el('defs');
    const clipId = 'plot-clip';
    const clipPath = el('clipPath', { id: clipId });
    clipPath.appendChild(el('rect', {
      x: M.left, y: M.top, width: plotW, height: plotH,
    }));
    defs.appendChild(clipPath);
    svg.appendChild(defs);

    // Mount points for add-on layers. Order matters for paint stack:
    //   ghost lines -> fit -> kinks -> data points -> hover guide on top.
    const gGhost = el('g', { class: 'ghost-layer', 'clip-path': `url(#${clipId})` });
    svg.appendChild(gGhost);

    const fitPath = el('polyline', { class: 'fit-curve' });
    svg.appendChild(fitPath);

    const gKinks = el('g', { class: 'kinks' });
    svg.appendChild(gKinks);

    const gPts = el('g', { class: 'points' });
    for (const [px, py] of D.points) {
      gPts.appendChild(el('circle', {
        class: 'data-point',
        cx: sx(px), cy: sy(py), r: 3.2,
      }));
    }
    svg.appendChild(gPts);

    // Mount group for the plot-hover module (vertical guide line + readout).
    // Lives on top of the data so the line crosses over points cleanly.
    const gHoverGuide = el('g', {
      class: 'hover-guide',
      'clip-path': `url(#${clipId})`,
    });
    svg.appendChild(gHoverGuide);

    // ---------------- Render bold fit + kinks -----------------------------

    function renderFit(fit) {
      const ys = evalFit(fit, xGrid);
      const pts = new Array(xGrid.length);
      for (let i = 0; i < xGrid.length; i++) {
        pts[i] = `${sx(xGrid[i]).toFixed(1)},${sy(ys[i]).toFixed(1)}`;
      }
      fitPath.setAttribute('points', pts.join(' '));
    }

    // Kink ticks: blue for fires-right, amber for fires-left. Length is
    // proportional to relative strength within the current width. Width 0
    // produces an empty list and the layer clears.
    function renderKinks(fit) {
      while (gKinks.firstChild) gKinks.removeChild(gKinks.firstChild);
      const kinks = kinkData(fit);
      if (kinks.length === 0) return;
      let maxStrength = 0;
      for (const k of kinks) if (k.inView && k.strength > maxStrength) maxStrength = k.strength;
      const baseY = M.top + plotH;
      const minLen = 5, maxLen = 14;
      for (const k of kinks) {
        if (!k.inView) continue;
        const t = maxStrength > 0 ? Math.sqrt(k.strength / maxStrength) : 0.5;
        const len = minLen + t * (maxLen - minLen);
        const cls = 'kink-tick ' + (k.firesRight ? 'fires-right' : 'fires-left');
        gKinks.appendChild(el('line', {
          class: cls,
          x1: sx(k.x), x2: sx(k.x),
          y1: baseY - 1, y2: baseY + len,
        }));
        gKinks.appendChild(el('circle', {
          class: 'kink-dot ' + (k.firesRight ? 'fires-right' : 'fires-left'),
          cx: sx(k.x), cy: baseY + len, r: 1.8,
        }));
      }
    }

    // ---------------- Caption ---------------------------------------------

    function captionFor(width) {
      if (width === 0)  return 'Width = 0 — a straight line. This is linear regression.';
      if (width === 1)  return 'Width = 1 — one ReLU adds one kink. One bend in the line.';
      if (width <= 4)   return `Width = ${width} — a coarse broken line. The bumps are emerging.`;
      if (width <= 9)   return `Width = ${width} — the broken line is starting to track the wave.`;
      if (width <= 15)  return `Width = ${width} — the fit tracks the wave well.`;
      if (width <= 22)  return `Width = ${width} — nearly indistinguishable from the ground truth.`;
      return `Width = ${width} — plenty of capacity; the fit hugs the noise.`;
    }

    // ---------------- State bus -------------------------------------------

    const subscribers = [];
    const bus = {
      width: 0,
      focusedNeuron: null,         // sticky single-neuron highlight (click)
      hover: null,                 // transient: null | {kind:'neuron',idx} | {kind:'x',x}
      fit: D.fits['0'],
      kinks: [],
      xGrid,
      sx, sy,
      M, plotW, plotH,
      data: D,
      onChange(fn) {
        subscribers.push(fn);
        try { fn(this); } catch (e) { console.error(e); }
      },
      _emit() {
        for (const fn of subscribers) {
          try { fn(this); } catch (e) { console.error(e); }
        }
      },
      setWidth(w, opts) {
        const cw = clampWidth(w);
        if (this.width === cw && opts && opts.silent) return;
        this.width = cw;
        this.fit = D.fits[String(cw)];
        this.kinks = kinkData(this.fit);
        this.focusedNeuron = null;
        this.hover = null;
        widthOut.textContent = String(cw);
        caption.textContent = captionFor(cw);
        renderFit(this.fit);
        renderKinks(this.fit);
        if (slider.value !== String(cw)) slider.value = String(cw);
        this._emit();
      },
      setFocus(idx) {
        if (idx != null && this.fit.kind !== 'mlp') return;
        if (idx != null) {
          if (idx < 0 || idx >= this.fit.w1.length) return;
        }
        if (this.focusedNeuron === idx) return;
        this.focusedNeuron = idx;
        this._emit();
      },
      setHover(spec) {
        // Normalize: null | {kind:'neuron',idx} | {kind:'x',x}
        let next = null;
        if (spec && spec.kind === 'neuron' && Number.isFinite(spec.idx)) {
          if (this.fit.kind === 'mlp' && spec.idx >= 0 && spec.idx < this.fit.w1.length) {
            next = { kind: 'neuron', idx: spec.idx | 0 };
          }
        } else if (spec && spec.kind === 'x' && Number.isFinite(spec.x)) {
          next = { kind: 'x', x: +spec.x };
        }
        if (this._sameHover(this.hover, next)) return;
        this.hover = next;
        this._emit();
      },
      _sameHover(a, b) {
        if (a === b) return true;
        if (!a || !b) return false;
        if (a.kind !== b.kind) return false;
        if (a.kind === 'neuron') return a.idx === b.idx;
        return Math.abs(a.x - b.x) < 1e-9;
      },
      effectiveFocus() {
        if (this.hover && this.hover.kind === 'neuron') return this.hover.idx;
        return this.focusedNeuron;
      },
      firingNeuronsAt(x) {
        if (!this.fit || this.fit.kind !== 'mlp') return [];
        const out = [];
        const W = this.fit.w1.length;
        for (let i = 0; i < W; i++) {
          if (this.fit.w1[i] * x + this.fit.bias[i] > 0) out.push(i);
        }
        return out;
      },
      evalNeuron(idx) {
        return this.fit.kind === 'mlp' ? evalNeuron(this.fit, idx, xGrid) : null;
      },
      evalFitAt(x) {
        if (!this.fit) return 0;
        if (this.fit.kind === 'linear') return this.fit.a * x + this.fit.b;
        let y = this.fit.b2;
        for (let i = 0; i < this.fit.w1.length; i++) {
          const pre = this.fit.w1[i] * x + this.fit.bias[i];
          if (pre > 0) y += this.fit.v[i] * pre;
        }
        return y;
      },
    };
    window.demo2 = bus;

    // ---------------- Slider + hash routing -------------------------------

    function clampWidth(v) {
      const n = parseInt(v, 10);
      if (!Number.isFinite(n)) return 0;
      return Math.max(0, Math.min(D.maxWidth, n));
    }

    function readHashWidth() {
      const m = (window.location.hash || '').match(/[#&?]w=(\d+)/);
      return m ? clampWidth(m[1]) : null;
    }

    function readHashFocus() {
      const m = (window.location.hash || '').match(/[#&?]focus=(\d+)/);
      return m ? parseInt(m[1], 10) : null;
    }

    function readHashHoverX() {
      const m = (window.location.hash || '').match(/[#&?]hovx=(-?\d+(?:\.\d+)?)/);
      if (!m) return null;
      const v = parseFloat(m[1]);
      return Number.isFinite(v) ? v : null;
    }

    function syncHash() {
      let h = `#w=${bus.width}`;
      if (bus.focusedNeuron != null) h += `&focus=${bus.focusedNeuron}`;
      if (window.location.hash !== h) history.replaceState(null, '', h);
    }

    slider.addEventListener('input', () => {
      bus.setWidth(slider.value);
      syncHash();
    });
    window.addEventListener('hashchange', () => {
      const w = readHashWidth();
      const f = readHashFocus();
      if (w != null && w !== bus.width) bus.setWidth(w);
      if (f != null) bus.setFocus(f);
    });
    // Click on empty plot area clears focus (subscribers may also do this).
    svg.addEventListener('click', (e) => {
      // Only clear when the click hits the SVG background, not a child layer.
      if (e.target === svg && bus.focusedNeuron != null) {
        bus.setFocus(null);
        syncHash();
      }
    });

    // Read URL hash before subscribing syncHash, otherwise the initial
    // onChange callback would rewrite #w=N back to #w=0.
    const initialW = readHashWidth();
    const initialF = readHashFocus();
    const initialHoverX = readHashHoverX();
    bus.setWidth(initialW != null ? initialW : 0);
    if (initialF != null) bus.setFocus(initialF);
    if (initialHoverX != null) bus.setHover({ kind: 'x', x: initialHoverX });

    bus.onChange(syncHash);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
