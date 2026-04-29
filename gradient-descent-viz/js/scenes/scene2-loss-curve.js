/* Scene 2 — Loss curve panel.

   Tiny SVG (viewBox 280x160) that plots log10(loss) vs iteration. On every
   bus.onUpdate it appends the latest (iter, loss) sample and redraws the
   polyline plus the y/x scale labels. Y range auto-fits to data; x range
   auto-extends. Subtle grid lines at log decades.

   Mounted via `buildLossCurve(root, bus)`. */
(function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const RENDER_MIN_DT_MS = 32;

  const VB_W = 280, VB_H = 160;
  const M = { top: 14, right: 12, bottom: 28, left: 36 };
  const PLOT_W = VB_W - M.left - M.right;
  const PLOT_H = VB_H - M.top - M.bottom;

  function el(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function buildLossCurve(root, bus) {
    if (!root) throw new Error('buildLossCurve: root missing');
    if (!bus || !bus.engine) throw new Error('buildLossCurve: bus.engine missing');

    while (root.firstChild) root.removeChild(root.firstChild);

    // Title above the SVG.
    const title = document.createElement('div');
    title.className = 's2-loss-title';
    title.textContent = 'training loss';
    root.appendChild(title);

    const svg = el('svg', {
      class: 's2-loss-curve',
      viewBox: `0 0 ${VB_W} ${VB_H}`,
      preserveAspectRatio: 'xMidYMid meet',
    });
    root.appendChild(svg);

    // Mounted layers.
    const gGrid = el('g', { class: 'grid' });
    svg.appendChild(gGrid);
    const gAxis = el('g', { class: 'axis' });
    svg.appendChild(gAxis);
    const polyline = el('polyline', { class: 'loss-line' });
    svg.appendChild(polyline);

    // Iteration / loss history. Capped to avoid unbounded memory if the
    // user mashes Train forever; we down-sample by stride at the cap.
    const MAX_SAMPLES = 4000;
    const iters = [];
    const losses = [];

    // Auto-scaling state: log10(loss) range (low/high), iter range (0..max).
    let logLo = -2, logHi = 1;     // initial guess; widens to fit data
    let iterMax = 100;              // initial x extent; extends with data

    function recomputeRange() {
      // Compute log10 of finite, positive losses; pad the y-range slightly.
      let lo = +Infinity, hi = -Infinity;
      for (let i = 0; i < losses.length; i++) {
        const v = losses[i];
        if (!(v > 0) || !Number.isFinite(v)) continue;
        const lg = Math.log10(v);
        if (lg < lo) lo = lg;
        if (lg > hi) hi = lg;
      }
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
        logLo = -2; logHi = 1; return;
      }
      // Pad by 0.2 log decades on each side (~58% on linear scale).
      const pad = 0.2;
      logLo = lo - pad;
      logHi = hi + pad;
      // Snap to whole decades for cleaner gridlines.
      logLo = Math.floor(logLo);
      logHi = Math.ceil(logHi);
      if (logHi - logLo < 1) logHi = logLo + 1;

      // X extent: snap up to the next "nice" round number above the latest iter.
      const last = iters.length ? iters[iters.length - 1] : 0;
      iterMax = Math.max(100, niceCeil(last));
    }

    function niceCeil(x) {
      if (x <= 0) return 100;
      const e = Math.pow(10, Math.floor(Math.log10(x)));
      const m = x / e;
      let mNice;
      if (m <= 1) mNice = 1;
      else if (m <= 2) mNice = 2;
      else if (m <= 5) mNice = 5;
      else mNice = 10;
      return mNice * e;
    }

    function sx(it) { return M.left + (it / iterMax) * PLOT_W; }
    function sy(logV) {
      // logHi at top, logLo at bottom (y grows downward in SVG).
      return M.top + (logHi - logV) / (logHi - logLo) * PLOT_H;
    }

    function renderAxes() {
      while (gGrid.firstChild) gGrid.removeChild(gGrid.firstChild);
      while (gAxis.firstChild) gAxis.removeChild(gAxis.firstChild);

      // Y gridlines + labels at every log decade.
      for (let lg = Math.floor(logLo); lg <= Math.ceil(logHi); lg++) {
        const y = sy(lg);
        gGrid.appendChild(el('line', {
          x1: M.left, x2: M.left + PLOT_W, y1: y, y2: y,
        }));
        const t = el('text', {
          class: 'tick-y',
          x: M.left - 4, y: y + 3, 'text-anchor': 'end',
        });
        // Format loss as 10^lg, e.g. 0.01, 0.1, 1, 10
        let label;
        if (lg <= -3) label = `1e${lg}`;
        else if (lg < 0) label = '0.' + '0'.repeat(-lg - 1) + '1';
        else if (lg === 0) label = '1';
        else if (lg <= 3) label = String(Math.pow(10, lg));
        else label = `1e${lg}`;
        t.textContent = label;
        gAxis.appendChild(t);
      }

      // X axis line.
      gAxis.appendChild(el('line', {
        class: 'x-axis-line',
        x1: M.left, x2: M.left + PLOT_W,
        y1: M.top + PLOT_H, y2: M.top + PLOT_H,
      }));
      // Y axis line.
      gAxis.appendChild(el('line', {
        class: 'y-axis-line',
        x1: M.left, x2: M.left,
        y1: M.top, y2: M.top + PLOT_H,
      }));

      // X-axis tick labels (4 evenly spaced positions including 0 and iterMax).
      const N_X_TICKS = 4;
      for (let k = 0; k <= N_X_TICKS; k++) {
        const it = Math.round((iterMax * k) / N_X_TICKS);
        const x = sx(it);
        gAxis.appendChild(el('line', {
          x1: x, x2: x,
          y1: M.top + PLOT_H, y2: M.top + PLOT_H + 3,
        }));
        const t = el('text', {
          class: 'tick-x',
          x: x, y: M.top + PLOT_H + 14, 'text-anchor': 'middle',
        });
        t.textContent = String(it);
        gAxis.appendChild(t);
      }
      // Axis labels.
      const xL = el('text', {
        class: 'axis-label',
        x: M.left + PLOT_W / 2, y: VB_H - 4, 'text-anchor': 'middle',
      });
      xL.textContent = 'iter';
      gAxis.appendChild(xL);

      const yL = el('text', {
        class: 'axis-label',
        x: 8, y: M.top + PLOT_H / 2,
        'text-anchor': 'middle',
        transform: `rotate(-90 8 ${M.top + PLOT_H / 2})`,
      });
      yL.textContent = 'loss';
      gAxis.appendChild(yL);
    }

    function renderPolyline() {
      const N = iters.length;
      if (N === 0) {
        polyline.setAttribute('points', '');
        return;
      }
      // Build the points string. Drop non-positive losses (can't take log).
      const parts = [];
      for (let i = 0; i < N; i++) {
        const v = losses[i];
        if (!(v > 0) || !Number.isFinite(v)) continue;
        const lg = Math.log10(v);
        parts.push(`${sx(iters[i]).toFixed(1)},${sy(lg).toFixed(1)}`);
      }
      polyline.setAttribute('points', parts.join(' '));
    }

    let lastRender = 0;
    function render() {
      const p = bus.engine.params();
      // Append latest sample if it advanced.
      const lastIter = iters.length ? iters[iters.length - 1] : -1;
      if (p.iter !== lastIter) {
        iters.push(p.iter);
        losses.push(p.loss);
        if (iters.length > MAX_SAMPLES) {
          // Stride-2 down-sample preserves shape and bounds memory.
          const newIters = [], newLosses = [];
          for (let i = 0; i < iters.length; i += 2) {
            newIters.push(iters[i]);
            newLosses.push(losses[i]);
          }
          iters.length = 0; losses.length = 0;
          for (let i = 0; i < newIters.length; i++) {
            iters.push(newIters[i]); losses.push(newLosses[i]);
          }
        }
      }

      const now = Date.now();
      if (now - lastRender < RENDER_MIN_DT_MS) return;
      lastRender = now;

      recomputeRange();
      renderAxes();
      renderPolyline();
    }

    function reset() {
      iters.length = 0;
      losses.length = 0;
      logLo = -2; logHi = 1;
      iterMax = 100;
      // Push the initial point so the curve starts somewhere useful.
      const p = bus.engine.params();
      iters.push(p.iter);
      losses.push(p.loss);
      recomputeRange();
      renderAxes();
      renderPolyline();
    }

    // Initial paint.
    reset();

    bus.onUpdate(render);

    return { renderNow: () => { renderAxes(); renderPolyline(); }, reset };
  }

  window.GDVScene2 = window.GDVScene2 || {};
  window.GDVScene2.buildLossCurve = buildLossCurve;
})();
