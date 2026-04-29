/* Scene 2 — Live plot panel.

   Mirrors the visual language of demo2-stack-neurons/js/main.js but driven
   by a TrainingEngine instead of precomputed fits. On every bus update the
   panel re-evaluates the bold fit, the per-neuron ghost contributions, and
   the kink ticks at the x-axis. The plot is mounted under root via
   `buildPlot(root, bus)`; the bus owns the engine and emits ticks.

   Static (rendered once): grid, axes, ground-truth dashed curve, scatter.
   Animated (re-rendered on bus.onUpdate): fit polyline, ghost lines,
   kink ticks. Internal throttle caps re-renders at ~30 fps even if the
   engine ticks faster. */
(function () {
  'use strict';

  const NS = 'http://www.w3.org/2000/svg';
  const RENDER_MIN_DT_MS = 32;   // ~30 fps throttle

  function el(tag, attrs) {
    const node = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  function defaultGhostOpacity(width) {
    if (width <= 0) return 0.18;
    const o = 1 / Math.sqrt(width);
    if (o < 0.06) return 0.06;
    if (o > 0.18) return 0.18;
    return o;
  }

  // -------- mounting -------------------------------------------------------

  function buildPlot(root, bus) {
    if (!root) throw new Error('buildPlot: root missing');
    if (!bus || !bus.engine) throw new Error('buildPlot: bus.engine missing');
    const D = window.GDV_DATA;
    if (!D) throw new Error('buildPlot: GDV_DATA missing');

    // Wipe any previous content (idempotent re-mount).
    while (root.firstChild) root.removeChild(root.firstChild);

    const VB_W = 720, VB_H = 320;
    const M = { top: 12, right: 24, bottom: 32, left: 48 };
    const plotW = VB_W - M.left - M.right;
    const plotH = VB_H - M.top - M.bottom;

    const N_GRID = 240;
    const xGrid = new Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      xGrid[i] = D.xMin + (D.xMax - D.xMin) * (i / (N_GRID - 1));
    }

    // Y range: pad around all observed y-values from the fixed scatter and
    // the ground truth curve. Stays static throughout training so the fit
    // can flop around without rescaling the axis.
    let yMin = Infinity, yMax = -Infinity;
    for (const [, py] of D.points) { if (py < yMin) yMin = py; if (py > yMax) yMax = py; }
    for (const [, py] of D.truthCurve) { if (py < yMin) yMin = py; if (py > yMax) yMax = py; }
    const yPad = 0.25 * (yMax - yMin);
    yMin -= yPad; yMax += yPad;

    function sx(x) { return M.left + (x - D.xMin) / (D.xMax - D.xMin) * plotW; }
    function sy(y) { return M.top + (yMax - y) / (yMax - yMin) * plotH; }

    // --- root SVG ---
    const svg = el('svg', {
      class: 's2-plot',
      viewBox: `0 0 ${VB_W} ${VB_H}`,
      preserveAspectRatio: 'xMidYMid meet',
    });
    root.appendChild(svg);

    // ------- static layers -----------------------------------------------

    // Grid lines.
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

    // X-axis.
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
      x: M.left + plotW / 2, y: M.top + plotH + 30, 'text-anchor': 'middle',
    });
    xLabel.textContent = 'x';
    gAx.appendChild(xLabel);

    // Y-axis.
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

    // Ground truth dashed curve.
    const truthPath = el('polyline', { class: 'truth-curve' });
    truthPath.setAttribute('points',
      D.truthCurve.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' '));
    svg.appendChild(truthPath);

    // Clip path for ghost lines (per-neuron contributions are unbounded).
    const defs = el('defs');
    const clipId = 's2-plot-clip-' + Math.floor(Math.random() * 1e9);
    const clipPath = el('clipPath', { id: clipId });
    clipPath.appendChild(el('rect', {
      x: M.left, y: M.top, width: plotW, height: plotH,
    }));
    defs.appendChild(clipPath);
    svg.appendChild(defs);

    // --- mount points (paint order matters) ---
    const gGhost = el('g', { class: 'ghost-layer', 'clip-path': `url(#${clipId})` });
    svg.appendChild(gGhost);

    const fitPath = el('polyline', { class: 'fit-curve' });
    svg.appendChild(fitPath);

    const gKinks = el('g', { class: 'kinks' });
    svg.appendChild(gKinks);

    // Static scatter. Always on top of fit/ghost/truth so points read clearly.
    const gPts = el('g', { class: 'points' });
    for (const [px, py] of D.points) {
      gPts.appendChild(el('circle', {
        class: 'data-point',
        cx: sx(px), cy: sy(py), r: 3.2,
      }));
    }
    svg.appendChild(gPts);

    // ------- dynamic render ----------------------------------------------

    // We re-use string buffers across renders to avoid per-tick allocations.
    function renderFit(p) {
      const xs = xGrid;
      const ys = bus.engine.evalAt(xs);
      const parts = new Array(xs.length);
      for (let i = 0; i < xs.length; i++) {
        parts[i] = `${sx(xs[i]).toFixed(1)},${sy(ys[i]).toFixed(1)}`;
      }
      fitPath.setAttribute('points', parts.join(' '));
    }

    function renderGhosts(p) {
      // Replace contents wholesale; cheaper than diffing for ~20 polylines.
      while (gGhost.firstChild) gGhost.removeChild(gGhost.firstChild);
      const W = p.w1.length;
      const op = defaultGhostOpacity(W);
      for (let j = 0; j < W; j++) {
        const ys = bus.engine.evalNeuron(j, xGrid);
        const parts = new Array(xGrid.length);
        for (let i = 0; i < xGrid.length; i++) {
          parts[i] = `${sx(xGrid[i]).toFixed(1)},${sy(ys[i]).toFixed(1)}`;
        }
        const poly = el('polyline', {
          class: 'ghost-line',
          points: parts.join(' '),
          opacity: op.toFixed(3),
        });
        gGhost.appendChild(poly);
      }
    }

    function renderKinks(p) {
      while (gKinks.firstChild) gKinks.removeChild(gKinks.firstChild);
      const W = p.w1.length;
      // Compute strengths and find max so tick lengths read relative.
      let maxStrength = 0;
      const ks = [];
      for (let i = 0; i < W; i++) {
        const w1 = p.w1[i], b = p.bias[i], v = p.v[i];
        if (Math.abs(w1) < 1e-8) continue;
        const x = -b / w1;
        const inView = x >= D.xMin && x <= D.xMax;
        const strength = Math.abs(w1 * v);
        if (inView && strength > maxStrength) maxStrength = strength;
        ks.push({ x, w1, v, inView, strength });
      }
      const baseY = M.top + plotH;
      const minLen = 5, maxLen = 14;
      for (const k of ks) {
        if (!k.inView) continue;
        const t = maxStrength > 0 ? Math.sqrt(k.strength / maxStrength) : 0.5;
        const len = minLen + t * (maxLen - minLen);
        const cls = 'kink-tick ' + (k.w1 > 0 ? 'fires-right' : 'fires-left');
        gKinks.appendChild(el('line', {
          class: cls,
          x1: sx(k.x), x2: sx(k.x),
          y1: baseY - 1, y2: baseY + len,
        }));
        gKinks.appendChild(el('circle', {
          class: 'kink-dot ' + (k.w1 > 0 ? 'fires-right' : 'fires-left'),
          cx: sx(k.x), cy: baseY + len, r: 1.8,
        }));
      }
    }

    let lastRender = 0;
    function render() {
      const now = Date.now();
      if (now - lastRender < RENDER_MIN_DT_MS) return;
      lastRender = now;
      const p = bus.engine.params();
      renderGhosts(p);
      renderFit(p);
      renderKinks(p);
    }

    // Initial paint — bypass throttle so cold entry is immediate.
    function renderNow() {
      lastRender = Date.now();
      const p = bus.engine.params();
      renderGhosts(p);
      renderFit(p);
      renderKinks(p);
    }
    renderNow();

    bus.onUpdate(render);

    // Expose a small handle so the orchestrator can force a paint when it
    // hot-swaps the engine (e.g. width slider) without waiting for a tick.
    return { renderNow };
  }

  // main.js owns window.GDV (the scene engine). We park sub-panel builders
  // on window.GDVScene2 to avoid the namespace collision when main.js
  // assigns window.GDV at the end of its IIFE.
  window.GDVScene2 = window.GDVScene2 || {};
  window.GDVScene2.buildPlot = buildPlot;
})();
