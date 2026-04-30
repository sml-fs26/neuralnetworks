/* Scene 3 — "Inside the high-D landscape"

   Demo 2's MLP has P = 3W + 1 = 37 parameters at width 12. We cannot
   draw 37-D, so we slice through θ* along two basis vectors:
       δ₁ = θ_init − θ*           (the descent direction, by construction)
       δ₂ = an orthogonal direction that has been filter-normalized
            (Hao Li et al., 2018 — the standard recipe for visualising
             neural-net loss landscapes)
   The slice is a 2D plane through θ*, and the loss restricted to that
   plane is a bowl with its minimum near the origin (α,β) = (0,0).
   That bowl is the whole reason gradient descent works on neural nets,
   even though the full 37-D landscape is famously non-convex.

   Live behaviour:
     - Spawn a TrainingEngine and a marble at the projection of θ_t.
     - Train button steps the engine at ~30 Hz; marble + trajectory +
       right-pane fit re-render each step.
     - Optimizer / lr / seed / Reset controls operate on the engine.

   Reads:
     window.GDV_DATA.projection (precomputed by precompute/build_projection.py).
     Falls back to a runtime-computed stub if missing — same math,
     coarser grid (50×50 instead of 100×100), suitable for development.

   Writes: nothing outside js/scenes/scene3.js and css/scene3.css. */
(function () {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const RENDER_MIN_DT_MS = 32;       // ~30 fps DOM throttle
  const TICK_MS = 33;                // ~30 Hz engine tick
  const DEFAULT_WIDTH = 12;
  const DEFAULT_LR = 0.05;
  const DEFAULT_SEED = 42;
  const DEFAULT_OPT = 'adam';
  const MAX_TRAJ = 400;

  // ---- URL-hash helpers ----------------------------------------------------

  function readHashFlag(name) {
    const re = new RegExp('[#&?]' + name + '(?:=([^&]*))?');
    const m = (window.location.hash || '').match(re);
    return m ? (m[1] != null ? m[1] : true) : null;
  }

  function readHashInt(name, fallback) {
    const v = readHashFlag(name);
    if (v == null || v === true) return fallback;
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : fallback;
  }

  // ---- DOM helpers ---------------------------------------------------------

  function el(tag, attrs, parent) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (k === 'html') node.innerHTML = attrs[k];
        else node.setAttribute(k, attrs[k]);
      }
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      for (const k in attrs) {
        if (k === 'class') node.setAttribute('class', attrs[k]);
        else if (k === 'text') node.textContent = attrs[k];
        else node.setAttribute(k, attrs[k]);
      }
    }
    if (parent) parent.appendChild(node);
    return node;
  }

  function svgPointFromEvent(svgEl, ev) {
    const rect = svgEl.getBoundingClientRect();
    const vb = svgEl.viewBox.baseVal;
    const sx = vb.width / rect.width;
    const sy = vb.height / rect.height;
    return {
      x: (ev.clientX - rect.left) * sx,
      y: (ev.clientY - rect.top) * sy,
    };
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  // ---- Projection stub fallback -------------------------------------------
  //
  // If precompute/build_projection.py hasn't run yet (no
  // window.GDV_DATA.projection), compute a slice on the fly. We:
  //   1) snapshot θ_init from a fresh engine (seed=42, width=12)
  //   2) train Adam for ~600 steps to land θ* near a converged solution
  //   3) δ₁ = θ_init − θ*; δ₂ = filter-normalized orthogonalised random
  //   4) evaluate L on a 50×50 (α, β) grid via direct MSE forward passes
  // The cached stub is parked on window.GDV_DATA.projection so subsequent
  // re-entries to the scene are O(1).

  function buildProjectionStub() {
    const W = DEFAULT_WIDTH;
    const SEED = DEFAULT_SEED;
    const STAR_STEPS = 600;
    const N_ALPHA = 50, N_BETA = 50;
    const A_MIN = -0.4, A_MAX = 1.4;
    const B_MIN = -1.0, B_MAX = 1.0;

    // 1) fresh engine, take θ_init
    const eng = new window.TrainingEngine({
      width: W, seed: SEED, lr: DEFAULT_LR, optimizer: DEFAULT_OPT,
    });
    const thetaInit = eng.params();
    const thetaInitFlat = window.Projection.flatten(thetaInit);
    const initLoss = thetaInit.loss;

    // 2) train to θ*
    for (let i = 0; i < STAR_STEPS; i++) eng.step();
    const thetaStar = eng.params();
    const thetaStarFlat = window.Projection.flatten(thetaStar);
    const starLoss = thetaStar.loss;

    // 3) δ₁ = θ_init − θ*
    const delta1Flat = window.Projection.diff(thetaInitFlat, thetaStarFlat);

    // 4) δ₂: random Gaussian (Mulberry32 + Box-Muller, deterministic),
    //    Gram-Schmidt against δ₁, then filter-normalise: rescale per
    //    parameter group to match the corresponding norm of θ*.
    const rand = makeRng(SEED * 7919 + 1);
    const randn = makeNormal(rand);
    const P = thetaStarFlat.length;
    const raw = new Array(P);
    for (let i = 0; i < P; i++) raw[i] = randn();

    // Gram-Schmidt: subtract projection onto δ₁.
    const d1d1 = window.Projection.dot(delta1Flat, delta1Flat);
    const rd1 = window.Projection.dot(raw, delta1Flat);
    const proj = window.Projection.scale(delta1Flat, rd1 / d1d1);
    let delta2Flat = window.Projection.diff(raw, proj);

    // Filter-normalise: scale δ₂ to match ‖δ₁‖. The Hao Li paper
    // normalises *per filter*, but for a tiny MLP the global rescale to
    // the same norm gives a faithful slice and the bowl reads clearly.
    const d1Norm = Math.sqrt(d1d1);
    const d2Norm = Math.sqrt(window.Projection.dot(delta2Flat, delta2Flat));
    if (d2Norm > 1e-12) {
      delta2Flat = window.Projection.scale(delta2Flat, d1Norm / d2Norm);
    }

    // 5) evaluate L on the (α, β) grid
    const xs = new Float64Array(window.GDV_DATA.points.length);
    const ys = new Float64Array(window.GDV_DATA.points.length);
    for (let i = 0; i < xs.length; i++) {
      xs[i] = +window.GDV_DATA.points[i][0];
      ys[i] = +window.GDV_DATA.points[i][1];
    }

    const values = new Float32Array(N_ALPHA * N_BETA);
    for (let i = 0; i < N_ALPHA; i++) {
      const a = A_MIN + (A_MAX - A_MIN) * i / (N_ALPHA - 1);
      for (let j = 0; j < N_BETA; j++) {
        const b = B_MIN + (B_MAX - B_MIN) * j / (N_BETA - 1);
        const theta = window.Projection.synth(thetaStarFlat, delta1Flat, delta2Flat, a, b);
        values[i * N_BETA + j] = mlpMSE(theta, W, xs, ys);
      }
    }

    return {
      width: W, P,
      thetaInit: {
        w1: thetaInit.w1, bias: thetaInit.bias, v: thetaInit.v, b2: thetaInit.b2,
      },
      thetaStar: {
        w1: thetaStar.w1, bias: thetaStar.bias, v: thetaStar.v, b2: thetaStar.b2,
      },
      delta1: window.Projection.unflatten(delta1Flat, W),
      delta2: window.Projection.unflatten(delta2Flat, W),
      thetaInitFlat, thetaStarFlat, delta1Flat, delta2Flat,
      grid: {
        alphaMin: A_MIN, alphaMax: A_MAX, nAlpha: N_ALPHA,
        betaMin: B_MIN, betaMax: B_MAX, nBeta: N_BETA,
        values: Array.from(values),
      },
      initPoint: [1.0, 0.0],
      optimumPoint: [0.0, 0.0],
      initLoss, starLoss,
      __stub: true,
    };
  }

  // Forward pass + MSE for a flattened parameter vector.
  function mlpMSE(theta, W, xs, ys) {
    const n = xs.length;
    let sse = 0;
    const b2 = theta[3 * W];
    for (let i = 0; i < n; i++) {
      const xi = xs[i];
      let s = b2;
      for (let j = 0; j < W; j++) {
        const p = theta[j] * xi + theta[W + j];
        if (p > 0) s += theta[2 * W + j] * p;
      }
      const e = s - ys[i];
      sse += e * e;
    }
    return sse / n;
  }

  function makeRng(seed) {
    let s = (seed | 0) || 1;
    return function () {
      s |= 0; s = (s + 0x6D2B79F5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeNormal(rng) {
    let cached = null;
    return function () {
      if (cached !== null) { const v = cached; cached = null; return v; }
      let u1 = rng(); if (u1 < 1e-12) u1 = 1e-12;
      const u2 = rng();
      const r = Math.sqrt(-2 * Math.log(u1));
      const a = 2 * Math.PI * u2;
      cached = r * Math.sin(a);
      return r * Math.cos(a);
    };
  }

  function ensureProjection() {
    if (window.GDV_DATA && window.GDV_DATA.projection
        && window.GDV_DATA.projection.grid
        && window.GDV_DATA.projection.thetaStarFlat) {
      return window.GDV_DATA.projection;
    }
    const proj = buildProjectionStub();
    window.GDV_DATA.projection = proj;
    return proj;
  }

  // ---- Bilinear grid sampler ----------------------------------------------
  // grid.values is row-major: values[i*nBeta + j] = L at (α[i], β[j]).
  function sampleGrid(grid, alpha, beta) {
    const { alphaMin, alphaMax, nAlpha, betaMin, betaMax, nBeta, values } = grid;
    const ai = (alpha - alphaMin) / (alphaMax - alphaMin) * (nAlpha - 1);
    const bj = (beta - betaMin) / (betaMax - betaMin) * (nBeta - 1);
    const i0 = Math.max(0, Math.min(nAlpha - 1, Math.floor(ai)));
    const j0 = Math.max(0, Math.min(nBeta - 1, Math.floor(bj)));
    const i1 = Math.min(nAlpha - 1, i0 + 1);
    const j1 = Math.min(nBeta - 1, j0 + 1);
    const ta = clamp(ai - i0, 0, 1);
    const tb = clamp(bj - j0, 0, 1);
    const v00 = values[i0 * nBeta + j0];
    const v10 = values[i1 * nBeta + j0];
    const v01 = values[i0 * nBeta + j1];
    const v11 = values[i1 * nBeta + j1];
    return (1 - ta) * (1 - tb) * v00
         + ta       * (1 - tb) * v10
         + (1 - ta) * tb       * v01
         + ta       * tb       * v11;
  }

  // ---- Loss-landscape pane (left) -----------------------------------------

  function buildLandscape(card, ctx) {
    const VW = 480, VH = 420;
    const padL = 56, padR = 16, padT = 18, padB = 46;
    const plotW = VW - padL - padR;
    const plotH = VH - padT - padB;

    const grid = ctx.proj.grid;
    const { alphaMin, alphaMax, nAlpha, betaMin, betaMax, nBeta } = grid;

    // (α, β) -> svg(x, y); β flipped so larger β is upward.
    function aToX(a) { return padL + (a - alphaMin) / (alphaMax - alphaMin) * plotW; }
    function bToY(b) { return padT + (1 - (b - betaMin) / (betaMax - betaMin)) * plotH; }
    function xToA(x) { return alphaMin + (x - padL) / plotW * (alphaMax - alphaMin); }
    function yToB(y) { return betaMin + (1 - (y - padT) / plotH) * (betaMax - betaMin); }

    // Contour transform: grid (i, j) -> (α, β) -> svg.
    function gridToSvg(i, j) {
      const a = alphaMin + (alphaMax - alphaMin) * i / (nAlpha - 1);
      const b = betaMin + (betaMax - betaMin) * j / (nBeta - 1);
      return [aToX(a), bToY(b)];
    }

    const root = svg('svg', {
      class: 's3-svg s3-landscape',
      viewBox: `0 0 ${VW} ${VH}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': '2D loss landscape in projected (α, β) coordinates',
    }, card);

    svg('rect', { class: 's3-pane-bg', x: 0, y: 0, width: VW, height: VH }, root);
    svg('rect', {
      class: 's3-plot-frame',
      x: padL, y: padT, width: plotW, height: plotH,
    }, root);

    // Contours
    const cgroup = svg('g', { class: 's3-contours' }, root);
    const levels = window.LossUtils.chooseLevels(grid.values, 9);
    const paths = window.LossUtils.contourPaths(
      grid.values, nAlpha, nBeta, levels, gridToSvg);
    for (let k = 0; k < paths.length; k++) {
      svg('path', {
        class: (k % 3 === 2) ? 's3-contour s3-contour-hi' : 's3-contour',
        d: paths[k].d,
      }, cgroup);
    }

    // Axes
    svg('line', { class: 's3-axis',
      x1: padL, y1: padT + plotH, x2: padL + plotW, y2: padT + plotH }, root);
    svg('line', { class: 's3-axis',
      x1: padL, y1: padT, x2: padL, y2: padT + plotH }, root);

    // Ticks: show min, 0, max on each axis where 0 is in range.
    function addTicks(values, axis) {
      values.forEach(v => {
        if (axis === 'a') {
          if (v < alphaMin - 1e-9 || v > alphaMax + 1e-9) return;
          const x = aToX(v);
          svg('line', { class: 's3-tick',
            x1: x, y1: padT + plotH, x2: x, y2: padT + plotH + 4 }, root);
          svg('text', { class: 's3-tick-label',
            x: x, y: padT + plotH + 16, 'text-anchor': 'middle',
            text: v.toFixed(1) }, root);
        } else {
          if (v < betaMin - 1e-9 || v > betaMax + 1e-9) return;
          const y = bToY(v);
          svg('line', { class: 's3-tick',
            x1: padL - 4, y1: y, x2: padL, y2: y }, root);
          svg('text', { class: 's3-tick-label',
            x: padL - 6, y: y + 4, 'text-anchor': 'end',
            text: v.toFixed(1) }, root);
        }
      });
    }
    addTicks([alphaMin, 0, alphaMax], 'a');
    addTicks([betaMin, 0, betaMax], 'b');

    // Axis labels (α and β)
    const axLabelA = svg('text', {
      class: 's3-axis-label',
      x: padL + plotW / 2, y: VH - 12,
      'text-anchor': 'middle',
      text: 'α   (init → trained direction)',
    }, root);
    const axLabelB = svg('text', {
      class: 's3-axis-label',
      x: 14, y: padT + plotH / 2,
      'text-anchor': 'middle',
      transform: `rotate(-90 14 ${padT + plotH / 2})`,
      text: 'β   (Adam-min → SGD-min direction)',
    }, root);
    void axLabelA; void axLabelB;

    // θ* marker (cross at origin)
    const optX = aToX(0), optY = bToY(0);
    const optG = svg('g', { class: 's3-optimum' }, root);
    svg('line', { class: 's3-optimum-mark',
      x1: optX - 6, y1: optY, x2: optX + 6, y2: optY }, optG);
    svg('line', { class: 's3-optimum-mark',
      x1: optX, y1: optY - 6, x2: optX, y2: optY + 6 }, optG);
    svg('text', {
      class: 's3-marker-label',
      x: optX + 8, y: optY - 8, 'text-anchor': 'start',
      text: 'θ*',
    }, optG);

    // init marker (small open ring at (1, 0))
    const initA = ctx.proj.initPoint[0], initB = ctx.proj.initPoint[1];
    const initX = aToX(initA), initY = bToY(initB);
    const initG = svg('g', { class: 's3-init' }, root);
    svg('circle', { class: 's3-init-ring',
      cx: initX, cy: initY, r: 5 }, initG);
    svg('text', {
      class: 's3-marker-label',
      x: initX + 8, y: initY - 8, 'text-anchor': 'start',
      text: 'θ₀',
    }, initG);

    // SGD-min marker (open cross + label) -- only if the precompute
    // shipped one. By construction it lies near (alpha_sgd, 1), inside
    // the rendered grid. Same for momentum-min.
    function placeMinMarker(point, label, klass) {
      if (!point) return;
      const a = point[0], b = point[1];
      const x = aToX(a), y = bToY(b);
      // Skip rendering if the point falls outside the visible grid.
      if (x < padL - 1 || x > padL + plotW + 1
          || y < padT - 1 || y > padT + plotH + 1) return;
      const g = svg('g', { class: klass }, root);
      svg('line', { class: 's3-altmin-mark',
        x1: x - 5, y1: y - 5, x2: x + 5, y2: y + 5 }, g);
      svg('line', { class: 's3-altmin-mark',
        x1: x - 5, y1: y + 5, x2: x + 5, y2: y - 5 }, g);
      svg('text', {
        class: 's3-marker-label',
        x: x + 8, y: y - 8, 'text-anchor': 'start',
        text: label,
      }, g);
    }
    placeMinMarker(ctx.proj.sgdMinPoint, 'θ★ SGD/Mom', 's3-altmin s3-altmin-sgd');
    // Suppress a separate momentum marker if it sits within ~0.12 of
    // SGD's (Euclidean) -- on this dataset they always do, and stacking
    // both labels just creates a cluttered overlap. The "SGD/Mom" label
    // above acknowledges they share the basin.
    const momP = ctx.proj.momentumMinPoint;
    const sgdP = ctx.proj.sgdMinPoint;
    const dx = (momP && sgdP) ? momP[0] - sgdP[0] : Infinity;
    const dy = (momP && sgdP) ? momP[1] - sgdP[1] : Infinity;
    const farFromSgd = (dx * dx + dy * dy) > 0.12 * 0.12;
    if (farFromSgd) {
      placeMinMarker(momP, 'θ★ Mom', 's3-altmin s3-altmin-mom');
    }

    // Trajectory polyline + marble (drawn after markers so they sit on top)
    const trajPath = svg('polyline', { class: 's3-trajectory', points: '' }, root);
    const marble = svg('circle', { class: 's3-marble',
      cx: initX, cy: initY, r: 5 }, root);

    // Hover readout box
    const hoverGroup = svg('g', { class: 's3-hover hidden' }, root);
    const hoverBg = svg('rect', { class: 's3-hover-bg',
      x: 0, y: 0, width: 130, height: 46, rx: 3 }, hoverGroup);
    const hoverL1 = svg('text', { class: 's3-hover-text',
      x: 6, y: 14, text: '' }, hoverGroup);
    const hoverL2 = svg('text', { class: 's3-hover-text',
      x: 6, y: 28, text: '' }, hoverGroup);
    const hoverL3 = svg('text', { class: 's3-hover-text',
      x: 6, y: 42, text: '' }, hoverGroup);

    root.addEventListener('mousemove', (ev) => {
      const pt = svgPointFromEvent(root, ev);
      if (pt.x < padL || pt.x > padL + plotW
          || pt.y < padT || pt.y > padT + plotH) {
        hoverGroup.setAttribute('class', 's3-hover hidden');
        return;
      }
      const a = xToA(pt.x);
      const b = yToB(pt.y);
      const L = sampleGrid(grid, a, b);
      hoverL1.textContent = 'α = ' + a.toFixed(3);
      hoverL2.textContent = 'β = ' + b.toFixed(3);
      hoverL3.textContent = 'L = ' + L.toFixed(4);
      let bx = pt.x + 12, by = pt.y + 12;
      if (bx + 130 > VW - 4) bx = pt.x - 130 - 12;
      if (by + 46 > VH - 4) by = pt.y - 46 - 12;
      hoverBg.setAttribute('x', bx.toFixed(1));
      hoverBg.setAttribute('y', by.toFixed(1));
      hoverL1.setAttribute('x', (bx + 8).toFixed(1));
      hoverL1.setAttribute('y', (by + 14).toFixed(1));
      hoverL2.setAttribute('x', (bx + 8).toFixed(1));
      hoverL2.setAttribute('y', (by + 28).toFixed(1));
      hoverL3.setAttribute('x', (bx + 8).toFixed(1));
      hoverL3.setAttribute('y', (by + 42).toFixed(1));
      hoverGroup.setAttribute('class', 's3-hover');
    });
    root.addEventListener('mouseleave', () => {
      hoverGroup.setAttribute('class', 's3-hover hidden');
    });

    // Render handle
    function renderMarbleAndTraj(traj) {
      // Marble: position at the most recent (α, β); clamp visually to keep
      // the SVG well-formed even if the marble drifts off the panel.
      const last = traj[traj.length - 1];
      if (last) {
        const x = aToX(last[0]);
        const y = bToY(last[1]);
        const cx = isFinite(x) ? Math.max(-50, Math.min(VW + 50, x)) : padL;
        const cy = isFinite(y) ? Math.max(-50, Math.min(VH + 50, y)) : padT;
        marble.setAttribute('cx', cx.toFixed(1));
        marble.setAttribute('cy', cy.toFixed(1));
      }
      // Trajectory polyline (decimate when long).
      const start = Math.max(0, traj.length - MAX_TRAJ);
      const parts = [];
      for (let i = start; i < traj.length; i++) {
        const [a, b] = traj[i];
        if (!isFinite(a) || !isFinite(b)) continue;
        const x = aToX(a), y = bToY(b);
        const cx = Math.max(-200, Math.min(VW + 200, x));
        const cy = Math.max(-200, Math.min(VH + 200, y));
        parts.push(cx.toFixed(1) + ',' + cy.toFixed(1));
      }
      trajPath.setAttribute('points', parts.join(' '));
    }

    return { renderMarbleAndTraj };
  }

  // ---- Fit pane (right) ---------------------------------------------------

  function buildFitPane(card, ctx) {
    const D = window.GDV_DATA;
    const VW = 480, VH = 420;
    const padL = 56, padR = 16, padT = 18, padB = 46;
    const plotW = VW - padL - padR;
    const plotH = VH - padT - padB;

    // Y range: pad around scatter + truth.
    let yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < D.points.length; i++) {
      const y = D.points[i][1];
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    for (let i = 0; i < D.truthCurve.length; i++) {
      const y = D.truthCurve[i][1];
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    const yPad = 0.20 * (yMax - yMin || 1);
    const yLo = yMin - yPad, yHi = yMax + yPad;
    const xLo = D.xMin, xHi = D.xMax;

    function xToPx(x) { return padL + (x - xLo) / (xHi - xLo) * plotW; }
    function yToPx(y) { return padT + (1 - (y - yLo) / (yHi - yLo)) * plotH; }

    const root = svg('svg', {
      class: 's3-svg s3-fit',
      viewBox: `0 0 ${VW} ${VH}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': 'Network fit at current parameters',
    }, card);

    svg('rect', { class: 's3-pane-bg', x: 0, y: 0, width: VW, height: VH }, root);
    svg('rect', { class: 's3-plot-frame',
      x: padL, y: padT, width: plotW, height: plotH }, root);

    // Axes
    svg('line', { class: 's3-axis',
      x1: padL, y1: padT + plotH, x2: padL + plotW, y2: padT + plotH }, root);
    svg('line', { class: 's3-axis',
      x1: padL, y1: padT, x2: padL, y2: padT + plotH }, root);

    // Ticks: x at xLo, 0, xHi; y at yLo, 0, yHi (each filtered to range).
    [xLo, 0, xHi].forEach((v) => {
      if (v < xLo - 1e-9 || v > xHi + 1e-9) return;
      const x = xToPx(v);
      svg('line', { class: 's3-tick',
        x1: x, y1: padT + plotH, x2: x, y2: padT + plotH + 4 }, root);
      svg('text', { class: 's3-tick-label',
        x: x, y: padT + plotH + 16, 'text-anchor': 'middle',
        text: v.toFixed(1) }, root);
    });
    [yLo, 0, yHi].forEach((v) => {
      if (v < yLo - 1e-9 || v > yHi + 1e-9) return;
      const y = yToPx(v);
      svg('line', { class: 's3-tick',
        x1: padL - 4, y1: y, x2: padL, y2: y }, root);
      svg('text', { class: 's3-tick-label',
        x: padL - 6, y: y + 4, 'text-anchor': 'end',
        text: v.toFixed(1) }, root);
    });

    svg('text', { class: 's3-axis-label',
      x: padL + plotW / 2, y: VH - 12, 'text-anchor': 'middle',
      text: 'x' }, root);
    svg('text', { class: 's3-axis-label',
      x: 14, y: padT + plotH / 2, 'text-anchor': 'middle',
      transform: `rotate(-90 14 ${padT + plotH / 2})`,
      text: 'y' }, root);

    // Truth (dashed)
    let truthD = '';
    for (let i = 0; i < D.truthCurve.length; i++) {
      const [tx, ty] = D.truthCurve[i];
      truthD += (i === 0 ? 'M' : 'L')
              + xToPx(tx).toFixed(1) + ',' + yToPx(ty).toFixed(1);
    }
    svg('path', { class: 's3-truth', d: truthD }, root);

    // Scatter
    const scat = svg('g', { class: 's3-scatter' }, root);
    for (let i = 0; i < D.points.length; i++) {
      svg('circle', { class: 's3-data-pt',
        cx: xToPx(D.points[i][0]).toFixed(1),
        cy: yToPx(D.points[i][1]).toFixed(1),
        r: 3 }, scat);
    }

    // Bold fit polyline (dynamic).
    const N_GRID = 200;
    const xGrid = new Float64Array(N_GRID);
    for (let i = 0; i < N_GRID; i++) {
      xGrid[i] = xLo + (xHi - xLo) * (i / (N_GRID - 1));
    }
    const fitPath = svg('polyline', { class: 's3-fit-curve' }, root);

    function renderFit(engine) {
      const ys = engine.evalAt(xGrid);
      const parts = new Array(xGrid.length);
      for (let i = 0; i < xGrid.length; i++) {
        const py = yToPx(ys[i]);
        // Clamp visually so a wildly diverged net doesn't break the SVG.
        const cy = Math.max(-200, Math.min(VH + 200, py));
        parts[i] = xToPx(xGrid[i]).toFixed(1) + ',' + cy.toFixed(1);
      }
      fitPath.setAttribute('points', parts.join(' '));
    }

    return { renderFit };
  }

  // ---- Math caption (KaTeX) -----------------------------------------------

  function renderMathCaption(target) {
    target.innerHTML = '';
    target.classList.add('s3-math-block');

    // Display equation: L(α, β) = MSE(θ* + α·δ₁ + β·δ₂)
    const eq = el('div', { class: 's3-eq' }, target);
    const latex =
      'L(\\alpha,\\,\\beta) \\;=\\; \\mathrm{MSE}\\bigl( ' +
      '\\theta^{*} \\,+\\, \\alpha\\,\\delta_{1} \\,+\\, \\beta\\,\\delta_{2} ' +
      '\\bigr)';
    if (window.katex) {
      try {
        window.katex.render(latex, eq, { throwOnError: false, displayMode: true });
      } catch (e) { eq.textContent = latex; }
    } else {
      eq.textContent = latex;
    }

    // Where-list (mirroring optimizer-info idiom).
    const where = el('div', { class: 's3-where' }, target);
    const lab = el('span', { class: 's3-where-lab', text: 'where' }, where);
    void lab;
    const items = [
      { tex: '\\theta^{*}',
        defn: 'the Adam-trained reference point in 37-D parameter space' },
      { tex: '\\delta_{1} = \\theta_{0} - \\theta^{*}',
        defn: 'descent direction from initialisation to the Adam minimum' },
      { tex: '\\delta_{2} = (\\theta^{*}_{\\mathrm{SGD}} - \\theta^{*})_{\\perp\\delta_{1}}',
        defn: 'displacement from Adam-min toward SGD-min, made orthogonal to δ₁ '
            + '(after Hao&nbsp;Li&nbsp;et&nbsp;al.&nbsp;2018)' },
      { tex: '(\\alpha,\\beta)',
        defn: 'planar coordinates: (1, 0) is initialisation, (0, 0) is Adam-min, '
            + '(α<sub>SGD</sub>, 1) is SGD-min' },
    ];
    items.forEach((v, i) => {
      const span = el('span', { class: 's3-var' }, where);
      const tex = document.createElement('span');
      tex.className = 's3-tex';
      if (window.katex) {
        try {
          window.katex.render(v.tex, tex, { throwOnError: false, displayMode: false });
        } catch (e) { tex.textContent = v.tex; }
      } else {
        tex.textContent = v.tex;
      }
      span.appendChild(tex);
      const dl = document.createElement('span');
      dl.className = 's3-defn';
      dl.innerHTML = ' — ' + v.defn;
      span.appendChild(dl);
      if (i < items.length - 1) {
        const sep = el('span', { class: 's3-sep', text: ' · ' }, where);
        void sep;
      }
    });
  }

  // ---- Scene assembly -----------------------------------------------------

  function captionFor(state) {
    const p = state.engine.params();
    const a = state.lastAlpha, b = state.lastBeta;
    const aBetaStr = (Number.isFinite(a) && Number.isFinite(b))
      ? `α = ${a.toFixed(3)}, β = ${b.toFixed(3)}`
      : 'α, β = —';
    if (state.isTraining) {
      return `Training… iter ${p.iter} · loss ${p.loss.toFixed(4)} · ${aBetaStr}`;
    }
    if (p.iter === 0) {
      return 'Press Train. Watch the marble roll across the slice toward θ*.';
    }
    return `Paused at iter ${p.iter} · loss ${p.loss.toFixed(4)} · ${aBetaStr}.`;
  }

  function buildScene(root) {
    if (!window.GDV_DATA || !Array.isArray(window.GDV_DATA.points)) {
      root.innerHTML = '<p style="opacity:0.5">GDV_DATA missing.</p>';
      return {};
    }
    if (!window.TrainingEngine || !window.Projection || !window.LossUtils) {
      root.innerHTML = '<p style="opacity:0.5">Required modules not loaded.</p>';
      return {};
    }

    // Compute or read the projection slice. Single shared instance for the
    // lifetime of the page so re-entries skip the ~100ms stub build.
    const proj = ensureProjection();

    // ---- DOM scaffolding -------------------------------------------------
    root.innerHTML = '';
    root.classList.add('s3-root');

    const wrap = el('div', { class: 's3-wrap' }, root);

    const hero = el('header', { class: 'hero s3-hero' }, wrap);
    el('h1', { text: 'Inside the high-D landscape' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Demo 2’s network has 37 parameters; we cannot draw 37-D, '
          + 'so we slice through θ* along two basis vectors. The slice '
          + 'is bowl-shaped — that is the whole reason gradient descent '
          + 'works on neural nets, even though the full landscape is famously '
          + 'non-convex. Watch SGD or Adam roll down the bowl.',
    }, hero);

    const grid2 = el('div', { class: 's3-grid' }, wrap);

    const leftCard = el('div', { class: 'card s3-pane' }, grid2);
    const rightCard = el('div', { class: 'card s3-pane' }, grid2);

    // Captions inside each card so they read with the figure they describe.
    const ctx = { proj };
    const landscape = buildLandscape(leftCard, ctx);
    el('div', {
      class: 'caption s3-caption',
      text: 'Loss restricted to the (α, β) plane. The marble is '
          + 'the projection of the live θₜ; the dashed amber line '
          + 'is its descent path.',
    }, leftCard);

    const fitPane = buildFitPane(rightCard, ctx);
    el('div', {
      class: 'caption s3-caption',
      text: 'The same network’s prediction at θₜ, drawn on the '
          + 'noisy data. As the marble rolls toward θ* the curve learns '
          + 'the wave.',
    }, rightCard);

    // ---- Controls -------------------------------------------------------
    const controls = el('div', { class: 'controls s3-controls' }, wrap);

    const optGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'Optimizer', for: 's3-opt' }, optGroup);
    const optSelect = el('select', { id: 's3-opt' }, optGroup);
    [['adam', 'Adam'], ['momentum', 'SGD + Momentum'], ['sgd', 'SGD']]
      .forEach(([v, l]) => {
        const o = el('option', { value: v, text: l }, optSelect);
        if (v === DEFAULT_OPT) o.selected = true;
      });

    const lrGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'LR', for: 's3-lr' }, lrGroup);
    const lrInput = el('input', {
      id: 's3-lr', type: 'range',
      min: '0.005', max: '0.2', step: '0.005',
      value: String(DEFAULT_LR),
    }, lrGroup);
    const lrOut = el('output', { class: 'control-value', text: DEFAULT_LR.toFixed(3) }, lrGroup);

    const seedGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'Seed', for: 's3-seed' }, seedGroup);
    const initialSeed = readHashInt('seed', DEFAULT_SEED);
    const seedInput = el('input', {
      id: 's3-seed', type: 'number',
      min: '0', step: '1',
      value: String(initialSeed),
      class: 's3-seed-input',
    }, seedGroup);

    const btnGroup = el('div', { class: 'control-group' }, controls);
    const trainBtn = el('button', { type: 'button', class: 'primary', text: 'Train' }, btnGroup);
    const pauseBtn = el('button', { type: 'button', text: 'Pause' }, btnGroup);
    const resetBtn = el('button', { type: 'button', text: 'Reset' }, btnGroup);

    // Optimizer-info block.
    const optInfo = el('p', { class: 'optimizer-info s3-opt-info' }, wrap);
    if (window.OptimizerInfo) window.OptimizerInfo.render(optInfo, DEFAULT_OPT);

    // Math caption (KaTeX block).
    const mathBlock = el('div', { class: 's3-math' }, wrap);
    renderMathCaption(mathBlock);

    // Live status caption.
    const statusCaption = el('p', { class: 'caption s3-status' }, wrap);

    // ---- Engine + state -------------------------------------------------
    // Optional &opt=adam|sgd|momentum URL flag for headless verification.
    const hashedOpt = readHashFlag('opt');
    const initialOpt = (hashedOpt === 'sgd' || hashedOpt === 'momentum'
      || hashedOpt === 'adam') ? hashedOpt : DEFAULT_OPT;
    if (initialOpt !== DEFAULT_OPT) {
      optSelect.value = initialOpt;
      if (window.OptimizerInfo) window.OptimizerInfo.render(optInfo, initialOpt);
    }
    const engine = new window.TrainingEngine({
      width: DEFAULT_WIDTH,
      seed: initialSeed,
      lr: DEFAULT_LR,
      optimizer: initialOpt,
    });

    // The precomputed projection was built against a specific θ_init (the
    // Python initializer with seed=42). The JS TrainingEngine's
    // initializer uses a different recipe, so the live engine and the
    // precomputed slice would otherwise be misaligned — and the marble
    // would not land at (1, 0). For the "canonical" seed we inject the
    // precomputed θ_init into the engine so the slice and the live
    // trajectory share a frame of reference. For any other user-picked
    // seed we keep the engine's natural init (the marble lands wherever
    // it lands; the bowl shape stays the same since the slice is fixed).
    function injectInitFromProjection(eng) {
      if (!proj || !proj.thetaInit) return;
      const ti = proj.thetaInit;
      if (!ti.w1 || !ti.bias || !ti.v) return;
      if (ti.w1.length !== eng.getWidth()) return;
      // Internal arrays: TrainingEngine names them _w1, _bias, _v, _b2.
      // Mutating in place keeps the engine's optimizer state (already
      // freshly zeroed by reset()) intact.
      const W = eng.getWidth();
      for (let i = 0; i < W; i++) {
        eng._w1[i] = ti.w1[i];
        eng._bias[i] = ti.bias[i];
        eng._v[i] = ti.v[i];
      }
      eng._b2 = ti.b2;
      // Recompute and stash the loss so caption / first paint read true.
      eng._lastLoss = eng._computeFullLoss();
    }
    if (initialSeed === DEFAULT_SEED) injectInitFromProjection(engine);

    const state = {
      engine,
      isTraining: false,
      lastAlpha: NaN,
      lastBeta: NaN,
      traj: [],            // [[α, β], ...]
      lastRender: 0,
      intervalId: null,
    };

    function projectCurrent() {
      const flat = window.Projection.flatten(state.engine.params());
      return window.Projection.project(
        flat, proj.thetaStarFlat, proj.delta1Flat, proj.delta2Flat);
    }

    function pushProjection() {
      const [a, b] = projectCurrent();
      state.lastAlpha = a;
      state.lastBeta = b;
      state.traj.push([a, b]);
      if (state.traj.length > MAX_TRAJ * 2) {
        state.traj.splice(0, state.traj.length - MAX_TRAJ);
      }
    }

    function fullRender() {
      landscape.renderMarbleAndTraj(state.traj);
      fitPane.renderFit(state.engine);
      statusCaption.textContent = captionFor(state);
    }

    function throttledRender() {
      const now = Date.now();
      if (now - state.lastRender < RENDER_MIN_DT_MS) return;
      state.lastRender = now;
      fullRender();
    }

    // Initial state: project θ_init and seed the trajectory.
    pushProjection();

    // Headless &epoch=N pre-step before first paint.
    const epochN = readHashInt('epoch', 0);
    if (epochN > 0) {
      for (let i = 0; i < epochN; i++) {
        state.engine.step();
        pushProjection();
      }
    }

    // First paint.
    state.lastRender = Date.now();
    fullRender();

    // ---- Train / Pause / Reset -----------------------------------------
    function startTraining() {
      if (state.intervalId != null) return;
      state.isTraining = true;
      statusCaption.textContent = captionFor(state);
      state.intervalId = setInterval(() => {
        try {
          state.engine.step();
          pushProjection();
          throttledRender();
        } catch (e) {
          console.error('scene3: training step failed', e);
          stopTraining();
        }
      }, TICK_MS);
    }

    function stopTraining() {
      if (state.intervalId != null) {
        clearInterval(state.intervalId);
        state.intervalId = null;
      }
      state.isTraining = false;
      // Final paint at pause to flush any throttled frame.
      fullRender();
    }

    function fullReset() {
      stopTraining();
      const seed = parseInt(seedInput.value, 10);
      const useSeed = Number.isFinite(seed) ? seed : DEFAULT_SEED;
      state.engine.reset(useSeed);
      // Same rule as on first construction: align to the precomputed
      // θ_init only for the canonical seed.
      if (useSeed === DEFAULT_SEED) injectInitFromProjection(state.engine);
      state.traj = [];
      pushProjection();
      fullRender();
    }

    trainBtn.addEventListener('click', startTraining);
    pauseBtn.addEventListener('click', stopTraining);
    resetBtn.addEventListener('click', fullReset);

    optSelect.addEventListener('change', () => {
      state.engine.setOptimizer(optSelect.value);
      if (window.OptimizerInfo) window.OptimizerInfo.render(optInfo, optSelect.value);
    });

    lrInput.addEventListener('input', () => {
      const v = parseFloat(lrInput.value);
      lrOut.textContent = Number.isFinite(v) ? v.toFixed(3) : '—';
      if (Number.isFinite(v) && v > 0) state.engine.setLR(v);
    });

    seedInput.addEventListener('change', () => {
      const v = parseInt(seedInput.value, 10);
      if (!Number.isFinite(v)) return;
      fullReset();
    });

    return {
      onEnter() {
        // Re-paint, but don't auto-resume training.
        fullRender();
      },
      onLeave() {
        // Hard rule: stop the interval so the engine doesn't keep stepping
        // while the user is on scene 1 or 2.
        stopTraining();
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene3 = function (root) { return buildScene(root); };
})();
