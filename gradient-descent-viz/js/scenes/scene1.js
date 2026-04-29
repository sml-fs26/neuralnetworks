/* Scene 1 -- "The bowl"

   Linear regression as a convex bowl. The user clicks anywhere in the
   (a, b) parameter pane to drop a marble; the chosen optimizer rolls it
   downhill toward the analytic optimum. The right pane mirrors the
   marble's (a, b) as a fitted line over the noisy scatter.

   Pedagogy this scene supports:
     - lr too small  -> marble creeps
     - lr too big    -> marble oscillates / diverges
     - good lr+Adam  -> smooth descent

   Hard rules followed: no fetch, no inline SVG colors, no JS-injected
   <style>, no CSS transitions on color/opacity/position. All shared
   globals: GDV_DATA, LossUtils, Optimizers. */
(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  // ----- URL hash helpers (dev affordances) ---------------------------------
  function readHashFlag(name) {
    const re = new RegExp('[#&]' + name + '(?:=([^&]*))?');
    const m = (window.location.hash || '').match(re);
    return m ? (m[1] != null ? m[1] : true) : null;
  }

  function readStartFromHash() {
    const m = (window.location.hash || '').match(/[#&]start=(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)/);
    if (!m) return null;
    return [parseFloat(m[1]), parseFloat(m[2])];
  }

  // ----- Small DOM helpers --------------------------------------------------
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

  // ----- Build the scene ----------------------------------------------------
  function build(root) {
    const data = window.GDV_DATA;
    const grid = data.lossGrid;
    // grid.values may be a plain JS array; LossUtils.contourPaths is index-
    // based so either works. Use as-is.
    const aMin = grid.aMin, aMax = grid.aMax;
    const bMin = grid.bMin, bMax = grid.bMax;
    const nA = grid.nA, nB = grid.nB;
    const points = data.points;

    // ----- Pane geometry (shared) ------------------------------------------
    const VW = 480, VH = 420;
    const padL = 56, padR = 16, padT = 18, padB = 46;
    const plotW = VW - padL - padR;
    const plotH = VH - padT - padB;

    // ----- Loss-pane transforms --------------------------------------------
    // (a, b) -> svg(x, y). a is x; b is y (flipped: bigger b nearer top).
    function aToX(a) { return padL + (a - aMin) / (aMax - aMin) * plotW; }
    function bToY(b) { return padT + (1 - (b - bMin) / (bMax - bMin)) * plotH; }
    function xToA(x) { return aMin + (x - padL) / plotW * (aMax - aMin); }
    function yToB(y) { return bMin + (1 - (y - padT) / plotH) * (bMax - bMin); }
    // contour transform: grid (i, j) -> (a, b) -> svg
    function gridToSvg(i, j) {
      const a = aMin + (aMax - aMin) * i / (nA - 1);
      const b = bMin + (bMax - bMin) * j / (nB - 1);
      return [aToX(a), bToY(b)];
    }

    // ----- Fit-pane transforms ---------------------------------------------
    // x range: data.xMin..data.xMax (= -3..3)
    // y range: padded around scatter+truth.
    let yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < points.length; i++) {
      const y = points[i][1];
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    for (let i = 0; i < data.truthCurve.length; i++) {
      const y = data.truthCurve[i][1];
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }
    const yPad = 0.15 * (yMax - yMin || 1);
    const yLo = yMin - yPad, yHi = yMax + yPad;
    const xLo = data.xMin, xHi = data.xMax;
    function xToPx(x) { return padL + (x - xLo) / (xHi - xLo) * plotW; }
    function yToPx(y) { return padT + (1 - (y - yLo) / (yHi - yLo)) * plotH; }

    // ----- DOM scaffolding -------------------------------------------------
    root.innerHTML = '';
    const wrap = el('div', { class: 's1-wrap' }, root);

    const hero = el('div', { class: 'hero s1-hero' }, wrap);
    el('h1', { text: 'The bowl', class: 's1-h1' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Linear regression has a convex bowl as its loss surface. ' +
            'Gradient descent rolls down it. The learning rate decides ' +
            'whether you arrive cleanly, oscillate, or diverge.',
    }, hero);

    const grid2 = el('div', { class: 's1-grid' }, wrap);

    // ----- Left card: loss landscape ---------------------------------------
    const leftCard = el('div', { class: 'card s1-pane' }, grid2);
    const leftSvg = svg('svg', {
      class: 's1-svg',
      viewBox: `0 0 ${VW} ${VH}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': 'Loss landscape in (a, b) parameter space',
    }, leftCard);

    // Background rect for click capture and styling.
    svg('rect', {
      class: 's1-pane-bg',
      x: 0, y: 0, width: VW, height: VH,
    }, leftSvg);

    // Plot frame (light box).
    svg('rect', {
      class: 's1-plot-frame',
      x: padL, y: padT, width: plotW, height: plotH,
    }, leftSvg);

    // Contour group (rendered once on build).
    const contourGroup = svg('g', { class: 's1-contours' }, leftSvg);
    const levels = window.LossUtils.chooseLevels(grid.values, 9);
    const paths = window.LossUtils.contourPaths(grid.values, nA, nB, levels, gridToSvg);
    for (let k = 0; k < paths.length; k++) {
      svg('path', {
        class: (k % 3 === 2) ? 's1-contour s1-contour-hi' : 's1-contour',
        d: paths[k].d,
      }, contourGroup);
    }

    // Axes (loss pane).
    // x-axis at bottom
    svg('line', {
      class: 's1-axis',
      x1: padL, y1: padT + plotH, x2: padL + plotW, y2: padT + plotH,
    }, leftSvg);
    // y-axis at left
    svg('line', {
      class: 's1-axis',
      x1: padL, y1: padT, x2: padL, y2: padT + plotH,
    }, leftSvg);

    // Tick marks + labels for a (x) at aMin, 0, aMax (if 0 is in range).
    const aTicks = [aMin, 0, aMax].filter(v => v >= aMin && v <= aMax);
    aTicks.forEach((v) => {
      const x = aToX(v);
      svg('line', {
        class: 's1-tick',
        x1: x, y1: padT + plotH, x2: x, y2: padT + plotH + 4,
      }, leftSvg);
      svg('text', {
        class: 's1-tick-label',
        x: x, y: padT + plotH + 16,
        'text-anchor': 'middle',
        text: v.toFixed(1),
      }, leftSvg);
    });
    const bTicks = [bMin, 0, bMax].filter(v => v >= bMin && v <= bMax);
    bTicks.forEach((v) => {
      const y = bToY(v);
      svg('line', {
        class: 's1-tick',
        x1: padL - 4, y1: y, x2: padL, y2: y,
      }, leftSvg);
      svg('text', {
        class: 's1-tick-label',
        x: padL - 6, y: y + 4,
        'text-anchor': 'end',
        text: v.toFixed(1),
      }, leftSvg);
    });

    // Axis labels.
    svg('text', {
      class: 's1-axis-label',
      x: padL + plotW / 2, y: VH - 12,
      'text-anchor': 'middle',
      text: 'a (slope)',
    }, leftSvg);
    svg('text', {
      class: 's1-axis-label',
      x: 14, y: padT + plotH / 2,
      'text-anchor': 'middle',
      transform: `rotate(-90 14 ${padT + plotH / 2})`,
      text: 'b (intercept)',
    }, leftSvg);

    // Optimum (a*, b*) marker -- a small + cross.
    const optA = data.optimum.a, optB = data.optimum.b;
    const optX = aToX(optA), optY = bToY(optB);
    const optGroup = svg('g', { class: 's1-optimum' }, leftSvg);
    svg('line', {
      class: 's1-optimum-mark',
      x1: optX - 6, y1: optY, x2: optX + 6, y2: optY,
    }, optGroup);
    svg('line', {
      class: 's1-optimum-mark',
      x1: optX, y1: optY - 6, x2: optX, y2: optY + 6,
    }, optGroup);

    // Trajectory polyline.
    const trajPath = svg('polyline', {
      class: 's1-trajectory',
      points: '',
    }, leftSvg);

    // Marble.
    const marble = svg('circle', {
      class: 's1-marble',
      cx: 0, cy: 0, r: 5,
    }, leftSvg);

    // Hover readout (text). Hidden by default via CSS class toggle.
    const hoverGroup = svg('g', { class: 's1-hover hidden' }, leftSvg);
    const hoverBg = svg('rect', {
      class: 's1-hover-bg',
      x: 0, y: 0, width: 130, height: 46, rx: 3,
    }, hoverGroup);
    const hoverLine1 = svg('text', {
      class: 's1-hover-text',
      x: 6, y: 14,
      text: '',
    }, hoverGroup);
    const hoverLine2 = svg('text', {
      class: 's1-hover-text',
      x: 6, y: 28,
      text: '',
    }, hoverGroup);
    const hoverLine3 = svg('text', {
      class: 's1-hover-text',
      x: 6, y: 42,
      text: '',
    }, hoverGroup);

    // Caption below the loss pane.
    el('div', {
      class: 'caption s1-caption',
      text: 'Drag a marble onto the bowl. Watch it roll down.',
    }, leftCard);

    // ----- Right card: fit on data ------------------------------------------
    const rightCard = el('div', { class: 'card s1-pane' }, grid2);
    const rightSvg = svg('svg', {
      class: 's1-svg',
      viewBox: `0 0 ${VW} ${VH}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': 'Linear fit y = a*x + b on noisy data',
    }, rightCard);

    svg('rect', {
      class: 's1-pane-bg',
      x: 0, y: 0, width: VW, height: VH,
    }, rightSvg);
    svg('rect', {
      class: 's1-plot-frame',
      x: padL, y: padT, width: plotW, height: plotH,
    }, rightSvg);

    // Axes for fit pane.
    svg('line', {
      class: 's1-axis',
      x1: padL, y1: padT + plotH, x2: padL + plotW, y2: padT + plotH,
    }, rightSvg);
    svg('line', {
      class: 's1-axis',
      x1: padL, y1: padT, x2: padL, y2: padT + plotH,
    }, rightSvg);

    // Ticks: x at xLo, 0, xHi; y at yLo (rounded), 0, yHi (rounded).
    const xTicks = [xLo, 0, xHi].filter(v => v >= xLo && v <= xHi);
    xTicks.forEach((v) => {
      const x = xToPx(v);
      svg('line', {
        class: 's1-tick',
        x1: x, y1: padT + plotH, x2: x, y2: padT + plotH + 4,
      }, rightSvg);
      svg('text', {
        class: 's1-tick-label',
        x: x, y: padT + plotH + 16,
        'text-anchor': 'middle',
        text: v.toFixed(1),
      }, rightSvg);
    });
    const yTickVals = [yLo, 0, yHi];
    yTickVals.forEach((v) => {
      if (v < yLo - 1e-9 || v > yHi + 1e-9) return;
      const y = yToPx(v);
      svg('line', {
        class: 's1-tick',
        x1: padL - 4, y1: y, x2: padL, y2: y,
      }, rightSvg);
      svg('text', {
        class: 's1-tick-label',
        x: padL - 6, y: y + 4,
        'text-anchor': 'end',
        text: v.toFixed(1),
      }, rightSvg);
    });

    svg('text', {
      class: 's1-axis-label',
      x: padL + plotW / 2, y: VH - 12,
      'text-anchor': 'middle',
      text: 'x',
    }, rightSvg);
    svg('text', {
      class: 's1-axis-label',
      x: 14, y: padT + plotH / 2,
      'text-anchor': 'middle',
      transform: `rotate(-90 14 ${padT + plotH / 2})`,
      text: 'y',
    }, rightSvg);

    // Truth curve (dashed).
    let truthD = '';
    for (let i = 0; i < data.truthCurve.length; i++) {
      const [tx, ty] = data.truthCurve[i];
      truthD += (i === 0 ? 'M' : 'L') + xToPx(tx).toFixed(1) + ',' + yToPx(ty).toFixed(1);
    }
    svg('path', {
      class: 's1-truth',
      d: truthD,
    }, rightSvg);

    // Scatter points.
    const scatterGroup = svg('g', { class: 's1-scatter' }, rightSvg);
    for (let i = 0; i < points.length; i++) {
      svg('circle', {
        class: 's1-data-pt',
        cx: xToPx(points[i][0]).toFixed(1),
        cy: yToPx(points[i][1]).toFixed(1),
        r: 3,
      }, scatterGroup);
    }

    // Fit line (dynamic).
    const fitLine = svg('line', {
      class: 's1-fit',
      x1: 0, y1: 0, x2: 0, y2: 0,
    }, rightSvg);

    el('div', {
      class: 'caption s1-caption',
      text: 'The line that the marble represents — same (a, b).',
    }, rightCard);

    // ----- Controls panel ---------------------------------------------------
    const controls = el('div', { class: 'controls s1-controls' }, wrap);

    const optGroupCtl = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'Optimizer', for: 's1-opt' }, optGroupCtl);
    const optSelect = el('select', { id: 's1-opt' }, optGroupCtl);
    el('option', { value: 'sgd', text: 'SGD' }, optSelect);
    el('option', { value: 'momentum', text: 'Momentum' }, optSelect);
    el('option', { value: 'adam', text: 'Adam' }, optSelect);

    const lrGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'Learning rate', for: 's1-lr' }, lrGroup);
    // Log slider 0.001..1.0; map slider 0..1000 -> 10^(log10(0.001) + t * (log10(1) - log10(0.001)))
    const lrSlider = el('input', {
      id: 's1-lr', type: 'range', min: '0', max: '1000', step: '1', value: '0',
    }, lrGroup);
    const lrOut = el('span', { class: 'control-value', text: 'lr = 0.050' }, lrGroup);

    function lrFromSlider(v) {
      const t = (+v) / 1000;
      const lo = Math.log10(0.001), hi = Math.log10(1.0);
      return Math.pow(10, lo + t * (hi - lo));
    }
    function sliderFromLr(lr) {
      const lo = Math.log10(0.001), hi = Math.log10(1.0);
      const t = (Math.log10(lr) - lo) / (hi - lo);
      return Math.round(t * 1000);
    }
    // Default lr = 0.05.
    lrSlider.value = String(sliderFromLr(0.05));

    const momGroup = el('div', { class: 'control-group s1-mom' }, controls);
    el('label', { text: 'Momentum', for: 's1-mom' }, momGroup);
    const momSlider = el('input', {
      id: 's1-mom', type: 'range', min: '0', max: '99', step: '1', value: '90',
    }, momGroup);
    const momOut = el('span', { class: 'control-value', text: 'β = 0.90' }, momGroup);

    const stepGroup = el('div', { class: 'control-group' }, controls);
    const stepBtn = el('button', { type: 'button', text: 'Step' }, stepGroup);

    const playGroup = el('div', { class: 'control-group' }, controls);
    const playBtn = el('button', { type: 'button', class: 'primary', text: 'Play' }, playGroup);

    const resetGroup = el('div', { class: 'control-group' }, controls);
    const resetBtn = el('button', { type: 'button', text: 'Reset' }, resetGroup);

    // Optimizer explanation -- updates when the dropdown changes.
    const optInfo = el('p', { class: 'optimizer-info s1-opt-info' }, wrap);
    if (window.OptimizerInfo) window.OptimizerInfo.render(optInfo, 'sgd');

    // Footnote with quick-reference for the dev hash flags.
    const foot = el('div', { class: 'footnote s1-foot' }, wrap);
    foot.innerHTML =
      'Click the bowl to drop a marble. Press <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> ' +
      'to change scenes. <kbd>t</kbd> toggles theme.';

    // ----- State ------------------------------------------------------------
    const ctx = {
      // Parameters & init.
      a: -0.4, b: 0.6,
      initA: -0.4, initB: 0.6,
      // Optimizer.
      optName: 'sgd',
      optState: window.Optimizers.OPTIMIZERS.sgd.init(2),
      // Trajectory.
      traj: [],         // array of [a, b]
      maxTraj: 150,
      // Play loop.
      playTimer: null,
      playSteps: 0,
      maxPlaySteps: 200,
      convergeEps: 1e-6,
      // DOM refs.
      marble, fitLine, trajPath,
      hoverGroup, hoverBg, hoverLine1, hoverLine2, hoverLine3,
      lrSlider, lrOut, momSlider, momOut, optSelect, momGroup, playBtn,
      // Geometry / data.
      aToX, bToY, xToA, yToB, aMin, aMax, bMin, bMax,
      xToPx, yToPx, xLo, xHi,
      points,
      stop: null, // assigned below
      render: null,
      reset: null,
    };

    // Apply optional &start=A,B hash override before first render.
    const startOverride = readStartFromHash();
    if (startOverride && Number.isFinite(startOverride[0]) && Number.isFinite(startOverride[1])) {
      ctx.initA = startOverride[0];
      ctx.initB = startOverride[1];
      ctx.a = ctx.initA;
      ctx.b = ctx.initB;
    }

    // ----- Render -----------------------------------------------------------
    function renderMarble() {
      // If marble strays outside the panel, clamp visually so the SVG
      // doesn't break; trajectory still records true coords.
      const x = ctx.aToX(ctx.a);
      const y = ctx.bToY(ctx.b);
      const cx = isFinite(x) ? Math.max(-50, Math.min(VW + 50, x)) : padL;
      const cy = isFinite(y) ? Math.max(-50, Math.min(VH + 50, y)) : padT;
      ctx.marble.setAttribute('cx', cx.toFixed(1));
      ctx.marble.setAttribute('cy', cy.toFixed(1));
    }

    function renderTrajectory() {
      const start = Math.max(0, ctx.traj.length - ctx.maxTraj);
      const parts = [];
      for (let i = start; i < ctx.traj.length; i++) {
        const [a, b] = ctx.traj[i];
        // Skip non-finite points (divergence).
        if (!isFinite(a) || !isFinite(b)) continue;
        const x = ctx.aToX(a);
        const y = ctx.bToY(b);
        // Clamp to a generous bbox to keep the polyline well-formed.
        const cx = Math.max(-200, Math.min(VW + 200, x));
        const cy = Math.max(-200, Math.min(VH + 200, y));
        parts.push(cx.toFixed(1) + ',' + cy.toFixed(1));
      }
      ctx.trajPath.setAttribute('points', parts.join(' '));
    }

    function renderFit() {
      // y = a*x + b at xLo and xHi, clamped to plot box (don't bother clamping
      // -- the SVG line is fine going off-screen a bit; the plot frame keeps
      // it visually anchored).
      const a = isFinite(ctx.a) ? ctx.a : 0;
      const b = isFinite(ctx.b) ? ctx.b : 0;
      ctx.fitLine.setAttribute('x1', ctx.xToPx(ctx.xLo).toFixed(1));
      ctx.fitLine.setAttribute('y1', ctx.yToPx(a * ctx.xLo + b).toFixed(1));
      ctx.fitLine.setAttribute('x2', ctx.xToPx(ctx.xHi).toFixed(1));
      ctx.fitLine.setAttribute('y2', ctx.yToPx(a * ctx.xHi + b).toFixed(1));
    }

    function render() {
      renderMarble();
      renderTrajectory();
      renderFit();
    }
    ctx.render = render;

    // ----- Optimizer step ---------------------------------------------------
    function currentHp() {
      return {
        lr: lrFromSlider(ctx.lrSlider.value),
        momentum: (+ctx.momSlider.value) / 100,
      };
    }

    function takeStep() {
      // If parameters have already diverged to non-finite, freeze.
      if (!isFinite(ctx.a) || !isFinite(ctx.b)) return 0;

      const grad = window.LossUtils.gradLinear(ctx.a, ctx.b, ctx.points);
      const opt = window.Optimizers.OPTIMIZERS[ctx.optName];
      const hp = currentHp();
      const out = opt.step(ctx.optState, grad, hp);
      ctx.optState = out.state;
      const newA = ctx.a + out.update[0];
      const newB = ctx.b + out.update[1];
      const delta = Math.hypot(out.update[0], out.update[1]);
      ctx.a = newA;
      ctx.b = newB;
      ctx.traj.push([ctx.a, ctx.b]);
      if (ctx.traj.length > ctx.maxTraj * 2) {
        // Trim occasionally to bound memory.
        ctx.traj.splice(0, ctx.traj.length - ctx.maxTraj);
      }
      render();
      return delta;
    }

    // ----- Play / pause -----------------------------------------------------
    function setPlayLabel(playing) {
      ctx.playBtn.textContent = playing ? 'Pause' : 'Play';
    }

    function startPlay(maxStepsOverride) {
      if (ctx.playTimer) return;
      ctx.playSteps = 0;
      const cap = maxStepsOverride || ctx.maxPlaySteps;
      setPlayLabel(true);
      ctx.playTimer = setInterval(() => {
        const delta = takeStep();
        ctx.playSteps += 1;
        if (
          ctx.playSteps >= cap ||
          (delta < ctx.convergeEps && ctx.playSteps > 5) ||
          !isFinite(ctx.a) || !isFinite(ctx.b)
        ) {
          stopPlay();
        }
      }, 33); // ~30 steps/sec
    }

    function stopPlay() {
      if (ctx.playTimer) {
        clearInterval(ctx.playTimer);
        ctx.playTimer = null;
      }
      setPlayLabel(false);
    }
    ctx.stop = stopPlay;

    // ----- Reset / re-init at click ----------------------------------------
    function resetOptimizerState() {
      const opt = window.Optimizers.OPTIMIZERS[ctx.optName];
      ctx.optState = opt.init(2);
    }

    function resetTo(a, b) {
      stopPlay();
      ctx.a = a;
      ctx.b = b;
      ctx.initA = a;
      ctx.initB = b;
      ctx.traj = [[a, b]];
      resetOptimizerState();
      render();
    }
    ctx.reset = function () {
      stopPlay();
      ctx.a = ctx.initA;
      ctx.b = ctx.initB;
      ctx.traj = [[ctx.a, ctx.b]];
      resetOptimizerState();
      render();
    };

    // ----- Wire interactions ------------------------------------------------
    // Click on left SVG -> drop marble.
    leftSvg.addEventListener('click', (ev) => {
      const pt = svgPointFromEvent(leftSvg, ev);
      // Only accept clicks inside the plot frame.
      if (pt.x < padL - 10 || pt.x > padL + plotW + 10) return;
      if (pt.y < padT - 10 || pt.y > padT + plotH + 10) return;
      const a = clamp(xToA(pt.x), aMin, aMax);
      const b = clamp(yToB(pt.y), bMin, bMax);
      resetTo(a, b);
    });

    // Hover readout.
    leftSvg.addEventListener('mousemove', (ev) => {
      const pt = svgPointFromEvent(leftSvg, ev);
      if (pt.x < padL || pt.x > padL + plotW || pt.y < padT || pt.y > padT + plotH) {
        hoverGroup.setAttribute('class', 's1-hover hidden');
        return;
      }
      const a = xToA(pt.x);
      const b = yToB(pt.y);
      const loss = window.LossUtils.mseLinear(a, b, points);
      hoverLine1.textContent = 'a = ' + a.toFixed(3);
      hoverLine2.textContent = 'b = ' + b.toFixed(3);
      hoverLine3.textContent = 'loss = ' + loss.toFixed(3);
      // Position the hover box near the cursor, but keep it inside the panel.
      let bx = pt.x + 12;
      let by = pt.y + 12;
      if (bx + 130 > VW - 4) bx = pt.x - 130 - 12;
      if (by + 46 > VH - 4) by = pt.y - 46 - 12;
      hoverBg.setAttribute('x', bx.toFixed(1));
      hoverBg.setAttribute('y', by.toFixed(1));
      hoverLine1.setAttribute('x', (bx + 8).toFixed(1));
      hoverLine1.setAttribute('y', (by + 14).toFixed(1));
      hoverLine2.setAttribute('x', (bx + 8).toFixed(1));
      hoverLine2.setAttribute('y', (by + 28).toFixed(1));
      hoverLine3.setAttribute('x', (bx + 8).toFixed(1));
      hoverLine3.setAttribute('y', (by + 42).toFixed(1));
      hoverGroup.setAttribute('class', 's1-hover');
    });
    leftSvg.addEventListener('mouseleave', () => {
      hoverGroup.setAttribute('class', 's1-hover hidden');
    });

    // Optimizer dropdown.
    optSelect.addEventListener('change', () => {
      ctx.optName = optSelect.value;
      // Show/hide momentum slider based on optimizer.
      momGroup.style.display = (ctx.optName === 'sgd') ? 'none' : '';
      if (window.OptimizerInfo) window.OptimizerInfo.render(optInfo, ctx.optName);
      resetOptimizerState();
      // Don't clear trajectory -- a curious user might want to compare.
      // But do reset the optimizer's internal state so velocity doesn't leak.
    });
    // Initial visibility (default optimizer = sgd -> hide momentum).
    momGroup.style.display = 'none';

    // Sliders -- live readouts.
    lrSlider.addEventListener('input', () => {
      const lr = lrFromSlider(lrSlider.value);
      lrOut.textContent = 'lr = ' + lr.toFixed(3);
    });
    momSlider.addEventListener('input', () => {
      const m = (+momSlider.value) / 100;
      momOut.textContent = 'β = ' + m.toFixed(2);
    });
    // Initialize lr readout for the default value.
    lrOut.textContent = 'lr = ' + lrFromSlider(lrSlider.value).toFixed(3);

    // Step button.
    stepBtn.addEventListener('click', () => {
      stopPlay();
      takeStep();
    });

    // Play / pause toggle.
    playBtn.addEventListener('click', () => {
      if (ctx.playTimer) stopPlay();
      else startPlay();
    });

    // Reset.
    resetBtn.addEventListener('click', () => ctx.reset());

    // ----- Initial render ---------------------------------------------------
    ctx.traj = [[ctx.a, ctx.b]];
    render();

    // ----- Hash dev-affordance: &play --------------------------------------
    if (readHashFlag('play')) {
      // Defer slightly so the scene is mounted/active by the time we run.
      setTimeout(() => startPlay(80), 60);
    }

    return ctx;
  }

  // ----- Helpers -----------------------------------------------------------
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

  function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
  }

  // ----- Scene registration ------------------------------------------------
  window.scenes = window.scenes || {};
  window.scenes.scene1 = function (root) {
    const ctx = build(root);
    return {
      onEnter() {
        // Re-render from current state. (No-op if scene was just built;
        // useful when re-entering after navigating away.)
        if (ctx && typeof ctx.render === 'function') ctx.render();
      },
      onLeave() {
        // Critical: stop the Play interval so the optimizer doesn't keep
        // stepping in the background.
        if (ctx && typeof ctx.stop === 'function') ctx.stop();
      },
    };
  };
})();
