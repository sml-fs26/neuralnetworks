/* Scene 4 -- "Local minima & the optimizer zoo"

   Four optimizers (GD, SGD, Momentum, Adam) run in parallel on a 1-D loss
   landscape with multiple local minima. The user picks a scenario, clicks
   theta_0 on the curve, and watches each optimizer's marble travel its own
   trajectory. The right panel plots loss vs step for all four runs.

   Pedagogical points:
     - GD is deterministic; it slides into the nearest local minimum and
       stops, regardless of where the global minimum lies.
     - SGD's noise can hop over small barriers (wiggly scenario).
     - Momentum builds velocity on long flat slopes (long_valley).
     - Adam's per-coordinate scaling tames sharp curvature jumps (asym_well).

   Hard rules followed: no fetch, no inline SVG colors, no JS-injected
   <style>, no CSS transitions on color/opacity/position. All shared
   globals: GDV_DATA, LossUtils, Optimizers, Loss1D, OptimizerInfo, katex. */
(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';

  const OPT_KEYS = ['gd', 'sgd', 'momentum', 'adam'];
  const OPT_LABELS = { gd: 'GD', sgd: 'SGD', momentum: 'Momentum', adam: 'Adam' };

  // ----- URL hash helpers (dev affordances) ---------------------------------
  function readHashFlag(name) {
    const re = new RegExp('[#&]' + name + '(?:=([^&]*))?');
    const m = (window.location.hash || '').match(re);
    return m ? (m[1] != null ? m[1] : true) : null;
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

  function niceCeil(v) {
    if (!isFinite(v) || v <= 0) return 1;
    const exp = Math.floor(Math.log10(v));
    const base = Math.pow(10, exp);
    const m = v / base;
    let nice;
    if (m <= 1) nice = 1;
    else if (m <= 2) nice = 2;
    else if (m <= 5) nice = 5;
    else nice = 10;
    return nice * base;
  }

  // ----- Build the scene ----------------------------------------------------
  function build(root) {
    const Loss1D = window.Loss1D;
    const Optimizers = window.Optimizers;

    // ----- Pane geometry (shared, both panes) -----------------------------
    const VW = 480, VH = 360;
    const padL = 56, padR = 16, padT = 18, padB = 46;
    const plotW = VW - padL - padR;
    const plotH = VH - padT - padB;

    // ----- DOM scaffolding -------------------------------------------------
    root.innerHTML = '';
    const wrap = el('div', { class: 's4-wrap' }, root);

    const hero = el('div', { class: 'hero s4-hero' }, wrap);
    el('h1', { text: 'Local minima & the optimizer zoo', class: 's4-h1' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Four optimizers race down the same 1-D loss curve. Pick a ' +
            'scenario, click anywhere on the curve to set theta_0, and ' +
            'press Play. Different curves expose different optimizer ' +
            'failure modes.',
    }, hero);

    const grid2 = el('div', { class: 's4-grid' }, wrap);

    // ----- Left card: loss curve + marbles ---------------------------------
    const leftCard = el('div', { class: 'card s4-pane' }, grid2);
    const leftSvg = svg('svg', {
      class: 's4-svg s4-svg-curve',
      viewBox: `0 0 ${VW} ${VH}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': '1-D loss curve with optimizer marbles',
    }, leftCard);

    svg('rect', { class: 's4-pane-bg', x: 0, y: 0, width: VW, height: VH }, leftSvg);
    svg('rect', {
      class: 's4-plot-frame',
      x: padL, y: padT, width: plotW, height: plotH,
    }, leftSvg);

    // x-axis (theta), y-axis (L)
    svg('line', {
      class: 's4-axis',
      x1: padL, y1: padT + plotH, x2: padL + plotW, y2: padT + plotH,
    }, leftSvg);
    svg('line', {
      class: 's4-axis',
      x1: padL, y1: padT, x2: padL, y2: padT + plotH,
    }, leftSvg);

    const xTicksGrp = svg('g', { class: 's4-ticks-x' }, leftSvg);
    const yTicksGrp = svg('g', { class: 's4-ticks-y' }, leftSvg);

    svg('text', {
      class: 's4-axis-label',
      x: padL + plotW / 2, y: VH - 12,
      'text-anchor': 'middle',
      text: 'theta',
    }, leftSvg);
    svg('text', {
      class: 's4-axis-label',
      x: 14, y: padT + plotH / 2,
      'text-anchor': 'middle',
      transform: `rotate(-90 14 ${padT + plotH / 2})`,
      text: 'L(theta)',
    }, leftSvg);

    // The loss curve path
    const curvePath = svg('path', { class: 's4-curve', d: '' }, leftSvg);

    // theta_0 vertical marker
    const theta0Line = svg('line', {
      class: 's4-theta0',
      x1: 0, y1: padT, x2: 0, y2: padT + plotH,
    }, leftSvg);

    // Per-optimizer trajectory polylines and marbles
    const trajPaths = {};
    const marbles = {};
    const trajGrp = svg('g', { class: 's4-trajectories' }, leftSvg);
    const marbleGrp = svg('g', { class: 's4-marbles' }, leftSvg);
    OPT_KEYS.forEach((k) => {
      trajPaths[k] = svg('polyline', {
        class: 's4-traj s4-traj-' + k,
        points: '',
      }, trajGrp);
    });
    OPT_KEYS.forEach((k) => {
      marbles[k] = svg('circle', {
        class: 's4-marble s4-marble-' + k,
        cx: 0, cy: 0, r: 5,
      }, marbleGrp);
    });

    el('div', {
      class: 'caption s4-caption',
      text: 'Click the curve to set theta_0. All four optimizers start there.',
    }, leftCard);

    // ----- Right card: loss vs step ----------------------------------------
    const rightCard = el('div', { class: 'card s4-pane' }, grid2);
    const rightSvg = svg('svg', {
      class: 's4-svg s4-svg-loss',
      viewBox: `0 0 ${VW} ${VH}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': 'Loss vs step for all four optimizers',
    }, rightCard);

    svg('rect', { class: 's4-pane-bg', x: 0, y: 0, width: VW, height: VH }, rightSvg);
    svg('rect', {
      class: 's4-plot-frame',
      x: padL, y: padT, width: plotW, height: plotH,
    }, rightSvg);

    // axes
    svg('line', {
      class: 's4-axis',
      x1: padL, y1: padT + plotH, x2: padL + plotW, y2: padT + plotH,
    }, rightSvg);
    svg('line', {
      class: 's4-axis',
      x1: padL, y1: padT, x2: padL, y2: padT + plotH,
    }, rightSvg);

    const lossXTicksGrp = svg('g', { class: 's4-ticks-x' }, rightSvg);
    const lossYTicksGrp = svg('g', { class: 's4-ticks-y' }, rightSvg);

    svg('text', {
      class: 's4-axis-label',
      x: padL + plotW / 2, y: VH - 12,
      'text-anchor': 'middle',
      text: 'step',
    }, rightSvg);
    const lossYLabel = svg('text', {
      class: 's4-axis-label',
      x: 14, y: padT + plotH / 2,
      'text-anchor': 'middle',
      transform: `rotate(-90 14 ${padT + plotH / 2})`,
      text: 'L (log scale)',
    }, rightSvg);

    const lossNote = svg('text', {
      class: 's4-loss-note hidden',
      x: padL + plotW / 2, y: padT - 6,
      'text-anchor': 'middle',
      text: '(loss can be negative — linear scale)',
    }, rightSvg);

    const lossLines = {};
    const lossLinesGrp = svg('g', { class: 's4-loss-lines' }, rightSvg);
    OPT_KEYS.forEach((k) => {
      lossLines[k] = svg('polyline', {
        class: 's4-loss-line s4-loss-line-' + k,
        points: '',
      }, lossLinesGrp);
    });

    el('div', {
      class: 'caption s4-caption',
      text: 'One line per optimizer; color matches its marble.',
    }, rightCard);

    // ----- Controls strip --------------------------------------------------
    const controls = el('div', { class: 'controls s4-controls' }, wrap);

    // Scenario dropdown
    const scenarioGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'Scenario', for: 's4-scenario' }, scenarioGroup);
    const scenarioSelect = el('select', { id: 's4-scenario' }, scenarioGroup);
    Object.keys(Loss1D.SCENARIOS).forEach((key) => {
      el('option', { value: key, text: Loss1D.SCENARIOS[key].label }, scenarioSelect);
    });

    // Optimizer enable checkboxes
    const optGroup = el('div', { class: 'control-group s4-opt-checks' }, controls);
    el('label', { text: 'Optimizers' }, optGroup);
    const checkBoxes = {};
    OPT_KEYS.forEach((k) => {
      const cbWrap = el('label', { class: 's4-cbwrap s4-cbwrap-' + k }, optGroup);
      const cb = el('input', { type: 'checkbox', id: 's4-cb-' + k }, cbWrap);
      cb.checked = true;
      checkBoxes[k] = cb;
      el('span', { class: 's4-swatch s4-swatch-' + k }, cbWrap);
      el('span', { class: 's4-cblabel', text: OPT_LABELS[k] }, cbWrap);
    });

    // Global lr factor slider (log over [0.1, 10])
    const lrGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'lr ×', for: 's4-lr' }, lrGroup);
    const lrSlider = el('input', {
      id: 's4-lr', type: 'range', min: '0', max: '1000', step: '1', value: '500',
    }, lrGroup);
    const lrOut = el('span', { class: 'control-value', text: '× 1.00' }, lrGroup);

    function lrFactorFromSlider(v) {
      const t = (+v) / 1000;
      const lo = Math.log10(0.1), hi = Math.log10(10);
      return Math.pow(10, lo + t * (hi - lo));
    }
    function sliderFromLrFactor(f) {
      const lo = Math.log10(0.1), hi = Math.log10(10);
      const t = (Math.log10(f) - lo) / (hi - lo);
      return Math.round(t * 1000);
    }

    // Sigma slider (linear, range scenario-dependent)
    const sigmaGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'σ', for: 's4-sigma' }, sigmaGroup);
    const sigmaSlider = el('input', {
      id: 's4-sigma', type: 'range', min: '0', max: '1000', step: '1', value: '333',
    }, sigmaGroup);
    const sigmaOut = el('span', { class: 'control-value', text: 'σ = 0.00' }, sigmaGroup);

    // Step / Play / Reset
    const stepGroup = el('div', { class: 'control-group' }, controls);
    const stepBtn = el('button', { type: 'button', text: 'Step' }, stepGroup);
    const playGroup = el('div', { class: 'control-group' }, controls);
    const playBtn = el('button', { type: 'button', class: 'primary', text: 'Play' }, playGroup);
    const resetGroup = el('div', { class: 'control-group' }, controls);
    const resetBtn = el('button', { type: 'button', text: 'Reset' }, resetGroup);

    // Log/linear toggle for loss y-axis
    const scaleGroup = el('div', { class: 'control-group' }, controls);
    const scaleBtn = el('button', { type: 'button', text: 'Linear y' }, scaleGroup);

    // Compact effective-lr legend (right side of controls strip)
    const legend = el('div', { class: 's4-legend' }, controls);

    // ----- Optimizer-info panel (4 cards stacked) --------------------------
    const infoCol = el('div', { class: 's4-info-col' }, wrap);
    const infoDivs = {};
    OPT_KEYS.forEach((k) => {
      const card = el('div', { class: 's4-info-card s4-info-card-' + k }, infoCol);
      const headBar = el('div', { class: 's4-info-head' }, card);
      el('span', { class: 's4-swatch s4-swatch-' + k }, headBar);
      el('span', { class: 's4-info-headlabel', text: OPT_LABELS[k] }, headBar);
      const body = el('div', { class: 'optimizer-info s4-info-body' }, card);
      if (window.OptimizerInfo) window.OptimizerInfo.render(body, k);
      infoDivs[k] = body;
    });

    // Footnote
    const foot = el('div', { class: 'footnote s4-foot' }, wrap);
    foot.innerHTML =
      'Click the curve to set θ<sub>0</sub>. Press <kbd>&larr;</kbd> / <kbd>&rarr;</kbd> ' +
      'to change scenes.';

    // ----- State ------------------------------------------------------------
    const ctx = {
      scenarioKey: 'wiggly',
      scenario: Loss1D.SCENARIOS.wiggly,
      theta0: Loss1D.SCENARIOS.wiggly.defaultStart,
      runs: {},   // populated by initRuns
      step: 0,
      maxPlaySteps: 200,
      convergeEps: 1e-7,
      convergeWindow: 5,
      convergeStreak: 0,
      playTimer: null,
      // log/linear scale
      yScaleMode: 'log',
      seenAnyNegative: false,
      seenLossMin: Infinity,
      seenLossMax: -Infinity,
      // RNG
      rng: window.LossUtils.makeRng(1),
      // DOM/geom transforms (re-set on scenario change)
      thetaToX: null, lossToY: null, xToTheta: null,
      // loss-axis transforms
      stepToX: null, lossYToY: null,
    };

    // ----- Curve transforms / rendering ------------------------------------
    function rebuildCurveAxes() {
      const sc = ctx.scenario;
      const tMin = sc.thetaMin, tMax = sc.thetaMax;

      // Sample curve so we know L's range over the visible domain.
      const samples = Loss1D.sampleCurve(sc, 400);
      let lMin = Infinity, lMax = -Infinity;
      for (let i = 0; i < samples.length; i++) {
        const L = samples[i][1];
        if (L < lMin) lMin = L;
        if (L > lMax) lMax = L;
      }
      // Pad a bit so curve doesn't kiss the frame.
      const lPad = 0.1 * (lMax - lMin || 1);
      const lLo = lMin - lPad, lHi = lMax + lPad;

      ctx.thetaToX = function (t) {
        return padL + (t - tMin) / (tMax - tMin) * plotW;
      };
      ctx.xToTheta = function (x) {
        return tMin + (x - padL) / plotW * (tMax - tMin);
      };
      ctx.lossToY = function (L) {
        return padT + (1 - (L - lLo) / (lHi - lLo)) * plotH;
      };

      // Build curve path
      let d = '';
      for (let i = 0; i < samples.length; i++) {
        const x = ctx.thetaToX(samples[i][0]);
        const y = ctx.lossToY(samples[i][1]);
        d += (i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1);
      }
      curvePath.setAttribute('d', d);

      // Rebuild ticks
      while (xTicksGrp.firstChild) xTicksGrp.removeChild(xTicksGrp.firstChild);
      while (yTicksGrp.firstChild) yTicksGrp.removeChild(yTicksGrp.firstChild);

      const xTicks = [tMin, 0, tMax].filter(v => v >= tMin && v <= tMax);
      xTicks.forEach((v) => {
        const x = ctx.thetaToX(v);
        svg('line', {
          class: 's4-tick',
          x1: x, y1: padT + plotH, x2: x, y2: padT + plotH + 4,
        }, xTicksGrp);
        svg('text', {
          class: 's4-tick-label',
          x: x, y: padT + plotH + 16,
          'text-anchor': 'middle',
          text: v.toFixed(1),
        }, xTicksGrp);
      });
      const yTicks = [lLo, (lLo + lHi) / 2, lHi];
      yTicks.forEach((v) => {
        const y = ctx.lossToY(v);
        svg('line', {
          class: 's4-tick',
          x1: padL - 4, y1: y, x2: padL, y2: y,
        }, yTicksGrp);
        svg('text', {
          class: 's4-tick-label',
          x: padL - 6, y: y + 4,
          'text-anchor': 'end',
          text: v.toFixed(2),
        }, yTicksGrp);
      });
    }

    // ----- Loss-vs-step axes -----------------------------------------------
    function rebuildLossAxes() {
      const sMax = ctx.maxPlaySteps;
      ctx.stepToX = function (s) {
        return padL + (s / sMax) * plotW;
      };

      while (lossXTicksGrp.firstChild) lossXTicksGrp.removeChild(lossXTicksGrp.firstChild);
      while (lossYTicksGrp.firstChild) lossYTicksGrp.removeChild(lossYTicksGrp.firstChild);

      // x ticks every 50
      for (let s = 0; s <= sMax; s += 50) {
        const x = ctx.stepToX(s);
        svg('line', {
          class: 's4-tick',
          x1: x, y1: padT + plotH, x2: x, y2: padT + plotH + 4,
        }, lossXTicksGrp);
        svg('text', {
          class: 's4-tick-label',
          x: x, y: padT + plotH + 16,
          'text-anchor': 'middle',
          text: String(s),
        }, lossXTicksGrp);
      }

      // y ticks: depend on scale mode and seen losses
      let lo, hi;
      const linearMode = (ctx.yScaleMode === 'linear') || ctx.seenAnyNegative;
      if (linearMode) {
        lo = isFinite(ctx.seenLossMin) ? ctx.seenLossMin : 0;
        hi = isFinite(ctx.seenLossMax) ? ctx.seenLossMax : 1;
        if (hi <= lo) { hi = lo + 1; }
        const pad = 0.1 * (hi - lo);
        lo -= pad;
        hi += pad;
        ctx.lossYToY = function (L) {
          return padT + (1 - (L - lo) / (hi - lo)) * plotH;
        };
        const ticks = [lo, (lo + hi) / 2, hi];
        ticks.forEach((v) => {
          const y = ctx.lossYToY(v);
          svg('line', {
            class: 's4-tick',
            x1: padL - 4, y1: y, x2: padL, y2: y,
          }, lossYTicksGrp);
          svg('text', {
            class: 's4-tick-label',
            x: padL - 6, y: y + 4,
            'text-anchor': 'end',
            text: v.toFixed(3),
          }, lossYTicksGrp);
        });
      } else {
        // Log scale.
        const seenMin = isFinite(ctx.seenLossMin) ? ctx.seenLossMin : 1e-2;
        const seenMax = isFinite(ctx.seenLossMax) ? ctx.seenLossMax : 100;
        lo = Math.max(1e-12, Math.min(1e-4, 0.5 * seenMin));
        hi = niceCeil(Math.max(seenMax, 1));
        if (hi <= lo) hi = lo * 100;
        const logLo = Math.log10(lo), logHi = Math.log10(hi);
        ctx.lossYToY = function (L) {
          const Lc = Math.max(L, lo * 1e-6);
          return padT + (1 - (Math.log10(Lc) - logLo) / (logHi - logLo)) * plotH;
        };
        // Ticks at decades.
        const lo10 = Math.ceil(logLo);
        const hi10 = Math.floor(logHi);
        for (let p = lo10; p <= hi10; p++) {
          const v = Math.pow(10, p);
          const y = ctx.lossYToY(v);
          svg('line', {
            class: 's4-tick',
            x1: padL - 4, y1: y, x2: padL, y2: y,
          }, lossYTicksGrp);
          svg('text', {
            class: 's4-tick-label',
            x: padL - 6, y: y + 4,
            'text-anchor': 'end',
            text: '1e' + p,
          }, lossYTicksGrp);
        }
      }

      // Update y-axis label and note
      lossYLabel.textContent = linearMode ? 'L (linear)' : 'L (log scale)';
      if (ctx.seenAnyNegative) {
        lossNote.setAttribute('class', 's4-loss-note');
      } else {
        lossNote.setAttribute('class', 's4-loss-note hidden');
      }
      // Toggle button state
      if (ctx.seenAnyNegative) {
        scaleBtn.disabled = true;
        scaleBtn.textContent = 'Linear y (forced)';
      } else {
        scaleBtn.disabled = false;
        scaleBtn.textContent = (ctx.yScaleMode === 'log') ? 'Linear y' : 'Log y';
      }
    }

    // ----- Run management --------------------------------------------------
    function freshOptState(k) {
      return Optimizers.OPTIMIZERS[k].init(1);
    }
    function effectiveLr(k) {
      const base = ctx.scenario.defaultLr[k];
      const factor = lrFactorFromSlider(lrSlider.value);
      return base * factor;
    }
    function currentSigma() {
      return +sigmaSlider.value / 1000 * (3 * ctx.scenario.defaultSigma);
    }

    function initRuns() {
      ctx.step = 0;
      ctx.convergeStreak = 0;
      ctx.seenAnyNegative = false;
      ctx.seenLossMin = Infinity;
      ctx.seenLossMax = -Infinity;
      ctx.rng = window.LossUtils.makeRng(1);
      const L0 = ctx.scenario.evaluate(ctx.theta0);
      OPT_KEYS.forEach((k) => {
        ctx.runs[k] = {
          theta: ctx.theta0,
          optState: freshOptState(k),
          traj: [[ctx.theta0, L0]],
          lossHist: [L0],
          lastDelta: Infinity,
          done: false,
        };
        if (L0 < ctx.seenLossMin) ctx.seenLossMin = L0;
        if (L0 > ctx.seenLossMax) ctx.seenLossMax = L0;
        if (L0 <= 0) ctx.seenAnyNegative = true;
      });
    }

    // ----- Step ------------------------------------------------------------
    function takeStep() {
      const sc = ctx.scenario;
      const sigma = currentSigma();
      let activeCount = 0;
      let allConverged = true;
      OPT_KEYS.forEach((k) => {
        if (!checkBoxes[k].checked) return;
        const run = ctx.runs[k];
        if (!run || run.done) return;
        activeCount++;
        const useSigma = (k === 'gd') ? 0 : sigma;
        const g = Loss1D.stochasticGrad(run.theta, sc, useSigma, ctx.rng);
        const opt = Optimizers.OPTIMIZERS[k];
        const hp = { lr: effectiveLr(k) };
        const out = opt.step(run.optState, [g], hp);
        run.optState = out.state;
        const du = out.update[0];
        run.theta = run.theta + du;
        run.lastDelta = Math.abs(du);
        const L = isFinite(run.theta) ? sc.evaluate(run.theta) : NaN;
        run.traj.push([run.theta, L]);
        run.lossHist.push(L);
        if (isFinite(L)) {
          if (L < ctx.seenLossMin) ctx.seenLossMin = L;
          if (L > ctx.seenLossMax) ctx.seenLossMax = L;
          if (L <= 0) ctx.seenAnyNegative = true;
        }
        if (run.lastDelta >= ctx.convergeEps) allConverged = false;
      });
      ctx.step++;
      if (activeCount > 0 && allConverged) ctx.convergeStreak++;
      else ctx.convergeStreak = 0;
      render();
      return { activeCount, allConverged };
    }

    // ----- Render ----------------------------------------------------------
    function render() {
      // theta_0 marker
      const x0 = ctx.thetaToX(ctx.theta0);
      theta0Line.setAttribute('x1', x0.toFixed(1));
      theta0Line.setAttribute('x2', x0.toFixed(1));

      OPT_KEYS.forEach((k) => {
        const run = ctx.runs[k];
        const enabled = checkBoxes[k].checked;
        if (!run || !enabled) {
          marbles[k].setAttribute('class', 's4-marble s4-marble-' + k + ' hidden');
          trajPaths[k].setAttribute('points', '');
          lossLines[k].setAttribute('points', '');
          return;
        }
        marbles[k].setAttribute('class', 's4-marble s4-marble-' + k);

        // Marble at (theta_t, L(theta_t))
        const theta = run.theta;
        const L = ctx.scenario.evaluate(theta);
        const cx = isFinite(theta) ? clamp(ctx.thetaToX(theta), -50, VW + 50) : padL;
        const cy = isFinite(L) ? clamp(ctx.lossToY(L), -50, VH + 50) : padT;
        marbles[k].setAttribute('cx', cx.toFixed(1));
        marbles[k].setAttribute('cy', cy.toFixed(1));

        // Trajectory polyline: (theta, L) projected to svg.
        const parts = [];
        const tj = run.traj;
        const start = Math.max(0, tj.length - 200);
        for (let i = start; i < tj.length; i++) {
          const t = tj[i][0], v = tj[i][1];
          if (!isFinite(t) || !isFinite(v)) continue;
          const xx = clamp(ctx.thetaToX(t), -200, VW + 200);
          const yy = clamp(ctx.lossToY(v), -200, VH + 200);
          parts.push(xx.toFixed(1) + ',' + yy.toFixed(1));
        }
        trajPaths[k].setAttribute('points', parts.join(' '));
      });

      // Right pane: rebuild axes (cheap; recomputes scaling against seen
      // losses so the polyline stays visible) and re-render polylines.
      rebuildLossAxes();
      OPT_KEYS.forEach((k) => {
        const run = ctx.runs[k];
        const enabled = checkBoxes[k].checked;
        if (!run || !enabled) {
          lossLines[k].setAttribute('points', '');
          return;
        }
        const parts = [];
        for (let i = 0; i < run.lossHist.length; i++) {
          const L = run.lossHist[i];
          if (!isFinite(L)) continue;
          const xx = ctx.stepToX(i);
          const yy = clamp(ctx.lossYToY(L), -200, VH + 200);
          parts.push(xx.toFixed(1) + ',' + yy.toFixed(1));
        }
        lossLines[k].setAttribute('points', parts.join(' '));
      });

      // Refresh effective-lr legend.
      legend.innerHTML = '';
      OPT_KEYS.forEach((k) => {
        const lr = effectiveLr(k);
        const row = el('span', { class: 's4-legend-row s4-legend-row-' + k }, legend);
        el('span', { class: 's4-swatch s4-swatch-' + k }, row);
        el('span', {
          class: 's4-legend-text',
          text: OPT_LABELS[k] + ' lr=' + lr.toFixed(3),
        }, row);
      });
    }

    // ----- Play / Pause ----------------------------------------------------
    function setPlayLabel(playing) {
      playBtn.textContent = playing ? 'Pause' : 'Play';
    }
    function startPlay() {
      if (ctx.playTimer) return;
      setPlayLabel(true);
      ctx.playTimer = setInterval(() => {
        const out = takeStep();
        if (
          ctx.step >= ctx.maxPlaySteps ||
          out.activeCount === 0 ||
          ctx.convergeStreak >= ctx.convergeWindow
        ) {
          stopPlay();
        }
      }, 33);
    }
    function stopPlay() {
      if (ctx.playTimer) {
        clearInterval(ctx.playTimer);
        ctx.playTimer = null;
      }
      setPlayLabel(false);
    }
    ctx.stop = stopPlay;

    // ----- Scenario change -------------------------------------------------
    function applyScenario(key) {
      stopPlay();
      ctx.scenarioKey = key;
      ctx.scenario = Loss1D.SCENARIOS[key];
      ctx.theta0 = ctx.scenario.defaultStart;
      // Reset sigma slider to default sigma (which is 1/3 of the slider's max
      // because we want the slider's default to land at scenario.defaultSigma).
      sigmaSlider.value = '333';
      // Reset lr factor to 1.0
      lrSlider.value = String(sliderFromLrFactor(1.0));
      lrOut.textContent = '× 1.00';
      sigmaOut.textContent = 'σ = ' + currentSigma().toFixed(2);
      rebuildCurveAxes();
      initRuns();
      render();
    }

    // ----- Wire interactions -----------------------------------------------
    leftSvg.addEventListener('click', (ev) => {
      const pt = svgPointFromEvent(leftSvg, ev);
      if (pt.x < padL - 10 || pt.x > padL + plotW + 10) return;
      if (pt.y < padT - 10 || pt.y > padT + plotH + 10) return;
      stopPlay();
      ctx.theta0 = clamp(ctx.xToTheta(pt.x), ctx.scenario.thetaMin, ctx.scenario.thetaMax);
      initRuns();
      render();
    });

    scenarioSelect.addEventListener('change', () => {
      applyScenario(scenarioSelect.value);
    });

    OPT_KEYS.forEach((k) => {
      checkBoxes[k].addEventListener('change', () => {
        render();
      });
    });

    lrSlider.addEventListener('input', () => {
      const f = lrFactorFromSlider(lrSlider.value);
      lrOut.textContent = '× ' + f.toFixed(2);
      // Re-render legend (effective lr changes)
      render();
    });
    sigmaSlider.addEventListener('input', () => {
      sigmaOut.textContent = 'σ = ' + currentSigma().toFixed(2);
    });

    stepBtn.addEventListener('click', () => {
      stopPlay();
      takeStep();
    });
    playBtn.addEventListener('click', () => {
      if (ctx.playTimer) stopPlay();
      else startPlay();
    });
    resetBtn.addEventListener('click', () => {
      stopPlay();
      initRuns();
      render();
    });
    scaleBtn.addEventListener('click', () => {
      if (ctx.seenAnyNegative) return;
      ctx.yScaleMode = (ctx.yScaleMode === 'log') ? 'linear' : 'log';
      render();
    });

    // ----- Hash dev affordances --------------------------------------------
    const hashScenario = readHashFlag('scenario');
    if (typeof hashScenario === 'string' && Loss1D.SCENARIOS[hashScenario]) {
      scenarioSelect.value = hashScenario;
      applyScenario(hashScenario);
    } else {
      // Initial scenario setup (default 'wiggly').
      applyScenario(scenarioSelect.value);
    }
    const hashTheta = readHashFlag('theta');
    if (typeof hashTheta === 'string') {
      const t = parseFloat(hashTheta);
      if (Number.isFinite(t)) {
        ctx.theta0 = clamp(t, ctx.scenario.thetaMin, ctx.scenario.thetaMax);
        initRuns();
        render();
      }
    }
    if (readHashFlag('play')) {
      setTimeout(() => startPlay(), 60);
    }

    return ctx;
  }

  // ----- Scene registration ------------------------------------------------
  window.scenes = window.scenes || {};
  window.scenes.scene4 = function (root) {
    const ctx = build(root);
    return {
      onEnter() {
        // No-op; state persists across re-entry.
      },
      onLeave() {
        if (ctx && typeof ctx.stop === 'function') ctx.stop();
      },
    };
  };
})();
