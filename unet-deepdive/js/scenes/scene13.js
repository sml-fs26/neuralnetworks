/* Scene 13 — "Watching the segmentation come into focus."

   A timeline slider over the training trajectory. As you scrub, the
   prediction snapshot for one fixed test sample updates on the left,
   and three live charts on the right move their markers: a loss curve,
   a pixel-accuracy curve, and a "weight magnitude" sparkline showing
   the up1 / up2 transposed-conv parameter norms. The viewer feels the
   network learning.

   Step engine:
     0 = epoch-0 frame (the "everything is sky" baseline)
     1 = the slider becomes scrubbable
     2 = loss curve overlays
     3 = pixel-accuracy curve overlays
     4 = up1/up2 weight-magnitude sparkline overlays
*/
(function () {
  'use strict';

  const NUM_STEPS = 5;
  const RUN_INTERVAL_MS = 700;
  const PRED_PX = 256;
  const INPUT_PX = 256;
  const GT_PX = 256;
  const CHART_W = 520;
  const CHART_H = 130;
  const SPARK_W = 520;
  const SPARK_H = 78;

  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];

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

  function readHashFlag(name) {
    const re = new RegExp('[#&?]' + name + '(?:=([^&]*))?');
    const m = (window.location.hash || '').match(re);
    return m ? (m[1] != null ? m[1] : true) : null;
  }

  /* Slice frame `f` out of the flat uint8 predFrames array.
     Returns a 64×64 array of class indices. */
  function frameSlice(framesObj, f) {
    const data = framesObj.data;       // Uint8Array length F*64*64
    const H = framesObj.shape[1];
    const W = framesObj.shape[2];
    const offset = f * H * W;
    const out = new Array(H);
    for (let i = 0; i < H; i++) {
      const row = new Array(W);
      for (let j = 0; j < W; j++) row[j] = data[offset + i * W + j];
      out[i] = row;
    }
    return out;
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.training) {
      root.innerHTML = '<p style="opacity:0.5">training data missing.</p>';
      return {};
    }
    const T = window.DATA.training;
    const framesObj = window.DATA._b64decode(T.predFrames);
    const F = framesObj.shape[0];
    const steps = T.steps;
    // Sanitize loss array: replace nulls / NaNs with "missing" markers.
    function clean(arr) {
      return arr.map(function (v) {
        return (v == null || !Number.isFinite(v)) ? null : v;
      });
    }
    const loss    = clean(T.loss);
    const pixAcc  = clean(T.pixAcc);
    const up1Norm = clean(T.up1Norm);
    const up2Norm = clean(T.up2Norm);

    root.innerHTML = '';
    root.classList.add('s13-root');
    const wrap = el('div', { class: 's13-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s13-hero' }, wrap);
    el('h1', { text: 'Watching the segmentation come into focus.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Same gradient descent as ever — but the artefact it produces is a 64×64 prediction map.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'The optimizer has ' + steps[steps.length - 1].toLocaleString() +
        ' steps to figure out how to encode, upsample, decode, and classify, ' +
        'all jointly. Scrub the slider to watch the prediction sharpen.',
    }, hero);

    /* ---- Body: prediction strip on the left, charts on the right --- */
    const body = el('div', { class: 's13-body' }, wrap);

    /* Left column: input + GT + current prediction */
    const leftCol = el('div', { class: 's13-left' }, body);
    function makePanel(parent, labelText, px) {
      const col = el('div', { class: 's13-panel' }, parent);
      el('div', { class: 's13-panel-label', text: labelText }, col);
      const host = el('div', {
        class: 'canvas-host s13-panel-host',
        style: 'width:' + px + 'px;height:' + px + 'px;',
      }, col);
      return { col: col, host: host };
    }
    const predPanel = makePanel(leftCol, 'prediction at this step', PRED_PX);
    // Below the prediction, side-by-side smaller input + GT for reference.
    const refRow = el('div', { class: 's13-refrow' }, leftCol);
    const inputPanel = makePanel(refRow, 'input', 132);
    const gtPanel = makePanel(refRow, 'ground truth', 132);

    // Step / accuracy badges next to the prediction.
    const badgeRow = el('div', { class: 's13-badges' }, leftCol);
    const stepBadge = el('div', { class: 's13-badge' }, badgeRow);
    el('span', { class: 's13-badge-label', text: 'training step' }, stepBadge);
    const stepBadgeVal = el('span', { class: 's13-badge-value', text: '0' }, stepBadge);
    const accBadge = el('div', { class: 's13-badge' }, badgeRow);
    el('span', { class: 's13-badge-label', text: 'pixel accuracy' }, accBadge);
    const accBadgeVal = el('span', { class: 's13-badge-value', text: '—' }, accBadge);
    const lossBadge = el('div', { class: 's13-badge' }, badgeRow);
    el('span', { class: 's13-badge-label', text: 'loss' }, lossBadge);
    const lossBadgeVal = el('span', { class: 's13-badge-value', text: '—' }, lossBadge);

    /* Right column: charts */
    const rightCol = el('div', { class: 's13-right' }, body);

    function makeChartCard(parent, title, w, h) {
      const card = el('div', { class: 's13-chart' }, parent);
      const head = el('div', { class: 's13-chart-head' }, card);
      el('span', { class: 's13-chart-title', text: title }, head);
      const note = el('span', { class: 's13-chart-note', text: '' }, head);
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
      svg.setAttribute('width', String(w));
      svg.setAttribute('height', String(h));
      svg.setAttribute('class', 's13-chart-svg');
      svg.style.width = '100%';
      svg.style.height = h + 'px';
      svg.style.maxWidth = w + 'px';
      card.appendChild(svg);
      return { card: card, svg: svg, note: note, w: w, h: h };
    }

    const lossChart = makeChartCard(rightCol, 'loss (cross-entropy, mean per pixel)', CHART_W, CHART_H);
    const accChart  = makeChartCard(rightCol, 'pixel accuracy on a held-out batch', CHART_W, CHART_H);
    const wmChart   = makeChartCard(rightCol, '‖W‖ — transposed-conv weight magnitude', SPARK_W, SPARK_H);

    /* ---- Timeline slider -------------------------------------------- */
    const timelineCard = el('div', { class: 's13-timeline' }, wrap);
    el('div', { class: 's13-timeline-label', text: 'timeline' }, timelineCard);
    const slider = el('input', {
      class: 's13-slider',
      type: 'range', min: '0', max: String(F - 1), step: '1', value: '0',
    }, timelineCard);
    const sliderOut = el('output', {
      class: 's13-slider-out', text: 'step 0',
    }, timelineCard);
    const playBtn = el('button', {
      type: 'button', class: 's13-play primary', text: '▶ play',
    }, timelineCard);

    /* ---- Caption + step controls ----------------------------------- */
    const caption = el('p', { class: 'caption s13-caption' }, wrap);

    const controls = el('div', { class: 'controls s13-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's13-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's13-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);

    /* ---- State ---------------------------------------------------- */
    const state = {
      step: 0,
      frame: 0,
      playing: false,
      playTimer: null,
    };

    /* ---- Painters ------------------------------------------------- */

    function paintFrame(f) {
      const lbl = frameSlice(framesObj, f);
      window.Drawing.paintLabelMap(predPanel.host, lbl, PRED_PX);
    }

    function paintInputAndGT() {
      window.Drawing.paintRGB(inputPanel.host, T.fixedSampleInput, 132);
      window.Drawing.paintLabelMap(gtPanel.host, T.fixedSampleLabel, 132);
    }

    /* Per-frame pixel accuracy: compare the prediction against the GT
       label. Cached so the chart marker is consistent with the slider. */
    function frameAccuracy(f) {
      const data = framesObj.data;
      const H = framesObj.shape[1];
      const W = framesObj.shape[2];
      const offset = f * H * W;
      const lbl = T.fixedSampleLabel;
      let correct = 0, total = H * W;
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          if (data[offset + i * W + j] === lbl[i][j]) correct++;
        }
      }
      return correct / total;
    }
    // Cache of per-frame accuracy on the fixed sample (cheap; F * 4096 ops).
    const fixedSampleAcc = new Array(F);
    for (let f = 0; f < F; f++) fixedSampleAcc[f] = frameAccuracy(f);

    /* ---- Chart drawing -------------------------------------------- */

    function drawChart(svg, ys, opts) {
      // ys: array of nullable numbers. opts:
      //   yMin, yMax, color, currentIdx, label1 (lo), label2 (hi),
      //   visible (bool), formatter (val -> string)
      svg.innerHTML = '';
      const w = parseFloat(svg.getAttribute('viewBox').split(' ')[2]);
      const h = parseFloat(svg.getAttribute('viewBox').split(' ')[3]);
      const padL = 38, padR = 12, padT = 6, padB = 18;
      const innerW = w - padL - padR;
      const innerH = h - padT - padB;
      const t = window.Drawing.tokens();
      const NS = 'http://www.w3.org/2000/svg';

      function svgEl(tag, attrs) {
        const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]);
        return e;
      }

      // Axis frame
      svg.appendChild(svgEl('rect', {
        x: padL, y: padT, width: innerW, height: innerH,
        fill: t.bg, stroke: t.rule, 'stroke-width': '1',
      }));

      // Y-axis labels (low / high)
      const fmt = opts.formatter || function (v) { return v.toFixed(2); };
      svg.appendChild(svgEl('text', {
        x: padL - 6, y: padT + 4,
        'text-anchor': 'end', 'font-size': '10', fill: t.inkSecondary,
        'font-family': '"SF Mono", Menlo, monospace',
      })).textContent = fmt(opts.yMax);
      svg.appendChild(svgEl('text', {
        x: padL - 6, y: padT + innerH,
        'text-anchor': 'end', 'font-size': '10', fill: t.inkSecondary,
        'font-family': '"SF Mono", Menlo, monospace',
      })).textContent = fmt(opts.yMin);

      // X-axis labels (first/last training step)
      svg.appendChild(svgEl('text', {
        x: padL, y: h - 4,
        'text-anchor': 'start', 'font-size': '10', fill: t.inkSecondary,
        'font-family': '"SF Mono", Menlo, monospace',
      })).textContent = String(steps[0]);
      svg.appendChild(svgEl('text', {
        x: padL + innerW, y: h - 4,
        'text-anchor': 'end', 'font-size': '10', fill: t.inkSecondary,
        'font-family': '"SF Mono", Menlo, monospace',
      })).textContent = String(steps[steps.length - 1]);

      if (!opts.visible) return;

      function xOf(i) { return padL + (i / Math.max(1, F - 1)) * innerW; }
      function yOf(v) {
        const tt = (opts.yMax !== opts.yMin) ?
          (v - opts.yMin) / (opts.yMax - opts.yMin) : 0;
        return padT + (1 - Math.max(0, Math.min(1, tt))) * innerH;
      }

      // Build path, breaking on nulls.
      let d = '', open = false;
      for (let i = 0; i < ys.length; i++) {
        const v = ys[i];
        if (v == null) { open = false; continue; }
        const x = xOf(i), y = yOf(v);
        d += (open ? ' L ' : ' M ') + x.toFixed(1) + ' ' + y.toFixed(1);
        open = true;
      }
      if (d.length) {
        const path = svgEl('path', {
          d: d, fill: 'none', stroke: opts.color, 'stroke-width': '1.6',
          'stroke-linecap': 'round', 'stroke-linejoin': 'round',
        });
        svg.appendChild(path);
      }

      // Marker at currentIdx
      const ci = opts.currentIdx;
      if (ci != null && ci >= 0 && ci < ys.length) {
        // Vertical guide
        const gx = xOf(ci);
        svg.appendChild(svgEl('line', {
          x1: gx, x2: gx, y1: padT, y2: padT + innerH,
          stroke: t.inkSecondary, 'stroke-width': '0.8',
          'stroke-dasharray': '3 3', opacity: '0.6',
        }));
        // Point
        const v = ys[ci];
        if (v != null) {
          const cy = yOf(v);
          svg.appendChild(svgEl('circle', {
            cx: gx, cy: cy, r: 4,
            fill: opts.color, stroke: t.bg, 'stroke-width': '1.5',
          }));
          // Numeric label near the marker
          const labelText = svgEl('text', {
            x: gx + 6, y: Math.max(padT + 10, cy - 6),
            'text-anchor': 'start', 'font-size': '10.5',
            fill: opts.color,
            'font-family': '"SF Mono", Menlo, monospace',
          });
          labelText.textContent = opts.markerLabel || fmt(v);
          svg.appendChild(labelText);
        }
      }
    }

    /* ---- Sparkline (two series overlaid) -------------------------- */

    function drawSparkline(svg, series, opts) {
      // series: [{ ys, color, label }, ...]
      svg.innerHTML = '';
      const w = parseFloat(svg.getAttribute('viewBox').split(' ')[2]);
      const h = parseFloat(svg.getAttribute('viewBox').split(' ')[3]);
      const padL = 38, padR = 70, padT = 6, padB = 14;
      const innerW = w - padL - padR;
      const innerH = h - padT - padB;
      const t = window.Drawing.tokens();
      const NS = 'http://www.w3.org/2000/svg';
      function svgEl(tag, attrs) {
        const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]);
        return e;
      }

      // Backing rect
      svg.appendChild(svgEl('rect', {
        x: padL, y: padT, width: innerW, height: innerH,
        fill: t.bg, stroke: t.rule, 'stroke-width': '1',
      }));

      // Find a shared y-range across both series.
      let yMin = Infinity, yMax = -Infinity;
      for (const s of series) {
        for (const v of s.ys) {
          if (v == null) continue;
          if (v < yMin) yMin = v;
          if (v > yMax) yMax = v;
        }
      }
      if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
      // Pad the range a hair so the lines don't kiss the box.
      const pad = (yMax - yMin) * 0.08 || 0.1;
      yMin -= pad; yMax += pad;

      // Y-axis labels
      function fmt(v) { return v.toFixed(2); }
      svg.appendChild(svgEl('text', {
        x: padL - 6, y: padT + 4,
        'text-anchor': 'end', 'font-size': '9.5', fill: t.inkSecondary,
        'font-family': '"SF Mono", Menlo, monospace',
      })).textContent = fmt(yMax);
      svg.appendChild(svgEl('text', {
        x: padL - 6, y: padT + innerH,
        'text-anchor': 'end', 'font-size': '9.5', fill: t.inkSecondary,
        'font-family': '"SF Mono", Menlo, monospace',
      })).textContent = fmt(yMin);

      if (!opts.visible) return;

      function xOf(i) { return padL + (i / Math.max(1, F - 1)) * innerW; }
      function yOf(v) {
        const tt = (yMax !== yMin) ? (v - yMin) / (yMax - yMin) : 0;
        return padT + (1 - tt) * innerH;
      }

      // Plot each series
      const legendX = padL + innerW + 8;
      let legendY = padT + 8;
      series.forEach(function (s) {
        let d = '', open = false;
        for (let i = 0; i < s.ys.length; i++) {
          const v = s.ys[i];
          if (v == null) { open = false; continue; }
          const x = xOf(i), y = yOf(v);
          d += (open ? ' L ' : ' M ') + x.toFixed(1) + ' ' + y.toFixed(1);
          open = true;
        }
        if (d.length) {
          svg.appendChild(svgEl('path', {
            d: d, fill: 'none', stroke: s.color, 'stroke-width': '1.4',
            'stroke-linecap': 'round',
          }));
        }
        // Marker at currentIdx
        const ci = opts.currentIdx;
        if (ci != null && s.ys[ci] != null) {
          svg.appendChild(svgEl('circle', {
            cx: xOf(ci), cy: yOf(s.ys[ci]), r: 3,
            fill: s.color, stroke: t.bg, 'stroke-width': '1.2',
          }));
        }
        // Legend swatch + label
        svg.appendChild(svgEl('rect', {
          x: legendX, y: legendY - 8, width: 10, height: 3,
          fill: s.color,
        }));
        const tx = svgEl('text', {
          x: legendX + 14, y: legendY - 4,
          'font-size': '10.5', fill: t.ink,
          'font-family': '"SF Mono", Menlo, monospace',
        });
        tx.textContent = s.label;
        svg.appendChild(tx);
        legendY += 16;
      });

      // Vertical guide
      const ci = opts.currentIdx;
      if (ci != null) {
        const gx = xOf(ci);
        svg.appendChild(svgEl('line', {
          x1: gx, x2: gx, y1: padT, y2: padT + innerH,
          stroke: t.inkSecondary, 'stroke-width': '0.8',
          'stroke-dasharray': '3 3', opacity: '0.6',
        }));
      }
    }

    /* ---- Update + render ----------------------------------------- */

    function updateBadges() {
      stepBadgeVal.textContent = steps[state.frame].toLocaleString();
      const a = fixedSampleAcc[state.frame];
      accBadgeVal.textContent = (a * 100).toFixed(2) + '%';
      const l = loss[state.frame];
      lossBadgeVal.textContent = (l == null) ? '—' : l.toFixed(3);
    }

    function captionFor(step) {
      switch (step) {
        case 0: return 'Step 0: random weights. The model predicts a single class for every pixel — whatever the random init happened to favour.';
        case 1: return 'Drag the slider to scrub training. Within a few hundred steps the sky/grass split arrives, then objects emerge.';
        case 2: return 'Loss curve, with a marker tracking the current step.';
        case 3: return 'Pixel accuracy climbs from a baseline (~20–60%) to >99% as the model learns.';
        case 4: return 'The transposed-conv weights grow steadily — the upsamplers are also being trained, not just the encoder.';
        default: return '';
      }
    }

    function render() {
      const step = state.step;
      slider.disabled = step < 1;
      playBtn.disabled = step < 1;

      paintFrame(state.frame);
      slider.value = String(state.frame);
      sliderOut.textContent = 'step ' + steps[state.frame].toLocaleString();
      updateBadges();

      // Loss chart
      drawChart(lossChart.svg, loss, {
        yMin: 0,
        // Use a generous yMax so the early spike is visible.
        yMax: Math.max(1.2, ...loss.filter(function (v) { return v != null; })),
        color: getCss('--cnn-pos'),
        currentIdx: state.frame,
        visible: step >= 2,
        formatter: function (v) { return v.toFixed(2); },
      });
      lossChart.note.textContent = (step >= 2)
        ? 'training loss · ' + steps[steps.length - 1].toLocaleString() + ' steps'
        : 'appears at step 2';

      // Accuracy chart
      drawChart(accChart.svg, pixAcc, {
        yMin: 0, yMax: 1.0,
        color: getCss('--cnn-green'),
        currentIdx: state.frame,
        visible: step >= 3,
        formatter: function (v) { return v.toFixed(2); },
        markerLabel: pixAcc[state.frame] != null
          ? (pixAcc[state.frame] * 100).toFixed(1) + '%' : '',
      });
      accChart.note.textContent = (step >= 3)
        ? 'mean pixel accuracy on a held-out training batch'
        : 'appears at step 3';

      // Weight magnitude sparkline
      drawSparkline(wmChart.svg, [
        { ys: up1Norm, color: getCss('--cnn-purple'), label: '‖up1‖' },
        { ys: up2Norm, color: getCss('--cnn-accent'), label: '‖up2‖' },
      ], { currentIdx: state.frame, visible: step >= 4 });
      wmChart.note.textContent = (step >= 4)
        ? 'L2 norm of the transposed-conv parameter tensors'
        : 'appears at step 4';

      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;
    }

    function getCss(name) {
      return getComputedStyle(document.documentElement)
        .getPropertyValue(name).trim() || '#888';
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      // Step 0 forces the frame to the start.
      if (state.step === 0) state.frame = 0;
      render();
    }

    function setFrame(f) {
      const ff = Math.max(0, Math.min(F - 1, f | 0));
      if (ff === state.frame) return;
      state.frame = ff;
      render();
    }

    function stopPlayback() {
      if (state.playTimer) { clearInterval(state.playTimer); state.playTimer = null; }
      state.playing = false;
      playBtn.textContent = '▶ play';
    }
    function startPlayback() {
      if (state.step < 1) applyStep(1);
      stopPlayback();
      state.playing = true;
      playBtn.textContent = '❚❚ pause';
      state.playTimer = setInterval(function () {
        if (state.frame >= F - 1) { stopPlayback(); return; }
        setFrame(state.frame + 1);
      }, 90);
    }

    /* ---- Wire controls ------------------------------------------- */

    slider.addEventListener('input', function () {
      stopPlayback();
      const v = parseInt(slider.value, 10);
      if (Number.isFinite(v)) setFrame(v);
    });
    playBtn.addEventListener('click', function () {
      if (state.playing) stopPlayback();
      else { if (state.frame >= F - 1) state.frame = 0; startPlayback(); }
    });
    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    /* ---- Initial paint ------------------------------------------- */
    paintInputAndGT();
    render();

    /* &run -> auto-advance through steps, ending with a playback. */
    let runTimer = null;
    function autoAdvance() {
      if (state.step >= NUM_STEPS - 1) {
        // Last action: scrub through training automatically.
        runTimer = setTimeout(function () {
          state.frame = 0;
          startPlayback();
          runTimer = null;
        }, 400);
        return;
      }
      applyStep(state.step + 1);
      runTimer = setTimeout(autoAdvance, RUN_INTERVAL_MS);
    }
    if (readHashFlag('run')) {
      runTimer = setTimeout(autoAdvance, 350);
    }

    return {
      onEnter: function () { paintInputAndGT(); render(); },
      onLeave: function () {
        stopPlayback();
        if (runTimer) { clearTimeout(runTimer); runTimer = null; }
      },
      onNextKey: function () {
        if (state.step < NUM_STEPS - 1) { applyStep(state.step + 1); return true; }
        return false;
      },
      onPrevKey: function () {
        if (state.step > 0) { applyStep(state.step - 1); return true; }
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene13 = function (root) { return buildScene(root); };
})();
