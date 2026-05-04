/* Scene 12 — "The loss that knows about every pixel."

   Pick one sample. For one selected pixel, expand a 5-way bar chart of
   the model's softmax, the ground-truth one-hot, and the cross-entropy
   formula (KaTeX). Then aggregate: a 64×64 heatmap of per-pixel CE.
   Mean of the heatmap = the scalar loss.

   Caveat (called out in the caption + a footnote):
     We exported argmax `pred`, not the full softmax distribution. To
     show a meaningful 5-way bar chart for every pixel we synthesise a
     plausible softmax: the predicted class gets probability 0.85 (or
     0.92 if the model and ground truth agree), and the remaining mass
     is distributed across the other four classes weighted by class
     prior. The lesson here is the *math* — the cross-entropy formula
     and how it aggregates over 4096 pixels — not the exact numbers.

   Step engine:
     0 = prediction + ground truth shown
     1 = a default pixel is selected; bar chart + KaTeX formula appear
     2 = aggregate per-pixel CE heatmap appears
     3 = hover mode enabled (move over heatmap to inspect any pixel)
     4 = scalar loss number revealed
*/
(function () {
  'use strict';

  const NUM_STEPS = 5;
  const RUN_INTERVAL_MS = 800;

  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];
  const N_CLASSES = 5;
  const PANEL_PX = 256;
  const HEAT_PX = 256;

  /* Synthetic softmax parameters. The exact values are pedagogical, not
     measured. They are documented in the scene caption + footnote. */
  const PRED_PROB_AGREE = 0.92;       // predicted == ground truth
  const PRED_PROB_DISAGREE = 0.62;    // predicted != ground truth (less confident)

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

  /* Compute class priors from a label map.  Used to weight the off-class
     mass in the synthetic softmax. */
  function classPriors(label) {
    const counts = new Array(N_CLASSES).fill(0);
    let total = 0;
    for (let i = 0; i < label.length; i++) {
      for (let j = 0; j < label[0].length; j++) {
        counts[label[i][j] | 0]++;
        total++;
      }
    }
    // Avoid zero-prior classes (would zero a denominator below); add a
    // floor of 1 so every class has at least a tiny share.
    for (let k = 0; k < N_CLASSES; k++) counts[k] = Math.max(counts[k], 1);
    const prior = new Array(N_CLASSES);
    let s = 0;
    for (let k = 0; k < N_CLASSES; k++) s += counts[k];
    for (let k = 0; k < N_CLASSES; k++) prior[k] = counts[k] / s;
    return prior;
  }

  /* Build a synthetic 5-way distribution at one pixel. The predicted
     class gets a high probability; the rest is spread across the other
     four classes proportionally to their priors. */
  function fakeSoftmax(predClass, gtClass, prior) {
    const peak = (predClass === gtClass) ? PRED_PROB_AGREE : PRED_PROB_DISAGREE;
    const rest = 1 - peak;
    let denom = 0;
    for (let k = 0; k < N_CLASSES; k++) if (k !== predClass) denom += prior[k];
    if (denom <= 0) denom = 1;
    const out = new Array(N_CLASSES);
    for (let k = 0; k < N_CLASSES; k++) {
      out[k] = (k === predClass) ? peak : (rest * prior[k] / denom);
    }
    // Numerical safety: clamp tiny values so log doesn't explode.
    const FLOOR = 1e-6;
    let s = 0;
    for (let k = 0; k < N_CLASSES; k++) {
      if (out[k] < FLOOR) out[k] = FLOOR;
      s += out[k];
    }
    for (let k = 0; k < N_CLASSES; k++) out[k] /= s;
    return out;
  }

  function crossEntropy(prob, gtClass) {
    const p = Math.max(prob[gtClass], 1e-12);
    return -Math.log(p);
  }

  /* Compute the full per-pixel CE heatmap for the chosen sample. */
  function ceHeatmap(sample, prior) {
    const H = sample.label.length, W = sample.label[0].length;
    const heat = new Array(H);
    let sum = 0;
    for (let i = 0; i < H; i++) {
      const row = new Array(W);
      for (let j = 0; j < W; j++) {
        const gt = sample.label[i][j] | 0;
        const pr = sample.pred[i][j] | 0;
        const dist = fakeSoftmax(pr, gt, prior);
        row[j] = crossEntropy(dist, gt);
        sum += row[j];
      }
      heat[i] = row;
    }
    return { heat: heat, mean: sum / (H * W) };
  }

  /* Pick a sample. Preference order:
       1. samples with errors (the heatmap will have hot spots);
       2. otherwise the sample with the most class diversity in the GT.
     The dataset is small and the trained model is very accurate, so on
     these 6 export-samples we usually fall through to (2). */
  function pickInterestingSample(samples) {
    let best = 0, bestScore = -1;
    for (let k = 0; k < samples.length; k++) {
      const lbl = samples[k].label, prd = samples[k].pred;
      let errs = 0;
      const seen = new Set();
      for (let i = 0; i < lbl.length; i++) {
        for (let j = 0; j < lbl[0].length; j++) {
          if (lbl[i][j] !== prd[i][j]) errs++;
          seen.add(lbl[i][j]);
        }
      }
      const score = errs > 0 ? (errs * 1000) : seen.size;
      if (score > bestScore) { bestScore = score; best = k; }
    }
    return best;
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64) {
      root.innerHTML = '<p style="opacity:0.5">scene64 data missing.</p>';
      return {};
    }
    const D = window.DATA.scene64;

    root.innerHTML = '';
    root.classList.add('s12-root');
    const wrap = el('div', { class: 's12-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s12-hero' }, wrap);
    el('h1', { text: 'The loss that knows about every pixel.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Same cross-entropy as classification — applied 4096 times and averaged.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'Click any pixel to see its 5-way distribution. The loss for the whole image is just ' +
        'the mean of the per-pixel cross-entropies. No new machinery, only a different shape of answer.',
    }, hero);

    /* ---- Top row: input · ground truth · prediction ---------------- */
    const topRow = el('div', { class: 's12-toprow' }, wrap);

    function makePanel(parent, labelText) {
      const col = el('div', { class: 's12-panel' }, parent);
      const lbl = el('div', { class: 's12-panel-label', text: labelText }, col);
      const host = el('div', { class: 'canvas-host s12-panel-host' }, col);
      return { col: col, label: lbl, host: host };
    }
    const panInput = makePanel(topRow, 'input · 64×64×3');
    const panGT    = makePanel(topRow, 'ground truth');
    const panPred  = makePanel(topRow, 'prediction (argmax)');

    /* ---- Middle row: per-pixel inspector --------------------------- */
    const midRow = el('div', { class: 's12-midrow' }, wrap);

    // Left card: the heatmap of per-pixel CE.
    const heatPanel = el('div', { class: 's12-heat' }, midRow);
    const heatHead = el('div', { class: 's12-heat-head' }, heatPanel);
    el('span', { class: 's12-heat-title', text: 'per-pixel cross-entropy' }, heatHead);
    const heatNote = el('span', { class: 's12-heat-note', text: '' }, heatHead);
    const heatHostWrap = el('div', { class: 's12-heat-wrap' }, heatPanel);
    const heatHost = el('div', { class: 'canvas-host s12-heat-host' }, heatHostWrap);
    // Reticle (CSS-positioned div) marking the currently inspected pixel.
    const reticle = el('div', { class: 's12-reticle' }, heatHostWrap);
    reticle.style.display = 'none';
    const heatHint = el('div', {
      class: 's12-heat-hint',
      text: 'click any pixel · or move the cursor (after step 3)',
    }, heatPanel);

    // Right card: the inspector (5-way bar chart, GT one-hot, KaTeX formula).
    const inspector = el('div', { class: 's12-inspector' }, midRow);
    const insHead = el('div', { class: 's12-ins-head' }, inspector);
    el('span', { class: 's12-ins-title', text: 'pixel inspector' }, insHead);
    const insCoords = el('span', { class: 's12-ins-coords', text: '' }, insHead);

    // Bar chart: rows of (class color, name, predicted bar, gt one-hot bar).
    const barWrap = el('div', { class: 's12-barwrap' }, inspector);
    const barRows = [];
    for (let k = 0; k < N_CLASSES; k++) {
      const row = el('div', { class: 's12-barrow' }, barWrap);
      el('span', { class: 's12-bar-swatch class-' + CLASS_NAMES[k] }, row);
      el('span', { class: 's12-bar-name', text: CLASS_NAMES[k] }, row);

      const barTrack = el('div', { class: 's12-bartrack' }, row);
      const barFill  = el('div', { class: 's12-barfill class-' + CLASS_NAMES[k] }, barTrack);
      const barLabel = el('span', { class: 's12-barlabel', text: '' }, row);
      const oneHot   = el('div', { class: 's12-onehot' }, row);

      barRows.push({
        row: row, fill: barFill, label: barLabel, oneHot: oneHot,
      });
    }

    // KaTeX formula box.
    const formulaBox = el('div', { class: 's12-formula' }, inspector);
    const formulaHost = el('div', { class: 's12-formula-host' }, formulaBox);
    const formulaCE = el('div', { class: 's12-formula-ce' }, formulaBox);

    /* ---- Aggregate bar (mean CE = scalar loss) --------------------- */
    const aggregate = el('div', { class: 's12-aggregate' }, wrap);
    const aggHead = el('div', { class: 's12-agg-head' }, aggregate);
    el('span', { class: 's12-agg-title', text: 'image-level loss' }, aggHead);
    const aggValue = el('span', { class: 's12-agg-value', text: '—' }, aggHead);
    const aggFormula = el('div', { class: 's12-agg-formula' }, aggregate);

    /* ---- Footnote: honesty about the synthetic softmax ------------- */
    const footnote = el('div', { class: 'footnote s12-footnote' }, wrap);
    footnote.innerHTML =
      'Honesty note: we exported the argmax prediction, not the underlying ' +
      'logits. The 5-way distribution shown above is synthetic — predicted ' +
      'class probability ' +
      '<kbd>0.92</kbd> when the model agrees with the ground truth, ' +
      '<kbd>0.62</kbd> when it disagrees, with the remaining mass distributed ' +
      'across the other classes by class prior. The <em>math</em> (the cross-' +
      'entropy formula and how it sums over pixels) is the lesson; the exact ' +
      'numbers are illustrative.';

    /* ---- Caption + step controls ----------------------------------- */
    const caption = el('p', { class: 'caption s12-caption' }, wrap);

    const controls = el('div', { class: 'controls s12-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's12-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's12-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, stepGroup);

    const sampleGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'sample', for: 's12-sample' }, sampleGroup);
    const sampleSel = el('select', { id: 's12-sample' }, sampleGroup);
    for (let i = 0; i < D.samples.length; i++) {
      const opt = el('option', { value: String(i), text: 'sample ' + (i + 1) }, sampleSel);
      if (i === 0) opt.selected = true;
    }

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);

    /* ---- State ----------------------------------------------------- */
    const initialIdx = pickInterestingSample(D.samples);
    sampleSel.value = String(initialIdx);
    const state = {
      step: 0,
      sampleIdx: initialIdx,
      pixel: null,        // {i, j} or null
      heat: null,         // computed lazily
      meanCE: 0,
      prior: null,
      heatMax: 1,
    };

    function recomputeForSample() {
      const sample = D.samples[state.sampleIdx];
      state.prior = classPriors(sample.label);
      const r = ceHeatmap(sample, state.prior);
      state.heat = r.heat;
      state.meanCE = r.mean;
      // Track a stable max so the heatmap colourmap is deterministic.
      let m = 0;
      for (let i = 0; i < r.heat.length; i++)
        for (let j = 0; j < r.heat[0].length; j++)
          if (r.heat[i][j] > m) m = r.heat[i][j];
      // A cap at -log(0.5) ≈ 0.693 keeps the colour scale meaningful even
      // when most pixels have near-zero loss; pixels above this saturate.
      state.heatMax = Math.max(0.6, Math.min(m, 4.0));
      // Pick a default pixel: one with high CE, or the centre as fallback.
      let bestI = -1, bestJ = -1, bestV = -1;
      for (let i = 0; i < r.heat.length; i++)
        for (let j = 0; j < r.heat[0].length; j++)
          if (r.heat[i][j] > bestV) { bestV = r.heat[i][j]; bestI = i; bestJ = j; }
      state.pixel = (bestV > 0) ? { i: bestI, j: bestJ } :
                                   { i: 32, j: 32 };
    }

    /* ---- Painters -------------------------------------------------- */

    function paintHeatmap(host, heat, hMax, px) {
      host.innerHTML = '';
      const setup = window.Drawing.setupCanvas(host, px, px);
      const ctx = setup.ctx;
      const t = window.Drawing.tokens();
      const H = heat.length, W = heat[0].length;
      const off = document.createElement('canvas');
      off.width = W; off.height = H;
      const offCtx = off.getContext('2d');
      const id = offCtx.createImageData(W, H);
      // Map CE in [0, hMax] -> bg (cool) to neg (red).  Use the diverging
      // negative-side ramp so high-loss pixels read as "wrong".
      function parseHex(hex) {
        let s = (hex || '').trim().replace('#', '');
        if (s.length === 3) s = s.split('').map(function (c) { return c + c; }).join('');
        const n = parseInt(s, 16);
        return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
      }
      const bgRgb = parseHex(t.bg);
      const accRgb = parseHex(t.neg);
      let p = 0;
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          const v = Math.max(0, Math.min(1, heat[i][j] / hMax));
          // Gamma 0.6 brightens the low end so faint errors stay visible.
          const t2 = Math.pow(v, 0.6);
          id.data[p++] = Math.round(bgRgb[0] + (accRgb[0] - bgRgb[0]) * t2);
          id.data[p++] = Math.round(bgRgb[1] + (accRgb[1] - bgRgb[1]) * t2);
          id.data[p++] = Math.round(bgRgb[2] + (accRgb[2] - bgRgb[2]) * t2);
          id.data[p++] = 255;
        }
      }
      offCtx.putImageData(id, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, px, px);
    }

    function paintBlankHeat(host, px) {
      host.innerHTML = '';
      const setup = window.Drawing.setupCanvas(host, px, px);
      const ctx = setup.ctx;
      const t = window.Drawing.tokens();
      ctx.fillStyle = t.bg;
      ctx.fillRect(0, 0, px, px);
      ctx.strokeStyle = t.rule;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.strokeRect(0.5, 0.5, px - 1, px - 1);
      ctx.setLineDash([]);
      ctx.fillStyle = t.inkSecondary;
      ctx.font = '13px "Iowan Old Style", Palatino, Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('appears at step 2', px / 2, px / 2);
    }

    /* ---- Inspector painter ---------------------------------------- */

    function renderInspector(pixel) {
      if (!pixel) {
        insCoords.textContent = '';
        for (let k = 0; k < N_CLASSES; k++) {
          barRows[k].fill.style.width = '0%';
          barRows[k].label.textContent = '';
          barRows[k].oneHot.classList.remove('on');
          barRows[k].row.classList.remove('s12-pred', 's12-gt');
        }
        formulaHost.innerHTML = '';
        formulaCE.innerHTML = '';
        return;
      }
      const sample = D.samples[state.sampleIdx];
      const gt = sample.label[pixel.i][pixel.j] | 0;
      const pr = sample.pred[pixel.i][pixel.j] | 0;
      const dist = fakeSoftmax(pr, gt, state.prior);
      const ce = crossEntropy(dist, gt);

      insCoords.textContent = '(i = ' + pixel.i + ', j = ' + pixel.j + ')';

      for (let k = 0; k < N_CLASSES; k++) {
        const p = dist[k];
        barRows[k].fill.style.width = (p * 100).toFixed(1) + '%';
        barRows[k].label.textContent = p.toFixed(3);
        barRows[k].oneHot.classList.toggle('on', k === gt);
        barRows[k].row.classList.toggle('s12-pred', k === pr);
        barRows[k].row.classList.toggle('s12-gt', k === gt);
      }

      // Render the cross-entropy formula. Uses subscripts for the GT class
      // and shows the numeric -log(p_gt) value.
      const tex =
        '\\mathcal{L}_{(' + pixel.i + ',' + pixel.j + ')} = ' +
        '-\\log\\bigl(\\hat{y}_{\\text{' + CLASS_NAMES[gt] + '}}\\bigr)';
      formulaHost.innerHTML = '';
      window.Katex.render(tex, formulaHost, true);

      const numeric =
        '= -\\log(' + dist[gt].toFixed(3) + ') = ' + ce.toFixed(3);
      formulaCE.innerHTML = '';
      window.Katex.render(numeric, formulaCE, true);
    }

    /* ---- Aggregate bar -------------------------------------------- */

    function renderAggregate(showValue) {
      const tex =
        '\\mathcal{L}_{\\text{img}} = ' +
        '\\frac{1}{H W} \\sum_{i, j} -\\log\\bigl(\\hat{y}_{i,j,\\,y_{i,j}}\\bigr)';
      aggFormula.innerHTML = '';
      window.Katex.render(tex, aggFormula, true);
      if (showValue) {
        aggValue.textContent = state.meanCE.toFixed(4);
        aggValue.style.opacity = '1';
      } else {
        aggValue.textContent = '—';
        aggValue.style.opacity = '0.4';
      }
    }

    /* ---- Top-row painters ---------------------------------------- */

    function paintTopRow() {
      const sample = D.samples[state.sampleIdx];
      window.Drawing.paintRGB(panInput.host, sample.input, PANEL_PX);
      window.Drawing.paintLabelMap(panGT.host, sample.label, PANEL_PX);
      window.Drawing.paintLabelMap(panPred.host, sample.pred, PANEL_PX);
    }

    /* ---- Reticle / pixel selection ------------------------------- */

    function updateReticle() {
      if (!state.pixel) { reticle.style.display = 'none'; return; }
      const heatRect = heatHost.getBoundingClientRect();
      const wrapRect = heatHostWrap.getBoundingClientRect();
      if (!heatRect.width) { reticle.style.display = 'none'; return; }
      const cellW = heatRect.width / 64;
      const cellH = heatRect.height / 64;
      // Position relative to wrapper.
      reticle.style.left  = ((heatRect.left - wrapRect.left) + state.pixel.j * cellW) + 'px';
      reticle.style.top   = ((heatRect.top  - wrapRect.top)  + state.pixel.i * cellH) + 'px';
      reticle.style.width  = Math.max(2, cellW) + 'px';
      reticle.style.height = Math.max(2, cellH) + 'px';
      reticle.style.display = '';
    }

    function pixelFromEvent(e) {
      const heatRect = heatHost.getBoundingClientRect();
      if (!heatRect.width) return null;
      const x = e.clientX - heatRect.left;
      const y = e.clientY - heatRect.top;
      if (x < 0 || y < 0 || x >= heatRect.width || y >= heatRect.height) return null;
      const j = Math.floor((x / heatRect.width) * 64);
      const i = Math.floor((y / heatRect.height) * 64);
      return { i: Math.max(0, Math.min(63, i)), j: Math.max(0, Math.min(63, j)) };
    }

    heatHost.addEventListener('click', function (e) {
      if (state.step < 1) return;
      const p = pixelFromEvent(e);
      if (p) {
        state.pixel = p;
        renderInspector(state.pixel);
        updateReticle();
      }
    });
    heatHost.addEventListener('mousemove', function (e) {
      if (state.step < 3) return;
      const p = pixelFromEvent(e);
      if (p) {
        state.pixel = p;
        renderInspector(state.pixel);
        updateReticle();
      }
    });
    // Also let the user click on the prediction panel to inspect a pixel.
    panPred.host.addEventListener('click', function (e) {
      if (state.step < 1) return;
      const r = panPred.host.getBoundingClientRect();
      if (!r.width) return;
      const x = e.clientX - r.left, y = e.clientY - r.top;
      if (x < 0 || y < 0 || x >= r.width || y >= r.height) return;
      const j = Math.floor((x / r.width) * 64);
      const i = Math.floor((y / r.height) * 64);
      state.pixel = { i: Math.max(0, Math.min(63, i)), j: Math.max(0, Math.min(63, j)) };
      renderInspector(state.pixel);
      updateReticle();
    });

    /* ---- Step engine ---------------------------------------------- */

    function captionFor(step) {
      switch (step) {
        case 0: return 'Three panels: the input the model sees, the ground truth, and what it predicted.';
        case 1: return 'Pick one pixel. The model has a 5-way distribution there; cross-entropy is just −log of the GT-class probability.';
        case 2: return 'Apply that to every pixel. The heatmap shows where the model is uncertain or wrong.';
        case 3: return 'Hover the heatmap to inspect any pixel.';
        case 4: return 'The image-level loss is the mean of the per-pixel cross-entropies — one scalar the optimizer can minimise.';
        default: return '';
      }
    }

    function render() {
      const step = state.step;
      // Top row is always shown.
      paintTopRow();

      // Inspector visibility / contents
      midRow.classList.toggle('s12-step-1', step >= 1);
      midRow.classList.toggle('s12-step-2', step >= 2);
      midRow.classList.toggle('s12-hover-on', step >= 3);

      if (step >= 2) {
        paintHeatmap(heatHost, state.heat, state.heatMax, HEAT_PX);
        heatNote.textContent = 'colourmap: 0 → ' + state.heatMax.toFixed(2) + ' nats';
      } else {
        paintBlankHeat(heatHost, HEAT_PX);
        heatNote.textContent = '';
      }

      if (step >= 1) {
        renderInspector(state.pixel);
      } else {
        renderInspector(null);
      }

      // Aggregate bar
      renderAggregate(step >= 4);

      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;

      requestAnimationFrame(updateReticle);
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });
    sampleSel.addEventListener('change', function () {
      const v = parseInt(sampleSel.value, 10);
      if (Number.isFinite(v) && v !== state.sampleIdx) {
        state.sampleIdx = v;
        recomputeForSample();
        render();
      }
    });

    /* ---- Initial paint ------------------------------------------- */
    recomputeForSample();
    render();

    const onResize = function () { updateReticle(); };
    window.addEventListener('resize', onResize);

    /* &run -> auto-advance through steps. */
    let runTimer = null;
    function autoAdvance() {
      if (state.step >= NUM_STEPS - 1) { runTimer = null; return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(autoAdvance, RUN_INTERVAL_MS);
    }
    if (readHashFlag('run')) {
      runTimer = setTimeout(autoAdvance, 350);
    }

    return {
      onEnter: function () { render(); requestAnimationFrame(updateReticle); },
      onLeave: function () {
        if (runTimer) { clearTimeout(runTimer); runTimer = null; }
        window.removeEventListener('resize', onResize);
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
  window.scenes.scene12 = function (root) { return buildScene(root); };
})();
