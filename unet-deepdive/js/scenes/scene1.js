/* Scene 1 — "From one label to a label per pixel".

   The pivot from CNN-deepdive's classification mindset to U-Net's
   per-pixel prediction. Two side-by-side panels: the left runs the same
   image through an illustrative classifier that emits one 5-way
   softmax; the right runs it through a segmenter that emits 4096
   independent 5-way softmaxes (one per pixel).

   The classifier output here is illustrative only — we do not have a
   separately trained classifier in scene64 data. We approximate it by
   counting class pixels in the segmenter prediction and renormalising;
   that is enough to convey the *shape* of the output, which is the
   pedagogical point.

   Step engine (5 steps):
     0 = both outputs blank
     1 = classifier output revealed (left bar chart + label)
     2 = segmenter label map fades in on the right
     3 = hover-pixel mode enabled (per-pixel softmax on the right)
     4 = the two cross-entropy formulas appear side-by-side (KaTeX)

   `&run` auto-advances 0 → 4 over ~3.2 s and pauses. */
(function () {
  'use strict';

  const NUM_STEPS = 5;
  const RUN_INTERVAL_MS = 800;

  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];
  const INPUT_PX = 220;       // RGB input thumbnail (both columns share)
  const OUTPUT_PX = 220;      // label map / classifier illustration
  const HOVER_PX = 96;        // per-pixel softmax mini-card on the right

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

  /* Pick a sample with at least three classes present so the contrast
     between "one label" and "many labels" is visceral. */
  function pickRichestSample(samples) {
    let best = 0, bestCount = -1;
    for (let k = 0; k < samples.length; k++) {
      const seen = new Set();
      const lbl = samples[k].label;
      for (let i = 0; i < lbl.length; i++) {
        for (let j = 0; j < lbl[0].length; j++) seen.add(lbl[i][j]);
      }
      if (seen.size > bestCount) { bestCount = seen.size; best = k; }
    }
    return best;
  }

  /* Illustrative classifier: count class pixels in the segmenter's pred,
     normalise to a 5-vector, and softmax-flatten it slightly so the
     dominant class still wins clearly. We are honest about this in the
     caption and a footnote. */
  function fakeClassifierProbs(pred) {
    const counts = [0, 0, 0, 0, 0];
    for (let i = 0; i < pred.length; i++) {
      for (let j = 0; j < pred[0].length; j++) counts[pred[i][j] | 0]++;
    }
    const tot = counts.reduce(function (a, b) { return a + b; }, 0) || 1;
    const raw = counts.map(function (c) { return c / tot; });
    // Soft-sharpen with a small temperature so the bar chart looks like a
    // classifier verdict rather than a raw histogram.
    const T = 0.35;
    const exps = raw.map(function (p) { return Math.exp(p / T); });
    const Z = exps.reduce(function (a, b) { return a + b; }, 0);
    return exps.map(function (e) { return e / Z; });
  }

  /* Synthesise a per-pixel 5-way softmax from the predicted class. We do
     not have full logits in scene64; we fake a confident distribution
     where the predicted class gets ~0.86 and the rest split the rest.
     This is illustrative — captioned as such. */
  function fakePerPixelProbs(predClass) {
    const p = [0.035, 0.035, 0.035, 0.035, 0.035];
    p[predClass] = 1 - 0.035 * 4;
    return p;
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64) {
      root.innerHTML = '<p style="opacity:0.5">scene64 data missing.</p>';
      return {};
    }
    const D = window.DATA.scene64;

    root.innerHTML = '';
    root.classList.add('s1-root');
    const wrap = el('div', { class: 's1-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s1-hero' }, wrap);
    el('h1', { text: 'From one label to a label per pixel.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Same image. Same convolutions. Different shape of answer.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'A classifier looks at this scene and emits <em>one</em> probability ' +
        'distribution over five classes — a single verdict for the whole image. ' +
        'A segmenter emits <em>4096</em> such distributions, one per pixel. ' +
        'Everything else in this deepdive follows from that shape change.',
    }, hero);

    /* ---- "You are here" mini-map ------------------------------------ */
    const miniHost = el('div', { class: 's1-mini-host' }, wrap);
    if (window.UNET && typeof window.UNET.mountUNetMiniMap === 'function') {
      const mm = window.UNET.mountUNetMiniMap(miniHost, {
        width: 280, title: 'you are here',
        label: '1×1 output head · turns the last decoder block into per-pixel class logits',
      });
      mm.setHighlight(['out']);
    }

    /* ---- Two panels ------------------------------------------------- */
    const panels = el('div', { class: 's1-panels' }, wrap);

    /* LEFT — classifier */
    const leftCol = el('section', { class: 's1-panel s1-panel-clf' }, panels);
    el('div', { class: 's1-panel-title', text: 'classifier' }, leftCol);
    el('div', { class: 's1-panel-sub', text: 'one image · one verdict' }, leftCol);

    const leftRow = el('div', { class: 's1-flow-row' }, leftCol);
    const leftInputCol = el('div', { class: 's1-flow-cell' }, leftRow);
    el('div', { class: 'col-label', text: 'input · 64×64×3' }, leftInputCol);
    const leftInputHost = el('div', { class: 'canvas-host s1-input-host' }, leftInputCol);

    const leftArrow = el('div', { class: 's1-arrow', text: '→' }, leftRow);

    const leftOutCol = el('div', { class: 's1-flow-cell' }, leftRow);
    el('div', { class: 'col-label', text: 'output · vector of 5 probabilities' }, leftOutCol);
    const leftBarsHost = el('div', { class: 's1-bars' }, leftOutCol);
    const leftVerdict = el('div', { class: 's1-verdict' }, leftOutCol);

    el('p', {
      class: 's1-panel-foot',
      html:
        'Output shape: <code>5</code>. Loss is one cross-entropy term over one ' +
        'distribution.',
    }, leftCol);

    /* RIGHT — segmenter */
    const rightCol = el('section', { class: 's1-panel s1-panel-seg' }, panels);
    el('div', { class: 's1-panel-title', text: 'segmenter (U-Net)' }, rightCol);
    el('div', { class: 's1-panel-sub', text: 'one image · 4096 verdicts' }, rightCol);

    const rightRow = el('div', { class: 's1-flow-row' }, rightCol);
    const rightInputCol = el('div', { class: 's1-flow-cell' }, rightRow);
    el('div', { class: 'col-label', text: 'input · 64×64×3' }, rightInputCol);
    const rightInputHost = el('div', { class: 'canvas-host s1-input-host' }, rightInputCol);

    const rightArrow = el('div', { class: 's1-arrow', text: '→' }, rightRow);

    const rightOutCol = el('div', { class: 's1-flow-cell' }, rightRow);
    el('div', { class: 'col-label', text: 'output · 64×64 label map' }, rightOutCol);
    const rightMapHost = el('div', { class: 'canvas-host s1-output-host' }, rightOutCol);

    /* Hover inspector — appears at step 3+. */
    const hoverWrap = el('div', { class: 's1-hover-wrap' }, rightCol);
    const hoverInfo = el('div', { class: 's1-hover-info' }, hoverWrap);
    el('div', { class: 's1-hover-title', text: 'hover any pixel' }, hoverInfo);
    const hoverPos = el('div', { class: 's1-hover-pos', text: '(–, –)' }, hoverInfo);
    const hoverBars = el('div', { class: 's1-bars s1-hover-bars' }, hoverInfo);
    const hoverArgmax = el('div', { class: 's1-hover-argmax', text: 'argmax: —' }, hoverInfo);

    el('p', {
      class: 's1-panel-foot',
      html:
        'Output shape: <code>64×64×5</code>. Loss is the sum (or mean) of ' +
        '4096 cross-entropy terms.',
    }, rightCol);

    /* ---- Class legend ----------------------------------------------- */
    const legend = el('div', { class: 's1-legend' }, wrap);
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const item = el('div', { class: 's1-legend-item' }, legend);
      el('span', { class: 's1-swatch class-' + CLASS_NAMES[c] }, item);
      el('span', { class: 's1-legend-name', text: CLASS_NAMES[c] }, item);
    }

    /* ---- Loss formulas (KaTeX, step 4) ------------------------------ */
    const formulas = el('div', { class: 's1-formulas' }, wrap);
    const fLeft = el('div', { class: 's1-formula s1-formula-clf' }, formulas);
    el('div', { class: 's1-formula-label', text: 'classifier loss (1 distribution)' }, fLeft);
    const fLeftMath = el('div', { class: 's1-formula-math' }, fLeft);

    const fRight = el('div', { class: 's1-formula s1-formula-seg' }, formulas);
    el('div', { class: 's1-formula-label', text: 'segmenter loss (H·W distributions, summed)' }, fRight);
    const fRightMath = el('div', { class: 's1-formula-math' }, fRight);

    /* ---- Caption ---------------------------------------------------- */
    const caption = el('p', { class: 'caption s1-caption' }, wrap);

    /* ---- Honest footnote ------------------------------------------- */
    el('p', {
      class: 's1-honest',
      html:
        '<strong>Note.</strong> The classifier panel is illustrative — we do not ' +
        'train a separate classifier in this deepdive. Its bar chart is computed ' +
        'from the segmenter\'s own pixel counts, sharpened with a temperature, ' +
        'so the “verdict” is the dominant class. The point of the panel is its ' +
        '<em>shape</em>, not its numerical fidelity.',
    }, wrap);

    /* ---- Step controls --------------------------------------------- */
    const controls = el('div', { class: 'controls s1-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's1-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's1-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', {
      class: 'control-value', text: '0 / ' + (NUM_STEPS - 1),
    }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State ------------------------------------------------------ */
    const initialIdx = pickRichestSample(D.samples);
    const state = {
      step: 0,
      sampleIdx: initialIdx,
      hoverI: -1,
      hoverJ: -1,
    };

    /* ---- Painters --------------------------------------------------- */
    function paintBars(host, probs, opts) {
      opts = opts || {};
      host.innerHTML = '';
      for (let c = 0; c < probs.length; c++) {
        const row = el('div', { class: 's1-bar-row' }, host);
        el('span', { class: 's1-bar-name', text: CLASS_NAMES[c] }, row);
        const track = el('div', { class: 's1-bar-track' }, row);
        const fill = el('div', { class: 's1-bar-fill class-' + CLASS_NAMES[c] }, track);
        fill.style.width = (probs[c] * 100).toFixed(1) + '%';
        el('span', {
          class: 's1-bar-val',
          text: probs[c].toFixed(2),
        }, row);
      }
    }

    function paintBlankBars(host) {
      host.innerHTML = '';
      for (let c = 0; c < CLASS_NAMES.length; c++) {
        const row = el('div', { class: 's1-bar-row s1-bar-row-blank' }, host);
        el('span', { class: 's1-bar-name', text: CLASS_NAMES[c] }, row);
        const track = el('div', { class: 's1-bar-track' }, row);
        el('div', { class: 's1-bar-fill s1-bar-fill-blank' }, track);
        el('span', { class: 's1-bar-val', text: '—' }, row);
      }
    }

    function captionFor(step) {
      switch (step) {
        case 0: return 'Same input on both sides. Click "next" to compare what each model emits.';
        case 1: return 'The classifier collapses the image to a single 5-way distribution. One verdict, one number per class.';
        case 2: return 'The segmenter keeps spatial structure. Every pixel gets its own 5-way distribution; the picture you see is the per-pixel argmax.';
        case 3: return 'Hover any pixel on the right. That single pixel had its own softmax — the segmenter ran 4096 of these in parallel.';
        case 4: return 'Same cross-entropy formula on both sides — applied to one distribution on the left, to 4096 on the right and summed.';
        default: return '';
      }
    }

    /* ---- Render ----------------------------------------------------- */
    function render() {
      const step = state.step;
      const sample = D.samples[state.sampleIdx];

      // Inputs (always painted)
      window.Drawing.paintRGB(leftInputHost, sample.input, INPUT_PX);
      window.Drawing.paintRGB(rightInputHost, sample.input, INPUT_PX);

      // Classifier output (step >= 1)
      if (step >= 1) {
        const probs = fakeClassifierProbs(sample.pred);
        paintBars(leftBarsHost, probs);
        let argmax = 0;
        for (let k = 1; k < probs.length; k++) if (probs[k] > probs[argmax]) argmax = k;
        leftVerdict.innerHTML = 'verdict: <span class="class-' + CLASS_NAMES[argmax] +
          '"><strong>' + CLASS_NAMES[argmax] + '</strong></span>';
        leftVerdict.classList.add('s1-visible');
      } else {
        paintBlankBars(leftBarsHost);
        leftVerdict.innerHTML = 'verdict: —';
        leftVerdict.classList.remove('s1-visible');
      }

      // Segmenter output (step >= 2)
      if (step >= 2) {
        window.Drawing.paintLabelMap(rightMapHost, sample.label, OUTPUT_PX);
        rightMapHost.classList.add('s1-visible');
      } else {
        window.Drawing.paintBlankCard(rightMapHost, OUTPUT_PX);
        rightMapHost.classList.remove('s1-visible');
      }

      // Hover mode (step >= 3)
      const hoverEnabled = step >= 3;
      hoverWrap.classList.toggle('s1-hover-enabled', hoverEnabled);
      rightMapHost.classList.toggle('s1-hover-on', hoverEnabled);
      if (hoverEnabled && state.hoverI >= 0) {
        renderHover(state.hoverI, state.hoverJ);
      } else {
        paintBlankBars(hoverBars);
        hoverPos.textContent = hoverEnabled ? '(–, –)' : '';
        hoverArgmax.textContent = hoverEnabled ? 'argmax: —' : '';
      }

      // Formulas (step >= 4)
      if (step >= 4) {
        // Inline KaTeX rendering. The classifier loss is single-pixel CE;
        // the segmenter loss is the sum (mean if normalised) over (i, j).
        window.katex.render(
          '\\mathcal{L}_{\\text{clf}} = -\\sum_{c=1}^{C} y_c \\log \\hat{p}_c',
          fLeftMath, { throwOnError: false, displayMode: true }
        );
        window.katex.render(
          '\\mathcal{L}_{\\text{seg}} = -\\sum_{i=1}^{H} \\sum_{j=1}^{W} \\sum_{c=1}^{C} y_{ijc} \\log \\hat{p}_{ijc}',
          fRightMath, { throwOnError: false, displayMode: true }
        );
        formulas.classList.add('s1-visible');
      } else {
        fLeftMath.innerHTML = '';
        fRightMath.innerHTML = '';
        formulas.classList.remove('s1-visible');
      }

      // Caption + controls
      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;

      // Re-attach hover listener on the right map host since paintLabelMap
      // replaces the canvas inside.
      attachHover();
    }

    function renderHover(i, j) {
      const sample = D.samples[state.sampleIdx];
      const cls = sample.label[i][j] | 0;
      const probs = fakePerPixelProbs(cls);
      paintBars(hoverBars, probs);
      hoverPos.textContent = '(' + i + ', ' + j + ')';
      hoverArgmax.innerHTML = 'argmax: <span class="class-' +
        CLASS_NAMES[cls] + '"><strong>' + CLASS_NAMES[cls] + '</strong></span>';
    }

    function attachHover() {
      const cv = rightMapHost.querySelector('canvas');
      if (!cv) return;
      cv.onmousemove = function (ev) {
        if (state.step < 3) return;
        const rect = cv.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        const j = Math.max(0, Math.min(63, Math.floor(x / rect.width * 64)));
        const i = Math.max(0, Math.min(63, Math.floor(y / rect.height * 64)));
        if (i !== state.hoverI || j !== state.hoverJ) {
          state.hoverI = i;
          state.hoverJ = j;
          renderHover(i, j);
        }
      };
      cv.onmouseleave = function () {
        if (state.step < 3) return;
        // Keep last reading visible — feels less jumpy than blanking out.
      };
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    /* Initial paint */
    render();

    /* &run -> auto-advance to step 4 over ~3.2s. */
    let runTimer = null;
    function autoAdvance(target) {
      if (state.step >= target) { runTimer = null; return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(function () { autoAdvance(target); }, RUN_INTERVAL_MS);
    }
    if (readHashFlag('run')) {
      runTimer = setTimeout(function () { autoAdvance(NUM_STEPS - 1); }, 200);
    }

    return {
      onEnter: function () { render(); },
      onLeave: function () {
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
  window.scenes.scene1 = function (root) { return buildScene(root); };
})();
