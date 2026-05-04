/* Scene 2 — "What a segmentation dataset looks like".

   Show what the model is actually shown: pairs of (RGB image, dense
   label map). 6 sample pairs side-by-side; a legend; a "data generator"
   inset that just rotates through the 6 samples we have (we are honest
   about that — there is no JS reimplementation of scene64_data.py); and
   a histogram of class pixel frequencies across the 6 samples we DO have
   (also captioned honestly: this is a 6-sample slice, not the full 600).

   Step engine:
     0 = grid only
     1 = legend appears
     2 = "another sample" button enabled in the data-generator inset
     3 = histogram fades in

   `&run` auto-advances 0 → 3 over ~3 s. */
(function () {
  'use strict';

  const NUM_STEPS = 4;
  const RUN_INTERVAL_MS = 850;

  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];
  const PAIR_PX = 96;       // image / label tile size in the gallery
  const GEN_PX = 168;       // image / label tile size in the data-generator inset
  const HIST_W = 540;       // histogram width
  const HIST_H = 140;       // histogram height (chart area only)

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

  /* Count class pixels across all available samples. Returns
     { counts: [c0..c4], total }. */
  function classHistogram(samples) {
    const counts = [0, 0, 0, 0, 0];
    let total = 0;
    for (let k = 0; k < samples.length; k++) {
      const lbl = samples[k].label;
      for (let i = 0; i < lbl.length; i++) {
        for (let j = 0; j < lbl[0].length; j++) {
          counts[lbl[i][j] | 0]++;
          total++;
        }
      }
    }
    return { counts: counts, total: total };
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64) {
      root.innerHTML = '<p style="opacity:0.5">scene64 data missing.</p>';
      return {};
    }
    const D = window.DATA.scene64;

    root.innerHTML = '';
    root.classList.add('s2-root');
    const wrap = el('div', { class: 's2-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s2-hero' }, wrap);
    el('h1', { text: 'What a segmentation dataset looks like.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Supervision is dense: every pixel has a known correct class.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        '600 training scenes, 100 test. Each scene is a 64×64×3 RGB image. ' +
        'Each label is a 64×64 array of integers in {0..4}. Every pixel ' +
        'is labelled — the model is told the answer at every position.',
    }, hero);

    /* ---- Sample gallery -------------------------------------------- */
    const galleryWrap = el('section', { class: 's2-gallery-wrap' }, wrap);
    el('div', {
      class: 's2-section-title',
      text: 'six sample pairs · image · label',
    }, galleryWrap);
    const gallery = el('div', { class: 's2-gallery' }, galleryWrap);

    const tileHosts = [];
    for (let k = 0; k < D.samples.length; k++) {
      const pair = el('div', { class: 's2-pair' }, gallery);
      el('div', { class: 's2-pair-idx', text: '#' + (k + 1) }, pair);
      const imgs = el('div', { class: 's2-pair-imgs' }, pair);
      const inputCol = el('div', { class: 's2-pair-col' }, imgs);
      el('div', { class: 's2-pair-sub', text: 'image' }, inputCol);
      const inputHost = el('div', { class: 'canvas-host s2-tile-host' }, inputCol);
      const labelCol = el('div', { class: 's2-pair-col' }, imgs);
      el('div', { class: 's2-pair-sub', text: 'label' }, labelCol);
      const labelHost = el('div', { class: 'canvas-host s2-tile-host' }, labelCol);
      tileHosts.push({ input: inputHost, label: labelHost });
    }

    /* ---- Class legend ---------------------------------------------- */
    const legend = el('div', { class: 's2-legend' }, wrap);
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const item = el('div', { class: 's2-legend-item' }, legend);
      el('span', { class: 's2-swatch class-' + CLASS_NAMES[c] }, item);
      el('span', { class: 's2-legend-name', text: CLASS_NAMES[c] }, item);
      el('span', {
        class: 's2-legend-int',
        text: '(int ' + c + ')',
      }, item);
    }

    /* ---- Data-generator inset -------------------------------------- */
    const genWrap = el('section', { class: 's2-gen-wrap' }, wrap);
    el('div', {
      class: 's2-section-title',
      text: 'data generator',
    }, genWrap);
    const genInner = el('div', { class: 's2-gen-inner' }, genWrap);
    const genCol = el('div', { class: 's2-gen-col' }, genInner);
    const genImgs = el('div', { class: 's2-gen-imgs' }, genCol);
    const genInputCell = el('div', { class: 's2-gen-cell' }, genImgs);
    el('div', { class: 's2-pair-sub', text: 'image' }, genInputCell);
    const genInputHost = el('div', { class: 'canvas-host s2-gen-host' }, genInputCell);
    const genLabelCell = el('div', { class: 's2-gen-cell' }, genImgs);
    el('div', { class: 's2-pair-sub', text: 'label' }, genLabelCell);
    const genLabelHost = el('div', { class: 'canvas-host s2-gen-host' }, genLabelCell);

    const genControls = el('div', { class: 's2-gen-controls' }, genCol);
    const genBtn = el('button', {
      type: 'button',
      class: 's2-gen-btn primary',
      text: 'another sample',
    }, genControls);
    const genIdx = el('span', { class: 's2-gen-idx', text: '#1 / ' + D.samples.length }, genControls);

    el('p', {
      class: 's2-gen-note',
      html:
        '<strong>Honest caveat.</strong> In the real precompute, this button ' +
        'would call <code>scene64_data.py</code> and synthesise a fresh scene. ' +
        'In-browser we do not have a JS port of the generator, so this just ' +
        'rotates through the six precomputed samples shown above.',
    }, genWrap);

    /* ---- Histogram ------------------------------------------------- */
    const histWrap = el('section', { class: 's2-hist-wrap' }, wrap);
    el('div', {
      class: 's2-section-title',
      text: 'class frequency · pixels per class',
    }, histWrap);
    const histHost = el('div', { class: 'canvas-host s2-hist-host' }, histWrap);
    el('p', {
      class: 's2-hist-note',
      html:
        'Across these six samples (≈ 24,576 pixels). The full 600-image ' +
        'training set follows the same shape: <strong>sky</strong> and ' +
        '<strong>grass</strong> dominate, <strong>sun</strong> is rare. ' +
        'This imbalance is the seed for the failure-modes scene at the end.',
    }, histWrap);

    /* ---- Caption + controls --------------------------------------- */
    const caption = el('p', { class: 'caption s2-caption' }, wrap);

    const controls = el('div', { class: 'controls s2-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's2-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's2-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', {
      class: 'control-value', text: '0 / ' + (NUM_STEPS - 1),
    }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State ----------------------------------------------------- */
    const state = {
      step: 0,
      genIdx: 0,
    };

    /* ---- Painters -------------------------------------------------- */
    function paintGallery() {
      for (let k = 0; k < D.samples.length; k++) {
        window.Drawing.paintRGB(tileHosts[k].input, D.samples[k].input, PAIR_PX);
        window.Drawing.paintLabelMap(tileHosts[k].label, D.samples[k].label, PAIR_PX);
      }
    }

    function paintGen() {
      const s = D.samples[state.genIdx];
      window.Drawing.paintRGB(genInputHost, s.input, GEN_PX);
      window.Drawing.paintLabelMap(genLabelHost, s.label, GEN_PX);
      genIdx.textContent = '#' + (state.genIdx + 1) + ' / ' + D.samples.length;
    }

    /* Histogram. Hand-drawn so it picks up class colors from CSS, and
       so we can label bars cleanly without pulling in d3. */
    function paintHistogram() {
      histHost.innerHTML = '';
      const margin = { top: 6, right: 12, bottom: 30, left: 70 };
      const W = HIST_W, H = HIST_H + margin.top + margin.bottom;
      const setup = window.Drawing.setupCanvas(histHost, W, H);
      const ctx = setup.ctx;
      const t = window.Drawing.tokens();
      const colors = window.Drawing.readClassColors();

      const hist = classHistogram(D.samples);
      const total = hist.total || 1;
      const fracs = hist.counts.map(function (c) { return c / total; });
      const maxFrac = Math.max.apply(null, fracs) || 1;

      // Background
      ctx.fillStyle = t.bg;
      ctx.fillRect(0, 0, W, H);

      // Plot area
      const innerW = W - margin.left - margin.right;
      const innerH = HIST_H;
      const x0 = margin.left, y0 = margin.top;

      // Axis line (left)
      ctx.strokeStyle = t.rule;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x0, y0 + innerH);
      ctx.lineTo(x0 + innerW, y0 + innerH);
      ctx.stroke();

      // Bars
      const n = CLASS_NAMES.length;
      const slot = innerW / n;
      const barW = slot * 0.62;
      ctx.font = '12px "SF Mono", Menlo, monospace';
      ctx.textBaseline = 'top';

      for (let c = 0; c < n; c++) {
        const f = fracs[c];
        const h = (f / maxFrac) * (innerH - 6);
        const bx = x0 + c * slot + (slot - barW) / 2;
        const by = y0 + innerH - h;
        // Bar fill
        ctx.fillStyle = colors[c] || t.ink;
        ctx.fillRect(bx, by, barW, h);
        // Bar outline
        ctx.strokeStyle = t.rule;
        ctx.strokeRect(bx + 0.5, by + 0.5, barW - 1, h - 1);

        // Class name (below)
        ctx.fillStyle = t.inkSecondary;
        ctx.textAlign = 'center';
        ctx.fillText(CLASS_NAMES[c], bx + barW / 2, y0 + innerH + 6);

        // Percentage (above bar)
        ctx.fillStyle = t.ink;
        ctx.textBaseline = 'bottom';
        ctx.fillText((f * 100).toFixed(1) + '%', bx + barW / 2, by - 3);
        ctx.textBaseline = 'top';
      }

      // Y-axis label
      ctx.save();
      ctx.translate(margin.left - 50, y0 + innerH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = t.inkSecondary;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '11px "SF Mono", Menlo, monospace';
      ctx.fillText('fraction of pixels', 0, 0);
      ctx.restore();
    }

    function captionFor(step) {
      switch (step) {
        case 0: return 'Six (image, label) pairs. The label map is the same shape as the image — every pixel is annotated.';
        case 1: return 'Five classes, color-coded the same way throughout the deepdive.';
        case 2: return 'In the real generator each call would synthesise a fresh scene. Here we cycle through the six precomputed samples — click "another sample".';
        case 3: return 'Class frequencies. Sky and grass dominate; sun is rare. The model will inherit this imbalance unless we correct for it.';
        default: return '';
      }
    }

    /* ---- Render ---------------------------------------------------- */
    function render() {
      const step = state.step;

      // Section visibility
      legend.classList.toggle('s2-visible', step >= 1);
      genWrap.classList.toggle('s2-visible', step >= 2);
      histWrap.classList.toggle('s2-visible', step >= 3);

      // Generator button gating
      genBtn.disabled = step < 2;

      // Repaint dynamic bits
      if (step >= 2) paintGen();
      if (step >= 3) paintHistogram();

      // Caption + controls
      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    /* ---- Wire-up --------------------------------------------------- */
    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });
    genBtn.addEventListener('click', function () {
      if (state.step < 2) return;
      state.genIdx = (state.genIdx + 1) % D.samples.length;
      paintGen();
    });

    /* ---- Initial paint -------------------------------------------- */
    paintGallery();
    paintGen();        // pre-paint so the inset is not empty when revealed
    render();

    /* &run -> auto-advance to step 3. */
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
      onEnter: function () {
        paintGallery();
        if (state.step >= 2) paintGen();
        if (state.step >= 3) paintHistogram();
        render();
      },
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
  window.scenes.scene2 = function (root) { return buildScene(root); };
})();
