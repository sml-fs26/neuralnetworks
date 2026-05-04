/* Scene 14 — "What this U-Net gets right, and what it doesn't."

   Honesty coda. A gallery sorted by accuracy: best on top, worst on the
   bottom. For each entry we show the per-sample pixel accuracy and any
   data-driven annotation. When the test-sample index also appears in
   scene64.samples (so we have inline input/label/pred arrays), we render
   the full visual triple; otherwise we fall back to a metadata-only card.

   At the bottom: a 5×5 row-normalized confusion matrix as a heatmap with
   class labels on both axes.

   Step engine:
     0  gallery sorted (best then worst), no annotations yet
     1  annotations + per-class accuracy bars on each card
     2  confusion matrix appears

   Reads:
     window.DATA.failures.{best, worst}       (index, accuracy, annotation)
     window.DATA.confusion.{classes, matrix}  (5x5 row-normalized)
     window.DATA.scene64.samples[i]           (full sample data when available) */
(function () {
  'use strict';

  const NUM_STEPS = 3;
  const RUN_INTERVAL_MS = 800;

  const PANEL_PX = 132;     // input / GT / pred panels in each gallery card
  const CM_CELL_PX = 64;    // confusion-matrix cell

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

  /* Map test-set index -> entry in scene64.samples, if present. We rely
     on scene64.chosenIndices (preferred) or fall back to assuming the
     first N samples are indices 0..N-1. */
  function buildIndexMap(scene64) {
    const map = {};
    if (!scene64 || !scene64.samples) return map;
    if (scene64.chosenIndices && scene64.chosenIndices.length === scene64.samples.length) {
      for (let i = 0; i < scene64.samples.length; i++) {
        map[scene64.chosenIndices[i]] = scene64.samples[i];
      }
    } else {
      for (let i = 0; i < scene64.samples.length; i++) {
        map[i] = scene64.samples[i];
      }
    }
    return map;
  }

  function diffMaskFor(sample) {
    const H = sample.label.length, W = sample.label[0].length;
    const out = new Array(H);
    for (let i = 0; i < H; i++) {
      const row = new Array(W);
      for (let j = 0; j < W; j++) {
        row[j] = (sample.pred[i][j] !== sample.label[i][j]) ? 1 : 0;
      }
      out[i] = row;
    }
    return out;
  }

  /* ---------------------------------------------------------------------
     Confusion-matrix painter (single canvas; rows = true, cols = pred).
     Theme-aware colors are pulled from CSS variables.
     --------------------------------------------------------------------- */
  function paintConfusionMatrix(host, classes, matrix) {
    host.innerHTML = '';
    const K = classes.length;
    const labelPad = 78;     // left + top label gutter
    const cell = CM_CELL_PX;
    const w = labelPad + K * cell + 12;
    const h = labelPad + K * cell + 12;
    const setup = window.Drawing.setupCanvas(host, w, h);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();

    // Background
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);

    // Cells: lerp between bg (0) and ink (1) for high contrast in both
    // themes. Diagonals will saturate to ink; off-diagonals stay near bg.
    function cellColor(v) {
      // Clamp v to [0, 1].
      const c = Math.max(0, Math.min(1, v));
      return window.Drawing.lerpHex(t.bg, t.ink, c);
    }

    for (let r = 0; r < K; r++) {
      for (let c = 0; c < K; c++) {
        const v = matrix[r][c];
        const x = labelPad + c * cell;
        const y = labelPad + r * cell;
        ctx.fillStyle = cellColor(v);
        ctx.fillRect(x, y, cell, cell);
        // Numeric label inside each cell — flip color so it's legible.
        const v100 = (v * 100);
        const txt = v >= 0.995 ? '100' :
                    v < 0.0005 ? '0' :
                    v >= 0.10 ? v100.toFixed(0) :
                    v100.toFixed(1);
        ctx.fillStyle = (v > 0.5) ? t.bg : t.ink;
        ctx.font = '12px "SF Mono", Menlo, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(txt, x + cell / 2, y + cell / 2);
        // Cell border
        ctx.strokeStyle = t.rule;
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
      }
    }

    // Axis labels: rows on the left (true class), columns on the top (pred class)
    ctx.fillStyle = t.ink;
    ctx.font = '11px "SF Mono", Menlo, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let r = 0; r < K; r++) {
      ctx.fillText(classes[r], labelPad - 8, labelPad + r * cell + cell / 2);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let c = 0; c < K; c++) {
      // Render top labels horizontally (short class names, fits in cell width)
      ctx.fillText(classes[c], labelPad + c * cell + cell / 2, labelPad - 14);
    }
    // Axis titles (italic)
    ctx.fillStyle = t.inkSecondary;
    ctx.font = 'italic 11px "Iowan Old Style", Palatino, Georgia, serif';
    ctx.textAlign = 'center';
    ctx.fillText('predicted →', labelPad + (K * cell) / 2, 12);
    ctx.save();
    ctx.translate(14, labelPad + (K * cell) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('true →', 0, 0);
    ctx.restore();
  }

  /* ---------------------------------------------------------------------
     One gallery card.
     If `sample` is provided, we paint input | GT | pred (with diff mask).
     If not, we render a compact metadata-only card.
     --------------------------------------------------------------------- */
  function buildGalleryCard(parent, entry, sample, kind) {
    const card = el('div', { class: 's14-card s14-card-' + kind }, parent);

    // Header strip: index + accuracy + kind tag
    const head = el('div', { class: 's14-card-head' }, card);
    el('span', { class: 's14-card-tag', text: kind === 'best' ? 'best' : 'worst' }, head);
    el('span', { class: 's14-card-idx', text: '#' + entry.index }, head);
    const accValue = el('span', {
      class: 's14-card-acc',
      text: (entry.accuracy * 100).toFixed(2) + '%',
    }, head);
    if (entry.accuracy >= 0.999) accValue.classList.add('s14-acc-good');
    else if (entry.accuracy < 0.997) accValue.classList.add('s14-acc-bad');

    // Visual body
    if (sample) {
      const visBody = el('div', { class: 's14-card-vis' }, card);
      const inputCol = el('div', { class: 's14-vis-col' }, visBody);
      el('div', { class: 's14-vis-label', text: 'input' }, inputCol);
      const inputHost = el('div', { class: 'canvas-host s14-vis-host' }, inputCol);
      window.Drawing.paintRGB(inputHost, sample.input, PANEL_PX);

      const gtCol = el('div', { class: 's14-vis-col' }, visBody);
      el('div', { class: 'col-label s14-vis-label', text: 'ground truth' }, gtCol);
      const gtHost = el('div', { class: 'canvas-host s14-vis-host' }, gtCol);
      window.Drawing.paintLabelMap(gtHost, sample.label, PANEL_PX);

      const predCol = el('div', { class: 's14-vis-col' }, visBody);
      el('div', { class: 'col-label s14-vis-label', text: 'prediction' }, predCol);
      const predHost = el('div', { class: 'canvas-host s14-vis-host' }, predCol);
      window.Drawing.paintLabelMap(predHost, sample.pred, PANEL_PX, {
        diffMask: diffMaskFor(sample),
      });
    } else {
      // Metadata-only fallback. Tells the viewer the sample exists and what
      // its character is, even though the inline pixels were not exported.
      const meta = el('div', { class: 's14-card-meta' }, card);
      el('div', {
        class: 's14-meta-note',
        text: 'Test sample #' + entry.index +
              ' (full pixel data not in scene64.samples; see annotation below).',
      }, meta);
    }

    // Annotation block (revealed at step >= 1)
    const ann = el('div', { class: 's14-card-ann' }, card);
    const annotation = entry.annotation || {};

    // Notes (auto-generated by precompute/confusion_and_picks.py)
    const notes = annotation.notes || [];
    if (notes.length > 0) {
      const notesEl = el('div', { class: 's14-ann-notes' }, ann);
      for (let i = 0; i < notes.length; i++) {
        el('span', { class: 's14-ann-tag', text: notes[i] }, notesEl);
      }
    } else {
      el('div', { class: 's14-ann-notes-empty', text: 'no notable failure modes flagged' }, ann);
    }

    // Per-class accuracy mini-bars
    const pca = annotation.per_class_accuracy || {};
    const present = annotation.classes_present || Object.keys(pca);
    if (present.length > 0) {
      const bars = el('div', { class: 's14-ann-bars' }, ann);
      for (let i = 0; i < present.length; i++) {
        const cn = present[i];
        const a = pca[cn];
        if (a == null) continue;
        const row = el('div', { class: 's14-bar-row' }, bars);
        el('span', { class: 's14-bar-name class-' + cn, text: cn }, row);
        const track = el('div', { class: 's14-bar-track' }, row);
        const fill = el('div', { class: 's14-bar-fill class-' + cn }, track);
        fill.style.width = (Math.max(0, Math.min(1, a)) * 100).toFixed(1) + '%';
        el('span', { class: 's14-bar-val', text: (a * 100).toFixed(1) + '%' }, row);
      }
    }

    return card;
  }

  /* ---------------------------------------------------------------------
     Builder
     --------------------------------------------------------------------- */
  function buildScene(root) {
    if (!window.DATA || !window.Drawing) {
      root.innerHTML = '<p style="opacity:0.5">Scene 14: missing globals.</p>';
      return {};
    }
    const failures = window.DATA.failures || { best: [], worst: [] };
    const confusion = window.DATA.confusion || null;
    const scene64 = window.DATA.scene64 || null;
    const indexMap = buildIndexMap(scene64);

    root.innerHTML = '';
    root.classList.add('s14-root');
    const wrap = el('div', { class: 's14-wrap' }, root);

    /* ---- Hero ----------------------------------------------------- */
    const hero = el('header', { class: 'hero s14-hero' }, wrap);
    el('h1', { text: "What this U-Net gets right, and what it doesn't." }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Every model has a shape of failure. The cartoon dataset is easy on the whole — the residual errors live on small objects and thin boundaries.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'Below: the three best test samples (top row) and the five worst ' +
        '(bottom row), sorted by per-pixel accuracy. The aggregate accuracy ' +
        'is high — but the U-Net is not magic. Its residual errors are concentrated ' +
        'on the rarest class (sun) and the thinnest objects (person, tree trunk).',
    }, hero);

    /* ---- Gallery (best row + worst row) -------------------------- */
    const gallery = el('div', { class: 's14-gallery' }, wrap);

    // Sort: best descending by accuracy, worst ascending by accuracy.
    const bestSorted = (failures.best || []).slice().sort(function (a, b) {
      return b.accuracy - a.accuracy;
    });
    const worstSorted = (failures.worst || []).slice().sort(function (a, b) {
      return a.accuracy - b.accuracy;
    });

    // BEST row
    const bestRow = el('div', { class: 's14-row s14-row-best' }, gallery);
    el('div', {
      class: 's14-row-label',
      text: 'BEST  ·  high accuracy, easy scenes',
    }, bestRow);
    const bestStrip = el('div', { class: 's14-row-strip' }, bestRow);
    bestSorted.forEach(function (entry) {
      buildGalleryCard(bestStrip, entry, indexMap[entry.index] || null, 'best');
    });

    // WORST row
    const worstRow = el('div', { class: 's14-row s14-row-worst' }, gallery);
    el('div', {
      class: 's14-row-label',
      text: 'WORST  ·  lowest accuracy, hardest scenes',
    }, worstRow);
    const worstStrip = el('div', { class: 's14-row-strip' }, worstRow);
    worstSorted.forEach(function (entry) {
      buildGalleryCard(worstStrip, entry, indexMap[entry.index] || null, 'worst');
    });

    /* ---- Confusion matrix --------------------------------------- */
    const cmSection = el('div', { class: 's14-cm-section' }, wrap);
    const cmHead = el('div', { class: 's14-cm-head' }, cmSection);
    el('h3', { class: 's14-cm-title', text: 'Confusion matrix · 5×5, rows = true class, cols = predicted' }, cmHead);
    el('p', {
      class: 's14-cm-sub',
      text: 'Row-normalized over all 100 test samples. Diagonal cells = correct; off-diagonal = the model said col when truth was row.',
    }, cmHead);
    const cmHost = el('div', { class: 'canvas-host s14-cm-host' }, cmSection);

    /* ---- Caption + controls ------------------------------------- */
    const caption = el('p', { class: 'caption s14-caption' }, wrap);

    const controls = el('div', { class: 'controls s14-controls' }, wrap);
    const navGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step' }, navGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, navGroup);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State + render ----------------------------------------- */
    const state = { step: 0, runTimer: null };

    function captionFor(step) {
      switch (step) {
        case 0: return 'Even the worst samples sit above 99.6% pixel accuracy on this cartoon dataset. The errors are concentrated, not diffuse.';
        case 1: return 'Per-class accuracy and auto-generated notes on each card. The pattern: tiny suns and thin people are where the model bleeds tenths of a percent.';
        case 2: return 'Confusion matrix. Sky and grass are nearly perfect. The model occasionally confuses sun, tree, and person — exactly where pixels are scarce.';
        default: return '';
      }
    }

    function render() {
      const step = state.step;

      // Reveal annotations on cards from step 1 onward.
      gallery.classList.toggle('s14-annotations-on', step >= 1);

      // Reveal confusion matrix at step 2.
      cmSection.classList.toggle('s14-visible', step >= 2);
      if (step >= 2 && confusion && confusion.matrix && !cmSection.dataset.painted) {
        paintConfusionMatrix(cmHost, confusion.classes, confusion.matrix);
        cmSection.dataset.painted = '1';
      } else if (step < 2) {
        cmHost.innerHTML = '';
        cmSection.dataset.painted = '';
      }

      caption.textContent = captionFor(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    /* ---- Re-paint the CM on theme change ------------------------ */
    function repaintCM() {
      if (state.step >= 2 && confusion && confusion.matrix) {
        paintConfusionMatrix(cmHost, confusion.classes, confusion.matrix);
        cmSection.dataset.painted = '1';
      }
    }

    /* ---- Wiring ------------------------------------------------- */
    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });

    // Theme observer: when [data-theme] flips on <html>, re-paint the CM
    // so its colors stay legible.
    const mo = new MutationObserver(function (records) {
      for (const r of records) {
        if (r.type === 'attributes' && r.attributeName === 'data-theme') {
          repaintCM();
        }
      }
    });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

    /* ---- First paint -------------------------------------------- */
    render();

    // &run -> auto-advance 0 -> NUM_STEPS-1.
    if (readHashFlag('run')) {
      function tick() {
        if (state.step >= NUM_STEPS - 1) { state.runTimer = null; return; }
        applyStep(state.step + 1);
        state.runTimer = setTimeout(tick, RUN_INTERVAL_MS);
      }
      state.runTimer = setTimeout(tick, 250);
    }

    return {
      onEnter: function () { render(); repaintCM(); },
      onLeave: function () {
        if (state.runTimer) { clearTimeout(state.runTimer); state.runTimer = null; }
        mo.disconnect();
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
  window.scenes.scene14 = function (root) { return buildScene(root); };
})();
