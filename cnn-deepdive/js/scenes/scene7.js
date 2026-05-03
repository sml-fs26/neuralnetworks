/* Scene 7 -- "Handcrafted vs. learned."

   The philosophical wow. Three steps:
     0  side-by-side conv1: 8 hand-designed (left), 8 learned (right).
     1  layer 2: hand column fades; the right shows 16 learned conv2 filters.
     2  layer 2 top-9: each of 8 conv2 filters with its 9 max-activating
        training images.

   No fabrication: hand filters from DATA.handFilters; learned filters
   from DATA.shapelets.conv1FiltersNormalized and conv2FiltersNormalized;
   top-9 indices via DATA.shapelets.conv2Top9 into trainImagesSample. */
(function () {
  'use strict';

  const NUM_STEPS = 3;

  // Hand filter labels + order (mirrors scene 1).
  const HAND = [
    { key: 'vertical',   label: 'vertical' },
    { key: 'horizontal', label: 'horizontal' },
    { key: 'diag_down',  label: 'diag down' },
    { key: 'diag_up',    label: 'diag up' },
    { key: 'dot',        label: 'centered dot' },
    { key: 'ring',       label: 'small ring' },
    { key: 'top_half',   label: 'top half' },
    { key: 'left_half',  label: 'left half' },
  ];

  const FILTER_PX = 78;       // conv1 cards (5x5, large enough to read)
  const FILTER_PX_DEEP = 60;  // conv2 cards in step 1 (4x4 grid, 16)
  const THUMB_PX = 40;        // top-9 thumbnails (28x28 -> ~40px)
  const FILTER_PX_TOP = 56;   // small filter shown alongside top-9 row
  const TOP_FILTERS_SHOWN = 8;

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

  /* Paint a kernel into a host.
     `mode`:
       'diverging'  -> use the diverging palette (hand filters, signed)
       'normalized' -> use sequential ink-on-bg (already in [0,1])           */
  function paintKernel(host, kernel, px, mode, opts) {
    host.innerHTML = '';
    const { ctx, w, h } = window.Drawing.setupCanvas(host, px, px);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    if (mode === 'diverging') {
      window.Drawing.drawGrid(ctx, kernel, 0, 0, w, h, {
        diverging: true,
        cellBorder: !!(opts && opts.cellBorder),
      });
    } else {
      // 'normalized': stretch to full [lo,hi] range using the diverging
      // palette but centred at the *mean* of the filter, so signed
      // structure remains visible. Most learned conv1 filters in the
      // shapelets net are signed; the pre-normalised version is in [0,1].
      // We map [0,1] -> [-1, 1] so 0.5 reads as neutral.
      const rows = kernel.length, cols = kernel[0].length;
      const signed = new Array(rows);
      for (let i = 0; i < rows; i++) {
        const row = new Array(cols);
        for (let j = 0; j < cols; j++) row[j] = kernel[i][j] * 2 - 1;
        signed[i] = row;
      }
      window.Drawing.drawGrid(ctx, signed, 0, 0, w, h, {
        diverging: true,
        valueRange: [-1, 1],
        cellBorder: !!(opts && opts.cellBorder),
      });
    }
    ctx.lineWidth = 1;
    ctx.strokeStyle = t.rule;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  }

  /* Paint a 28x28 grayscale image (values in [0, 1]). */
  function paintImage(host, img, px) {
    host.innerHTML = '';
    const { ctx, w, h } = window.Drawing.setupCanvas(host, px, px);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    window.Drawing.drawGrid(ctx, img, 0, 0, w, h, {
      diverging: false, valueRange: [0, 1],
    });
    ctx.lineWidth = 1;
    ctx.strokeStyle = t.rule;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.handFilters || !window.DATA.shapelets) {
      root.innerHTML = '<p style="opacity:0.5">Scene 7: data missing.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s7-root');
    const wrap = el('div', { class: 's7-wrap' }, root);

    // ---- Hero ---------------------------------------------------------
    const hero = el('header', { class: 'hero s7-hero' }, wrap);
    el('h1', { text: 'Handcrafted vs. learned.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'You built eight filters by hand. The network built its own. Look at what it built.',
    }, hero);
    el('p', {
      class: 'lede',
      text: 'On the left: the eight 5x5 filters from scene one. On the right: the eight conv1 filters of a network trained from scratch on shapelets. Same job, different origin.',
    }, hero);

    // ---- Two-column board ---------------------------------------------
    const board = el('div', { class: 's7-board' }, wrap);

    // Left column ("hand-designed").
    const leftCol = el('div', { class: 's7-col s7-col-hand' }, board);
    const leftLabel = el('div', { class: 's7-col-title' }, leftCol);
    el('span', { class: 's7-col-eyebrow', text: 'human' }, leftLabel);
    el('span', { class: 's7-col-name', text: 'hand-designed' }, leftLabel);
    const handGrid = el('div', { class: 's7-grid s7-grid-2x4' }, leftCol);
    const leftCaption = el('p', {
      class: 'caption s7-col-caption',
      text: 'Eight filters chosen on purpose. Each one is a shape: a vertical bar, a ring, a top half.',
    }, leftCol);

    // Right column ("learned").
    const rightCol = el('div', { class: 's7-col s7-col-learned' }, board);
    const rightLabel = el('div', { class: 's7-col-title' }, rightCol);
    el('span', { class: 's7-col-eyebrow s7-col-eyebrow-learned', text: 'network' }, rightLabel);
    const rightName = el('span', { class: 's7-col-name', text: 'learned' }, rightLabel);
    const learnedGrid = el('div', { class: 's7-grid s7-grid-2x4' }, rightCol);
    const rightCaption = el('p', {
      class: 'caption s7-col-caption',
      text: 'Eight filters learned by gradient descent on a few thousand training images. The shapes match.',
    }, rightCol);

    // ---- Step controls -------------------------------------------------
    const controls = el('div', { class: 'controls s7-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'depth', for: 's7-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's7-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '1 / 3' }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const goBtn = el('button', {
      type: 'button', class: 'primary', text: 'go deeper',
    }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    // ---- How-were-these-learned card ----------------------------------
    // Pinned context so the audience knows the dataset, architecture, and
    // optimizer that produced the "learned" filters they're looking at.
    const ctxCard = el('div', { class: 's7-context card' }, wrap);
    el('div', { class: 's7-context-title', text: 'How were these learned?' }, ctxCard);
    const ctxRow = el('div', { class: 's7-context-row' }, ctxCard);

    function ctxBlock(parent, title, lines) {
      const block = el('div', { class: 's7-context-block' }, parent);
      el('div', { class: 's7-context-block-title', text: title }, block);
      lines.forEach((ln) => {
        el('div', { class: 's7-context-block-line', text: ln }, block);
      });
    }

    ctxBlock(ctxRow, 'dataset', [
      'shapelets28 — 1500 train · 300 test',
      '6 classes: cross · L · vert / horiz line · circle · triangle',
      '28×28 grayscale · procedurally generated, random rotation/position/thickness',
    ]);
    ctxBlock(ctxRow, 'architecture', [
      'conv1 5×5 → 8 filters, pad 2 → ReLU',
      'max-pool 2×2 (stride 2)',
      'conv2 5×5 → 16 filters, pad 2 → ReLU',
      'max-pool 2×2 (stride 2)',
      'conv3 3×3 → 24 filters, pad 1 → ReLU',
      'global avg-pool · linear → 6 logits',
    ]);
    ctxBlock(ctxRow, 'optimizer', [
      'Adam · lr 1e-3 · batch 64',
      '30 epochs · cross-entropy loss',
      'test accuracy ≈ 99%',
    ]);

    // Caption beneath the whole scene.
    const bottomCaption = el('p', { class: 'caption s7-bottom-caption' }, wrap);

    // Pedagogical sub-caption (the closing line).
    el('p', {
      class: 's7-coda',
      text: 'You built one. The network built dozens. With more data and more layers, the same machinery becomes ImageNet.',
    }, wrap);

    // ---- State --------------------------------------------------------
    const state = { step: 0 };

    function captionFor(step) {
      switch (step) {
        case 0:
          return 'Compare. The network rediscovered most of your toolkit -- not because we told it to, but because these are the shapes that pay off when you reduce loss on shape data.';
        case 1:
          return 'These are not edges. They are curves, junctions, fragments -- the pieces of a shape, not the shape itself.';
        case 2:
          return 'Beyond layer two, filters are best understood by what excites them. Each row: one filter and the nine training images that made it fire hardest.';
        default:
          return '';
      }
    }

    // ---- Painters per step --------------------------------------------

    function renderStep0() {
      // Left column: 8 hand-designed cards (2 rows x 4 cols).
      handGrid.innerHTML = '';
      handGrid.classList.add('s7-grid-2x4');
      handGrid.classList.remove('s7-grid-4x4');
      HAND.forEach((f) => {
        const card = el('div', { class: 's7-card' }, handGrid);
        const cvHost = el('div', { class: 'canvas-host' }, card);
        cvHost.style.width = FILTER_PX + 'px';
        cvHost.style.height = FILTER_PX + 'px';
        paintKernel(cvHost, window.DATA.handFilters[f.key], FILTER_PX, 'diverging',
          { cellBorder: true });
        el('div', { class: 's7-card-label', text: f.label }, card);
      });
      leftCol.classList.remove('s7-faded');
      leftCol.classList.remove('s7-erased');
      leftCaption.textContent =
        'Eight filters chosen on purpose. Each one is a shape: a vertical bar, a ring, a top half.';

      // Right column: 8 learned conv1 cards.
      learnedGrid.innerHTML = '';
      learnedGrid.classList.add('s7-grid-2x4');
      learnedGrid.classList.remove('s7-grid-4x4');
      const conv1Norm = window.DATA.shapelets.conv1FiltersNormalized;
      for (let i = 0; i < conv1Norm.length; i++) {
        const card = el('div', { class: 's7-card s7-card-learned' }, learnedGrid);
        const cvHost = el('div', { class: 'canvas-host' }, card);
        cvHost.style.width = FILTER_PX + 'px';
        cvHost.style.height = FILTER_PX + 'px';
        paintKernel(cvHost, conv1Norm[i], FILTER_PX, 'normalized',
          { cellBorder: true });
        el('div', { class: 's7-card-label', text: 'filter ' + (i + 1) }, card);
      }
      rightName.textContent = 'learned (conv1)';
      rightCaption.textContent =
        'Eight filters learned by gradient descent on a few thousand training images. The shapes match.';
    }

    function renderStep1() {
      // Left column: keep the 8 hand cards but fade them.
      // Caption replaced.
      handGrid.innerHTML = '';
      handGrid.classList.add('s7-grid-2x4');
      handGrid.classList.remove('s7-grid-4x4');
      HAND.forEach((f) => {
        const card = el('div', { class: 's7-card' }, handGrid);
        const cvHost = el('div', { class: 'canvas-host' }, card);
        cvHost.style.width = FILTER_PX + 'px';
        cvHost.style.height = FILTER_PX + 'px';
        paintKernel(cvHost, window.DATA.handFilters[f.key], FILTER_PX, 'diverging',
          { cellBorder: true });
        el('div', { class: 's7-card-label', text: f.label }, card);
      });
      leftCol.classList.add('s7-faded');
      leftCol.classList.remove('s7-erased');
      leftCaption.innerHTML =
        '<em>There were no hand-designed layer-two filters. We never imagined them.</em>';

      // Right column: 16 conv2 filters in a 4x4 grid.
      learnedGrid.innerHTML = '';
      learnedGrid.classList.remove('s7-grid-2x4');
      learnedGrid.classList.add('s7-grid-4x4');
      const conv2Norm = window.DATA.shapelets.conv2FiltersNormalized;
      for (let i = 0; i < conv2Norm.length; i++) {
        const card = el('div', { class: 's7-card s7-card-learned' }, learnedGrid);
        const cvHost = el('div', { class: 'canvas-host' }, card);
        cvHost.style.width = FILTER_PX_DEEP + 'px';
        cvHost.style.height = FILTER_PX_DEEP + 'px';
        paintKernel(cvHost, conv2Norm[i], FILTER_PX_DEEP, 'normalized',
          { cellBorder: false });
        el('div', { class: 's7-card-label', text: 'f' + (i + 1) }, card);
      }
      rightName.textContent = 'learned (conv2)';
      rightCaption.textContent =
        'Sixteen filters at layer two. Mean-collapsed across the eight input channels for display.';
    }

    function renderStep2() {
      // Left column: hide hand grid; show the philosophical caption only.
      handGrid.innerHTML = '';
      leftCol.classList.add('s7-faded');
      leftCol.classList.add('s7-erased');
      leftCaption.innerHTML =
        '<em>You cannot draw a layer-two filter the way you can draw a vertical line. Show it what fires it instead.</em>';

      // Right column: top-9 rows for up to 8 conv2 filters.
      learnedGrid.innerHTML = '';
      learnedGrid.classList.remove('s7-grid-2x4');
      learnedGrid.classList.remove('s7-grid-4x4');
      learnedGrid.classList.add('s7-top9-stack');

      const conv2Norm = window.DATA.shapelets.conv2FiltersNormalized;
      const top9 = window.DATA.shapelets.conv2Top9;
      const train = window.DATA.shapelets.trainImagesSample;
      const total = Math.min(TOP_FILTERS_SHOWN, conv2Norm.length, top9.length);

      for (let i = 0; i < total; i++) {
        const row = el('div', { class: 's7-top9-row' }, learnedGrid);
        const filterCell = el('div', { class: 's7-top9-filter' }, row);
        const fHost = el('div', { class: 'canvas-host' }, filterCell);
        fHost.style.width = FILTER_PX_TOP + 'px';
        fHost.style.height = FILTER_PX_TOP + 'px';
        paintKernel(fHost, conv2Norm[i], FILTER_PX_TOP, 'normalized',
          { cellBorder: false });
        el('div', { class: 's7-top9-tag', text: 'f' + (i + 1) }, filterCell);

        const arrow = el('div', { class: 's7-top9-arrow', text: 'fires on:' }, row);
        void arrow;

        const thumbs = el('div', { class: 's7-top9-thumbs' }, row);
        const indices = top9[i];
        for (let j = 0; j < indices.length; j++) {
          const idx = indices[j];
          const img = train[idx % train.length];
          const tHost = el('div', { class: 'canvas-host s7-top9-thumb' }, thumbs);
          tHost.style.width = THUMB_PX + 'px';
          tHost.style.height = THUMB_PX + 'px';
          paintImage(tHost, img, THUMB_PX);
        }
      }
      rightName.textContent = 'learned (conv2) -- by what excites them';
      rightCaption.textContent =
        'A filter is what makes it fire. These nine training images, in each row, are the network\'s answer to "what does this neuron want to see?".';
    }

    function render() {
      switch (state.step) {
        case 0: renderStep0(); break;
        case 1: renderStep1(); break;
        case 2: renderStep2(); break;
      }
      stepInput.value = String(state.step);
      stepOut.textContent = (state.step + 1) + ' / ' + NUM_STEPS;
      prevBtn.disabled = state.step <= 0;
      goBtn.disabled = state.step >= NUM_STEPS - 1;
      goBtn.textContent =
        state.step === 0 ? 'show layer 2' :
        state.step === 1 ? 'show top-9' : 'go deeper';
      bottomCaption.textContent = captionFor(state.step);
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    prevBtn.addEventListener('click', () => applyStep(state.step - 1));
    goBtn.addEventListener('click',   () => applyStep(state.step + 1));
    resetBtn.addEventListener('click', () => applyStep(0));
    stepInput.addEventListener('input', () => {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    // ---- &run auto-advance --------------------------------------------
    let runTimer = null;
    function autoAdvance() {
      if (state.step >= NUM_STEPS - 1) { runTimer = null; return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(autoAdvance, 1100);
    }

    // First paint.
    render();

    if (readHashFlag('run')) {
      runTimer = setTimeout(autoAdvance, 250);
    }

    return {
      onEnter() { render(); },
      onLeave() {
        if (runTimer) { clearTimeout(runTimer); runTimer = null; }
      },
      onNextKey() {
        if (state.step < NUM_STEPS - 1) { applyStep(state.step + 1); return true; }
        return false;
      },
      onPrevKey() {
        if (state.step > 0) { applyStep(state.step - 1); return true; }
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene7 = function (root) { return buildScene(root); };
})();
