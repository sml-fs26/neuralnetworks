/* Scene 3 -- "One filter, one feature map."

   Consolidation moment after scenes 1-2: each of the eight hand-designed
   filters produces a different feature map of the SAME input. Switching
   the input reveals which filter responds to which structure.

   No internal step engine. ArrowRight advances normally.

   `&run` cycles through the six sample shapes once for headless capture. */
(function () {
  'use strict';

  const FILTER_KEYS = [
    'vertical', 'horizontal', 'diag_down', 'diag_up',
    'dot', 'ring', 'top_half', 'left_half',
  ];

  const FILTER_LABELS = {
    vertical: 'vertical',
    horizontal: 'horizontal',
    diag_down: 'diag ↘',     // down-right arrow
    diag_up: 'diag ↙',       // down-left arrow
    dot: 'dot',
    ring: 'ring',
    top_half: 'top half',
    left_half: 'left half',
  };

  const SAMPLE_KEYS = ['cross', 'L', 'vertical', 'horizontal', 'circle', 'triangle'];

  const SAMPLE_LABELS = {
    cross: 'cross',
    L: 'L-shape',
    vertical: 'vertical line',
    horizontal: 'horizontal line',
    circle: 'circle',
    triangle: 'triangle',
  };

  // Canvas dimensions (logical px)
  const INPUT_PX = 140;          // 28x28 -> 5px per cell
  const KERNEL_PX = 56;           // 5x5  -> ~11.2px per cell
  const FMAP_PX = 140;            // 28x28 -> 5px per cell
  const MODAL_INPUT_PX = 280;
  const MODAL_KERNEL_PX = 140;
  const MODAL_FMAP_PX = 280;

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

  /* Raw conv2d. We deliberately do NOT apply ReLU here so the diverging
     palette shows both red (negative response) and blue (positive
     response) — keeping the color story consistent with scenes 1 and 2,
     where the response heatmap was also raw conv output. */
  function computeFmap(input, filter) {
    return window.CNN.conv2d(input, filter, 2);
  }

  function symmetricRange(map) {
    const r = window.CNN.range2D(map);
    return Math.max(Math.abs(r.lo), Math.abs(r.hi)) || 1;
  }

  function drawInputCanvas(host, input) {
    host.innerHTML = '';
    const { ctx, w, h } = window.Drawing.setupCanvas(host, INPUT_PX, INPUT_PX);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    window.Drawing.drawGrid(ctx, input, 0, 0, w, h, {
      diverging: false,
      valueRange: [0, 1],
    });
  }

  function drawKernelCanvas(host, filter, size) {
    host.innerHTML = '';
    const px = size || KERNEL_PX;
    const { ctx, w, h } = window.Drawing.setupCanvas(host, px, px);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    window.Drawing.drawGrid(ctx, filter, 0, 0, w, h, {
      diverging: true,
      cellBorder: true,
    });
  }

  function drawFmapCanvas(host, fmap, vmax, size) {
    host.innerHTML = '';
    const px = size || FMAP_PX;
    const { ctx, w, h } = window.Drawing.setupCanvas(host, px, px);
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, w, h);
    window.Drawing.drawGrid(ctx, fmap, 0, 0, w, h, {
      diverging: true,
      valueRange: [-vmax, vmax],
    });
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.handFilters) {
      root.innerHTML = '<p style="opacity:0.5">handFilters missing.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s3-root');

    const wrap = el('div', { class: 's3-wrap' }, root);

    const hero = el('header', { class: 'hero s3-hero' }, wrap);
    el('h1', { text: 'One filter, one feature map.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Eight hand-designed filters. One input. Eight different views of the same picture.',
    }, hero);

    // ---- Top row: input picker + input canvas ---------------------------
    const top = el('div', { class: 's3-top' }, wrap);

    const pickerGroup = el('div', { class: 's3-picker' }, top);
    el('label', { class: 's3-picker-label', for: 's3-input-select', text: 'Input shape' }, pickerGroup);
    const select = el('select', { id: 's3-input-select', class: 's3-select' }, pickerGroup);
    SAMPLE_KEYS.forEach(k => {
      const o = el('option', { value: k, text: SAMPLE_LABELS[k] }, select);
      if (k === 'cross') o.selected = true;
    });

    const inputBlock = el('div', { class: 's3-input-block' }, top);
    el('div', { class: 's3-input-cap', text: 'input  →' }, inputBlock);
    const inputHost = el('div', { class: 'canvas-host s3-input-host' }, inputBlock);

    // ---- Gallery: 4 columns x 2 rows ------------------------------------
    const gallery = el('div', { class: 's3-gallery' }, wrap);

    const cards = {};
    FILTER_KEYS.forEach(key => {
      const card = el('div', { class: 'card s3-fcard', tabindex: '0',
        role: 'button', 'aria-label': 'Inspect ' + FILTER_LABELS[key] + ' feature map' }, gallery);
      const head = el('div', { class: 's3-fcard-head' }, card);
      const kHost = el('div', { class: 'canvas-host s3-kernel-host' }, head);
      const headLabel = el('div', { class: 's3-fcard-name', text: FILTER_LABELS[key] }, head);
      void headLabel;
      const fHost = el('div', { class: 'canvas-host s3-fmap-host' }, card);
      cards[key] = { card, kHost, fHost };
    });

    el('p', {
      class: 'caption s3-caption',
      text: 'Same input, eight different views.',
    }, wrap);

    el('p', {
      class: 'footnote',
      html: 'Click any card to enlarge. Press <kbd>Esc</kbd> or click outside to close. ' +
            'Use <kbd>→</kbd> to advance to the next scene.',
    }, wrap);

    // ---- Modal overlay (built once, hidden by default) ------------------
    const overlay = el('div', { class: 's3-overlay hidden', role: 'dialog',
      'aria-hidden': 'true' }, root);
    const modal = el('div', { class: 's3-modal' }, overlay);
    const modalClose = el('button', { class: 's3-modal-close', type: 'button',
      'aria-label': 'Close detail view', text: '×' }, modal);
    void modalClose;
    const modalGrid = el('div', { class: 's3-modal-grid' }, modal);

    const mInputCol = el('div', { class: 's3-modal-col' }, modalGrid);
    el('div', { class: 's3-modal-label', text: 'input' }, mInputCol);
    const mInputHost = el('div', { class: 'canvas-host' }, mInputCol);

    const mKernelCol = el('div', { class: 's3-modal-col' }, modalGrid);
    el('div', { class: 's3-modal-label', text: 'filter' }, mKernelCol);
    const mKernelHost = el('div', { class: 'canvas-host' }, mKernelCol);
    const mKernelName = el('div', { class: 's3-modal-fname', text: '' }, mKernelCol);

    const mFmapCol = el('div', { class: 's3-modal-col' }, modalGrid);
    el('div', { class: 's3-modal-label', text: 'feature map' }, mFmapCol);
    const mFmapHost = el('div', { class: 'canvas-host' }, mFmapCol);

    // ---- State ----------------------------------------------------------
    const state = {
      inputName: 'cross',
      input: null,
      fmaps: {},          // key -> 2D array
      vmaxes: {},         // key -> per-card vmax
      modalKey: null,
      runTimer: null,
    };

    function recompute() {
      state.input = window.Drawing.makeSample(state.inputName, 28);
      state.fmaps = {};
      state.vmaxes = {};
      FILTER_KEYS.forEach(k => {
        const m = computeFmap(state.input, window.DATA.handFilters[k]);
        state.fmaps[k] = m;
        // Per-card normalization: each filter's strongest response at
        // full intensity. Cross-filter intensity comparison is not the
        // story here — pattern shape is.
        state.vmaxes[k] = symmetricRange(m);
      });
    }

    function render() {
      drawInputCanvas(inputHost, state.input);
      FILTER_KEYS.forEach(k => {
        drawKernelCanvas(cards[k].kHost, window.DATA.handFilters[k], KERNEL_PX);
        drawFmapCanvas(cards[k].fHost, state.fmaps[k], state.vmaxes[k], FMAP_PX);
      });
      if (state.modalKey) renderModal(state.modalKey);
    }

    function renderModal(key) {
      drawInputCanvas(mInputHost, state.input);
      // override input size for modal
      const inHostCanvas = mInputHost.querySelector('canvas');
      void inHostCanvas;
      // redraw at modal size by clearing host and re-running
      mInputHost.innerHTML = '';
      const inSetup = window.Drawing.setupCanvas(mInputHost, MODAL_INPUT_PX, MODAL_INPUT_PX);
      const it = window.Drawing.tokens();
      inSetup.ctx.fillStyle = it.bg;
      inSetup.ctx.fillRect(0, 0, inSetup.w, inSetup.h);
      window.Drawing.drawGrid(inSetup.ctx, state.input, 0, 0, inSetup.w, inSetup.h,
        { diverging: false, valueRange: [0, 1] });

      drawKernelCanvas(mKernelHost, window.DATA.handFilters[key], MODAL_KERNEL_PX);
      mKernelName.textContent = FILTER_LABELS[key];
      drawFmapCanvas(mFmapHost, state.fmaps[key], state.vmaxes[key], MODAL_FMAP_PX);
    }

    function openModal(key) {
      state.modalKey = key;
      overlay.classList.remove('hidden');
      overlay.setAttribute('aria-hidden', 'false');
      renderModal(key);
    }

    function closeModal() {
      state.modalKey = null;
      overlay.classList.add('hidden');
      overlay.setAttribute('aria-hidden', 'true');
    }

    overlay.addEventListener('click', (e) => {
      // Click outside the modal box (or on the close button) closes it.
      if (e.target === overlay || e.target === modalClose) closeModal();
    });
    window.addEventListener('keydown', (e) => {
      if (state.modalKey != null && e.key === 'Escape') {
        e.preventDefault();
        closeModal();
      }
    });

    // Card click handlers
    FILTER_KEYS.forEach(k => {
      const c = cards[k].card;
      c.addEventListener('click', () => openModal(k));
      c.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          openModal(k);
        }
      });
    });

    // Input picker
    select.addEventListener('change', () => {
      state.inputName = select.value;
      recompute();
      render();
    });

    // ---- &run cycle ----------------------------------------------------
    function startRunCycle() {
      let i = 0;
      function tick() {
        state.inputName = SAMPLE_KEYS[i % SAMPLE_KEYS.length];
        select.value = state.inputName;
        recompute();
        render();
        i++;
        if (i < SAMPLE_KEYS.length) {
          state.runTimer = setTimeout(tick, 700);
        } else {
          state.runTimer = null;
        }
      }
      tick();
    }

    // Initial paint
    recompute();
    render();

    if (readHashFlag('run')) {
      // Headless: defer slightly so the first frame paints, then cycle.
      state.runTimer = setTimeout(startRunCycle, 100);
    }

    return {
      onEnter() { render(); },
      onLeave() {
        if (state.runTimer != null) {
          clearTimeout(state.runTimer);
          state.runTimer = null;
        }
        closeModal();
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene3 = function (root) { return buildScene(root); };
})();
