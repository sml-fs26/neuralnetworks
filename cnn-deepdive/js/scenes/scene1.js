/* Scene 1 — "A filter is a little picture."

   The pedagogical opening. No math yet; pure pattern matching.

   Layout (3 columns):
     left   – 2×4 grid of 8 hand-designed 5×5 filter cards
     center – 28×28 input canvas + sample selector
     right  – 28×28 response heatmap canvas

   Interaction: click a filter → the kernel sweeps over the input from
   top-left to bottom-right; the heatmap fills in cell by cell. Switching
   the input clears the heatmap and the selection.

   &run flag: auto-clicks the vertical filter 200 ms after build for
   headless capture.
*/
(function () {
  'use strict';

  // ----- DOM helpers -------------------------------------------------------
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

  function hasRunFlag() {
    return /[#&?]run\b/.test(window.location.hash || '');
  }

  // ----- Constants ---------------------------------------------------------
  const FILTER_PX = 88;       // 5 cells × ~17.6 px (we round in drawGrid)
  const FILTER_CELL = 17;
  const FIELD_PX = 308;       // 28 cells × 11 px
  const FIELD_CELL = 11;
  const FIELD_SIZE = 28;
  const KERNEL_K = 5;

  // Filter catalog. The keys must match window.DATA.handFilters; the labels
  // are what the audience sees. Order is hand-tuned so the 2×4 grid reads
  // left-to-right, top-to-bottom, simple-to-complex.
  const FILTERS = [
    { key: 'vertical',   label: 'vertical line' },
    { key: 'horizontal', label: 'horizontal line' },
    { key: 'diag_down',  label: 'down-right diagonal' },
    { key: 'diag_up',    label: 'up-right diagonal' },
    { key: 'dot',        label: 'centered dot' },
    { key: 'ring',       label: 'small ring' },
    { key: 'top_half',   label: 'top half' },
    { key: 'left_half',  label: 'left half' },
  ];

  const SAMPLES = [
    { key: 'cross',      label: 'cross' },
    { key: 'L',          label: 'L shape' },
    { key: 'vertical',   label: 'vertical line' },
    { key: 'horizontal', label: 'horizontal line' },
    { key: 'circle',     label: 'circle' },
    { key: 'triangle',   label: 'triangle' },
  ];

  function buildScene(root) {
    if (!window.DATA || !window.DATA.handFilters || !window.Drawing || !window.CNN) {
      root.innerHTML = '<p style="opacity:0.5">Scene 1: missing globals.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s1c-root');

    const wrap = el('div', { class: 's1c-wrap' }, root);

    // ----- Hero text -------------------------------------------------------
    const hero = el('div', { class: 'hero s1c-hero' }, wrap);
    el('h1', { text: 'A filter is a little picture.', class: 's1c-h1' }, hero);
    el('p', {
      class: 'subtitle s1c-subtitle',
      text: 'Pick one. Watch where it lights up.',
    }, hero);

    // ----- Three-column layout --------------------------------------------
    const grid = el('div', { class: 's1c-grid' }, wrap);

    // --- Left: filter library ---
    const leftCol = el('div', { class: 's1c-col s1c-filters-col' }, grid);
    el('div', { class: 'col-label s1c-col-label', text: 'filters' }, leftCol);
    const filterGrid = el('div', { class: 's1c-filter-grid' }, leftCol);

    // --- Center: input + sample selector ---
    const centerCol = el('div', { class: 's1c-col s1c-input-col' }, grid);
    el('div', { class: 'col-label s1c-col-label', text: 'input' }, centerCol);
    const inputHost = el('div', { class: 'canvas-host s1c-input-host' }, centerCol);
    const selectorWrap = el('div', { class: 's1c-selector' }, centerCol);
    el('label', { text: 'shape', for: 's1c-sample-select' }, selectorWrap);
    const sampleSelect = el('select', { id: 's1c-sample-select' }, selectorWrap);
    SAMPLES.forEach((s) => {
      const opt = el('option', { value: s.key, text: s.label }, sampleSelect);
      if (s.key === 'cross') opt.selected = true;
    });

    // --- Right: response heatmap ---
    const rightCol = el('div', { class: 's1c-col s1c-resp-col' }, grid);
    el('div', { class: 'col-label s1c-col-label', text: 'response' }, rightCol);
    const respHost = el('div', { class: 'canvas-host s1c-resp-host' }, rightCol);
    const respHint = el('p', { class: 's1c-resp-hint', text: 'click a filter →' }, rightCol);

    // ----- Caption ---------------------------------------------------------
    const caption = el('p', { class: 'caption s1c-caption' }, wrap);
    caption.textContent =
      'A filter is a 5×5 picture. Slide it across the input. ' +
      'Blue is a match, red is the opposite, cream is indifferent.';

    // ----- Footnote --------------------------------------------------------
    const foot = el('div', { class: 'footnote s1c-foot' }, wrap);
    foot.innerHTML =
      'Click a filter; the kernel sweeps the image and the response fills in. ' +
      'Switch the shape to see how each filter responds. ' +
      '<kbd>&rarr;</kbd> goes to the next scene.';

    // ----- Canvas setup ----------------------------------------------------
    const ic = window.Drawing.setupCanvas(inputHost, FIELD_PX, FIELD_PX);
    const rc = window.Drawing.setupCanvas(respHost, FIELD_PX, FIELD_PX);
    // Lazy: filter cards each get their own small canvas, created below.

    // ----- State -----------------------------------------------------------
    const state = {
      sampleKey: 'cross',
      input: window.Drawing.makeSample('cross', FIELD_SIZE),
      activeFilter: null,        // 'vertical' | ... | null
      response: null,            // 28×28 conv output (full)
      respRange: { lo: -1, hi: 1 },
      partial: null,             // 28×28 with NaN until drawn
      r: 0, c: 0,                // current scan position
      kernelPos: null,           // {r, c} for the input kernel-rect overlay
      timerId: null,
      finished: false,
    };

    // ----- Filter card construction ----------------------------------------
    // Each card hosts a small canvas (filter weight grid) plus a label.
    const filterCanvases = {};

    function buildFilterCard(f) {
      const card = el('button', {
        class: 's1c-filter-card',
        type: 'button',
        'data-filter': f.key,
        'aria-label': 'Apply filter: ' + f.label,
      }, filterGrid);
      const cvHost = el('div', { class: 'canvas-host s1c-filter-canvas-host' }, card);
      const cv = window.Drawing.setupCanvas(cvHost, FILTER_PX, FILTER_PX);
      filterCanvases[f.key] = cv;
      el('div', { class: 's1c-filter-label', text: f.label }, card);
      card.addEventListener('click', () => onFilterClick(f.key));
      return card;
    }

    FILTERS.forEach(buildFilterCard);

    // ----- Drawing ---------------------------------------------------------
    function drawAllFilters() {
      const t = window.Drawing.tokens();
      FILTERS.forEach((f) => {
        const cv = filterCanvases[f.key];
        cv.ctx.fillStyle = t.bg;
        cv.ctx.fillRect(0, 0, FILTER_PX, FILTER_PX);
        const k = window.DATA.handFilters[f.key];
        window.Drawing.drawGrid(cv.ctx, k, 0, 0, FILTER_PX, FILTER_PX, {
          cellSize: FILTER_CELL,
          diverging: true,
          cellBorder: true,
          valueRange: [-2, 2],
        });
      });
    }

    function drawInput() {
      const t = window.Drawing.tokens();
      ic.ctx.fillStyle = t.bg;
      ic.ctx.fillRect(0, 0, FIELD_PX, FIELD_PX);
      window.Drawing.drawGrid(ic.ctx, state.input, 0, 0, FIELD_PX, FIELD_PX, {
        cellSize: FIELD_CELL,
        diverging: false,
        cellBorder: false,
        valueRange: [0, 1],
      });
      // Faint outer border to keep the input visible against the bg.
      ic.ctx.lineWidth = 1;
      ic.ctx.strokeStyle = t.rule;
      ic.ctx.strokeRect(0.5, 0.5, FIELD_PX - 1, FIELD_PX - 1);

      // Overlay the kernel rectangle if a scan is in progress.
      if (state.kernelPos) {
        const { r, c } = state.kernelPos;
        const top = (r - 2) * FIELD_CELL;
        const left = (c - 2) * FIELD_CELL;
        const w = KERNEL_K * FIELD_CELL;
        const h = KERNEL_K * FIELD_CELL;
        ic.ctx.lineWidth = 2;
        ic.ctx.strokeStyle = t.pos;
        ic.ctx.strokeRect(left + 0.5, top + 0.5, w, h);
      }
    }

    function drawResponse() {
      const t = window.Drawing.tokens();
      rc.ctx.fillStyle = t.bg;
      rc.ctx.fillRect(0, 0, FIELD_PX, FIELD_PX);
      if (!state.partial) {
        // Empty placeholder. Render a faint frame so the slot is visible.
        rc.ctx.lineWidth = 1;
        rc.ctx.strokeStyle = t.rule;
        rc.ctx.strokeRect(0.5, 0.5, FIELD_PX - 1, FIELD_PX - 1);
        return;
      }
      // Build a sanitized copy with NaN -> 0; we'll wash un-swept cells.
      const data = new Array(FIELD_SIZE);
      for (let i = 0; i < FIELD_SIZE; i++) {
        const row = new Array(FIELD_SIZE);
        for (let j = 0; j < FIELD_SIZE; j++) {
          const v = state.partial[i][j];
          row[j] = isNaN(v) ? 0 : v;
        }
        data[i] = row;
      }
      window.Drawing.drawGrid(rc.ctx, data, 0, 0, FIELD_PX, FIELD_PX, {
        cellSize: FIELD_CELL,
        diverging: true,
        cellBorder: false,
        valueRange: [state.respRange.lo, state.respRange.hi],
      });
      // Wash over unrendered cells.
      rc.ctx.fillStyle = t.bg;
      for (let i = 0; i < FIELD_SIZE; i++) {
        for (let j = 0; j < FIELD_SIZE; j++) {
          if (isNaN(state.partial[i][j])) {
            rc.ctx.fillRect(j * FIELD_CELL, i * FIELD_CELL, FIELD_CELL, FIELD_CELL);
          }
        }
      }
      rc.ctx.lineWidth = 1;
      rc.ctx.strokeStyle = t.rule;
      rc.ctx.strokeRect(0.5, 0.5, FIELD_PX - 1, FIELD_PX - 1);
    }

    function refreshFilterCardSelection() {
      const cards = filterGrid.querySelectorAll('.s1c-filter-card');
      cards.forEach((c) => {
        if (c.getAttribute('data-filter') === state.activeFilter) {
          c.classList.add('selected');
        } else {
          c.classList.remove('selected');
        }
      });
    }

    function refreshHint() {
      if (state.activeFilter) {
        respHint.classList.add('hidden');
      } else {
        respHint.classList.remove('hidden');
      }
    }

    function fullRender() {
      drawInput();
      drawResponse();
      refreshFilterCardSelection();
      refreshHint();
    }

    // ----- Convolution sweep animation ------------------------------------
    function stopSweep() {
      if (state.timerId) {
        clearInterval(state.timerId);
        state.timerId = null;
      }
    }

    function onFilterClick(key) {
      stopSweep();
      state.activeFilter = key;
      const k = window.DATA.handFilters[key];
      state.response = window.CNN.conv2d(state.input, k, 2);
      const r = window.CNN.range2D(state.response);
      const m = Math.max(Math.abs(r.lo), Math.abs(r.hi)) || 1;
      state.respRange = { lo: -m, hi: m };
      state.partial = window.CNN.zeros2D(FIELD_SIZE, FIELD_SIZE);
      for (let i = 0; i < FIELD_SIZE; i++) {
        for (let j = 0; j < FIELD_SIZE; j++) state.partial[i][j] = NaN;
      }
      state.r = 0;
      state.c = 0;
      state.kernelPos = { r: 0, c: 0 };
      state.finished = false;
      refreshFilterCardSelection();
      refreshHint();
      // Begin scanning. ~28×28 = 784 cells; we want the sweep to feel
      // present but not slow. 12 cells/tick at 30 ms tick ≈ 2.0 s total.
      const CELLS_PER_TICK = 12;
      const TICK_MS = 30;
      state.timerId = setInterval(() => {
        for (let s = 0; s < CELLS_PER_TICK; s++) {
          state.partial[state.r][state.c] = state.response[state.r][state.c];
          state.kernelPos = { r: state.r, c: state.c };
          state.c += 1;
          if (state.c >= FIELD_SIZE) {
            state.c = 0;
            state.r += 1;
            if (state.r >= FIELD_SIZE) {
              state.r = FIELD_SIZE - 1;
              state.c = FIELD_SIZE - 1;
              state.kernelPos = null; // hide the rect once done
              state.finished = true;
              stopSweep();
              fullRender();
              return;
            }
          }
        }
        fullRender();
      }, TICK_MS);
      fullRender();
    }

    // ----- Sample switch ---------------------------------------------------
    function onSampleChange() {
      stopSweep();
      const key = sampleSelect.value;
      state.sampleKey = key;
      state.input = window.Drawing.makeSample(key, FIELD_SIZE);
      state.activeFilter = null;
      state.response = null;
      state.partial = null;
      state.kernelPos = null;
      state.finished = false;
      fullRender();
    }
    sampleSelect.addEventListener('change', onSampleChange);

    // ----- First paint -----------------------------------------------------
    drawAllFilters();
    fullRender();

    // ----- &run dev affordance --------------------------------------------
    let runTimer = null;
    if (hasRunFlag()) {
      runTimer = setTimeout(() => onFilterClick('vertical'), 200);
    }

    return {
      onEnter() {
        // If returning, repaint without losing state.
        drawAllFilters();
        fullRender();
      },
      onLeave() {
        stopSweep();
        if (runTimer) {
          clearTimeout(runTimer);
          runTimer = null;
        }
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene1 = function (root) { return buildScene(root); };
})();
