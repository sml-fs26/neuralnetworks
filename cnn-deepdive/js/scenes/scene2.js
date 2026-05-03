/* Scene 2 — "The dot product as a match score."

   Interactive playground. The student picks an input image, picks a
   filter, and clicks anywhere on the input. The kernel rectangle moves
   to the click; the patch / kernel / product cards update; the running
   sum lands on the score thermometer; the scalar drops into the
   response heatmap at that position. "Compute everywhere" fast-fills
   the rest of the heatmap with the full convolution.

   Click to add positions one at a time; the heatmap accumulates.

   The convolution formula sits at the top with a tiny glossary so the
   audience can read S, I, K.

   `&run` walks a small predetermined click sequence for headless
   capture. */
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
  const FIELD_SIZE = 28;
  const KERNEL_K = 5;
  const CARD_PX = 170;
  const CARD_CELL = 34;
  const FIELD_PX = 196;
  const FIELD_CELL = 7;
  const REVEAL_MS = 25;       // per cell during product reveal animation
  const FAST_FORWARD_MS = 6;  // per heatmap cell during fast-fill

  const SAMPLES = [
    { key: 'cross',      label: 'cross' },
    { key: 'L',          label: 'L shape' },
    { key: 'vertical',   label: 'vertical line' },
    { key: 'horizontal', label: 'horizontal line' },
    { key: 'circle',     label: 'circle' },
    { key: 'triangle',   label: 'triangle' },
  ];

  const FILTERS = [
    { key: 'horizontal', label: 'horizontal' },
    { key: 'vertical',   label: 'vertical' },
    { key: 'diag_down',  label: 'diag ↘' },
    { key: 'diag_up',    label: 'diag ↙' },
    { key: 'dot',        label: 'centered dot' },
    { key: 'ring',       label: 'small ring' },
    { key: 'top_half',   label: 'top half' },
    { key: 'left_half',  label: 'left half' },
  ];

  // Default position: on the horizontal arm of the cross with the
  // horizontal filter — gives a clean positive score on first paint.
  const DEFAULT_POS = { r: 14, c: 13 };

  function readVar(name) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || null;
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.handFilters || !window.Drawing || !window.CNN) {
      root.innerHTML = '<p style="opacity:0.5">Scene 2: missing globals.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s2c-root');

    const wrap = el('div', { class: 's2c-wrap' }, root);

    // ----- Hero ------------------------------------------------------------
    const hero = el('div', { class: 'hero s2c-hero' }, wrap);
    el('h1', { text: 'The dot product as a match score.', class: 's2c-h1' }, hero);
    el('p', {
      class: 'subtitle s2c-subtitle',
      text: 'Pick a position. Multiply, sum, drop.',
    }, hero);

    // ----- KaTeX formula ---------------------------------------------------
    const formulaHost = el('div', { class: 'formula-block s2c-formula' }, wrap);
    window.Katex.render(
      'S_{i,j} \\;=\\; \\sum_{u=0}^{4}\\sum_{v=0}^{4} I_{i+u,\\,j+v} \\,\\cdot\\, K_{u,v}',
      formulaHost, true
    );

    // ----- Glossary --------------------------------------------------------
    const gloss = el('div', { class: 's2c-gloss' }, wrap);
    const glossEntries = [
      ['S', 'response — one number per kernel position'],
      ['I', 'input image — the picture being scanned'],
      ['K', 'kernel — the filter being applied'],
    ];
    glossEntries.forEach(([sym, desc]) => {
      const item = el('div', { class: 's2c-gloss-item' }, gloss);
      el('span', { class: 's2c-gloss-sym', text: sym }, item);
      el('span', { class: 's2c-gloss-desc', text: desc }, item);
    });

    // ----- Picker controls -------------------------------------------------
    const picker = el('div', { class: 's2c-picker' }, wrap);
    const sampleGroup = el('div', { class: 's2c-picker-group' }, picker);
    el('label', { for: 's2c-sample', text: 'image' }, sampleGroup);
    const sampleSel = el('select', { id: 's2c-sample' }, sampleGroup);
    SAMPLES.forEach((s) => {
      const opt = el('option', { value: s.key, text: s.label }, sampleSel);
      if (s.key === 'cross') opt.selected = true;
    });

    const filterGroup = el('div', { class: 's2c-picker-group' }, picker);
    el('label', { for: 's2c-filter', text: 'filter' }, filterGroup);
    const filterSel = el('select', { id: 's2c-filter' }, filterGroup);
    FILTERS.forEach((f) => {
      const opt = el('option', { value: f.key, text: f.label }, filterSel);
      if (f.key === 'horizontal') opt.selected = true;
    });

    const fillBtn = el('button', {
      type: 'button', class: 's2c-fill-btn primary',
      text: 'Compute everywhere',
    }, picker);

    // ----- Top row: patch × filter = product -------------------------------
    const topRow = el('div', { class: 's2c-toprow' }, wrap);

    function buildCard(parent, labelText) {
      const card = el('div', { class: 's2c-card' }, parent);
      el('div', { class: 's2c-card-label', text: labelText }, card);
      const host = el('div', { class: 'canvas-host s2c-card-canvas-host' }, card);
      const cv = window.Drawing.setupCanvas(host, CARD_PX, CARD_PX);
      return { card, cv };
    }

    const patchCard = buildCard(topRow, 'patch');
    el('div', { class: 's2c-op', text: '×' }, topRow);
    const filterCard = buildCard(topRow, 'filter');
    el('div', { class: 's2c-op', text: '=' }, topRow);
    const productCard = buildCard(topRow, 'product');

    // ----- Bottom row: input + heatmap + thermometer -----------------------
    const botRow = el('div', { class: 's2c-botrow' }, wrap);

    const inputCol = el('div', { class: 's2c-bot-col' }, botRow);
    el('div', { class: 's2c-card-label', text: 'input · click to pick a position' }, inputCol);
    const inputHost = el('div', {
      class: 'canvas-host s2c-bot-canvas-host s2c-input-clickable',
    }, inputCol);
    const ic = window.Drawing.setupCanvas(inputHost, FIELD_PX, FIELD_PX);

    const heatCol = el('div', { class: 's2c-bot-col' }, botRow);
    el('div', { class: 's2c-card-label', text: 'response so far' }, heatCol);
    const heatHost = el('div', { class: 'canvas-host s2c-bot-canvas-host' }, heatCol);
    const hc = window.Drawing.setupCanvas(heatHost, FIELD_PX, FIELD_PX);

    const scoreCol = el('div', { class: 's2c-bot-col s2c-score-col' }, botRow);
    el('div', { class: 's2c-card-label', text: 'match score' }, scoreCol);
    const thermHost = el('div', { class: 's2c-therm-host' }, scoreCol);
    el('div', { class: 's2c-therm-bg' }, thermHost);
    const thermPosFill = el('div', { class: 's2c-therm-fill s2c-therm-pos' }, thermHost);
    const thermNegFill = el('div', { class: 's2c-therm-fill s2c-therm-neg' }, thermHost);
    el('div', { class: 's2c-therm-zero' }, thermHost);
    const scoreReadout = el('div', { class: 's2c-score-readout', text: '—' }, scoreCol);

    // ----- Caption + footnote ----------------------------------------------
    const caption = el('p', { class: 'caption s2c-caption' }, wrap);
    caption.textContent =
      'Click anywhere on the input. The kernel lands there; the 25 products fill in; the dot product becomes one cell of the response.';

    const foot = el('div', { class: 'footnote s2c-foot' }, wrap);
    foot.innerHTML =
      'Click any position on the input to compute its score. Press <kbd>Compute everywhere</kbd> to fill the whole response.';

    // ----- State -----------------------------------------------------------
    const state = {
      sampleKey: 'cross',
      filterKey: 'horizontal',
      input: null,
      kernel: null,
      pos: { r: DEFAULT_POS.r, c: DEFAULT_POS.c },
      heatmap: null,
      fullResponse: null,
      fullM: 1,
      scoreScale: 1,
      revealed: KERNEL_K * KERNEL_K,
      timerId: null,
    };

    function makeNaNGrid() {
      const g = window.CNN.zeros2D(FIELD_SIZE, FIELD_SIZE);
      for (let i = 0; i < FIELD_SIZE; i++) {
        for (let j = 0; j < FIELD_SIZE; j++) g[i][j] = NaN;
      }
      return g;
    }

    function clampPos(r, c) {
      return {
        r: Math.max(0, Math.min(FIELD_SIZE - 1, r)),
        c: Math.max(0, Math.min(FIELD_SIZE - 1, c)),
      };
    }

    function clearTimer() {
      if (state.timerId) { clearInterval(state.timerId); state.timerId = null; }
    }

    function rebuild() {
      clearTimer();
      state.input = window.Drawing.makeSample(state.sampleKey, FIELD_SIZE);
      state.kernel = window.DATA.handFilters[state.filterKey];
      state.fullResponse = window.CNN.conv2d(state.input, state.kernel, 2);
      const r = window.CNN.range2D(state.fullResponse);
      state.fullM = Math.max(Math.abs(r.lo), Math.abs(r.hi)) || 1;
      state.scoreScale = Math.max(state.fullM, 1) * 1.05;
      state.heatmap = makeNaNGrid();
      state.pos = { r: DEFAULT_POS.r, c: DEFAULT_POS.c };
      state.revealed = KERNEL_K * KERNEL_K;
      // Seed the heatmap so the score is shown at the default position.
      const br = curBreakdown();
      state.heatmap[state.pos.r][state.pos.c] = br.sum;
    }

    // ----- Breakdown for current position ---------------------------------
    function curBreakdown() {
      const patch = window.CNN.extractPatch(
        state.input, state.pos.r - 2, state.pos.c - 2, KERNEL_K, KERNEL_K
      );
      const br = window.CNN.dotProductBreakdown(patch, state.kernel);
      return { patch, product: br.product, sum: br.sum };
    }

    // ----- Drawing ---------------------------------------------------------
    function drawPatch(br) {
      const t = window.Drawing.tokens();
      const cv = patchCard.cv;
      cv.ctx.fillStyle = t.bg;
      cv.ctx.fillRect(0, 0, CARD_PX, CARD_PX);
      window.Drawing.drawGrid(cv.ctx, br.patch, 0, 0, CARD_PX, CARD_PX, {
        cellSize: CARD_CELL, diverging: false, cellBorder: true, valueRange: [0, 1],
      });
      cv.ctx.font = '12px "SF Mono", Menlo, monospace';
      cv.ctx.textAlign = 'center';
      cv.ctx.textBaseline = 'middle';
      for (let i = 0; i < KERNEL_K; i++) {
        for (let j = 0; j < KERNEL_K; j++) {
          const v = br.patch[i][j];
          cv.ctx.fillStyle = v > 0.5 ? t.bg : t.inkSecondary;
          cv.ctx.fillText(v > 0.5 ? '1' : '0',
            (j + 0.5) * CARD_CELL, (i + 0.5) * CARD_CELL);
        }
      }
    }

    function drawFilter() {
      const t = window.Drawing.tokens();
      const cv = filterCard.cv;
      cv.ctx.fillStyle = t.bg;
      cv.ctx.fillRect(0, 0, CARD_PX, CARD_PX);
      window.Drawing.drawGrid(cv.ctx, state.kernel, 0, 0, CARD_PX, CARD_PX, {
        cellSize: CARD_CELL, diverging: true, cellBorder: true,
        valueRange: [-2, 2], labels: true, labelDecimals: 0,
      });
    }

    function drawProduct(br) {
      const t = window.Drawing.tokens();
      const cv = productCard.cv;
      cv.ctx.fillStyle = t.bg;
      cv.ctx.fillRect(0, 0, CARD_PX, CARD_PX);
      const product = br.product;
      const pmRange = window.CNN.range2D(product);
      const m = Math.max(Math.abs(pmRange.lo), Math.abs(pmRange.hi)) || 1;
      const greenHex = readVar('--cnn-green') || t.pos;
      const redHex = readVar('--cnn-neg') || t.neg;
      const neutralHex = readVar('--cnn-neutral') || t.neutral;
      for (let i = 0; i < KERNEL_K; i++) {
        for (let j = 0; j < KERNEL_K; j++) {
          const k = i * KERNEL_K + j;
          const v = (k < state.revealed) ? product[i][j] : null;
          let fill;
          if (v == null) fill = t.bg;
          else if (v >= 0) fill = window.Drawing.lerpHex(neutralHex, greenHex, Math.min(1, v / m));
          else fill = window.Drawing.lerpHex(neutralHex, redHex, Math.min(1, -v / m));
          cv.ctx.fillStyle = fill;
          cv.ctx.fillRect(j * CARD_CELL, i * CARD_CELL, CARD_CELL, CARD_CELL);
        }
      }
      cv.ctx.strokeStyle = t.rule;
      cv.ctx.lineWidth = 1;
      for (let i = 0; i <= KERNEL_K; i++) {
        cv.ctx.beginPath();
        cv.ctx.moveTo(0, i * CARD_CELL);
        cv.ctx.lineTo(CARD_PX, i * CARD_CELL);
        cv.ctx.stroke();
      }
      for (let j = 0; j <= KERNEL_K; j++) {
        cv.ctx.beginPath();
        cv.ctx.moveTo(j * CARD_CELL, 0);
        cv.ctx.lineTo(j * CARD_CELL, CARD_PX);
        cv.ctx.stroke();
      }
      cv.ctx.font = '12px "SF Mono", Menlo, monospace';
      cv.ctx.textAlign = 'center';
      cv.ctx.textBaseline = 'middle';
      cv.ctx.fillStyle = t.ink;
      for (let i = 0; i < KERNEL_K; i++) {
        for (let j = 0; j < KERNEL_K; j++) {
          const k = i * KERNEL_K + j;
          if (k >= state.revealed) continue;
          const v = product[i][j];
          const s = (Math.abs(v) < 1e-9) ? '0' : v.toFixed(0);
          cv.ctx.fillText(s, (j + 0.5) * CARD_CELL, (i + 0.5) * CARD_CELL);
        }
      }
    }

    function drawInput() {
      const t = window.Drawing.tokens();
      ic.ctx.fillStyle = t.bg;
      ic.ctx.fillRect(0, 0, FIELD_PX, FIELD_PX);
      window.Drawing.drawGrid(ic.ctx, state.input, 0, 0, FIELD_PX, FIELD_PX, {
        cellSize: FIELD_CELL, diverging: false, cellBorder: false, valueRange: [0, 1],
      });
      ic.ctx.lineWidth = 1;
      ic.ctx.strokeStyle = t.rule;
      ic.ctx.strokeRect(0.5, 0.5, FIELD_PX - 1, FIELD_PX - 1);
      const top = (state.pos.r - 2) * FIELD_CELL;
      const left = (state.pos.c - 2) * FIELD_CELL;
      ic.ctx.lineWidth = 2;
      ic.ctx.strokeStyle = t.pos;
      ic.ctx.strokeRect(left + 0.5, top + 0.5, KERNEL_K * FIELD_CELL, KERNEL_K * FIELD_CELL);
    }

    function drawHeatmap() {
      const t = window.Drawing.tokens();
      hc.ctx.fillStyle = t.bg;
      hc.ctx.fillRect(0, 0, FIELD_PX, FIELD_PX);
      const data = new Array(FIELD_SIZE);
      for (let i = 0; i < FIELD_SIZE; i++) {
        const row = new Array(FIELD_SIZE);
        for (let j = 0; j < FIELD_SIZE; j++) {
          const v = state.heatmap[i][j];
          row[j] = isNaN(v) ? 0 : v;
        }
        data[i] = row;
      }
      window.Drawing.drawGrid(hc.ctx, data, 0, 0, FIELD_PX, FIELD_PX, {
        cellSize: FIELD_CELL, diverging: true, cellBorder: false,
        valueRange: [-state.fullM, state.fullM],
      });
      hc.ctx.fillStyle = t.bg;
      for (let i = 0; i < FIELD_SIZE; i++) {
        for (let j = 0; j < FIELD_SIZE; j++) {
          if (isNaN(state.heatmap[i][j])) {
            hc.ctx.fillRect(j * FIELD_CELL, i * FIELD_CELL, FIELD_CELL, FIELD_CELL);
          }
        }
      }
      hc.ctx.lineWidth = 1;
      hc.ctx.strokeStyle = t.rule;
      hc.ctx.strokeRect(0.5, 0.5, FIELD_PX - 1, FIELD_PX - 1);
      hc.ctx.lineWidth = 2;
      hc.ctx.strokeStyle = t.ink;
      hc.ctx.strokeRect(
        state.pos.c * FIELD_CELL + 0.5, state.pos.r * FIELD_CELL + 0.5,
        FIELD_CELL - 1, FIELD_CELL - 1
      );
    }

    function drawThermometer(br) {
      const v = br ? br.sum : null;
      const m = state.scoreScale;
      let posFrac = 0, negFrac = 0;
      if (v != null) {
        if (v > 0) posFrac = Math.min(1, v / m);
        if (v < 0) negFrac = Math.min(1, -v / m);
      }
      thermPosFill.style.width = (posFrac * 50) + '%';
      thermNegFill.style.width = (negFrac * 50) + '%';
      if (v == null) {
        scoreReadout.textContent = '—';
        scoreReadout.classList.remove('s2c-score-pos', 's2c-score-neg');
      } else {
        scoreReadout.textContent = (v >= 0 ? '+' : '') + v.toFixed(1);
        scoreReadout.classList.toggle('s2c-score-pos', v > 0);
        scoreReadout.classList.toggle('s2c-score-neg', v < 0);
      }
    }

    function fullRender() {
      const br = curBreakdown();
      drawPatch(br);
      drawFilter();
      drawProduct(br);
      drawInput();
      drawHeatmap();
      drawThermometer(br);
    }

    // ----- Place at current position --------------------------------------
    function placeAtCurrent(animate) {
      clearTimer();
      const br = curBreakdown();
      state.heatmap[state.pos.r][state.pos.c] = br.sum;
      if (animate) {
        state.revealed = 0;
        const total = KERNEL_K * KERNEL_K;
        state.timerId = setInterval(() => {
          state.revealed += 1;
          if (state.revealed >= total) {
            state.revealed = total;
            clearTimer();
            fullRender();
            return;
          }
          fullRender();
        }, REVEAL_MS);
      } else {
        state.revealed = KERNEL_K * KERNEL_K;
      }
      fullRender();
    }

    // ----- Click handler on input -----------------------------------------
    inputHost.addEventListener('click', (e) => {
      const rect = ic.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const c = Math.floor((x / rect.width) * FIELD_SIZE);
      const r = Math.floor((y / rect.height) * FIELD_SIZE);
      state.pos = clampPos(r, c);
      placeAtCurrent(true);
    });

    // ----- Selectors -------------------------------------------------------
    sampleSel.addEventListener('change', () => {
      state.sampleKey = sampleSel.value;
      rebuild();
      fullRender();
    });
    filterSel.addEventListener('change', () => {
      state.filterKey = filterSel.value;
      rebuild();
      fullRender();
    });

    // ----- Compute-everywhere fast-fill -----------------------------------
    fillBtn.addEventListener('click', () => {
      clearTimer();
      const placements = [];
      for (let i = 0; i < FIELD_SIZE; i++) {
        for (let j = 0; j < FIELD_SIZE; j++) {
          if (isNaN(state.heatmap[i][j])) placements.push([i, j]);
        }
      }
      let k = 0;
      const BATCH = 14;
      state.timerId = setInterval(() => {
        for (let b = 0; b < BATCH && k < placements.length; b++, k++) {
          const [i, j] = placements[k];
          state.heatmap[i][j] = state.fullResponse[i][j];
        }
        if (k >= placements.length) {
          clearTimer();
          fullRender();
          return;
        }
        fullRender();
      }, FAST_FORWARD_MS);
    });

    // ----- First paint -----------------------------------------------------
    rebuild();
    fullRender();

    // ----- &run dev affordance --------------------------------------------
    let runChain = null;
    if (hasRunFlag()) {
      runChain = setTimeout(() => {
        state.pos = { r: 14, c: 13 };
        placeAtCurrent(true);
        runChain = setTimeout(() => {
          state.pos = { r: 11, c: 14 };
          placeAtCurrent(true);
          runChain = setTimeout(() => fillBtn.click(), 1500);
        }, 1700);
      }, 250);
    }

    return {
      onEnter() { fullRender(); },
      onLeave() {
        clearTimer();
        if (runChain) { clearTimeout(runChain); runChain = null; }
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene2 = function (root) { return buildScene(root); };
})();
