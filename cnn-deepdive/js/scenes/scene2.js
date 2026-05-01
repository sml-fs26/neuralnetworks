/* Scene 2 — "The dot product as a match score."

   Pause mid-slide. Pick one (i, j) position; show the 5×5 patch and the
   5×5 kernel; multiply them elementwise (green for positive products,
   red for negative); sum to a scalar; drop that scalar into the response
   heatmap at the corresponding cell. Then slide one position over and
   repeat. Then fast-forward.

   Step engine: state.cursor ∈ {0, …, 8}. `applyStep` is idempotent on
   `state` so prev = (reset → replay up to cursor−1) and cold entry just
   works. `render(state)` paints the scene from scratch.

   Steps:
     0   patch + filter shown at position A; product blank; score blank
     1   reveal the 25 product cells in waves at A
     2   tick the running sum up to s_A
     3   drop s_A into the heatmap at A
     4   kernel slides to position B (one cell right of A)
     5   reveal product cells at B
     6   tick sum at B
     7   drop s_B into heatmap at B
     8   fast-forward: fill the full conv response heatmap

   We use the vertical filter on the cross sample because the vertical bar
   of the cross gives a strong positive at the chosen positions, and a
   visibly negative response at off-bar positions, so green/red read
   pedagogically.
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
  const FIELD_SIZE = 28;
  const KERNEL_K = 5;
  // 5×5 patch / filter / product cards.
  const CARD_PX = 170;
  const CARD_CELL = 34;
  // Bottom row: input thumbnail and response heatmap. 28 cells × 7 px.
  const FIELD_PX = 196;
  const FIELD_CELL = 7;
  // Thermometer.
  const THERM_W = 220;
  const THERM_H = 24;

  // Position A sits the kernel directly on the horizontal arm of the
  // cross so the horizontal filter gives a strong positive (≥ 0).
  // Position B sits the kernel just above the arm so only the suppression
  // row hits the bar — strong negative. Audience sees a green-dominant
  // first match, then a red-dominant second match: the filter selecting
  // what it matches.
  const POS_A = { r: 14, c: 8 };
  const POS_B = { r: 11, c: 8 };

  // Total number of steps the scene exposes. cursor ∈ [0, MAX_CURSOR].
  const MAX_CURSOR = 8;

  // Per-step animation timings (ms). Keep short so the click-step rhythm
  // doesn't lull the audience to sleep.
  const PRODUCT_REVEAL_MS = 60;       // per cell
  const SUM_TICK_MS = 50;             // per cell added to running total
  const FAST_FORWARD_MS = 8;          // per heatmap cell during fast-fwd

  // Step captions — single italic sentence each.
  const CAPTIONS = [
    'A 5×5 patch under the kernel. The match is the dot product of these two pictures.',
    'Multiply, cell by cell. Green where they agree, red where they disagree.',
    'Add the 25 numbers together. The running total grows.',
    'That single scalar is one cell of the response.',
    'Move the kernel to a new spot. The patch underneath is different.',
    'Multiply again. Same kernel, different patch.',
    'Add again. A different scalar — and now negative.',
    'A second cell of the response.',
    'Repeat for every position. The whole response is the kernel sliding everywhere.',
  ];

  function buildScene(root) {
    if (!window.DATA || !window.DATA.handFilters || !window.Drawing || !window.CNN) {
      root.innerHTML = '<p style="opacity:0.5">Scene 2: missing globals.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s2c-root');

    const wrap = el('div', { class: 's2c-wrap' }, root);

    // ----- Hero text -------------------------------------------------------
    const hero = el('div', { class: 'hero s2c-hero' }, wrap);
    el('h1', { text: 'The dot product as a match score.', class: 's2c-h1' }, hero);
    el('p', {
      class: 'subtitle s2c-subtitle',
      text: 'Pause the kernel. Multiply, sum, drop.',
    }, hero);

    // ----- KaTeX formula ---------------------------------------------------
    const formulaHost = el('div', { class: 'formula-block s2c-formula' }, wrap);
    window.Katex.render(
      's_{i,j} \\;=\\; \\sum_{u=0}^{4}\\sum_{v=0}^{4} I_{i+u,\\,j+v} \\,\\cdot\\, K_{u,v}',
      formulaHost, true
    );

    // ----- Top row: patch × filter = product ------------------------------
    const topRow = el('div', { class: 's2c-toprow' }, wrap);

    function buildCard(parent, labelText) {
      const card = el('div', { class: 's2c-card' }, parent);
      el('div', { class: 's2c-card-label', text: labelText }, card);
      const host = el('div', { class: 'canvas-host s2c-card-canvas-host' }, card);
      const cv = window.Drawing.setupCanvas(host, CARD_PX, CARD_PX);
      return { card, cv };
    }

    const patchCard = buildCard(topRow, 'patch');
    const opEq1 = el('div', { class: 's2c-op', text: '×' }, topRow);
    const filterCard = buildCard(topRow, 'filter');
    const opEq2 = el('div', { class: 's2c-op', text: '=' }, topRow);
    const productCard = buildCard(topRow, 'product');
    void opEq1; void opEq2;

    // ----- Bottom row: input thumb + heatmap + thermometer -----------------
    const botRow = el('div', { class: 's2c-botrow' }, wrap);

    const inputCol = el('div', { class: 's2c-bot-col' }, botRow);
    el('div', { class: 's2c-card-label', text: 'input · kernel position' }, inputCol);
    const inputHost = el('div', { class: 'canvas-host s2c-bot-canvas-host' }, inputCol);
    const ic = window.Drawing.setupCanvas(inputHost, FIELD_PX, FIELD_PX);

    const heatCol = el('div', { class: 's2c-bot-col' }, botRow);
    el('div', { class: 's2c-card-label', text: 'response so far' }, heatCol);
    const heatHost = el('div', { class: 'canvas-host s2c-bot-canvas-host' }, heatCol);
    const hc = window.Drawing.setupCanvas(heatHost, FIELD_PX, FIELD_PX);

    const scoreCol = el('div', { class: 's2c-bot-col s2c-score-col' }, botRow);
    el('div', { class: 's2c-card-label', text: 'match score' }, scoreCol);
    const thermHost = el('div', { class: 's2c-therm-host' }, scoreCol);
    const thermBg = el('div', { class: 's2c-therm-bg' }, thermHost);
    const thermPosFill = el('div', { class: 's2c-therm-fill s2c-therm-pos' }, thermHost);
    const thermNegFill = el('div', { class: 's2c-therm-fill s2c-therm-neg' }, thermHost);
    el('div', { class: 's2c-therm-zero' }, thermHost);
    void thermBg;
    const scoreReadout = el('div', { class: 's2c-score-readout', text: '—' }, scoreCol);

    // ----- Caption + footer cue --------------------------------------------
    const caption = el('p', { class: 'caption s2c-caption' }, wrap);

    const foot = el('div', { class: 'footnote s2c-foot' }, wrap);
    foot.innerHTML =
      'Click <kbd>&rarr;</kbd> to take the next step; <kbd>&larr;</kbd> rewinds. ' +
      'There are nine steps in this scene.';

    // ----- Persistent data -------------------------------------------------
    // We use the horizontal filter because it gives a clean +/- contrast
    // between an on-arm position (positive) and an off-arm position
    // (negative) on the cross sample. The filter's middle row is the
    // "match" row; the top and bottom rows suppress non-horizontal stuff.
    const filterKey = 'horizontal';
    const kernel = window.DATA.handFilters[filterKey];
    const inputImg = window.Drawing.makeSample('cross', FIELD_SIZE);
    const fullResponse = window.CNN.conv2d(inputImg, kernel, 2);
    const fullRange = window.CNN.range2D(fullResponse);
    const fullM = Math.max(Math.abs(fullRange.lo), Math.abs(fullRange.hi)) || 1;

    // Patches at A and B (extracted with padding-style top-left of kernel).
    function patchAt(pos) {
      return window.CNN.extractPatch(inputImg, pos.r - 2, pos.c - 2, KERNEL_K, KERNEL_K);
    }
    const patchA = patchAt(POS_A);
    const patchB = patchAt(POS_B);
    const breakA = window.CNN.dotProductBreakdown(patchA, kernel);
    const breakB = window.CNN.dotProductBreakdown(patchB, kernel);

    // Symmetric value range for the product grid so green/red read on the
    // same scale at A and B.
    const prodM = Math.max(
      Math.abs(window.CNN.range2D(breakA.product).lo),
      Math.abs(window.CNN.range2D(breakA.product).hi),
      Math.abs(window.CNN.range2D(breakB.product).lo),
      Math.abs(window.CNN.range2D(breakB.product).hi)
    ) || 1;

    // Score range — used by the thermometer. Symmetric around zero.
    const scoreScale = Math.max(Math.abs(breakA.sum), Math.abs(breakB.sum), 1) * 1.05;

    // ----- Reveal-order plan ----------------------------------------------
    // For each of A and B, the product reveal walks the 5×5 in row-major
    // order; the running sum ticks along the same order.
    function makeOrder() {
      const order = [];
      for (let i = 0; i < KERNEL_K; i++) {
        for (let j = 0; j < KERNEL_K; j++) order.push([i, j]);
      }
      return order;
    }
    const ORDER = makeOrder();

    // ----- State -----------------------------------------------------------
    // The state is fully derivable from `cursor` and the static data above.
    // We also carry sub-step animation progress so render() can paint
    // intermediate frames without re-running the engine.
    const state = {
      cursor: 0,
      // 'subProgress' ∈ [0..25]: how many product cells revealed (during
      // step 1 or 5) or how many cells have ticked into the running sum
      // (during step 2 or 6).
      subProgress: 0,
      // Heatmap accumulator: 28×28 array, NaN where not yet placed.
      heatmap: makeNaNGrid(),
      timerId: null,
      runIntent: hasRunFlag(),    // whether we're auto-stepping
    };

    function makeNaNGrid() {
      const g = window.CNN.zeros2D(FIELD_SIZE, FIELD_SIZE);
      for (let i = 0; i < FIELD_SIZE; i++) {
        for (let j = 0; j < FIELD_SIZE; j++) g[i][j] = NaN;
      }
      return g;
    }

    // ----- Geometry helpers ------------------------------------------------
    function curPos() {
      // Step 0..3 → A; 4..8 → B (during step 4 the kernel slides; we
      // simply show B's position from then on).
      return state.cursor < 4 ? POS_A : POS_B;
    }
    function curPatch() { return state.cursor < 4 ? patchA : patchB; }
    function curBreakdown() { return state.cursor < 4 ? breakA : breakB; }

    // ----- Drawing ---------------------------------------------------------
    function drawPatch() {
      const t = window.Drawing.tokens();
      const cv = patchCard.cv;
      cv.ctx.fillStyle = t.bg;
      cv.ctx.fillRect(0, 0, CARD_PX, CARD_PX);
      // The patch is values in [0, 1] so render with the sequential palette
      // (ink-on-bg). Cell border + integer labels look noisy on a binary
      // image, so we only label nonzero cells lightly.
      const patch = curPatch();
      window.Drawing.drawGrid(cv.ctx, patch, 0, 0, CARD_PX, CARD_PX, {
        cellSize: CARD_CELL,
        diverging: false,
        cellBorder: true,
        valueRange: [0, 1],
      });
      // Overlay numeric labels (0 or 1) so the audience can read off the
      // patch values when comparing against the kernel.
      cv.ctx.fillStyle = t.inkSecondary;
      cv.ctx.font = '12px "SF Mono", Menlo, monospace';
      cv.ctx.textAlign = 'center';
      cv.ctx.textBaseline = 'middle';
      for (let i = 0; i < KERNEL_K; i++) {
        for (let j = 0; j < KERNEL_K; j++) {
          const v = patch[i][j];
          // White text on filled cells, grey on bg cells.
          cv.ctx.fillStyle = v > 0.5 ? t.bg : t.inkSecondary;
          cv.ctx.fillText(
            v > 0.5 ? '1' : '0',
            (j + 0.5) * CARD_CELL,
            (i + 0.5) * CARD_CELL
          );
        }
      }
    }

    function drawFilter() {
      const t = window.Drawing.tokens();
      const cv = filterCard.cv;
      cv.ctx.fillStyle = t.bg;
      cv.ctx.fillRect(0, 0, CARD_PX, CARD_PX);
      window.Drawing.drawGrid(cv.ctx, kernel, 0, 0, CARD_PX, CARD_PX, {
        cellSize: CARD_CELL,
        diverging: true,
        cellBorder: true,
        valueRange: [-2, 2],
        labels: true,
        labelDecimals: 0,
      });
    }

    function drawProduct() {
      const t = window.Drawing.tokens();
      const cv = productCard.cv;
      cv.ctx.fillStyle = t.bg;
      cv.ctx.fillRect(0, 0, CARD_PX, CARD_PX);

      // How many cells have been revealed?
      let revealed;
      if (state.cursor === 0 || state.cursor === 4) {
        revealed = 0;
      } else if (state.cursor === 1 || state.cursor === 5) {
        revealed = state.subProgress;       // animating
      } else {
        // From step 2 onward at A, all 25 are visible until the next slide.
        revealed = ORDER.length;
      }

      const product = curBreakdown().product;
      // Build a partial product grid for drawGrid; un-revealed cells render
      // as bg.
      const partial = window.CNN.zeros2D(KERNEL_K, KERNEL_K);
      for (let k = 0; k < ORDER.length; k++) {
        const [i, j] = ORDER[k];
        partial[i][j] = (k < revealed) ? product[i][j] : 0;
      }

      // Use a custom green/red palette here: positive products → green,
      // negative → red. drawGrid uses --cnn-pos (blue) for positives, so
      // we render the palette manually to stay on tone with the brief.
      const greenHex = readVar('--cnn-green') || t.pos;
      const redHex = readVar('--cnn-neg') || t.neg;
      const neutralHex = readVar('--cnn-neutral') || t.neutral;
      for (let i = 0; i < KERNEL_K; i++) {
        for (let j = 0; j < KERNEL_K; j++) {
          const k = i * KERNEL_K + j;
          const v = (k < revealed) ? product[i][j] : null;
          let fill;
          if (v == null) {
            fill = t.bg;
          } else {
            const m = prodM;
            if (v >= 0) fill = window.Drawing.lerpHex(neutralHex, greenHex, Math.min(1, v / m));
            else fill = window.Drawing.lerpHex(neutralHex, redHex, Math.min(1, -v / m));
          }
          cv.ctx.fillStyle = fill;
          cv.ctx.fillRect(j * CARD_CELL, i * CARD_CELL, CARD_CELL, CARD_CELL);
        }
      }
      // Cell borders.
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
      // Numeric labels for revealed cells.
      cv.ctx.font = '12px "SF Mono", Menlo, monospace';
      cv.ctx.textAlign = 'center';
      cv.ctx.textBaseline = 'middle';
      cv.ctx.fillStyle = t.ink;
      for (let i = 0; i < KERNEL_K; i++) {
        for (let j = 0; j < KERNEL_K; j++) {
          const k = i * KERNEL_K + j;
          if (k >= revealed) continue;
          const v = product[i][j];
          const s = (Math.abs(v) < 1e-9) ? '0' : v.toFixed(0);
          cv.ctx.fillText(s, (j + 0.5) * CARD_CELL, (i + 0.5) * CARD_CELL);
        }
      }
    }

    // Read a CSS variable from :root. Returns trimmed value or null.
    function readVar(name) {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || null;
    }

    function drawInput() {
      const t = window.Drawing.tokens();
      ic.ctx.fillStyle = t.bg;
      ic.ctx.fillRect(0, 0, FIELD_PX, FIELD_PX);
      window.Drawing.drawGrid(ic.ctx, inputImg, 0, 0, FIELD_PX, FIELD_PX, {
        cellSize: FIELD_CELL,
        diverging: false,
        cellBorder: false,
        valueRange: [0, 1],
      });
      ic.ctx.lineWidth = 1;
      ic.ctx.strokeStyle = t.rule;
      ic.ctx.strokeRect(0.5, 0.5, FIELD_PX - 1, FIELD_PX - 1);

      // Kernel rectangle highlight at the current position. The kernel's
      // top-left in input coords is (r - 2, c - 2).
      const pos = curPos();
      const top = (pos.r - 2) * FIELD_CELL;
      const left = (pos.c - 2) * FIELD_CELL;
      const w = KERNEL_K * FIELD_CELL;
      const h = KERNEL_K * FIELD_CELL;
      ic.ctx.lineWidth = 2;
      ic.ctx.strokeStyle = t.pos;
      ic.ctx.strokeRect(left + 0.5, top + 0.5, w, h);
    }

    function drawHeatmap() {
      const t = window.Drawing.tokens();
      hc.ctx.fillStyle = t.bg;
      hc.ctx.fillRect(0, 0, FIELD_PX, FIELD_PX);
      // Grid with NaNs treated as bg.
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
        cellSize: FIELD_CELL,
        diverging: true,
        cellBorder: false,
        valueRange: [-fullM, fullM],
      });
      // Wash unrendered cells.
      hc.ctx.fillStyle = t.bg;
      for (let i = 0; i < FIELD_SIZE; i++) {
        for (let j = 0; j < FIELD_SIZE; j++) {
          if (isNaN(state.heatmap[i][j])) {
            hc.ctx.fillRect(j * FIELD_CELL, i * FIELD_CELL, FIELD_CELL, FIELD_CELL);
          }
        }
      }
      // Outer frame.
      hc.ctx.lineWidth = 1;
      hc.ctx.strokeStyle = t.rule;
      hc.ctx.strokeRect(0.5, 0.5, FIELD_PX - 1, FIELD_PX - 1);
      // Mark the position(s) just placed.
      const just = positionJustPlaced();
      if (just) {
        hc.ctx.lineWidth = 2;
        hc.ctx.strokeStyle = t.ink;
        hc.ctx.strokeRect(
          just.c * FIELD_CELL + 0.5, just.r * FIELD_CELL + 0.5,
          FIELD_CELL - 1, FIELD_CELL - 1
        );
      }
    }

    // Position whose scalar was placed most recently — used to mark the
    // cell on the heatmap. Returns null if no scalar has been placed yet
    // or we've fast-forwarded past individual placements.
    function positionJustPlaced() {
      if (state.cursor === 3) return POS_A;
      if (state.cursor === 4 || state.cursor === 5 || state.cursor === 6) return POS_A;
      if (state.cursor === 7) return POS_B;
      return null;
    }

    function currentRunningSum() {
      // During step 2 (or 6), animate the partial sum from 0 → final;
      // step 3+ shows the full sum at A; step 7+ shows the full sum at B.
      const br = curBreakdown();
      if (state.cursor === 2 || state.cursor === 6) {
        let s = 0;
        const order = ORDER;
        for (let k = 0; k < state.subProgress && k < order.length; k++) {
          const [i, j] = order[k];
          s += br.product[i][j];
        }
        return s;
      }
      if (state.cursor >= 3 && state.cursor < 7) return breakA.sum;
      if (state.cursor >= 7) return breakB.sum;
      return null;
    }

    function drawThermometer() {
      const v = currentRunningSum();
      // Bar widths are positioned with two absolutely-stacked fills (pos
      // grows right, neg grows left) so the zero-line stays anchored.
      // Both fills have width 0 unless the value is on their side.
      let posFrac = 0, negFrac = 0;
      if (v != null) {
        if (v > 0) posFrac = Math.min(1, v / scoreScale);
        if (v < 0) negFrac = Math.min(1, -v / scoreScale);
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

    function drawCaption() {
      caption.textContent = CAPTIONS[Math.min(state.cursor, CAPTIONS.length - 1)];
    }

    function fullRender() {
      drawPatch();
      drawFilter();
      drawProduct();
      drawInput();
      drawHeatmap();
      drawThermometer();
      drawCaption();
    }

    // ----- Step engine -----------------------------------------------------
    // applyStep(targetCursor, animate?) advances the state to `targetCursor`,
    // mutating heatmap and subProgress as appropriate. If `animate` is
    // true, we drive the sub-step animation with setInterval so the user
    // sees the multiply / sum reveal. If false (as in prev → reset+replay),
    // we apply the side-effects instantly.
    function clearTimer() {
      if (state.timerId) {
        clearInterval(state.timerId);
        state.timerId = null;
      }
    }

    function startProductReveal(then) {
      clearTimer();
      state.subProgress = 0;
      const total = ORDER.length;
      state.timerId = setInterval(() => {
        state.subProgress += 1;
        if (state.subProgress >= total) {
          state.subProgress = total;
          clearTimer();
          fullRender();
          if (then) then();
          return;
        }
        fullRender();
      }, PRODUCT_REVEAL_MS);
    }

    function startSumReveal(then) {
      clearTimer();
      state.subProgress = 0;
      const total = ORDER.length;
      state.timerId = setInterval(() => {
        state.subProgress += 1;
        if (state.subProgress >= total) {
          state.subProgress = total;
          clearTimer();
          fullRender();
          if (then) then();
          return;
        }
        fullRender();
      }, SUM_TICK_MS);
    }

    function startFastForward(then) {
      clearTimer();
      // Build the placement order: row-major over the whole 28×28, but
      // skip cells already placed.
      const placements = [];
      for (let i = 0; i < FIELD_SIZE; i++) {
        for (let j = 0; j < FIELD_SIZE; j++) {
          if (isNaN(state.heatmap[i][j])) placements.push([i, j]);
        }
      }
      let k = 0;
      // Fill in batches of 14 (≈ half a row) per tick to feel zippy.
      const BATCH = 14;
      state.timerId = setInterval(() => {
        for (let b = 0; b < BATCH && k < placements.length; b++, k++) {
          const [i, j] = placements[k];
          state.heatmap[i][j] = fullResponse[i][j];
        }
        if (k >= placements.length) {
          clearTimer();
          fullRender();
          if (then) then();
          return;
        }
        fullRender();
      }, FAST_FORWARD_MS);
    }

    // Apply the side effects of step `c` — used both for forward animation
    // and for instant rebuilds (prev / cold entry). `instant` collapses
    // any animation to its final state and skips intervals.
    function applyStepInstant(c) {
      // Effect rules:
      //   step 0: nothing (initial).
      //   step 1: subProgress = 25 (full product visible at A).
      //   step 2: subProgress = 25 (full sum at A).
      //   step 3: heatmap[A] = breakA.sum.
      //   step 4: still subProgress = 25 conceptually but now showing B's
      //           patch+filter; since B's product hasn't been multiplied
      //           we set subProgress = 0 to indicate the product card is
      //           blank again.
      //   step 5: subProgress = 25 at B.
      //   step 6: subProgress = 25 at B.
      //   step 7: heatmap[B] = breakB.sum.
      //   step 8: heatmap fully filled with fullResponse.
      switch (c) {
        case 0: state.subProgress = 0; break;
        case 1: state.subProgress = ORDER.length; break;
        case 2: state.subProgress = ORDER.length; break;
        case 3:
          state.subProgress = ORDER.length;
          state.heatmap[POS_A.r][POS_A.c] = breakA.sum;
          break;
        case 4:
          state.subProgress = 0;
          state.heatmap[POS_A.r][POS_A.c] = breakA.sum;
          break;
        case 5:
          state.subProgress = ORDER.length;
          state.heatmap[POS_A.r][POS_A.c] = breakA.sum;
          break;
        case 6:
          state.subProgress = ORDER.length;
          state.heatmap[POS_A.r][POS_A.c] = breakA.sum;
          break;
        case 7:
          state.subProgress = ORDER.length;
          state.heatmap[POS_A.r][POS_A.c] = breakA.sum;
          state.heatmap[POS_B.r][POS_B.c] = breakB.sum;
          break;
        case 8:
          state.subProgress = ORDER.length;
          for (let i = 0; i < FIELD_SIZE; i++) {
            for (let j = 0; j < FIELD_SIZE; j++) {
              state.heatmap[i][j] = fullResponse[i][j];
            }
          }
          break;
      }
    }

    // Replay the entire history up to (and including) `c` from a clean
    // slate. Used for prev() and cold render().
    function rebuildTo(c) {
      clearTimer();
      state.heatmap = makeNaNGrid();
      state.subProgress = 0;
      for (let k = 0; k <= c; k++) applyStepInstant(k);
      state.cursor = c;
    }

    // Forward step with animation.
    function nextStep() {
      if (state.cursor >= MAX_CURSOR) return false;
      // If a sub-animation is mid-flight, snap it to the end and stop
      // here; the next press starts the following step. This mirrors the
      // behaviour of the gradient-descent viz.
      if (state.timerId) {
        clearTimer();
        applyStepInstant(state.cursor);
        fullRender();
        return true;
      }
      const next = state.cursor + 1;
      state.cursor = next;
      // Side-effect schedule for forward animation:
      switch (next) {
        case 1: case 5: startProductReveal(); break;
        case 2: case 6: startSumReveal(); break;
        case 3: case 7:
          // Drop scalar — no animation needed; just place it.
          applyStepInstant(next);
          fullRender();
          break;
        case 4:
          // Slide kernel — clear product card by resetting subProgress.
          applyStepInstant(next);
          fullRender();
          break;
        case 8:
          startFastForward();
          break;
        default:
          applyStepInstant(next);
          fullRender();
      }
      // Even when we kicked off an interval, render once now so the
      // caption + thermometer reflect the new step number.
      fullRender();
      return true;
    }

    function prevStep() {
      if (state.cursor <= 0) return false;
      rebuildTo(state.cursor - 1);
      fullRender();
      return true;
    }

    // ----- First paint -----------------------------------------------------
    rebuildTo(0);
    fullRender();

    // ----- &run dev affordance --------------------------------------------
    let runTimer = null;
    let runChain = null;
    if (state.runIntent) {
      // Step through cursors 0 → 8 with delays sized to each substep.
      const delays = [400, 1900, 1500, 700, 700, 1900, 1500, 700, 1700];
      let i = 0;
      function tick() {
        if (state.cursor >= MAX_CURSOR) return;
        nextStep();
        i += 1;
        runChain = setTimeout(tick, delays[Math.min(i, delays.length - 1)]);
      }
      runTimer = setTimeout(tick, 250);
    }

    return {
      onEnter() {
        // Repaint without losing state.
        fullRender();
      },
      onLeave() {
        clearTimer();
        if (runTimer) { clearTimeout(runTimer); runTimer = null; }
        if (runChain) { clearTimeout(runChain); runChain = null; }
      },
      onNextKey() {
        // Consume the keystroke if there is more to step through.
        if (state.cursor < MAX_CURSOR) {
          nextStep();
          return true;
        }
        return false;
      },
      onPrevKey() {
        if (state.cursor > 0) {
          prevStep();
          return true;
        }
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene2 = function (root) { return buildScene(root); };
})();
