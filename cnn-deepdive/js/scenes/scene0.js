/* Scene 0 — "A convolutional network, end to end."

   The hero. Single panel, no internal steps. A looping decorative banner
   scans one of the hand-designed filters across a sample input while the
   response heatmap fills in. The animation is non-interactive; it sets
   the visual register for the rest of the deepdive.

   Reads:
     window.DATA.handFilters.vertical (and `dot` as a second pass)
     window.Drawing.makeSample, setupCanvas, drawGrid, tokens
     window.CNN.conv2d, range2D
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

  // Detect dev `&run` flag (auto-animate on cold entry).
  function hasRunFlag() {
    return /[#&?]run\b/.test(window.location.hash || '');
  }

  // ----- Banner geometry ---------------------------------------------------
  // The banner is a horizontal triptych:
  //   [filter 5x5]   [input 28x28 with kernel rect]   [heatmap 28x28]
  // Each canvas size is locked in the hero CSS (.s0-banner-canvas) but the
  // logical drawing size below must agree.
  const FILTER_PX = 110;     // 5 cells × 22 px
  const FILTER_CELL = 22;
  const FIELD_PX = 280;      // 28 cells × 10 px
  const FIELD_CELL = 10;
  const KERNEL_K = 5;        // 5×5 hand filters
  const FIELD_SIZE = 28;

  function buildScene(root) {
    if (!window.DATA || !window.DATA.handFilters || !window.Drawing || !window.CNN) {
      root.innerHTML = '<p style="opacity:0.5">Scene 0: missing globals.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s0-root');

    const wrap = el('div', { class: 's0-wrap' }, root);

    // ----- Hero text -------------------------------------------------------
    const hero = el('div', { class: 'hero s0-hero' }, wrap);
    el('h1', { text: 'A convolutional network, end to end.', class: 's0-h1' }, hero);
    el('p', {
      class: 'subtitle s0-subtitle',
      text: 'Eight little pictures. One image. Match scores everywhere.',
    }, hero);
    el('p', {
      class: 'lede s0-lede',
      html:
        'A convolutional network looks for small patterns and reports where it found them. ' +
        'Over the next ten scenes we lay it open: what a filter is, how the dot product turns ' +
        'it into a match score, why stacking filters lets the network see in the abstract, ' +
        'what these things actually learn from data, and how the same machinery paints ' +
        'a label onto every pixel of an image.',
    }, hero);

    // ----- Animated banner -------------------------------------------------
    const banner = el('div', { class: 's0-banner card' }, wrap);

    // Three columns inside the banner.
    const filterCol = el('div', { class: 's0-banner-col' }, banner);
    const filterHost = el('div', { class: 'canvas-host s0-banner-canvas-host' }, filterCol);
    el('div', { class: 'col-label s0-banner-label', text: 'filter' }, filterCol);

    const inputCol = el('div', { class: 's0-banner-col' }, banner);
    const inputHost = el('div', { class: 'canvas-host s0-banner-canvas-host' }, inputCol);
    el('div', { class: 'col-label s0-banner-label', text: 'input' }, inputCol);

    const respCol = el('div', { class: 's0-banner-col' }, banner);
    const respHost = el('div', { class: 'canvas-host s0-banner-canvas-host' }, respCol);
    el('div', { class: 'col-label s0-banner-label', text: 'response' }, respCol);

    // Canvases — locked sizes so headless renders identically.
    const fc = window.Drawing.setupCanvas(filterHost, FILTER_PX, FILTER_PX);
    const ic = window.Drawing.setupCanvas(inputHost, FIELD_PX, FIELD_PX);
    const rc = window.Drawing.setupCanvas(respHost, FIELD_PX, FIELD_PX);

    // ----- Footer cue ------------------------------------------------------
    const cue = el('p', { class: 's0-cue' }, wrap);
    cue.innerHTML =
      'Press <kbd>&rarr;</kbd> to begin, or <kbd>t</kbd> to flip the theme.';

    // ----- Animation state -------------------------------------------------
    // We loop over a small playlist of (filter name, sample name) pairs so
    // re-watching the hero stays visually interesting without losing focus.
    const playlist = [
      { filter: 'vertical', sample: 'cross' },
      { filter: 'horizontal', sample: 'L' },
      { filter: 'dot', sample: 'circle' },
      { filter: 'diag_down', sample: 'triangle' },
    ];

    const state = {
      pi: 0,                 // playlist index
      input: null,           // 28×28 sample
      kernel: null,          // 5×5 filter
      response: null,        // 28×28 conv output
      respRange: { lo: -1, hi: 1 },
      partial: null,         // 28×28 with NaN for unrendered cells
      // Scan position: row, col over the FULL 28×28 with padding=2 so the
      // kernel sweeps the entire image.
      r: 0, c: 0,
      step: 0,
      anim: null,            // requestAnimationFrame id
      timer: null,           // setTimeout id between cycles
      stopped: false,
      lastFrame: 0,
    };

    function loadPair(prefill) {
      const p = playlist[state.pi % playlist.length];
      state.input = window.Drawing.makeSample(p.sample, FIELD_SIZE);
      state.kernel = window.DATA.handFilters[p.filter];
      state.response = window.CNN.conv2d(state.input, state.kernel, 2);
      state.respRange = window.CNN.range2D(state.response);
      // Symmetric range for the diverging palette so positives read clearly.
      const m = Math.max(Math.abs(state.respRange.lo), Math.abs(state.respRange.hi)) || 1;
      state.respRange = { lo: -m, hi: m };
      state.partial = window.CNN.zeros2D(FIELD_SIZE, FIELD_SIZE);
      // If `prefill` is true, copy the full response in so the heatmap
      // reads as "already computed". The kernel rect then just sweeps
      // over the finished result. Otherwise fill with NaN so the response
      // builds up cell by cell during the first sweep.
      if (prefill) {
        for (let i = 0; i < FIELD_SIZE; i++) {
          for (let j = 0; j < FIELD_SIZE; j++) state.partial[i][j] = state.response[i][j];
        }
      } else {
        for (let i = 0; i < FIELD_SIZE; i++) {
          for (let j = 0; j < FIELD_SIZE; j++) state.partial[i][j] = NaN;
        }
      }
      state.r = 0;
      state.c = 0;
    }

    // ----- Drawing ---------------------------------------------------------
    function drawFilter() {
      const t = window.Drawing.tokens();
      fc.ctx.fillStyle = t.bg;
      fc.ctx.fillRect(0, 0, FILTER_PX, FILTER_PX);
      window.Drawing.drawGrid(fc.ctx, state.kernel, 0, 0, FILTER_PX, FILTER_PX, {
        cellSize: FILTER_CELL,
        diverging: true,
        cellBorder: true,
        valueRange: [-2, 2],
      });
    }

    function drawInputAndKernel() {
      const t = window.Drawing.tokens();
      ic.ctx.fillStyle = t.bg;
      ic.ctx.fillRect(0, 0, FIELD_PX, FIELD_PX);
      // Draw the binary input as ink-on-bg.
      window.Drawing.drawGrid(ic.ctx, state.input, 0, 0, FIELD_PX, FIELD_PX, {
        cellSize: FIELD_CELL,
        diverging: false,
        cellBorder: false,
        valueRange: [0, 1],
      });
      // Kernel rectangle outline. The kernel is 5×5 and centered on (r, c)
      // when used with padding=2: top-left corner in input coords is
      // (r - 2, c - 2). Skip drawing when the kernel sits mostly outside
      // the field — the partial-edge overhang is visually noisy.
      if (state.r >= 1 && state.r <= FIELD_SIZE - 2
          && state.c >= 1 && state.c <= FIELD_SIZE - 2) {
        const top = (state.r - 2) * FIELD_CELL;
        const left = (state.c - 2) * FIELD_CELL;
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
      // Draw only the cells that have been swept so far. We render the
      // partial array with the symmetric value range so the colormap is
      // stable across frames.
      // Replace NaNs with 0 for the palette call, but draw a neutral wash
      // over them afterwards.
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
      // Wash over the un-swept cells with the bg color so they read empty.
      rc.ctx.fillStyle = t.bg;
      for (let i = 0; i < FIELD_SIZE; i++) {
        for (let j = 0; j < FIELD_SIZE; j++) {
          if (isNaN(state.partial[i][j])) {
            rc.ctx.fillRect(j * FIELD_CELL, i * FIELD_CELL, FIELD_CELL, FIELD_CELL);
          }
        }
      }
    }

    function fullDraw() {
      drawFilter();
      drawInputAndKernel();
      drawResponse();
    }

    // ----- Animation loop --------------------------------------------------
    // We sweep row-major. To make the heatmap fill quickly we reveal a small
    // batch of cells per animation frame.
    const CELLS_PER_FRAME = 6;
    const SCAN_FRAME_MS = 30;

    function tick(ts) {
      if (state.stopped) return;
      if (!state.lastFrame || ts - state.lastFrame >= SCAN_FRAME_MS) {
        state.lastFrame = ts;
        for (let k = 0; k < CELLS_PER_FRAME; k++) {
          // The partial is pre-filled, so we don't actually need to "reveal"
          // the value — but for the very first sweep on a cold pair we
          // still want the heatmap to BUILD from empty for visual interest.
          // A pre-filled partial array means this assignment is idempotent.
          state.partial[state.r][state.c] = state.response[state.r][state.c];
          state.c += 1;
          if (state.c >= FIELD_SIZE) {
            state.c = 0;
            state.r += 1;
            if (state.r >= FIELD_SIZE) {
              // Cycle complete. Hold for a beat, then advance the playlist
              // and start the next pair (with empty heatmap to rebuild).
              state.r = FIELD_SIZE - 1;
              state.c = FIELD_SIZE - 1;
              fullDraw();
              state.timer = setTimeout(() => {
                if (state.stopped) return;
                state.pi = (state.pi + 1) % playlist.length;
                loadPair(false);  // empty: rebuild for the new pair
                fullDraw();
                state.lastFrame = 0;
                state.anim = requestAnimationFrame(tick);
              }, 1400);
              return;
            }
          }
        }
        fullDraw();
      }
      state.anim = requestAnimationFrame(tick);
    }

    function start() {
      stop();
      state.stopped = false;
      // First sweep starts with the heatmap already filled. The kernel
      // rect sweeps over the finished response; visually this reads as
      // "we already convolved; here's where the kernel is right now."
      // Subsequent pairs (after the playlist advances) start empty and
      // build up, so the audience sees the heatmap *forming* at least
      // once.
      loadPair(true);
      fullDraw();
      state.lastFrame = 0;
      state.anim = requestAnimationFrame(tick);
    }

    function stop() {
      state.stopped = true;
      if (state.anim) {
        cancelAnimationFrame(state.anim);
        state.anim = null;
      }
      if (state.timer) {
        clearTimeout(state.timer);
        state.timer = null;
      }
    }

    // ----- First paint -----------------------------------------------------
    // Always render the first pair fully so cold entry captures something.
    loadPair(true);
    state.r = Math.floor(FIELD_SIZE * 0.55);
    state.c = Math.floor(FIELD_SIZE * 0.55);
    fullDraw();

    // ----- Auto-start on entry ---------------------------------------------
    // Always animate on enter (the hero is the hero). The &run flag just
    // shortens the start delay so headless captures see motion.
    const startDelay = hasRunFlag() ? 200 : 350;
    state.timer = setTimeout(start, startDelay);

    return {
      onEnter() {
        // Kick the loop again if we're returning to the hero.
        if (!state.anim) {
          state.timer = setTimeout(start, 120);
        }
      },
      onLeave() {
        // Always halt the rAF loop; the loop is purely decorative and
        // should not run while the user is on a later scene.
        stop();
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene0 = function (root) { return buildScene(root); };
})();
