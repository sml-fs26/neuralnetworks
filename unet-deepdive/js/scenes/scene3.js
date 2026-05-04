/* Scene 3 — "Pooling collapses. The decoder must uncollapse."

   The bridge from the bottleneck (scene 5) to the upsampling story
   (scene 6). Makes the symmetry of the U-Net visible: the encoder pools
   (4 cells -> 1 cell, throwing information away); the decoder must
   undo this (1 cell -> 4 cells, but it has to *guess* what was there).
   This is the conceptual frame for the next three scenes.

   Step engine (5 steps):
     0: input shown (a 4×4 toy feature map). Hero + caption.
     1: pooling animation — highlight each 2×2 block and collapse it
        to its max value, building a 2×2 output.
     2: the loss is named explicitly (we kept 4 numbers out of 16).
     3: expanding back — take the 2×2 and "uncollapse" by repeating
        each cell into a 2×2 block, building a 4×4 output (lossy).
     4: side-by-side: original input vs reconstruction, with the diff
        cells marked. Caption foreshadows scene 6.

   No external data dependencies. */
(function () {
  'use strict';

  // ---- DOM helper ----
  function el(tag, attrs, parent) {
    const node = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === 'class') node.className = attrs[k];
      else if (k === 'text') node.textContent = attrs[k];
      else if (k === 'html') node.innerHTML = attrs[k];
      else node.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(node);
    return node;
  }
  function readHashFlag(name) {
    const re = new RegExp('[#&?]' + name + '(?:=([^&]*))?');
    const m = (window.location.hash || '').match(re);
    return m ? (m[1] != null ? m[1] : true) : null;
  }

  // ---- Toy data ----
  // Hand-picked so the max-of-each-block is interesting: each 2×2 block
  // has one clear winner and several losers, making the "info lost" beat
  // visceral.
  const INPUT4x4 = [
    [3, 5, 1, 2],
    [6, 4, 0, 7],
    [2, 8, 5, 1],
    [4, 6, 3, 9],
  ];

  function maxPool(input) {
    const H = input.length, W = input[0].length;
    const oh = H / 2 | 0, ow = W / 2 | 0;
    const out = [];
    for (let i = 0; i < oh; i++) {
      const row = [];
      for (let j = 0; j < ow; j++) {
        let m = -Infinity;
        for (let u = 0; u < 2; u++)
          for (let v = 0; v < 2; v++)
            if (input[i*2+u][j*2+v] > m) m = input[i*2+u][j*2+v];
        row.push(m);
      }
      out.push(row);
    }
    return out;
  }

  function nearestUpsample2x(input) {
    const H = input.length, W = input[0].length;
    const out = [];
    for (let i = 0; i < H * 2; i++) {
      const row = [];
      for (let j = 0; j < W * 2; j++) row.push(input[i >> 1][j >> 1]);
      out.push(row);
    }
    return out;
  }

  const POOLED = maxPool(INPUT4x4);                   // 2×2
  const RECONSTRUCTED = nearestUpsample2x(POOLED);    // 4×4

  // ---- Painter ----
  function paintGrid(host, data, opts) {
    host.innerHTML = '';
    opts = opts || {};
    const rows = data.length, cols = data[0].length;
    const cell = opts.cell || 56;
    const W = cell * cols, H = cell * rows;
    const setup = window.Drawing.setupCanvas(host, W, H);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();
    const accent = opts.accent || '#d97a1f';

    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, W, H);

    let m = 0;
    for (const row of data) for (const v of row) m = Math.max(m, Math.abs(v));
    if (!m) m = 1;
    if (opts.vmax) m = opts.vmax;

    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        ctx.fillStyle = window.Drawing.divergingColor(data[i][j] / m, t);
        ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }

    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    for (let i = 0; i <= rows; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * cell);
      ctx.lineTo(W, i * cell);
      ctx.stroke();
    }
    for (let j = 0; j <= cols; j++) {
      ctx.beginPath();
      ctx.moveTo(j * cell, 0);
      ctx.lineTo(j * cell, H);
      ctx.stroke();
    }

    if (opts.labels !== false) {
      ctx.font = `${Math.max(12, Math.floor(cell * 0.36))}px "SF Mono", Menlo, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const intensity = Math.min(1, Math.abs(data[i][j]) / m);
          ctx.fillStyle = intensity > 0.55 ? t.bg : t.ink;
          ctx.fillText(String(data[i][j]), (j + 0.5) * cell, (i + 0.5) * cell);
        }
      }
    }

    if (opts.diffMask) {
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = accent;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (opts.diffMask[i][j]) {
            ctx.strokeRect(j * cell + 1.25, i * cell + 1.25, cell - 2.5, cell - 2.5);
          }
        }
      }
    }

    if (opts.highlight) {
      const h = opts.highlight;
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = accent;
      ctx.strokeRect(
        h.col * cell + 1.75,
        h.row * cell + 1.75,
        (h.cols || 1) * cell - 3.5,
        (h.rows || 1) * cell - 3.5
      );
    }

    return { W, H };
  }

  function diffMask(a, b) {
    const out = [];
    for (let i = 0; i < a.length; i++) {
      const row = [];
      for (let j = 0; j < a[0].length; j++) row.push(a[i][j] !== b[i][j] ? 1 : 0);
      out.push(row);
    }
    return out;
  }

  // ---- Builder ----
  function buildScene(root) {
    if (!window.Drawing) {
      root.innerHTML = '<p style="opacity:0.5">Scene 3: Drawing missing.</p>';
      return {};
    }
    root.innerHTML = '';
    root.classList.add('s3-root');
    const wrap = el('div', { class: 's3-wrap' }, root);

    const hero = el('header', { class: 'hero s3-hero' }, wrap);
    el('h1', { text: 'Pooling collapses. The decoder must uncollapse.' }, hero);
    el('p', {
      class: 'subtitle',
      text:
        'The encoder used max-pool to halve the resolution at every level. ' +
        'The decoder is going to do the opposite — turn one cell back into many. ' +
        'Three things to understand here, then we look at how it actually does it.',
    }, hero);

    /* "You are here" mini-map: highlight the pool ops on the encoder
       side and the upsample ops on the decoder side, so the viewer sees
       which architectural pieces this scene is talking about. */
    const miniHost = el('div', { class: 's3-mini-host' }, wrap);
    if (window.UNET && typeof window.UNET.mountUNetMiniMap === 'function') {
      const mm = window.UNET.mountUNetMiniMap(miniHost, {
        width: 280, title: 'you are here',
        label: 'pool1, pool2 (encoder · down) ↔ up1, up2 (decoder · up)',
      });
      mm.setHighlight(['pool1', 'pool2', 'up1', 'up2']);
    }

    const main = el('section', { class: 's3-main' }, wrap);

    // ---- Encoder column (pooling) ----
    const encCol = el('div', { class: 's3-col s3-col-enc' }, main);
    el('div', { class: 's3-col-eyebrow', text: 'encoder · max-pool 2×2' }, encCol);
    const encStack = el('div', { class: 's3-col-stack' }, encCol);
    const encInPanel = el('div', { class: 's3-panel' }, encStack);
    el('div', { class: 's3-panel-cap', text: '4 × 4 input feature map' }, encInPanel);
    const encInHost = el('div', { class: 'canvas-host s3-canvas-host' }, encInPanel);
    el('div', { class: 's3-arrow', text: '↓ pool' }, encStack);
    const encOutPanel = el('div', { class: 's3-panel' }, encStack);
    el('div', { class: 's3-panel-cap', text: '2 × 2 pooled (max of each block)' }, encOutPanel);
    const encOutHost = el('div', { class: 'canvas-host s3-canvas-host' }, encOutPanel);

    // ---- Decoder column (upsample) ----
    const decCol = el('div', { class: 's3-col s3-col-dec' }, main);
    el('div', { class: 's3-col-eyebrow', text: 'decoder · upsample 2×' }, decCol);
    const decStack = el('div', { class: 's3-col-stack' }, decCol);
    const decInPanel = el('div', { class: 's3-panel' }, decStack);
    el('div', { class: 's3-panel-cap', text: '2 × 2 input from below' }, decInPanel);
    const decInHost = el('div', { class: 'canvas-host s3-canvas-host' }, decInPanel);
    el('div', { class: 's3-arrow', text: '↑ upsample' }, decStack);
    const decOutPanel = el('div', { class: 's3-panel' }, decStack);
    el('div', { class: 's3-panel-cap', text: '4 × 4 reconstruction' }, decOutPanel);
    const decOutHost = el('div', { class: 'canvas-host s3-canvas-host' }, decOutPanel);

    // ---- Comparison strip (revealed at step 4) ----
    const compare = el('section', { class: 's3-compare' }, wrap);
    el('div', { class: 's3-compare-eyebrow', text: 'what got lost' }, compare);
    const compareBody = el('div', { class: 's3-compare-body' }, compare);
    const cmpOrigPanel = el('div', { class: 's3-cmp-panel' }, compareBody);
    el('div', { class: 's3-panel-cap', text: 'original 4 × 4' }, cmpOrigPanel);
    const cmpOrigHost = el('div', { class: 'canvas-host s3-canvas-host' }, cmpOrigPanel);
    el('div', { class: 's3-vs', text: 'vs' }, compareBody);
    const cmpReconPanel = el('div', { class: 's3-cmp-panel' }, compareBody);
    el('div', { class: 's3-panel-cap', text: 'after pool → upsample' }, cmpReconPanel);
    const cmpReconHost = el('div', { class: 'canvas-host s3-canvas-host' }, cmpReconPanel);
    const cmpNote = el('p', { class: 's3-cmp-note' }, compare);

    const caption = el('p', { class: 'caption s3-caption' }, wrap);
    const controls = el('section', { class: 's3-controls' }, wrap);
    const stepGroup = el('div', { class: 's3-control-group' }, controls);
    el('label', { class: 's3-control-label', text: 'step', for: 's3-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's3-step', type: 'range', min: '0', max: '4', step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 's3-control-value', text: '0 / 4' }, stepGroup);
    const navGroup = el('div', { class: 's3-control-group' }, controls);
    const prevBtn = el('button', { type: 'button', class: 's3-btn', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 's3-btn s3-btn-primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', class: 's3-btn', text: 'reset' }, navGroup);

    const state = { step: 0, poolBlock: -1, decBlock: -1 };

    function captionFor(step) {
      switch (step) {
        case 0: return 'A 4×4 toy feature map. Imagine this is one channel of an encoder activation.';
        case 1: return 'Max-pool walks through 2×2 blocks. Each block keeps only its largest value.';
        case 2: return 'After pooling we have 4 numbers instead of 16. The other 12 are gone — we kept only the maxes.';
        case 3: return 'Now the decoder. To get back to 4×4 it has to put 4 cells where there used to be 1. The simplest guess: copy the value into all 4 cells.';
        case 4: return 'Side-by-side. The reconstruction matches at the max cells (orange-outlined are the misses). The next scene shows three smarter ways to do this — including the one the U-Net actually uses.';
        default: return '';
      }
    }

    function render() {
      const s = state.step;

      let encHL = null;
      if (s === 1 && state.poolBlock >= 0 && state.poolBlock < 4) {
        const r = (state.poolBlock >> 1) * 2;
        const c = (state.poolBlock & 1) * 2;
        encHL = { row: r, col: c, rows: 2, cols: 2 };
      }
      paintGrid(encInHost, INPUT4x4, { highlight: encHL });

      const encOutData = (function () {
        if (s < 1) return [[0, 0], [0, 0]];
        if (s >= 2) return POOLED;
        if (state.poolBlock < 0) return [[0, 0], [0, 0]];
        const out = [[0, 0], [0, 0]];
        for (let k = 0; k <= state.poolBlock && k < 4; k++) {
          const r = (k >> 1), c = (k & 1);
          out[r][c] = POOLED[r][c];
        }
        return out;
      })();
      paintGrid(encOutHost, encOutData, { cell: 64 });

      paintGrid(decInHost, s >= 3 ? POOLED : [[0, 0], [0, 0]], { cell: 64 });

      const decOutData = (function () {
        if (s < 3) return [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
        if (s >= 4) return RECONSTRUCTED;
        if (state.decBlock < 0) return [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
        const out = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
        for (let k = 0; k <= state.decBlock && k < 4; k++) {
          const r = (k >> 1), c = (k & 1);
          for (let u = 0; u < 2; u++)
            for (let v = 0; v < 2; v++)
              out[r*2+u][c*2+v] = POOLED[r][c];
        }
        return out;
      })();
      let decHL = null;
      if (s === 3 && state.decBlock >= 0 && state.decBlock < 4) {
        const r = (state.decBlock >> 1) * 2;
        const c = (state.decBlock & 1) * 2;
        decHL = { row: r, col: c, rows: 2, cols: 2 };
      }
      paintGrid(decOutHost, decOutData, { highlight: decHL });

      if (s >= 4) {
        compare.classList.add('s3-visible');
        paintGrid(cmpOrigHost, INPUT4x4);
        paintGrid(cmpReconHost, RECONSTRUCTED, {
          diffMask: diffMask(INPUT4x4, RECONSTRUCTED),
        });
        let miss = 0;
        const dm = diffMask(INPUT4x4, RECONSTRUCTED);
        for (const r of dm) for (const v of r) if (v) miss++;
        cmpNote.innerHTML =
          '<strong>' + (16 - miss) + ' of 16</strong> cells match the original ' +
          '(the four max-cells of each block). The other <strong>' + miss + '</strong> are wrong — ' +
          'the upsample <em>guessed</em> a value (here, just by copying), and there is no way to recover ' +
          'the truth without extra information. <span class="s3-cmp-handoff">That extra information is exactly what the encoder skip connections will provide later.</span>';
      } else {
        compare.classList.remove('s3-visible');
        cmpNote.textContent = '';
      }

      caption.textContent = captionFor(s);
      stepInput.value = String(s);
      stepOut.textContent = s + ' / 4';
      prevBtn.disabled = s <= 0;
      nextBtn.disabled = s >= 4;
    }

    let blockTimer = null;
    function stopBlockTimer() { if (blockTimer) { clearTimeout(blockTimer); blockTimer = null; } }
    function startPoolBlocks() {
      stopBlockTimer();
      state.poolBlock = -1;
      function tick() {
        state.poolBlock = state.poolBlock + 1;
        if (state.poolBlock >= 4) { state.poolBlock = 3; render(); blockTimer = null; return; }
        render();
        blockTimer = setTimeout(tick, 700);
      }
      blockTimer = setTimeout(tick, 250);
    }
    function startDecBlocks() {
      stopBlockTimer();
      state.decBlock = -1;
      function tick() {
        state.decBlock = state.decBlock + 1;
        if (state.decBlock >= 4) { state.decBlock = 3; render(); blockTimer = null; return; }
        render();
        blockTimer = setTimeout(tick, 700);
      }
      blockTimer = setTimeout(tick, 250);
    }

    function applyStep(n) {
      const wasStep = state.step;
      state.step = Math.max(0, Math.min(4, n));
      if (state.step === 1 && wasStep !== 1) state.poolBlock = -1;
      if (state.step === 3 && wasStep !== 3) state.decBlock = -1;
      render();
      if (state.step === 1) startPoolBlocks();
      else if (state.step === 3) startDecBlocks();
      else stopBlockTimer();
    }

    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    render();

    let runTimer = null;
    function autoAdvance() {
      if (state.step >= 4) { runTimer = null; return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(autoAdvance, 2400);
    }
    if (readHashFlag('run')) {
      runTimer = setTimeout(autoAdvance, 600);
    }

    return {
      onEnter: function () { render(); },
      onLeave: function () { stopBlockTimer(); if (runTimer) clearTimeout(runTimer); runTimer = null; },
      onNextKey: function () { if (state.step < 4) { applyStep(state.step + 1); return true; } return false; },
      onPrevKey: function () { if (state.step > 0) { applyStep(state.step - 1); return true; } return false; },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene3 = function (root) { return buildScene(root); };
})();
