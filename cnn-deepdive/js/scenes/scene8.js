/* Scene 8 — "What does this neuron want to see?"

   The activation-maximization payoff. For each of 20 neurons (8 conv1 +
   8 conv2 + 4 conv3) the precomputed `image` field is the synthesized AM
   image. The right panel shows the 9 training images that excite the
   neuron most.

   Implementation note. The brief promises that DATA.AM.neurons[k].top9Indices
   index into DATA.shapelets.trainImagesSample, but the export pipeline
   stored full Xtr indices (max ~1500) while the sample only contains 80
   images. To keep the pedagogical promise without fabricating, we
   recompute top-9 directly: run the actual shapelets CNN on the 80 sample
   images and rank by mean pre-pool activation of (layer, channel). The
   result is real ("what excites this neuron most among the train images
   we have at hand") and uses only DATA + CNN globals.

   Reads:
     window.DATA.AM.neurons       // 20 neurons w/ AM image, layer, channel
     window.DATA.shapelets.{conv1,conv2,conv3,trainImagesSample}
     window.Drawing.drawGrid, setupCanvas, tokens
     window.CNN.shapeletsForward
*/
(function () {
  'use strict';

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

  // --- Geometry --------------------------------------------------------------
  const TAB_PX = 38;             // thumbnail size inside each tab
  const PANEL_PX = 168;          // 6 × 28
  const TOP9_CELL = 52;          // each top-9 thumbnail
  const TOP9_GAP = 6;            // gap between cells
  const TOP9_PX = TOP9_CELL * 3 + TOP9_GAP * 2;  // 168

  // --- Top-9 cache.  Computed once on first need. -----------------------------
  // For each (layer, channel) we cache the indices (into trainImagesSample)
  // sorted by descending mean activation, plus the per-image score. We compute
  // it by running shapeletsForward on every sample image once.
  let _scoreCache = null;

  function computeScores() {
    if (_scoreCache) return _scoreCache;
    const sh = window.DATA.shapelets;
    const samples = sh.trainImagesSample;        // [N][28][28]
    const N = samples.length;
    const weights = sh;                          // shapeletsForward reads .conv1, .conv2, .conv3
    const layers = ['conv1', 'conv2', 'conv3'];
    // scores[layer][channel] = Float64Array(N)
    const scores = { conv1: [], conv2: [], conv3: [] };
    for (let n = 0; n < N; n++) {
      const fmaps = window.CNN.shapeletsForward(samples[n], weights);
      for (const L of layers) {
        const fL = fmaps[L];                     // [C][H][W]
        for (let c = 0; c < fL.length; c++) {
          if (!scores[L][c]) scores[L][c] = new Float64Array(N);
          let s = 0;
          const fmap = fL[c];
          const H = fmap.length, W = fmap[0].length;
          for (let i = 0; i < H; i++) {
            const row = fmap[i];
            for (let j = 0; j < W; j++) s += row[j];
          }
          scores[L][c][n] = s / (H * W);
        }
      }
    }
    _scoreCache = scores;
    return scores;
  }

  function topKIndices(layer, channel, k) {
    const scores = computeScores()[layer][channel];
    if (!scores) return [];
    const order = new Array(scores.length);
    for (let i = 0; i < scores.length; i++) order[i] = i;
    order.sort((a, b) => scores[b] - scores[a]);
    return order.slice(0, k);
  }

  // --- Build ----------------------------------------------------------------

  function buildScene(root) {
    if (!window.DATA || !window.DATA.AM || !window.DATA.shapelets || !window.Drawing || !window.CNN) {
      root.innerHTML = '<p style="opacity:0.5">Scene 8: missing globals.</p>';
      return {};
    }

    const NEURONS = window.DATA.AM.neurons;
    const TRAIN_SAMPLES = window.DATA.shapelets.trainImagesSample;
    const N_NEURONS = NEURONS.length;             // 20

    root.innerHTML = '';
    root.classList.add('s8-root');
    const wrap = el('div', { class: 's8-wrap' }, root);

    // Hero
    const hero = el('div', { class: 'hero s8-hero' }, wrap);
    el('h1', { class: 's8-h1', text: 'What does this neuron want to see?' }, hero);
    el('p', {
      class: 'subtitle s8-subtitle',
      text: 'Activation maximization, on a network you understand end to end.',
    }, hero);
    el('p', {
      class: 'lede s8-lede',
      text: 'For each neuron, find the input that screams the loudest. The result is an image — the essence of what excites that neuron.',
    }, hero);

    // Selector card.
    const selectorCard = el('div', { class: 'card s8-selector' }, wrap);
    const tabRefs = [];
    function addRow(labelText, indices) {
      const row = el('div', { class: 's8-selector-row' }, selectorCard);
      el('div', { class: 's8-selector-label', text: labelText }, row);
      const tabs = el('div', { class: 's8-selector-tabs' }, row);
      for (const i of indices) {
        const btn = el('button', { class: 's8-tab', type: 'button', 'aria-label': `${NEURONS[i].layer} channel ${NEURONS[i].channel}` }, tabs);
        btn.addEventListener('click', () => selectNeuron(i, true));
        const host = el('div', { class: 'canvas-host' }, btn);
        host.style.width = TAB_PX + 'px';
        host.style.height = TAB_PX + 'px';
        const cv = window.Drawing.setupCanvas(host, TAB_PX, TAB_PX);
        tabRefs[i] = { btn, ctx: cv.ctx };
      }
    }
    const conv1Idx = []; const conv2Idx = []; const conv3Idx = [];
    for (let i = 0; i < N_NEURONS; i++) {
      const L = NEURONS[i].layer;
      if (L === 'conv1') conv1Idx.push(i);
      else if (L === 'conv2') conv2Idx.push(i);
      else if (L === 'conv3') conv3Idx.push(i);
    }
    addRow('conv1', conv1Idx);
    addRow('conv2', conv2Idx);
    addRow('conv3', conv3Idx);

    // Display card.
    const display = el('div', { class: 'card s8-display' }, wrap);

    // Left panel — AM image.
    const leftPanel = el('div', { class: 's8-panel' }, display);
    el('div', { class: 's8-panel-label', text: 'what excites this neuron' }, leftPanel);
    const leftHost = el('div', { class: 's8-canvas-host' }, leftPanel);
    const leftCv = window.Drawing.setupCanvas(leftHost, PANEL_PX, PANEL_PX);
    const leftIdent = el('div', { class: 's8-identity' }, leftPanel);

    // Right panel — top-9 grid.
    const rightPanel = el('div', { class: 's8-panel' }, display);
    el('div', { class: 's8-panel-label', text: 'training images that excite it most' }, rightPanel);
    const rightHost = el('div', { class: 's8-canvas-host' }, rightPanel);
    const rightCv = window.Drawing.setupCanvas(rightHost, TOP9_PX, TOP9_PX);
    const rightCount = el('div', { class: 's8-identity' }, rightPanel);

    // Pedagogical caption.
    const pedagogy = el('p', {
      class: 's8-pedagogy',
      text: 'The synthesized image on the left is the essence of what those nine training images share.',
    }, wrap);

    // Navigation.
    const nav = el('div', { class: 's8-nav' }, wrap);
    const prevBtn = el('button', { class: 's8-nav-btn', type: 'button', text: 'Previous neuron' }, nav);
    const nextBtn = el('button', { class: 's8-nav-btn', type: 'button', text: 'Next neuron' }, nav);
    const navKeys = el('span', { class: 's8-nav-keys' }, nav);
    navKeys.innerHTML = 'or use <kbd>&larr;</kbd> <kbd>&rarr;</kbd>';
    prevBtn.addEventListener('click', () => selectNeuron(state.idx - 1, true));
    nextBtn.addEventListener('click', () => selectNeuron(state.idx + 1, true));

    // Footnote.
    el('p', {
      class: 's8-footnote',
      text: 'Activation maximization optimizes a single neuron’s pre-ReLU response by gradient ascent on the input image. Total-variation regularization keeps the result interpretable.',
    }, wrap);

    // ---- Drawing helpers ----
    function drawTab(i) {
      const ref = tabRefs[i];
      if (!ref) return;
      const t = window.Drawing.tokens();
      ref.ctx.fillStyle = t.bg;
      ref.ctx.fillRect(0, 0, TAB_PX, TAB_PX);
      window.Drawing.drawGrid(ref.ctx, NEURONS[i].image, 0, 0, TAB_PX, TAB_PX, {
        diverging: false,
        valueRange: [0, 1],
      });
      ref.btn.classList.toggle('selected', i === state.idx);
    }

    function drawAMPanel() {
      const t = window.Drawing.tokens();
      leftCv.ctx.fillStyle = t.bg;
      leftCv.ctx.fillRect(0, 0, PANEL_PX, PANEL_PX);
      const n = NEURONS[state.idx];
      window.Drawing.drawGrid(leftCv.ctx, n.image, 0, 0, PANEL_PX, PANEL_PX, {
        diverging: false,
        valueRange: [0, 1],
      });
      leftIdent.textContent = `${n.layer} · channel ${n.channel}`;
    }

    function drawTop9Panel() {
      const t = window.Drawing.tokens();
      rightCv.ctx.fillStyle = t.bg;
      rightCv.ctx.fillRect(0, 0, TOP9_PX, TOP9_PX);
      const n = NEURONS[state.idx];
      const idxs = topKIndices(n.layer, n.channel, 9);
      for (let k = 0; k < 9; k++) {
        const r = Math.floor(k / 3), c = k % 3;
        const x = c * (TOP9_CELL + TOP9_GAP);
        const y = r * (TOP9_CELL + TOP9_GAP);
        if (k < idxs.length) {
          const img = TRAIN_SAMPLES[idxs[k]];
          if (img) {
            window.Drawing.drawGrid(rightCv.ctx, img, x, y, TOP9_CELL, TOP9_CELL, {
              diverging: false,
              valueRange: [0, 1],
            });
          }
          rightCv.ctx.strokeStyle = t.rule;
          rightCv.ctx.lineWidth = 1;
          rightCv.ctx.strokeRect(x + 0.5, y + 0.5, TOP9_CELL - 1, TOP9_CELL - 1);
        }
      }
      rightCount.textContent = `top 9 of ${TRAIN_SAMPLES.length} sample images`;
    }

    function renderAll() {
      for (let i = 0; i < N_NEURONS; i++) drawTab(i);
      drawAMPanel();
      drawTop9Panel();
      prevBtn.disabled = state.idx <= 0;
      nextBtn.disabled = state.idx >= N_NEURONS - 1;
    }

    // ---- State + selection ----
    const state = { idx: 0, runTimer: null, runStop: false };

    function selectNeuron(i, stopRun) {
      if (i < 0 || i >= N_NEURONS) return;
      if (stopRun) stopRunMode();
      state.idx = i;
      drawAMPanel();
      drawTop9Panel();
      // Repaint just the two tabs whose selected-class flips. Cheaper.
      for (let k = 0; k < N_NEURONS; k++) {
        if (tabRefs[k]) tabRefs[k].btn.classList.toggle('selected', k === state.idx);
      }
      prevBtn.disabled = state.idx <= 0;
      nextBtn.disabled = state.idx >= N_NEURONS - 1;
    }

    // ---- &run mode: cycle through all 20 neurons ----
    function startRunMode() {
      stopRunMode();
      state.runStop = false;
      const period = 6000 / N_NEURONS;            // ~300ms per neuron
      let k = 0;
      const tick = () => {
        if (state.runStop) return;
        selectNeuron(k % N_NEURONS, false);
        k += 1;
        state.runTimer = setTimeout(tick, period);
      };
      tick();
    }

    function stopRunMode() {
      state.runStop = true;
      if (state.runTimer) {
        clearTimeout(state.runTimer);
        state.runTimer = null;
      }
    }

    // ---- First paint ----
    // computeScores is heavy-ish (80 forward passes); let it run synchronously
    // so the very first draw of the right panel is correct (no flash).
    try { computeScores(); } catch (e) { console.error('Scene 8 score precompute failed:', e); }
    renderAll();

    if (hasRunFlag()) {
      startRunMode();
    }

    // Repaint on theme toggle (palette changes).
    const themeObserver = new MutationObserver(() => renderAll());
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

    return {
      onEnter() {
        renderAll();
      },
      onLeave() {
        stopRunMode();
      },
      onNextKey() {
        if (state.idx < N_NEURONS - 1) {
          selectNeuron(state.idx + 1, true);
          return true;
        }
        return false;
      },
      onPrevKey() {
        if (state.idx > 0) {
          selectNeuron(state.idx - 1, true);
          return true;
        }
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene8 = function (root) { return buildScene(root); };
})();
