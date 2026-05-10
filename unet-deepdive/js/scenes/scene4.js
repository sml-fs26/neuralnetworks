/* Scene 4 -- "The encoder is the CNN you already know."

   Brisk reuse, not a rebuild. Three encoder levels, each shown as a pair:
   the post-conv-block feature map, then the pool output that feeds the
   next level. Sizes are proportional to spatial extent so the halving
   reads at a glance.

   For one selected sample (a small thumb selector lets the viewer switch
   among the 6), light each tensor in sequence using `paintFeatureCard`
   (4 channels per level, top-variance picks).

   Step engine:
     0 = input only
     1 = enc1 lit (64x64x16, 4 ch)
     2 = pool1 lit (32x32x16, 4 ch)   -- max-pool of enc1
     3 = enc2 lit (32x32x32, 4 ch)
     4 = pool2 lit (16x16x32, 4 ch)   -- max-pool of enc2
     5 = enc3 lit (16x16x64, 4 ch)    -- the bottleneck

   Pool outputs are not stored in scene64; we compute them on-the-fly
   per channel from the conv-block previews. That is faithful per
   channel (the model uses nn.MaxPool2d(2)) but is technically the
   pool of a top-variance subset rather than the literal post-permutation
   indexing -- a footnote in the caption acknowledges this. */
(function () {
  'use strict';

  const NUM_STEPS = 6;
  const RUN_INTERVAL_MS = 600;

  // Sizes proportional to spatial extent. Crucially, pool1 (32x32) is
  // drawn at the same px as enc2 (32x32), so the eye reads pool1 as
  // "the same shape that enters enc2". Same for pool2 vs enc3.
  const INPUT_PX = 220;  // primary input view, slightly bigger than enc1
  const ENC1_PX  = 200;  // 64x64 -- the largest
  const POOL1_PX = 156;  // 32x32 -- matches enc2
  const ENC2_PX  = 156;  // 32x32
  const POOL2_PX = 120;  // 16x16 -- matches enc3
  const ENC3_PX  = 120;  // 16x16
  const THUMB_PX = 56;

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

  function pickRichestSample(samples) {
    let best = 0, bestCount = -1;
    for (let k = 0; k < samples.length; k++) {
      const seen = new Set();
      const lbl = samples[k].label;
      for (let i = 0; i < lbl.length; i++) {
        for (let j = 0; j < lbl[0].length; j++) seen.add(lbl[i][j]);
      }
      if (seen.size > bestCount) { bestCount = seen.size; best = k; }
    }
    return best;
  }

  /* 2x2 max-pool stride 2 on a single 2D channel. Mirrors nn.MaxPool2d(2). */
  function maxPool2x2(featMap) {
    const h = featMap.length, w = featMap[0].length;
    const out = [];
    for (let i = 0; i < h; i += 2) {
      const row = [];
      for (let j = 0; j < w; j += 2) {
        const a = featMap[i][j];
        const b = (j + 1 < w) ? featMap[i][j + 1] : a;
        const c = (i + 1 < h) ? featMap[i + 1][j] : a;
        const d = (i + 1 < h && j + 1 < w) ? featMap[i + 1][j + 1] : a;
        row.push(Math.max(a, b, c, d));
      }
      out.push(row);
    }
    return out;
  }

  /* Apply maxPool2x2 to each channel of a [C][H][W] stack independently. */
  function poolStack(stack) {
    const out = new Array(stack.length);
    for (let c = 0; c < stack.length; c++) out[c] = maxPool2x2(stack[c]);
    return out;
  }

  function makeLevelCard(parent, name, sub, px, kind) {
    // `kind` is 'conv' or 'pool' -- used for a small CSS accent only.
    const card = el('div', { class: 's4-card s4-card-' + (kind || 'conv') }, parent);
    const head = el('div', { class: 's4-card-head' }, card);
    el('span', { class: 's4-card-name', text: name }, head);
    el('span', { class: 's4-card-sub', text: sub }, head);
    const body = el('div', {
      class: 'canvas-host s4-card-body',
      style: 'width:' + px + 'px;height:' + px + 'px;',
    }, card);
    return { card: card, body: body };
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64) {
      root.innerHTML = '<p style="opacity:0.5">scene64 data missing.</p>';
      return {};
    }
    const D = window.DATA.scene64;
    const Drawing = window.Drawing;

    root.innerHTML = '';
    root.classList.add('s4-root');
    const wrap = el('div', { class: 's4-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s4-hero' }, wrap);
    el('h1', { text: 'The encoder is the CNN you already know.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Conv, ReLU, conv, ReLU, pool. Repeat. Doubling channels, halving resolution.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'If you watched the CNN deepdive, this is the encoder you ' +
        'already understand. Three blocks of two convolutions and a ReLU, ' +
        'with a max-pool between blocks. Watch each tensor light up &mdash; ' +
        'including the pool outputs, which halve the spatial resolution ' +
        'while leaving the channel count alone.',
    }, hero);

    /* ---- "You are here" mini-map ------------------------------------- */
    const miniHost = el('div', { class: 's4-mini-host' }, wrap);
    let mm = null;
    if (window.UNET && typeof window.UNET.mountUNetMiniMap === 'function') {
      mm = window.UNET.mountUNetMiniMap(miniHost, {
        width: 280, title: 'you are here',
        label: 'encoder · enc1 → pool → enc2 → pool → enc3',
      });
    }

    /* ---- Sample selector ------------------------------------------- */
    const selectorStrip = el('div', { class: 's4-selector' }, wrap);
    el('div', { class: 's4-selector-label', text: 'sample' }, selectorStrip);
    const selectorRow = el('div', { class: 's4-selector-row' }, selectorStrip);
    const thumbHosts = [];
    for (let i = 0; i < D.samples.length; i++) {
      const btn = el('button', {
        type: 'button',
        class: 's4-thumb',
        'data-sample-index': String(i),
        'aria-label': 'Select sample ' + (i + 1),
      }, selectorRow);
      const tHost = el('div', { class: 'canvas-host s4-thumb-host' }, btn);
      thumbHosts.push(tHost);
      btn.addEventListener('click', function () {
        const idx = parseInt(btn.getAttribute('data-sample-index'), 10);
        switchSample(idx);
      });
    }

    /* ---- Pipeline ---------------------------------------------------- */
    const pipeline = el('div', { class: 's4-pipeline' }, wrap);

    // Input column.
    const inputCol = el('div', { class: 's4-stage s4-stage-input' }, pipeline);
    const inputHead = el('div', { class: 's4-card-head' }, inputCol);
    el('span', { class: 's4-card-name', text: 'input' }, inputHead);
    el('span', { class: 's4-card-sub', text: '64×64×3 RGB' }, inputHead);
    const inputHost = el('div', {
      class: 'canvas-host s4-input-host',
      style: 'width:' + INPUT_PX + 'px;height:' + INPUT_PX + 'px;',
    }, inputCol);

    el('div', {
      class: 's4-arrow s4-arrow-conv',
      html: '&rarr;<div class="s4-arrow-label">conv·relu·conv·relu</div>',
    }, pipeline);

    const enc1Stage = el('div', { class: 's4-stage' }, pipeline);
    const enc1Card = makeLevelCard(enc1Stage, 'enc1', '64×64×16', ENC1_PX, 'conv');
    el('div', { class: 's4-stage-note', text: '4 of 16 channels' }, enc1Stage);

    el('div', {
      class: 's4-arrow s4-arrow-pool',
      html: '&rarr;<div class="s4-arrow-label">max-pool 2×2</div>',
    }, pipeline);

    const pool1Stage = el('div', { class: 's4-stage' }, pipeline);
    const pool1Card = makeLevelCard(pool1Stage, 'pool₁', '32×32×16', POOL1_PX, 'pool');
    el('div', { class: 's4-stage-note', text: 'channels unchanged' }, pool1Stage);

    el('div', {
      class: 's4-arrow s4-arrow-conv',
      html: '&rarr;<div class="s4-arrow-label">conv·relu·conv·relu</div>',
    }, pipeline);

    const enc2Stage = el('div', { class: 's4-stage' }, pipeline);
    const enc2Card = makeLevelCard(enc2Stage, 'enc2', '32×32×32', ENC2_PX, 'conv');
    el('div', { class: 's4-stage-note', text: '4 of 32 channels' }, enc2Stage);

    el('div', {
      class: 's4-arrow s4-arrow-pool',
      html: '&rarr;<div class="s4-arrow-label">max-pool 2×2</div>',
    }, pipeline);

    const pool2Stage = el('div', { class: 's4-stage' }, pipeline);
    const pool2Card = makeLevelCard(pool2Stage, 'pool₂', '16×16×32', POOL2_PX, 'pool');
    el('div', { class: 's4-stage-note', text: 'channels unchanged' }, pool2Stage);

    el('div', {
      class: 's4-arrow s4-arrow-conv',
      html: '&rarr;<div class="s4-arrow-label">conv·relu·conv·relu</div>',
    }, pipeline);

    const enc3Stage = el('div', { class: 's4-stage' }, pipeline);
    const enc3Card = makeLevelCard(enc3Stage, 'enc3', '16×16×64', ENC3_PX, 'conv');
    el('div', { class: 's4-stage-note', text: '4 of 64 channels — the bottleneck' }, enc3Stage);

    /* ---- Caption + footnote ----------------------------------------- */
    const caption = el('p', { class: 'caption s4-caption' }, wrap);
    el('p', {
      class: 's4-footnote',
      html:
        'Pool panels are computed by max-pooling the four previewed ' +
        'channels independently &mdash; per-channel faithful to ' +
        '<code>nn.MaxPool2d(2)</code>, just rendered on the top-variance ' +
        'subset shown above.',
    }, wrap);

    /* ---- Step controls --------------------------------------------- */
    const controls = el('div', { class: 'controls s4-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's4-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's4-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State ------------------------------------------------------ */
    const initialIdx = pickRichestSample(D.samples);
    const state = { step: 0, sampleIdx: initialIdx };

    // Pool outputs are sample-dependent. Cache per sampleIdx so we don't
    // recompute on every step click.
    const poolCache = {};
    function poolsFor(idx) {
      if (poolCache[idx]) return poolCache[idx];
      const s = D.samples[idx];
      const p1 = poolStack(s.enc1);
      const p2 = poolStack(s.enc2);
      poolCache[idx] = { pool1: p1, pool2: p2 };
      return poolCache[idx];
    }

    function renderThumbs() {
      for (let i = 0; i < D.samples.length; i++) {
        Drawing.paintRGB(thumbHosts[i], D.samples[i].input, THUMB_PX);
      }
      updateThumbActive();
    }
    function updateThumbActive() {
      const btns = selectorRow.querySelectorAll('.s4-thumb');
      btns.forEach(function (b, i) {
        b.classList.toggle('active', i === state.sampleIdx);
      });
    }

    function captionFor(step) {
      switch (step) {
        case 0: return 'A 64×64 RGB scene. Three colors per pixel — that is the whole input.';
        case 1: return 'enc1: two 3×3 convolutions and a ReLU. 16 feature maps at full resolution (64×64).';
        case 2: return 'pool₁: max-pool 2×2, stride 2. 64×64 → 32×32. Channels unchanged: 16 in, 16 out.';
        case 3: return 'enc2: two more convolutions on the half-size tensor. Channel count doubles to 32.';
        case 4: return 'pool₂: again 2×2 max-pool. 32×32 → 16×16. Still 32 channels.';
        case 5: return 'enc3: 16×16 with 64 channels — the bottleneck. Each cell now sees a wide patch of the input.';
        default: return '';
      }
    }

    /* The mini-map highlight set per step. We accumulate already-traversed
       tensors so the path "lights up" rather than blinking one-at-a-time. */
    function highlightFor(step) {
      switch (step) {
        case 0: return [];
        case 1: return ['enc1'];
        case 2: return ['enc1', 'pool1'];
        case 3: return ['enc1', 'pool1', 'enc2'];
        case 4: return ['enc1', 'pool1', 'enc2', 'pool2'];
        case 5: return ['enc1', 'pool1', 'enc2', 'pool2', 'enc3'];
        default: return [];
      }
    }

    function paintCard(card, host, stack, px, lit) {
      if (lit) Drawing.paintFeatureCard(host, stack, px);
      else Drawing.paintBlankCard(host, px);
      card.classList.toggle('s4-visible', !!lit);
    }

    function render() {
      const step = state.step;
      const sample = D.samples[state.sampleIdx];
      const pools = poolsFor(state.sampleIdx);

      updateThumbActive();
      Drawing.paintRGB(inputHost, sample.input, INPUT_PX);

      paintCard(enc1Card.card,  enc1Card.body,  sample.enc1,  ENC1_PX,  step >= 1);
      paintCard(pool1Card.card, pool1Card.body, pools.pool1,  POOL1_PX, step >= 2);
      paintCard(enc2Card.card,  enc2Card.body,  sample.enc2,  ENC2_PX,  step >= 3);
      paintCard(pool2Card.card, pool2Card.body, pools.pool2,  POOL2_PX, step >= 4);
      paintCard(enc3Card.card,  enc3Card.body,  sample.enc3,  ENC3_PX,  step >= 5);

      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;

      if (mm) mm.setHighlight(highlightFor(step));
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }
    function switchSample(idx) {
      if (idx < 0 || idx >= D.samples.length) return;
      state.sampleIdx = idx;
      render();
    }

    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    renderThumbs();
    render();

    /* &run -> auto-advance to last step over a few seconds. */
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
      onEnter: function () { renderThumbs(); render(); },
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
  window.scenes.scene4 = function (root) { return buildScene(root); };
})();
