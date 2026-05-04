/* Scene 4 -- "The encoder is the CNN you already know."

   Brisk reuse, not a rebuild. Three encoder cards (64x64x16, 32x32x32,
   16x16x64). For one selected sample (a small thumb selector lets the
   viewer switch among the 6), light each level in sequence using
   `paintFeatureCard` (4 channels per level, top-variance picks).

   Step engine:
     0 = input only
     1 = enc1 lit (64x64, 4 ch)
     2 = enc2 lit (32x32, 4 ch)
     3 = enc3 lit (16x16, 4 ch -- the bottleneck) */
(function () {
  'use strict';

  const NUM_STEPS = 4;
  const RUN_INTERVAL_MS = 700;

  const ENC1_PX = 200;   // 64x64 -- the largest
  const ENC2_PX = 156;   // 32x32
  const ENC3_PX = 120;   // 16x16
  const INPUT_PX = 220;  // primary input view, slightly bigger than enc1
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

  function makeLevelCard(parent, name, sub, px) {
    const card = el('div', { class: 's4-card' }, parent);
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
        'with a max-pool between blocks. The interesting half of a U-Net ' +
        'is the <em>other</em> half &mdash; what we do on the way back up.',
    }, hero);

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

    // Input column
    const inputCol = el('div', { class: 's4-stage s4-stage-input' }, pipeline);
    const inputHead = el('div', { class: 's4-card-head' }, inputCol);
    el('span', { class: 's4-card-name', text: 'input' }, inputHead);
    el('span', { class: 's4-card-sub', text: '64×64×3 RGB' }, inputHead);
    const inputHost = el('div', {
      class: 'canvas-host s4-input-host',
      style: 'width:' + INPUT_PX + 'px;height:' + INPUT_PX + 'px;',
    }, inputCol);

    el('div', { class: 's4-arrow', html: '&rarr;<div class="s4-arrow-label">conv·relu·conv·relu</div>' }, pipeline);

    const enc1Stage = el('div', { class: 's4-stage' }, pipeline);
    const enc1Card = makeLevelCard(enc1Stage, 'enc1', '64×64×16', ENC1_PX);
    el('div', { class: 's4-stage-note', text: '4 of 16 channels' }, enc1Stage);

    el('div', { class: 's4-arrow', html: '&rarr;<div class="s4-arrow-label">pool · 2×</div>' }, pipeline);

    const enc2Stage = el('div', { class: 's4-stage' }, pipeline);
    const enc2Card = makeLevelCard(enc2Stage, 'enc2', '32×32×32', ENC2_PX);
    el('div', { class: 's4-stage-note', text: '4 of 32 channels' }, enc2Stage);

    el('div', { class: 's4-arrow', html: '&rarr;<div class="s4-arrow-label">pool · 2×</div>' }, pipeline);

    const enc3Stage = el('div', { class: 's4-stage' }, pipeline);
    const enc3Card = makeLevelCard(enc3Stage, 'enc3', '16×16×64', ENC3_PX);
    el('div', { class: 's4-stage-note', text: '4 of 64 channels — the bottleneck' }, enc3Stage);

    /* ---- Caption ---------------------------------------------------- */
    const caption = el('p', { class: 'caption s4-caption' }, wrap);

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
        case 1: return 'enc1: two 3×3 convolutions and a ReLU. 16 feature maps at full resolution.';
        case 2: return 'pool, then enc2. Half the resolution, twice as many channels. The trade we established in the CNN deepdive.';
        case 3: return 'pool again, enc3. 16×16 with 64 channels: the bottleneck. Each cell now sees a wide patch of the input.';
        default: return '';
      }
    }

    function render() {
      const step = state.step;
      const sample = D.samples[state.sampleIdx];

      updateThumbActive();
      Drawing.paintRGB(inputHost, sample.input, INPUT_PX);

      if (step >= 1) Drawing.paintFeatureCard(enc1Card.body, sample.enc1, ENC1_PX);
      else Drawing.paintBlankCard(enc1Card.body, ENC1_PX);
      enc1Card.card.classList.toggle('s4-visible', step >= 1);

      if (step >= 2) Drawing.paintFeatureCard(enc2Card.body, sample.enc2, ENC2_PX);
      else Drawing.paintBlankCard(enc2Card.body, ENC2_PX);
      enc2Card.card.classList.toggle('s4-visible', step >= 2);

      if (step >= 3) Drawing.paintFeatureCard(enc3Card.body, sample.enc3, ENC3_PX);
      else Drawing.paintBlankCard(enc3Card.body, ENC3_PX);
      enc3Card.card.classList.toggle('s4-visible', step >= 3);

      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;
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
