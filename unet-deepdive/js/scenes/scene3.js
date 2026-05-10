/* Scene 3 -- "Three stages: encoder -> bottleneck -> decoder."

   The big-picture conceptual frame between the architecture overview
   (scene 0) and the detailed encoder walk-through (scene 4). Names the
   bottleneck and shows the input -> encoder -> bottleneck -> decoder
   -> segmentation pipeline as five labelled panels in a single row.

   We deliberately do NOT show layers, channels, or convolutions inside
   the encoder/decoder boxes -- those are scene 4's and scene 8's job.
   Here the encoder and decoder are just labelled rectangles that the
   reader treats as black boxes; the real content is the *named output*
   between them: the bottleneck.

   Step engine (4 steps):
     0 = input only; encoder/bottleneck/decoder/segmentation panels dim
     1 = encoder block lights up + bottleneck panel revealed
     2 = decoder block lights up + segmentation panel revealed
     3 = the longer "what is the bottleneck" caption appears

   Reads:
     window.DATA.scene64.samples[0..5]     (input + label + pred + enc3)
     window.DATA.scene64.classes
     window.Drawing.{paintRGB, paintLabelMap, paintFeatureCard}
     window.UNET.mountUNetMiniMap                                            */
(function () {
  'use strict';

  const NUM_STEPS = 4;
  const RUN_INTERVAL_MS = 900;

  /* Panel sizes. Input/segmentation are 64x64 -- equal so the eye reads
     them as "before / after". The bottleneck is 16x16x4 -- visibly smaller
     so the compression is felt, not just stated. */
  const INPUT_PX = 168;
  const BOTTLENECK_PX = 132;
  const SEG_PX = 168;
  const THUMB_PX = 52;

  /* ---- DOM helpers ---- */
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

  /* Pick the sample with the most distinct classes -- richer label maps
     make the encoder/decoder transformation visibly more interesting. */
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

  /* ---- Builder ---- */
  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64 || !window.Drawing) {
      root.innerHTML = '<p style="opacity:0.5">Scene 3: missing globals.</p>';
      return {};
    }
    const D = window.DATA.scene64;
    if (!D.samples || !D.samples.length) {
      root.innerHTML = '<p style="opacity:0.5">Scene 3: no samples.</p>';
      return {};
    }
    const Drawing = window.Drawing;

    root.innerHTML = '';
    root.classList.add('s3-root');
    const wrap = el('div', { class: 's3-wrap' }, root);

    /* ---- Hero ---- */
    const hero = el('header', { class: 'hero s3-hero' }, wrap);
    el('h1', { text: 'Three stages: encoder → bottleneck → decoder.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'input ↦ compressed features ↦ per-pixel labels',
    }, hero);

    /* ---- Mini-map ---- */
    const miniHost = el('div', { class: 's3-mini-host' }, wrap);
    if (window.UNET && typeof window.UNET.mountUNetMiniMap === 'function') {
      const mm = window.UNET.mountUNetMiniMap(miniHost, {
        width: 280, title: 'you are here',
        label: 'the whole U — input → encoder → bottleneck → decoder → segmentation',
      });
      mm.setHighlight(['enc1', 'enc2', 'enc3', 'dec2', 'dec1', 'out']);
    }

    /* ---- Pipeline (the hero diagram of this scene) ----

       Five panels, four arrows:
         INPUT  -> [encoder]  -> BOTTLENECK  -> [decoder]  -> SEGMENTATION
       The two stage boxes (encoder, decoder) are labelled rectangles that
       only carry the stage name + a one-line summary -- intentionally
       opaque, since their internals are the subject of later scenes. */
    const pipeline = el('div', { class: 's3-pipeline' }, wrap);

    function makePanel(parent, klass, name, sub) {
      const panel = el('div', { class: 's3-panel ' + klass }, parent);
      const head = el('div', { class: 's3-panel-head' }, panel);
      el('span', { class: 's3-panel-name', text: name }, head);
      el('span', { class: 's3-panel-sub', text: sub }, head);
      const body = el('div', { class: 'canvas-host s3-panel-body' }, panel);
      return { panel: panel, body: body };
    }

    function makeStageBox(parent, klass, name, sub) {
      const box = el('div', { class: 's3-stage ' + klass }, parent);
      el('span', { class: 's3-stage-name', text: name }, box);
      el('span', { class: 's3-stage-sub', text: sub }, box);
      return box;
    }

    function makeArrow(parent) {
      const a = el('div', { class: 's3-arrow' }, parent);
      el('span', { class: 's3-arrow-shaft' }, a);
      el('span', { class: 's3-arrow-head', text: '▶' }, a);
      return a;
    }

    const inputPanel = makePanel(pipeline, 's3-input', 'input', '64×64×3 RGB');
    inputPanel.body.style.width = INPUT_PX + 'px';
    inputPanel.body.style.height = INPUT_PX + 'px';

    const arrowA = makeArrow(pipeline);
    const encStage = makeStageBox(pipeline, 's3-enc', 'encoder', 'conv · pool · repeat');
    const arrowB = makeArrow(pipeline);

    const bottleneckPanel = makePanel(pipeline, 's3-bottleneck', 'bottleneck', '16×16×64');
    bottleneckPanel.body.style.width = BOTTLENECK_PX + 'px';
    bottleneckPanel.body.style.height = BOTTLENECK_PX + 'px';

    const arrowC = makeArrow(pipeline);
    const decStage = makeStageBox(pipeline, 's3-dec', 'decoder', 'upsample · conv · repeat');
    const arrowD = makeArrow(pipeline);

    const segPanel = makePanel(pipeline, 's3-seg', 'segmentation', '64×64 labels');
    segPanel.body.style.width = SEG_PX + 'px';
    segPanel.body.style.height = SEG_PX + 'px';

    /* ---- Sample picker (matches scene 4 conventions) ---- */
    const selectorStrip = el('div', { class: 's3-selector' }, wrap);
    el('div', { class: 's3-selector-label', text: 'sample' }, selectorStrip);
    const selectorRow = el('div', { class: 's3-selector-row' }, selectorStrip);
    const thumbHosts = [];
    for (let i = 0; i < D.samples.length; i++) {
      const btn = el('button', {
        type: 'button',
        class: 's3-thumb',
        'data-sample-index': String(i),
        'aria-label': 'Select sample ' + (i + 1),
      }, selectorRow);
      const tHost = el('div', { class: 'canvas-host s3-thumb-host' }, btn);
      thumbHosts.push(tHost);
      btn.addEventListener('click', function () {
        const idx = parseInt(btn.getAttribute('data-sample-index'), 10);
        switchSample(idx);
      });
    }

    /* ---- Caption + extended explainer ---- */
    const caption = el('p', { class: 'caption s3-caption' }, wrap);
    const explainer = el('section', { class: 's3-explainer' }, wrap);
    el('p', { class: 's3-explainer-title', text: 'what the bottleneck is' }, explainer);
    el('p', {
      class: 's3-explainer-body',
      html:
        'A <strong>16×16 grid of cells</strong>, with <strong>64 numbers per cell</strong>. ' +
        'Each cell summarises a wide patch of the input — it knows ' +
        '<em>what is around here</em> (sky? grass? the edge of a tree?) ' +
        'but it has thrown away <em>where exactly</em>. The encoder distilled the ' +
        'image down to this; the decoder will reconstruct a full-resolution label ' +
        'map from it.',
    }, explainer);
    el('p', {
      class: 's3-explainer-body',
      html:
        'That is the U-Net in one sentence: <em>compress to semantics, then ' +
        'expand back to per-pixel labels.</em> The next scenes open up each of ' +
        'these three stages in turn.',
    }, explainer);

    /* ---- Step controls ---- */
    const controls = el('div', { class: 'controls s3-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's3-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's3-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State ---- */
    const state = { step: 0, sampleIdx: pickRichestSample(D.samples) };

    function captionFor(step) {
      switch (step) {
        case 0: return 'Start with the input: a 64×64 RGB scene.';
        case 1: return 'The encoder (the left half of the U) chews through the input and produces “the bottleneck” — a 16×16×64 tensor that holds the main features of the image.';
        case 2: return 'The decoder (the right half) takes the bottleneck and expands it back to a 64×64 grid — one predicted class per pixel.';
        case 3: return 'Three stages, three named tensors: input, bottleneck, segmentation. Everything in this deepdive lives between them.';
        default: return '';
      }
    }

    function renderThumbs() {
      for (let i = 0; i < D.samples.length; i++) {
        Drawing.paintRGB(thumbHosts[i], D.samples[i].input, THUMB_PX);
      }
      updateThumbActive();
    }
    function updateThumbActive() {
      const btns = selectorRow.querySelectorAll('.s3-thumb');
      btns.forEach(function (b, i) {
        b.classList.toggle('active', i === state.sampleIdx);
      });
    }

    function render() {
      const step = state.step;
      const sample = D.samples[state.sampleIdx];

      updateThumbActive();

      // Input is always painted (it's the source of truth at every step).
      Drawing.paintRGB(inputPanel.body, sample.input, INPUT_PX);

      // Bottleneck panel reveals at step 1. The 4-channel preview lives at
      // sample.enc3 (top level), matching scene 4's convention.
      if (step >= 1 && sample.enc3) {
        Drawing.paintFeatureCard(bottleneckPanel.body, sample.enc3, BOTTLENECK_PX);
      } else {
        Drawing.paintBlankCard(bottleneckPanel.body, BOTTLENECK_PX);
      }
      bottleneckPanel.panel.classList.toggle('s3-revealed', step >= 1);

      // Segmentation panel reveals at step 2.
      if (step >= 2) {
        Drawing.paintLabelMap(segPanel.body, sample.pred, SEG_PX);
      } else {
        Drawing.paintBlankCard(segPanel.body, SEG_PX);
      }
      segPanel.panel.classList.toggle('s3-revealed', step >= 2);

      // Stage boxes light up the moment their *output* panel is revealed.
      encStage.classList.toggle('s3-lit', step >= 1);
      decStage.classList.toggle('s3-lit', step >= 2);

      // Arrows: brighten in lockstep with the stage boxes either side.
      arrowA.classList.toggle('s3-lit', step >= 1);
      arrowB.classList.toggle('s3-lit', step >= 1);
      arrowC.classList.toggle('s3-lit', step >= 2);
      arrowD.classList.toggle('s3-lit', step >= 2);

      // The explainer block is the payoff at step 3.
      explainer.classList.toggle('s3-revealed', step >= 3);

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

    /* &run -> auto-advance through the steps. */
    let runTimer = null;
    function autoAdvance() {
      if (state.step >= NUM_STEPS - 1) { runTimer = null; return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(autoAdvance, RUN_INTERVAL_MS);
    }
    if (readHashFlag('run')) {
      runTimer = setTimeout(autoAdvance, 300);
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
  window.scenes.scene3 = function (root) { return buildScene(root); };
})();
