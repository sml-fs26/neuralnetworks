/* Scene 8 -- "Walking up the right side of the U."

   Mirror of the encoder, on the way back up. We walk:
     bottleneck (16x16x64) -> up2 (32x32x32, transposed conv)
                           -> dec2 (32x32x32, conv-block)
                           -> up1 (64x64x16, transposed conv)
                           -> dec1 (64x64x16, conv-block)
                           -> output (64x64x5 logits -> argmax label map).

   The skip arcs are deliberately ABSENT here -- this scene introduces
   the decoder without skips. The output at the final step is therefore
   the no-skip prediction (window.DATA.noskip.samples[i].pred). It is
   visibly worse than the ground truth: that bad prediction is the hook
   for scene 9, which introduces skip connections.

   We do not have intermediates for the no-skip model; we use the
   with-skip enc3/dec2/dec1 feature maps for the decoder feature
   visualization. The caption is honest about this.

   Step engine:
     0 = bottleneck only
     1 = up2 lit                (32x32x32, "transposed conv")
     2 = dec2 lit               (after the 32->32 conv-block)
     3 = up1 lit                (64x64x16)
     4 = dec1 lit               (after the conv-block)
     5 = output revealed        (no-skip prediction; visibly off) */
(function () {
  'use strict';

  const NUM_STEPS = 6;
  const RUN_INTERVAL_MS = 700;

  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];

  const LEVEL1_PX = 132;   // 64x64
  const LEVEL2_PX = 100;   // 32x32
  const LEVEL3_PX = 80;    // 16x16 bottleneck
  const OVERLAY_PX = 256;  // big prediction strip
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
    const card = el('div', { class: 's8-card' }, parent);
    const head = el('div', { class: 's8-card-head' }, card);
    el('span', { class: 's8-card-name', text: name }, head);
    el('span', { class: 's8-card-sub', text: sub }, head);
    const body = el('div', {
      class: 'canvas-host s8-card-body',
      style: 'width:' + px + 'px;height:' + px + 'px;',
    }, card);
    return { card: card, body: body };
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64 || !window.DATA.noskip) {
      root.innerHTML = '<p style="opacity:0.5">scene8 data missing.</p>';
      return {};
    }
    const D = window.DATA.scene64;
    const NS = window.DATA.noskip;
    const Drawing = window.Drawing;

    root.innerHTML = '';
    root.classList.add('s8-root');
    const wrap = el('div', { class: 's8-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s8-hero' }, wrap);
    el('h1', { text: 'Walking up the right side of the U.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Up, conv, up, conv. The decoder is the encoder run backwards.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'Each decoder level <em>upsamples</em> (transposed conv, scene 7) and ' +
        'then runs a <em>conv-block</em>. We climb from 16×16 back to 64×64. ' +
        'But notice: there are no skip arcs in this diagram. Hold that picture &mdash; ' +
        'the prediction at the end will be <em>kind of right</em>, with mushy ' +
        'boundaries. The next scene explains why.',
    }, hero);

    /* ---- Sample selector ------------------------------------------- */
    const selectorStrip = el('div', { class: 's8-selector' }, wrap);
    el('div', { class: 's8-selector-label', text: 'sample' }, selectorStrip);
    const selectorRow = el('div', { class: 's8-selector-row' }, selectorStrip);
    const thumbHosts = [];
    for (let i = 0; i < D.samples.length; i++) {
      const btn = el('button', {
        type: 'button',
        class: 's8-thumb',
        'data-sample-index': String(i),
        'aria-label': 'Select sample ' + (i + 1),
      }, selectorRow);
      const tHost = el('div', { class: 'canvas-host s8-thumb-host' }, btn);
      thumbHosts.push(tHost);
      btn.addEventListener('click', function () {
        const idx = parseInt(btn.getAttribute('data-sample-index'), 10);
        switchSample(idx);
      });
    }

    /* ---- U-shape diagram (NO SKIP ARCS) ----------------------------- */
    const arch = el('div', { class: 's8-arch' }, wrap);

    // SVG overlay for the up-only flow arrows. No "skip" group at all.
    const arrowsSvg = el('div', { class: 's8-arrows' }, arch);
    arrowsSvg.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
      '<defs>' +
        '<marker id="s8-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
        '<path d="M 0 0 L 10 5 L 0 10 z" />' +
        '</marker>' +
      '</defs>' +
      '<g class="s8-flow s8-flow-u1">' +
        '<path class="s8-up" fill="none" marker-end="url(#s8-arrow)" />' +
        '<text class="s8-up-label" text-anchor="middle">upsample</text>' +
      '</g>' +
      '<g class="s8-flow s8-flow-c1">' +
        '<path class="s8-conv" fill="none" marker-end="url(#s8-arrow)" />' +
        '<text class="s8-conv-label" text-anchor="middle">conv-block</text>' +
      '</g>' +
      '<g class="s8-flow s8-flow-u2">' +
        '<path class="s8-up" fill="none" marker-end="url(#s8-arrow)" />' +
        '<text class="s8-up-label" text-anchor="middle">upsample</text>' +
      '</g>' +
      '<g class="s8-flow s8-flow-c2">' +
        '<path class="s8-conv" fill="none" marker-end="url(#s8-arrow)" />' +
        '<text class="s8-conv-label" text-anchor="middle">conv-block</text>' +
      '</g>' +
      '<g class="s8-flow s8-flow-out">' +
        '<path class="s8-conv" fill="none" marker-end="url(#s8-arrow)" />' +
        '<text class="s8-conv-label" text-anchor="middle">1×1 conv → softmax</text>' +
      '</g>' +
      '</svg>';

    // Bottleneck (centre/left) + decoder column (right).
    // We place the bottleneck on the LEFT of the diagram and walk
    // strictly RIGHTWARD to make "no skips" visually unmistakable.
    const botCol = el('div', { class: 's8-col s8-col-bot' }, arch);
    const enc3Card = makeLevelCard(botCol, 'enc3', '16×16×64 · bottleneck', LEVEL3_PX);

    const dec2Col = el('div', { class: 's8-col s8-col-dec2' }, arch);
    const dec2Card = makeLevelCard(dec2Col, 'dec2', '32×32×32', LEVEL2_PX);

    const dec1Col = el('div', { class: 's8-col s8-col-dec1' }, arch);
    const dec1Card = makeLevelCard(dec1Col, 'dec1', '64×64×16', LEVEL1_PX);

    enc3Card.card.dataset.s8level = 'enc3';
    dec2Card.card.dataset.s8level = 'dec2';
    dec1Card.card.dataset.s8level = 'dec1';

    /* ---- Prediction overlay panel ----------------------------------- */
    const overlay = el('div', { class: 's8-overlay' }, wrap);

    const ovInputCol = el('div', { class: 's8-ov-col' }, overlay);
    el('div', { class: 's8-ov-label', text: 'input · 64×64×3' }, ovInputCol);
    const ovInputHost = el('div', { class: 'canvas-host s8-ov-host' }, ovInputCol);

    const ovGtCol = el('div', { class: 's8-ov-col' }, overlay);
    el('div', { class: 's8-ov-label', text: 'ground truth' }, ovGtCol);
    const ovGtHost = el('div', { class: 'canvas-host s8-ov-host' }, ovGtCol);

    const ovPredCol = el('div', { class: 's8-ov-col' }, overlay);
    const predLabelEl = el('div', { class: 's8-ov-label', text: 'prediction (no skips)' }, ovPredCol);
    const ovPredHost = el('div', { class: 'canvas-host s8-ov-host' }, ovPredCol);

    /* ---- Class legend ----------------------------------------------- */
    const legend = el('div', { class: 's8-legend' }, wrap);
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const item = el('div', { class: 's8-legend-item' }, legend);
      el('span', { class: 's8-swatch class-' + CLASS_NAMES[c] }, item);
      el('span', { class: 's8-legend-name', text: CLASS_NAMES[c] }, item);
    }

    /* ---- Caption + honesty footnote --------------------------------- */
    const caption = el('p', { class: 'caption s8-caption' }, wrap);
    const honesty = el('p', { class: 's8-honesty' }, wrap);
    honesty.innerHTML =
      '<span class="s8-honesty-tag">a quiet honesty note:</span> ' +
      'feature maps in the decoder cards are taken from the <em>with-skip</em> ' +
      'model (we did not export the no-skip intermediates). The prediction ' +
      'shown below at step 5 is the <em>no-skip</em> model&#39;s actual output, ' +
      'so the qualitative point — fuzzy boundaries without skips — is honest.';

    /* ---- Step controls --------------------------------------------- */
    const controls = el('div', { class: 'controls s8-controls' }, wrap);

    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's8-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's8-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* Compute SVG arrow paths from real DOM positions. */
    function layoutArrows() {
      const archRect = arch.getBoundingClientRect();
      if (!archRect.width || !archRect.height) return;

      const svg = arrowsSvg.querySelector('svg');
      svg.setAttribute('viewBox',
        '0 0 ' + Math.round(archRect.width) + ' ' + Math.round(archRect.height));
      svg.setAttribute('width', String(Math.round(archRect.width)));
      svg.setAttribute('height', String(Math.round(archRect.height)));

      function rectOf(node) {
        const r = node.getBoundingClientRect();
        return {
          x: r.left - archRect.left,
          y: r.top - archRect.top,
          w: r.width, h: r.height,
          right: r.right - archRect.left,
          bottom: r.bottom - archRect.top,
          cx: (r.left + r.right) / 2 - archRect.left,
          cy: (r.top + r.bottom) / 2 - archRect.top,
        };
      }

      const r3  = rectOf(enc3Card.card);
      const r2d = rectOf(dec2Card.card);
      const r1d = rectOf(dec1Card.card);

      function setPath(group, d) {
        group.querySelector('path').setAttribute('d', d);
      }
      function setText(group, x, y, s) {
        const t = group.querySelector('text');
        t.setAttribute('x', String(x));
        t.setAttribute('y', String(y));
        if (s != null) t.textContent = s;
      }

      const flowU1 = arrowsSvg.querySelector('.s8-flow-u1');
      const flowC1 = arrowsSvg.querySelector('.s8-flow-c1');
      const flowU2 = arrowsSvg.querySelector('.s8-flow-u2');
      const flowC2 = arrowsSvg.querySelector('.s8-flow-c2');
      const flowOut = arrowsSvg.querySelector('.s8-flow-out');

      // Strategy: we visually split each "level transition" into TWO arrows
      // (upsample then conv-block), drawn over different parts of the gap
      // between adjacent cards. Top arrow = upsample, bottom arrow = conv.
      // For the first transition (enc3 -> dec2):
      //   upsample arrow:  from r3 right-edge upper -> r2d left-edge upper
      //   conv arrow:      from r2d left-edge lower -> r2d left-edge lower (a tiny self-loop)
      // That self-loop reads as "and then the conv-block also runs here";
      // which is a slight cheat. A cleaner approach: draw upsample as the
      // arrow into the card and draw the conv-block as a small loop arrow
      // sitting JUST to the left of the card's body. We do the latter.

      function arrowBetween(a, b, group, label, dy) {
        // Simple curve from a.right midpoint to b.left midpoint, with a
        // vertical offset (dy in viewBox coords) so two arrows can stack.
        const ax = a.right;
        const ay = a.cy + dy;
        const bx = b.x;
        const by = b.cy + dy;
        const mx = (ax + bx) / 2;
        setPath(group,
          'M ' + ax + ' ' + ay + ' ' +
          'C ' + mx + ' ' + ay + ', ' +
                mx + ' ' + by + ', ' +
                bx + ' ' + by);
        // Label sits just above the arrow.
        setText(group, mx, by - 6, label);
      }

      // up1: bottleneck enc3 -> dec2 (the first upsample, called up2 in
      // the model -- model names "up2" because it goes 16->32. We call
      // the *animation* arrow "u1" since it's the first upsample we walk).
      arrowBetween(r3, r2d, flowU1, 'upsample', -12);
      // conv-block sitting in front of dec2 (a tiny down-arc into the same card).
      const c1ax = r2d.x - 18;
      const c1ay = r2d.cy + 18;
      const c1bx = r2d.x;
      const c1by = r2d.cy + 18;
      setPath(flowC1,
        'M ' + c1ax + ' ' + c1ay + ' ' +
        'C ' + (c1ax - 18) + ' ' + (c1ay - 18) + ', ' +
              (c1ax - 18) + ' ' + (c1ay + 18) + ', ' +
              c1bx + ' ' + c1by);
      setText(flowC1, c1ax - 8, c1ay + 28, 'conv-block');

      // up2: dec2 -> dec1
      arrowBetween(r2d, r1d, flowU2, 'upsample', -12);
      const c2ax = r1d.x - 18;
      const c2ay = r1d.cy + 18;
      const c2bx = r1d.x;
      const c2by = r1d.cy + 18;
      setPath(flowC2,
        'M ' + c2ax + ' ' + c2ay + ' ' +
        'C ' + (c2ax - 18) + ' ' + (c2ay - 18) + ', ' +
              (c2ax - 18) + ' ' + (c2ay + 18) + ', ' +
              c2bx + ' ' + c2by);
      setText(flowC2, c2ax - 8, c2ay + 28, 'conv-block');

      // out arrow: dec1 right edge -> a small label off to the right
      // pointing into the prediction overlay area.
      const outAx = r1d.right;
      const outAy = r1d.cy;
      const outBx = r1d.right + 60;
      const outBy = r1d.cy;
      setPath(flowOut,
        'M ' + outAx + ' ' + outAy + ' ' +
        'L ' + outBx + ' ' + outBy);
      setText(flowOut, (outAx + outBx) / 2, outAy - 8, '1×1 conv → softmax');
    }

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
      const btns = selectorRow.querySelectorAll('.s8-thumb');
      btns.forEach(function (b, i) {
        b.classList.toggle('active', i === state.sampleIdx);
      });
    }

    function captionFor(step) {
      switch (step) {
        case 0: return 'The bottleneck. 16×16 with 64 channels — wide receptive fields, no spatial precision.';
        case 1: return 'up2: a transposed convolution lifts the bottleneck from 16×16 to 32×32 (32 channels).';
        case 2: return 'dec2: a conv-block (two convs + ReLU) refines those 32×32×32 features.';
        case 3: return 'up1: another transposed convolution doubles the resolution again, to 64×64.';
        case 4: return 'dec1: one more conv-block. We are back to full resolution, with 16 channels.';
        case 5: return 'A 1×1 conv plus softmax. But notice: the boundaries are mush. Hold that picture for scene 9.';
        default: return '';
      }
    }

    function render() {
      const step = state.step;
      const sample = D.samples[state.sampleIdx];
      const noskipPred = NS.samples[state.sampleIdx].pred;

      updateThumbActive();

      // Bottleneck always lit at step >= 0 (it's the starting point).
      Drawing.paintFeatureCard(enc3Card.body, sample.enc3, LEVEL3_PX);
      enc3Card.card.classList.add('s8-visible');

      // dec2 lit at step >= 2 (after the conv-block on the upsampled
      // tensor). Step 1 (just upsample) lights only the up-arrow into dec2;
      // dec2's body remains blank to signal "upsampled but not yet conv'd".
      // For visual clarity we light dec2 at step 1 too with a "stamped"
      // dimmer style; but to keep code simple we use:
      //   step 1 -> blank-but-bordered dec2 (just outline)
      //   step 2 -> populated dec2.
      if (step >= 2) {
        Drawing.paintFeatureCard(dec2Card.body, sample.dec2, LEVEL2_PX);
        dec2Card.card.classList.add('s8-visible');
        dec2Card.card.classList.remove('s8-half');
      } else if (step >= 1) {
        Drawing.paintBlankCard(dec2Card.body, LEVEL2_PX);
        dec2Card.card.classList.remove('s8-visible');
        dec2Card.card.classList.add('s8-half');
      } else {
        Drawing.paintBlankCard(dec2Card.body, LEVEL2_PX);
        dec2Card.card.classList.remove('s8-visible');
        dec2Card.card.classList.remove('s8-half');
      }

      if (step >= 4) {
        Drawing.paintFeatureCard(dec1Card.body, sample.dec1, LEVEL1_PX);
        dec1Card.card.classList.add('s8-visible');
        dec1Card.card.classList.remove('s8-half');
      } else if (step >= 3) {
        Drawing.paintBlankCard(dec1Card.body, LEVEL1_PX);
        dec1Card.card.classList.remove('s8-visible');
        dec1Card.card.classList.add('s8-half');
      } else {
        Drawing.paintBlankCard(dec1Card.body, LEVEL1_PX);
        dec1Card.card.classList.remove('s8-visible');
        dec1Card.card.classList.remove('s8-half');
      }

      // Flow arrows lighting.
      arch.classList.toggle('s8-up1-lit', step >= 1);
      arch.classList.toggle('s8-c1-lit', step >= 2);
      arch.classList.toggle('s8-up2-lit', step >= 3);
      arch.classList.toggle('s8-c2-lit', step >= 4);
      arch.classList.toggle('s8-out-lit', step >= 5);

      // Prediction overlay -- always show input, GT/Pred from step 5 only.
      // Per user feedback: no diff overlay, no accuracy badge -- the
      // highlighted "wrong pixel" markers caused confusion. Just show
      // ground truth and the no-skip prediction side by side; the viewer
      // compares them visually.
      Drawing.paintRGB(ovInputHost, sample.input, OVERLAY_PX);
      if (step >= 5) {
        Drawing.paintLabelMap(ovGtHost, sample.label, OVERLAY_PX);
        Drawing.paintLabelMap(ovPredHost, noskipPred, OVERLAY_PX);
        ovGtHost.classList.add('s8-visible');
        ovPredHost.classList.add('s8-visible');
        predLabelEl.textContent = 'prediction (no skips)';
      } else {
        Drawing.paintBlankCard(ovGtHost, OVERLAY_PX);
        Drawing.paintBlankCard(ovPredHost, OVERLAY_PX);
        ovGtHost.classList.remove('s8-visible');
        ovPredHost.classList.remove('s8-visible');
        predLabelEl.textContent = 'prediction (no skips)';
      }

      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;

      requestAnimationFrame(layoutArrows);
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    function switchSample(idx) {
      if (idx < 0 || idx >= D.samples.length) return;
      state.sampleIdx = idx;
      // If we are past step 5, stay there so the user immediately sees the
      // new prediction; otherwise keep the current step.
      if (state.step < 5) state.step = 5;
      render();
    }

    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    /* Initial paint */
    renderThumbs();
    render();
    // Belt + suspenders for layoutArrows in headless renders.
    requestAnimationFrame(function () {
      requestAnimationFrame(layoutArrows);
    });
    setTimeout(layoutArrows, 0);
    setTimeout(layoutArrows, 50);
    setTimeout(layoutArrows, 200);

    const onResize = function () { layoutArrows(); };
    window.addEventListener('resize', onResize);

    /* &run -> auto-advance to step 5 over ~5s. */
    let runTimer = null;
    function autoAdvance(target) {
      if (state.step >= target) { runTimer = null; return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(function () { autoAdvance(target); }, RUN_INTERVAL_MS);
    }
    if (readHashFlag('run')) {
      runTimer = setTimeout(function () { autoAdvance(5); }, 200);
    }

    return {
      onEnter: function () {
        renderThumbs();
        render();
        requestAnimationFrame(function () {
          requestAnimationFrame(layoutArrows);
        });
      },
      onLeave: function () {
        if (runTimer) { clearTimeout(runTimer); runTimer = null; }
        window.removeEventListener('resize', onResize);
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
  window.scenes.scene8 = function (root) { return buildScene(root); };
})();
