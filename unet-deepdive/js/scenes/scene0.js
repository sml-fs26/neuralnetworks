/* Scene 0 — "A U-Net, end to end."

   The overture. Hero text on top; below it the full U-Net diagram on the
   left and an input -> output strip on the right. A single auto-advance
   "play" pulse lights up encoder -> bottleneck -> decoder -> skips ->
   output over ~3 seconds. Honors `&run`.

   Step engine (6 frames):
     0  static U, all cards faintly drawn, no skips lit, output hidden
     1  sweep encoder (enc1, enc2 light up)
     2  sweep bottleneck (enc3 lights up)
     3  sweep skip arcs (both skips light)
     4  sweep decoder (dec2, dec1 light up)
     5  reveal output prediction map

   Reads:
     window.DATA.scene64.samples[0..5]  (input/label/pred + intermediates)
     window.DATA.scene64.classes
     window.Drawing.{paintRGB, paintLabelMap, paintFeatureCard,
                      paintBlankCard, setupCanvas, tokens} */
(function () {
  'use strict';

  const NUM_STEPS = 6;
  const RUN_TOTAL_MS = 3000;
  const RUN_INTERVAL_MS = Math.round(RUN_TOTAL_MS / (NUM_STEPS - 1));  // ~600ms

  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];

  // Card sizes — more compact than scene9, since we need to fit alongside
  // the input/output strip on the right.
  const LEVEL1_PX = 96;     // enc1 / dec1
  const LEVEL2_PX = 72;     // enc2 / dec2
  const LEVEL3_PX = 56;     // enc3 (bottleneck)
  const STRIP_PX  = 220;    // input + label panel

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

  function makeLevelCard(parent, name, sub, px) {
    const card = el('div', { class: 's0-card' }, parent);
    const head = el('div', { class: 's0-card-head' }, card);
    el('span', { class: 's0-card-name', text: name }, head);
    el('span', { class: 's0-card-sub', text: sub }, head);
    const body = el('div', {
      class: 'canvas-host s0-card-body',
      style: 'width:' + px + 'px;height:' + px + 'px;',
    }, card);
    return { card: card, body: body };
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64 || !window.Drawing) {
      root.innerHTML = '<p style="opacity:0.5">Scene 0: missing globals.</p>';
      return {};
    }
    const D = window.DATA.scene64;
    if (!D.samples || !D.samples.length) {
      root.innerHTML = '<p style="opacity:0.5">Scene 0: no samples.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s0-root');
    const wrap = el('div', { class: 's0-wrap' }, root);

    /* ---- Hero ----------------------------------------------------- */
    const hero = el('header', { class: 'hero s0-hero' }, wrap);
    el('h1', { text: 'A U-Net, end to end.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Image in. A label for every pixel out. The next fourteen scenes earn the diagram on the left.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'A classifier says <em>"this image is a circle"</em>. ' +
        'A segmenter says <em>"this pixel is a tree, this strip is grass, ' +
        'that little dot is the sun"</em>. Same convolutional machinery, ' +
        'but a different output shape — one prediction per pixel — and a ' +
        'tidy U-shaped wiring that makes it work.',
    }, hero);

    /* ---- Main split: U diagram (left) + I/O strip (right) -------- */
    const main = el('div', { class: 's0-main' }, wrap);

    /* ---- Left: full U diagram ------------------------------------ */
    const arch = el('div', { class: 's0-arch' }, main);

    // SVG overlay for skip + flow arrows.
    const arrowsSvg = el('div', { class: 's0-arrows' }, arch);
    arrowsSvg.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
      '<defs>' +
        '<marker id="s0-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
          'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
          '<path d="M 0 0 L 10 5 L 0 10 z" />' +
        '</marker>' +
      '</defs>' +
      '<g class="s0-skip s0-skip1">' +
        '<path class="s0-skip-path" fill="none" marker-end="url(#s0-arrow)" />' +
      '</g>' +
      '<g class="s0-skip s0-skip2">' +
        '<path class="s0-skip-path" fill="none" marker-end="url(#s0-arrow)" />' +
      '</g>' +
      '<g class="s0-flow s0-flow-d1">' +
        '<path class="s0-down" fill="none" marker-end="url(#s0-arrow)" />' +
      '</g>' +
      '<g class="s0-flow s0-flow-d2">' +
        '<path class="s0-down" fill="none" marker-end="url(#s0-arrow)" />' +
      '</g>' +
      '<g class="s0-flow s0-flow-u1">' +
        '<path class="s0-up" fill="none" marker-end="url(#s0-arrow)" />' +
      '</g>' +
      '<g class="s0-flow s0-flow-u2">' +
        '<path class="s0-up" fill="none" marker-end="url(#s0-arrow)" />' +
      '</g>' +
      '</svg>';

    // Encoder column (left)
    const encCol = el('div', { class: 's0-col s0-col-enc' }, arch);
    const enc1Card = makeLevelCard(encCol, 'enc1', '64×64', LEVEL1_PX);
    const enc2Card = makeLevelCard(encCol, 'enc2', '32×32', LEVEL2_PX);

    // Bottleneck (centre, bottom)
    const botCol = el('div', { class: 's0-col s0-col-bot' }, arch);
    const enc3Card = makeLevelCard(botCol, 'enc3', '16×16 · bottleneck', LEVEL3_PX);

    // Decoder column (right)
    const decCol = el('div', { class: 's0-col s0-col-dec' }, arch);
    const dec1Card = makeLevelCard(decCol, 'dec1', '64×64', LEVEL1_PX);
    const dec2Card = makeLevelCard(decCol, 'dec2', '32×32', LEVEL2_PX);

    enc1Card.card.dataset.level = 'enc1';
    enc2Card.card.dataset.level = 'enc2';
    enc3Card.card.dataset.level = 'enc3';
    dec2Card.card.dataset.level = 'dec2';
    dec1Card.card.dataset.level = 'dec1';

    /* ---- Right: input -> output strip ---------------------------- */
    const strip = el('div', { class: 's0-strip' }, main);

    const inputCol = el('div', { class: 's0-strip-col' }, strip);
    el('div', { class: 's0-strip-label', text: 'input · 64×64×3' }, inputCol);
    const inputHost = el('div', { class: 'canvas-host s0-strip-host' }, inputCol);

    const arrowCell = el('div', { class: 's0-strip-arrow' }, strip);
    arrowCell.innerHTML =
      '<svg viewBox="0 0 60 24" aria-hidden="true">' +
      '<line x1="2" y1="12" x2="50" y2="12" />' +
      '<polygon points="50,4 58,12 50,20" />' +
      '</svg>' +
      '<div class="s0-strip-arrow-label">U-Net</div>';

    const outputCol = el('div', { class: 's0-strip-col' }, strip);
    const outputLabel = el('div', { class: 's0-strip-label', text: 'output · per-pixel labels' }, outputCol);
    const outputHost = el('div', { class: 'canvas-host s0-strip-host' }, outputCol);

    /* ---- Class legend -------------------------------------------- */
    const legend = el('div', { class: 's0-legend' }, wrap);
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const item = el('div', { class: 's0-legend-item' }, legend);
      el('span', { class: 's0-swatch class-' + CLASS_NAMES[c] }, item);
      el('span', { class: 's0-legend-name', text: CLASS_NAMES[c] }, item);
    }

    /* ---- Caption ------------------------------------------------- */
    const caption = el('p', { class: 'caption s0-caption' }, wrap);

    /* ---- Controls ------------------------------------------------ */
    const controls = el('div', { class: 'controls s0-controls' }, wrap);
    const navGroup = el('div', { class: 'control-group' }, controls);
    const playBtn = el('button', { type: 'button', class: 'primary s0-play', text: 'play sweep' }, navGroup);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    const cue = el('p', { class: 's0-cue' }, wrap);
    cue.innerHTML =
      'We will spend the next 14 scenes building this. ' +
      'Press <kbd>&rarr;</kbd> to begin, or <kbd>play sweep</kbd> for the tour.';

    /* ---- State --------------------------------------------------- */
    // Pick a sample with all the interesting classes if we can.
    function pickRichest(samples) {
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
    const state = {
      step: 0,
      sampleIdx: pickRichest(D.samples),
      runTimer: null,
      autoPlaying: false,
    };
    const sample = function () { return D.samples[state.sampleIdx]; };

    /* ---- Layout helpers ----------------------------------------- */
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
          w: r.width,
          h: r.height,
          right: r.right - archRect.left,
          bottom: r.bottom - archRect.top,
          cx: (r.left + r.right) / 2 - archRect.left,
          cy: (r.top + r.bottom) / 2 - archRect.top,
        };
      }

      const r1e = rectOf(enc1Card.card);
      const r2e = rectOf(enc2Card.card);
      const r3  = rectOf(enc3Card.card);
      const r2d = rectOf(dec2Card.card);
      const r1d = rectOf(dec1Card.card);

      function setPath(group, d) {
        group.querySelector('path').setAttribute('d', d);
      }
      const skip1 = arrowsSvg.querySelector('.s0-skip1');
      const skip2 = arrowsSvg.querySelector('.s0-skip2');
      const flowD1 = arrowsSvg.querySelector('.s0-flow-d1');
      const flowD2 = arrowsSvg.querySelector('.s0-flow-d2');
      const flowU1 = arrowsSvg.querySelector('.s0-flow-u1');
      const flowU2 = arrowsSvg.querySelector('.s0-flow-u2');

      // skip1: enc1 right edge -> dec1 left edge, arc above
      const s1ax = r1e.right - 4;
      const s1ay = r1e.y - 4;
      const s1bx = r1d.x + 4;
      const s1by = r1d.y - 4;
      const peak1 = 14;
      const Yctrl1 = (8 * peak1 - s1ay - s1by) / 6;
      const s1ctrl = Math.max(80, (s1bx - s1ax) * 0.30);
      setPath(skip1,
        'M ' + s1ax + ' ' + s1ay + ' ' +
        'C ' + (s1ax + s1ctrl) + ' ' + Yctrl1 + ', ' +
              (s1bx - s1ctrl) + ' ' + Yctrl1 + ', ' +
              s1bx + ' ' + s1by);

      // skip2: enc2 right edge -> dec2 left edge, arc bows up into central gap
      const s2ax = r2e.right;
      const s2ay = r2e.cy;
      const s2bx = r2d.x;
      const s2by = r2d.cy;
      const peak2 = (r1e.bottom + r2e.y) / 2 - 6;
      const Yctrl2 = (8 * peak2 - s2ay - s2by) / 6;
      const s2ctrl = Math.max(70, (s2bx - s2ax) * 0.30);
      setPath(skip2,
        'M ' + s2ax + ' ' + s2ay + ' ' +
        'C ' + (s2ax + s2ctrl) + ' ' + Yctrl2 + ', ' +
              (s2bx - s2ctrl) + ' ' + Yctrl2 + ', ' +
              s2bx + ' ' + s2by);

      // Flow D1: enc1 bottom -> enc2 top
      setPath(flowD1,
        'M ' + r1e.cx + ' ' + r1e.bottom + ' ' +
        'L ' + r2e.cx + ' ' + r2e.y);

      // Flow D2: enc2 bottom -> enc3 top (curved)
      const d2ax = r2e.cx;
      const d2ay = r2e.bottom;
      const d2bx = r3.x + 4;
      const d2by = r3.y;
      setPath(flowD2,
        'M ' + d2ax + ' ' + d2ay + ' ' +
        'C ' + d2ax + ' ' + (d2ay + 24) + ', ' +
              d2bx + ' ' + (d2by - 16) + ', ' +
              d2bx + ' ' + d2by);

      // Flow U1: enc3 top -> dec2 bottom (curved)
      const u1ax = r3.right - 4;
      const u1ay = r3.y;
      const u1bx = r2d.cx;
      const u1by = r2d.bottom;
      setPath(flowU1,
        'M ' + u1ax + ' ' + u1ay + ' ' +
        'C ' + u1ax + ' ' + (u1ay - 16) + ', ' +
              u1bx + ' ' + (u1by + 24) + ', ' +
              u1bx + ' ' + u1by);

      // Flow U2: dec2 top -> dec1 bottom
      setPath(flowU2,
        'M ' + r2d.cx + ' ' + r2d.y + ' ' +
        'L ' + r1d.cx + ' ' + r1d.bottom);
    }

    /* ---- Painting helpers --------------------------------------- */
    function paintLevelCard(card, stack, px, lit) {
      if (lit && stack) {
        window.Drawing.paintFeatureCard(card.body, stack, px);
      } else {
        window.Drawing.paintBlankCard(card.body, px);
      }
      card.card.classList.toggle('s0-visible', !!lit);
    }

    function captionFor(step) {
      switch (step) {
        case 0: return 'A 64×64×3 image enters on the left. The U-Net will paint a class onto every pixel.';
        case 1: return 'Encoder: two convolutional blocks shrink the image while growing the channel count.';
        case 2: return 'Bottleneck: a small 16×16 map where each cell encodes the semantics of a large patch.';
        case 3: return 'Skip connections carry early, sharp feature maps across the U so the decoder can see them.';
        case 4: return 'Decoder: transposed convolutions upsample, conv blocks merge skips. Resolution returns.';
        case 5: return 'A 1×1 conv plus softmax. Every pixel gets a class. Same image, painted by class.';
        default: return '';
      }
    }

    /* ---- Render ------------------------------------------------- */
    function render() {
      const step = state.step;
      const s = sample();

      // Encoder cards (step >= 1 lights enc1, enc2; step >= 2 lights enc3)
      paintLevelCard(enc1Card, s.enc1, LEVEL1_PX, step >= 1);
      paintLevelCard(enc2Card, s.enc2, LEVEL2_PX, step >= 1);
      paintLevelCard(enc3Card, s.enc3, LEVEL3_PX, step >= 2);
      paintLevelCard(dec2Card, s.dec2, LEVEL2_PX, step >= 4);
      paintLevelCard(dec1Card, s.dec1, LEVEL1_PX, step >= 4);

      // Skip arrows light at step 3 (and stay lit through 5)
      arch.classList.toggle('s0-skips-lit', step >= 3);
      // Flow arrows: encoder pools light at step 1, second pool at step 2,
      // upsample arrows at step 4.
      arch.classList.toggle('s0-down1-lit', step >= 1);
      arch.classList.toggle('s0-down2-lit', step >= 2);
      arch.classList.toggle('s0-up1-lit', step >= 4);
      arch.classList.toggle('s0-up2-lit', step >= 4);

      // Input is always painted; output reveals at step 5.
      window.Drawing.paintRGB(inputHost, s.input, STRIP_PX);
      if (step >= 5) {
        window.Drawing.paintLabelMap(outputHost, s.pred, STRIP_PX);
        outputHost.classList.add('s0-visible');
        outputLabel.textContent = 'output · per-pixel labels';
      } else {
        window.Drawing.paintBlankCard(outputHost, STRIP_PX);
        outputHost.classList.remove('s0-visible');
        outputLabel.textContent = 'output · pending';
      }

      caption.textContent = captionFor(step);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;

      // Re-position arrows once layout has settled.
      requestAnimationFrame(layoutArrows);
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    /* ---- Auto-play sweep ---------------------------------------- */
    function stopRun() {
      if (state.runTimer) {
        clearTimeout(state.runTimer);
        state.runTimer = null;
      }
      state.autoPlaying = false;
      playBtn.textContent = 'play sweep';
    }
    function startRun() {
      stopRun();
      state.autoPlaying = true;
      playBtn.textContent = 'playing…';
      // Start from 0 so the sweep is the full tour.
      applyStep(0);
      function tick() {
        if (!state.autoPlaying) return;
        if (state.step >= NUM_STEPS - 1) {
          stopRun();
          return;
        }
        applyStep(state.step + 1);
        state.runTimer = setTimeout(tick, RUN_INTERVAL_MS);
      }
      state.runTimer = setTimeout(tick, RUN_INTERVAL_MS);
    }

    /* ---- Wiring ------------------------------------------------- */
    playBtn.addEventListener('click', function () {
      if (state.autoPlaying) stopRun();
      else startRun();
    });
    prevBtn.addEventListener('click', function () { stopRun(); applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { stopRun(); applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { stopRun(); applyStep(0); });

    /* ---- First paint -------------------------------------------- */
    render();
    // Belt-and-suspenders: layoutArrows fires after browser measures.
    requestAnimationFrame(function () {
      requestAnimationFrame(layoutArrows);
    });
    setTimeout(layoutArrows, 0);
    setTimeout(layoutArrows, 50);
    setTimeout(layoutArrows, 200);
    const onResize = function () { layoutArrows(); };
    window.addEventListener('resize', onResize);

    // &run -> auto-play the sweep on entry.
    if (readHashFlag('run')) {
      state.runTimer = setTimeout(startRun, 250);
    }

    return {
      onEnter: function () {
        render();
        requestAnimationFrame(function () {
          requestAnimationFrame(layoutArrows);
        });
      },
      onLeave: function () {
        stopRun();
        window.removeEventListener('resize', onResize);
      },
      onNextKey: function () {
        if (state.step < NUM_STEPS - 1) {
          stopRun();
          applyStep(state.step + 1);
          return true;
        }
        return false;
      },
      onPrevKey: function () {
        if (state.step > 0) {
          stopRun();
          applyStep(state.step - 1);
          return true;
        }
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene0 = function (root) { return buildScene(root); };
})();
