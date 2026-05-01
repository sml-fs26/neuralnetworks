/* Scene 9 -- "Segmentation -- same machinery, per pixel."

   The U-Net coda. The classifier said "this image is a circle". The
   segmenter says "this pixel is a tree, this one a person, this strip is
   grass". Same filters, same units, same composition. The output shape is
   the only thing that changed.

   Layout:
     - Hero (h1, italic subtitle, lede)
     - Sample selector strip: 6 thumbnails, click to switch sample.
     - U-shape architecture diagram:
         encoder column on the left going down,
         bottleneck card centred at the bottom,
         decoder column on the right going up,
         skip arrows arc horizontally across the top.
     - Prediction overlay: input | ground truth | prediction (256x256 each).
     - Class legend.
     - Step engine: 0..7 (zero shows input only; 1..3 reveal encoder; 4..5
       reveal decoder + skip highlight; 6 reveals prediction; 7 swaps sample
       and replays rapidly).
     - Caption.

   `&run` auto-advances to step 6 over ~5s. */
(function () {
  'use strict';

  const NUM_STEPS = 8;
  const RUN_INTERVAL_MS = 700;

  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];

  // Card sizes (logical px) for the U-shape diagram.
  const LEVEL1_PX = 132;   // encoder/decoder level 1 (64x64) -- larger
  const LEVEL2_PX = 100;   // encoder/decoder level 2 (32x32)
  const LEVEL3_PX = 80;    // bottleneck (16x16)
  const OVERLAY_PX = 256;  // big prediction strip
  const THUMB_PX = 56;     // sample selector

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

  /* Read class colors live from CSS so theme switching follows automatically. */
  function readClassColors() {
    const cs = getComputedStyle(document.documentElement);
    return CLASS_NAMES.map(function (name) {
      return cs.getPropertyValue('--class-' + name).trim() || '#888';
    });
  }

  /* Choose the default sample: the one with the most distinct classes
     present in its label. Ties broken by lower index. */
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

  /* ---------------------------------------------------------------------
     Painters
     --------------------------------------------------------------------- */

  /* Paint a 64x64 RGB array onto a canvas at logical size [px, px].
     Uses ImageData for speed. Colors are NOT theme-affected -- the input
     is the input. */
  function paintRGB(host, rgb, px) {
    host.innerHTML = '';
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const H = rgb.length, W = rgb[0].length;
    // Build a small ImageData at native resolution then scale via drawImage.
    const off = document.createElement('canvas');
    off.width = W; off.height = H;
    const offCtx = off.getContext('2d');
    const id = offCtx.createImageData(W, H);
    let p = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const r = rgb[i][j][0], g = rgb[i][j][1], b = rgb[i][j][2];
        id.data[p++] = Math.max(0, Math.min(255, Math.round(r * 255)));
        id.data[p++] = Math.max(0, Math.min(255, Math.round(g * 255)));
        id.data[p++] = Math.max(0, Math.min(255, Math.round(b * 255)));
        id.data[p++] = 255;
      }
    }
    offCtx.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, px, px);
  }

  /* Paint an integer label map [H][W] of class indices using the per-class
     CSS colors. Optionally outline pixels where pred != label. */
  function paintLabelMap(host, lbl, px, opts) {
    host.innerHTML = '';
    opts = opts || {};
    const colors = opts.colors || readClassColors();
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const H = lbl.length, W = lbl[0].length;
    const off = document.createElement('canvas');
    off.width = W; off.height = H;
    const offCtx = off.getContext('2d');
    const id = offCtx.createImageData(W, H);
    let p = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const c = lbl[i][j] | 0;
        const hex = colors[c] || '#888';
        const rgb = parseHex(hex);
        id.data[p++] = rgb[0];
        id.data[p++] = rgb[1];
        id.data[p++] = rgb[2];
        id.data[p++] = 255;
      }
    }
    offCtx.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, px, px);

    // If a diff mask is supplied, outline disagreements.
    if (opts.diffMask) {
      const cw = px / W, ch = px / H;
      const t = window.Drawing.tokens();
      ctx.strokeStyle = t.ink;
      ctx.lineWidth = 1.4;
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          if (opts.diffMask[i][j]) {
            ctx.strokeRect(j * cw + 0.5, i * ch + 0.5, cw - 1, ch - 1);
          }
        }
      }
    }
  }

  function parseHex(hex) {
    let s = (hex || '').trim().replace('#', '');
    if (s.length === 3) s = s.split('').map(function (c) { return c + c; }).join('');
    if (!/^[0-9a-fA-F]{6}$/.test(s)) return [136, 136, 136];
    const n = parseInt(s, 16);
    return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff];
  }

  /* Paint a 4-channel feature stack [4][H][W] as a 2x2 grid of small
     thumbnails using the diverging colormap. `host` should be the card body. */
  function paintFeatureCard(host, stack4, px) {
    host.innerHTML = '';
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, px, px);

    // Symmetric range across all 4 channels for fair contrast.
    let m = 0;
    for (let c = 0; c < stack4.length; c++) {
      const r = window.CNN.range2D(stack4[c]);
      m = Math.max(m, Math.abs(r.lo), Math.abs(r.hi));
    }
    if (!m) m = 1;

    const half = px / 2;
    const gap = 2;
    const cellW = half - gap;
    const positions = [
      [0, 0], [half + gap, 0],
      [0, half + gap], [half + gap, half + gap],
    ];
    for (let c = 0; c < 4; c++) {
      const pos = positions[c];
      window.Drawing.drawGrid(ctx, stack4[c], pos[0], pos[1], cellW, cellW, {
        diverging: true, valueRange: [-m, m],
      });
    }
    // Thin separator lines so the 2x2 reads as a grid of channels.
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(half, 0); ctx.lineTo(half, px);
    ctx.moveTo(0, half); ctx.lineTo(px, half);
    ctx.stroke();
  }

  function paintBlankCard(host, px) {
    host.innerHTML = '';
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();
    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, px, px);
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.strokeRect(0.5, 0.5, px - 1, px - 1);
    ctx.beginPath();
    ctx.moveTo(0, px / 2); ctx.lineTo(px, px / 2);
    ctx.moveTo(px / 2, 0); ctx.lineTo(px / 2, px);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  /* ---------------------------------------------------------------------
     Builder
     --------------------------------------------------------------------- */

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64) {
      root.innerHTML = '<p style="opacity:0.5">scene64 data missing.</p>';
      return {};
    }
    const D = window.DATA.scene64;

    root.innerHTML = '';
    root.classList.add('s9-root');
    const wrap = el('div', { class: 's9-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s9-hero' }, wrap);
    el('h1', { text: 'Segmentation — same machinery, per pixel.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'A tiny U-Net runs the same forward pass we have built. Only the output shape changes.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'The classifier said <em>"this image is a circle"</em>. The segmenter ' +
        'says <em>"this pixel is a tree, this one a person, this strip is grass"</em>. ' +
        'Same filters, same units, same composition. The output shape is the only thing that changed.',
    }, hero);

    /* ---- Sample selector ------------------------------------------- */
    const selectorStrip = el('div', { class: 's9-selector' }, wrap);
    el('div', { class: 's9-selector-label', text: 'sample' }, selectorStrip);
    const selectorRow = el('div', { class: 's9-selector-row' }, selectorStrip);
    const thumbHosts = [];
    for (let i = 0; i < D.samples.length; i++) {
      const btn = el('button', {
        type: 'button',
        class: 's9-thumb',
        'data-sample-index': String(i),
        'aria-label': 'Select sample ' + (i + 1),
      }, selectorRow);
      const tHost = el('div', { class: 'canvas-host s9-thumb-host' }, btn);
      thumbHosts.push(tHost);
      btn.addEventListener('click', function () {
        const idx = parseInt(btn.getAttribute('data-sample-index'), 10);
        switchSample(idx);
      });
    }

    /* ---- U-shape diagram -------------------------------------------- */
    const arch = el('div', { class: 's9-arch' }, wrap);

    // SVG overlay for skip + flow arrows. The viewBox is set to the actual
    // pixel size of the .s9-arch container after first layout (see
    // layoutArrows()), so paths can be drawn in real DOM coordinates.
    const arrowsSvg = el('div', { class: 's9-arrows' }, arch);
    arrowsSvg.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
      '<defs>' +
        '<marker id="s9-arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
        '<path d="M 0 0 L 10 5 L 0 10 z" />' +
        '</marker>' +
      '</defs>' +
      '<g class="s9-skip s9-skip1">' +
        '<path class="s9-skip-path" fill="none" marker-end="url(#s9-arrow)" />' +
        '<text class="s9-skip-label" text-anchor="middle">skip</text>' +
      '</g>' +
      '<g class="s9-skip s9-skip2">' +
        '<path class="s9-skip-path" fill="none" marker-end="url(#s9-arrow)" />' +
        '<text class="s9-skip-label" text-anchor="middle">skip</text>' +
      '</g>' +
      '<g class="s9-flow s9-flow-d1">' +
        '<path class="s9-down" fill="none" marker-end="url(#s9-arrow)" />' +
        '<text class="s9-down-label" text-anchor="start">pool</text>' +
      '</g>' +
      '<g class="s9-flow s9-flow-d2">' +
        '<path class="s9-down" fill="none" marker-end="url(#s9-arrow)" />' +
        '<text class="s9-down-label" text-anchor="start">pool</text>' +
      '</g>' +
      '<g class="s9-flow s9-flow-u1">' +
        '<path class="s9-up" fill="none" marker-end="url(#s9-arrow)" />' +
        '<text class="s9-up-label" text-anchor="start">upsample</text>' +
      '</g>' +
      '<g class="s9-flow s9-flow-u2">' +
        '<path class="s9-up" fill="none" marker-end="url(#s9-arrow)" />' +
        '<text class="s9-up-label" text-anchor="start">upsample</text>' +
      '</g>' +
      '</svg>';

    // Encoder column (left)
    const encCol = el('div', { class: 's9-col s9-col-enc' }, arch);
    const enc1Card = makeLevelCard(encCol, 'enc1', '64×64, 4 ch', LEVEL1_PX);
    const enc2Card = makeLevelCard(encCol, 'enc2', '32×32, 4 ch', LEVEL2_PX);

    // Bottleneck (centre, bottom)
    const botCol = el('div', { class: 's9-col s9-col-bot' }, arch);
    const enc3Card = makeLevelCard(botCol, 'enc3', '16×16 · bottleneck', LEVEL3_PX);

    // Decoder column (right)
    const decCol = el('div', { class: 's9-col s9-col-dec' }, arch);
    const dec1Card = makeLevelCard(decCol, 'dec1', '64×64, 4 ch', LEVEL1_PX);
    const dec2Card = makeLevelCard(decCol, 'dec2', '32×32, 4 ch', LEVEL2_PX);

    // Mark cards by data attr so we can drive 'visible' state from CSS.
    enc1Card.card.dataset.s9level = 'enc1';
    enc2Card.card.dataset.s9level = 'enc2';
    enc3Card.card.dataset.s9level = 'enc3';
    dec2Card.card.dataset.s9level = 'dec2';
    dec1Card.card.dataset.s9level = 'dec1';

    /* ---- Prediction overlay panel ----------------------------------- */
    const overlay = el('div', { class: 's9-overlay' }, wrap);

    const ovInputCol = el('div', { class: 's9-ov-col' }, overlay);
    el('div', { class: 's9-ov-label', text: 'input · 64×64×3' }, ovInputCol);
    const ovInputHost = el('div', { class: 'canvas-host s9-ov-host' }, ovInputCol);

    const ovGtCol = el('div', { class: 's9-ov-col' }, overlay);
    el('div', { class: 's9-ov-label', text: 'ground truth' }, ovGtCol);
    const ovGtHost = el('div', { class: 'canvas-host s9-ov-host' }, ovGtCol);

    const ovPredCol = el('div', { class: 's9-ov-col' }, overlay);
    const predLabelEl = el('div', { class: 's9-ov-label', text: 'prediction' }, ovPredCol);
    const ovPredHost = el('div', { class: 'canvas-host s9-ov-host' }, ovPredCol);

    /* ---- Class legend ----------------------------------------------- */
    const legend = el('div', { class: 's9-legend' }, wrap);
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const item = el('div', { class: 's9-legend-item' }, legend);
      el('span', { class: 's9-swatch class-' + CLASS_NAMES[c] }, item);
      el('span', { class: 's9-legend-name', text: CLASS_NAMES[c] }, item);
    }

    /* ---- Caption ---------------------------------------------------- */
    const caption = el('p', { class: 'caption s9-caption' }, wrap);

    /* ---- Step controls --------------------------------------------- */
    const controls = el('div', { class: 'controls s9-controls' }, wrap);

    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's9-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's9-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    // Footnote: model accuracy on its tiny held-out set.
    const accNote = el('div', { class: 's9-acc-note' }, wrap);
    el('span', { class: 's9-acc-label', text: 'mean pixel accuracy' }, accNote);
    el('span', {
      class: 's9-acc-value',
      text: (D.meanPixelAccuracy * 100).toFixed(2) + '%',
    }, accNote);

    /* Compute SVG arrow paths from the actual DOM positions of the level
       cards. Idempotent and cheap to call after every render. */
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
      function setText(group, x, y, s) {
        const t = group.querySelector('text');
        t.setAttribute('x', String(x));
        t.setAttribute('y', String(y));
        if (s != null) t.textContent = s;
      }

      const skip1 = arrowsSvg.querySelector('.s9-skip1');
      const skip2 = arrowsSvg.querySelector('.s9-skip2');
      const flowD1 = arrowsSvg.querySelector('.s9-flow-d1');
      const flowD2 = arrowsSvg.querySelector('.s9-flow-d2');
      const flowU1 = arrowsSvg.querySelector('.s9-flow-u1');
      const flowU2 = arrowsSvg.querySelector('.s9-flow-u2');

      // skip1: enc1 right edge -> dec1 left edge, arcing high above the cards.
      // For a cubic Bezier with endpoints at y=Yend and control y=Yctrl,
      // the midpoint y is (Yend*2 + Yctrl*6) / 8. To get the visual peak
      // ~30px below the arch top (Yend ~= 90), we set Yctrl much lower
      // (smaller y). With Yctrl = -100 and Yend = 95, midpoint y is
      // (190 - 600) / 8 = -51 -- which means the arc would bow ABOVE the
      // viewBox top. Use overflow: visible on the SVG so this still draws.
      const s1ax = r1e.right - 6;
      const s1ay = r1e.y - 8;
      const s1bx = r1d.x + 6;
      const s1by = r1d.y - 8;
      // Aim the actual arc PEAK at y=20 (viewBox coords). Solve:
      //   peakY = (s1ay + 3*Yctrl + 3*Yctrl + s1by) / 8 = (s1ay + s1by + 6*Yctrl) / 8
      //   => Yctrl = (8*peakY - s1ay - s1by) / 6
      const desiredPeakY = 20;
      const Yctrl = (8 * desiredPeakY - s1ay - s1by) / 6;
      const s1mid = (s1ax + s1bx) / 2;
      const s1ctrl = Math.max(160, (s1bx - s1ax) * 0.26);
      setPath(skip1,
        'M ' + s1ax + ' ' + s1ay + ' ' +
        'C ' + (s1ax + s1ctrl) + ' ' + Yctrl + ', ' +
              (s1bx - s1ctrl) + ' ' + Yctrl + ', ' +
              s1bx + ' ' + s1by);
      setText(skip1, s1mid, desiredPeakY - 8, 'skip');

      // skip2: enc2 right edge centre -> dec2 left edge centre.
      // Arc bows up into the EMPTY centre of the U, not above enc1.
      const s2ax = r2e.right;
      const s2ay = r2e.cy;
      const s2bx = r2d.x;
      const s2by = r2d.cy;
      // Peak ~30px above the highest of (enc1.bottom, dec1.bottom) -- i.e.
      // up into the dead space between row1 and row2 of cards.
      const peakY2 = (r1e.bottom + r2e.y) / 2 - 12;
      const Yctrl2 = (8 * peakY2 - s2ay - s2by) / 6;
      const s2mid = (s2ax + s2bx) / 2;
      const s2ctrl = Math.max(140, (s2bx - s2ax) * 0.30);
      setPath(skip2,
        'M ' + s2ax + ' ' + s2ay + ' ' +
        'C ' + (s2ax + s2ctrl) + ' ' + Yctrl2 + ', ' +
              (s2bx - s2ctrl) + ' ' + Yctrl2 + ', ' +
              s2bx + ' ' + s2by);
      // Only the first skip carries a text label; clutter-free.
      setText(skip2, s2mid, peakY2 - 4, '');

      // Flow D1: enc1 bottom centre -> enc2 top centre
      setPath(flowD1,
        'M ' + r1e.cx + ' ' + r1e.bottom + ' ' +
        'L ' + r2e.cx + ' ' + r2e.y);
      setText(flowD1, r1e.cx + 8, (r1e.bottom + r2e.y) / 2 + 4, 'pool');

      // Flow D2: enc2 bottom-right -> enc3 top-left
      const d2ax = r2e.cx;
      const d2ay = r2e.bottom;
      const d2bx = r3.x + 6;
      const d2by = r3.y;
      setPath(flowD2,
        'M ' + d2ax + ' ' + d2ay + ' ' +
        'C ' + d2ax + ' ' + (d2ay + 30) + ', ' +
              d2bx + ' ' + (d2by - 20) + ', ' +
              d2bx + ' ' + d2by);
      setText(flowD2, (d2ax + d2bx) / 2 - 18, (d2ay + d2by) / 2 + 14, 'pool');

      // Flow U1: enc3 top-right -> dec2 bottom-centre
      const u1ax = r3.right - 6;
      const u1ay = r3.y;
      const u1bx = r2d.cx;
      const u1by = r2d.bottom;
      setPath(flowU1,
        'M ' + u1ax + ' ' + u1ay + ' ' +
        'C ' + u1ax + ' ' + (u1ay - 20) + ', ' +
              u1bx + ' ' + (u1by + 30) + ', ' +
              u1bx + ' ' + u1by);
      setText(flowU1, (u1ax + u1bx) / 2 + 6, (u1ay + u1by) / 2 + 14, 'upsample');

      // Flow U2: dec2 top centre -> dec1 bottom centre
      setPath(flowU2,
        'M ' + r2d.cx + ' ' + r2d.y + ' ' +
        'L ' + r1d.cx + ' ' + r1d.bottom);
      setText(flowU2, r1d.cx + 8, (r1d.bottom + r2d.y) / 2 + 4, 'upsample');
    }

    /* ---- State ------------------------------------------------------ */
    const initialIdx = pickRichestSample(D.samples);
    const state = {
      step: 0,
      sampleIdx: initialIdx,
      replayedIdx: initialIdx,
    };

    /* Render the small thumbnails once -- they are static. */
    function renderThumbs() {
      for (let i = 0; i < D.samples.length; i++) {
        paintRGB(thumbHosts[i], D.samples[i].input, THUMB_PX);
      }
      updateThumbActive();
    }
    function updateThumbActive() {
      const btns = selectorRow.querySelectorAll('.s9-thumb');
      btns.forEach(function (b, i) {
        b.classList.toggle('active', i === state.sampleIdx);
      });
    }

    function captionFor(step) {
      switch (step) {
        case 0: return 'A cartoon scene. Sky, grass, maybe a sun, a tree, a person.';
        case 1: return 'Encoder level 1. Same convolutions, same ReLU. Four channels of feature maps at full resolution.';
        case 2: return 'Pool, then encoder level 2. Half the resolution, deeper features.';
        case 3: return 'Pool again. The bottleneck holds a 16×16 summary.';
        case 4: return 'Decoder level 2. Upsample the bottleneck and concatenate the encoder-2 skip.';
        case 5: return 'Decoder level 1. Upsample again, add the encoder-1 skip. We are back to full resolution.';
        case 6: return 'A 1×1 conv plus softmax. Every pixel gets a class. The output looks like the input.';
        case 7: return 'Same machinery, different sample. The forward pass is the only thing that runs.';
        default: return '';
      }
    }

    function diffMaskFor(sample) {
      const H = sample.label.length, W = sample.label[0].length;
      const m = window.CNN.zeros2D(H, W);
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          m[i][j] = (sample.pred[i][j] !== sample.label[i][j]) ? 1 : 0;
        }
      }
      return m;
    }

    function render() {
      const step = state.step;
      const sample = D.samples[state.sampleIdx];

      // Sample thumbnails -- toggle active class
      updateThumbActive();

      // Encoder cards: enc1 at step >= 1, enc2 at step >= 2, enc3 at step >= 3.
      if (step >= 1) paintFeatureCard(enc1Card.body, sample.enc1, LEVEL1_PX);
      else paintBlankCard(enc1Card.body, LEVEL1_PX);
      enc1Card.card.classList.toggle('s9-visible', step >= 1);

      if (step >= 2) paintFeatureCard(enc2Card.body, sample.enc2, LEVEL2_PX);
      else paintBlankCard(enc2Card.body, LEVEL2_PX);
      enc2Card.card.classList.toggle('s9-visible', step >= 2);

      if (step >= 3) paintFeatureCard(enc3Card.body, sample.enc3, LEVEL3_PX);
      else paintBlankCard(enc3Card.body, LEVEL3_PX);
      enc3Card.card.classList.toggle('s9-visible', step >= 3);

      // Decoder
      if (step >= 4) paintFeatureCard(dec2Card.body, sample.dec2, LEVEL2_PX);
      else paintBlankCard(dec2Card.body, LEVEL2_PX);
      dec2Card.card.classList.toggle('s9-visible', step >= 4);

      if (step >= 5) paintFeatureCard(dec1Card.body, sample.dec1, LEVEL1_PX);
      else paintBlankCard(dec1Card.body, LEVEL1_PX);
      dec1Card.card.classList.toggle('s9-visible', step >= 5);

      // Skip arrows: lit only on the steps that introduce them.
      arch.classList.toggle('s9-skip2-lit', step >= 4);
      arch.classList.toggle('s9-skip2-active', step === 4);
      arch.classList.toggle('s9-skip1-lit', step >= 5);
      arch.classList.toggle('s9-skip1-active', step === 5);

      // Encoder/decoder flow arrows.
      arch.classList.toggle('s9-down1-lit', step >= 2);
      arch.classList.toggle('s9-down2-lit', step >= 3);
      arch.classList.toggle('s9-up1-lit', step >= 4);
      arch.classList.toggle('s9-up2-lit', step >= 5);

      // Prediction overlay -- always paint the input, GT/Pred from step 6 on.
      paintRGB(ovInputHost, sample.input, OVERLAY_PX);
      if (step >= 6) {
        paintLabelMap(ovGtHost, sample.label, OVERLAY_PX);
        paintLabelMap(ovPredHost, sample.pred, OVERLAY_PX, { diffMask: diffMaskFor(sample) });
        ovGtHost.classList.add('s9-visible');
        ovPredHost.classList.add('s9-visible');
        // Pixel-accuracy badge for THIS sample
        const dm = diffMaskFor(sample);
        let diff = 0, tot = 0;
        for (let i = 0; i < dm.length; i++) for (let j = 0; j < dm[0].length; j++) { tot++; if (dm[i][j]) diff++; }
        const acc = tot ? (1 - diff / tot) : 1;
        predLabelEl.textContent = 'prediction · '
          + (acc * 100).toFixed(2) + '% match';
      } else {
        paintBlankCard(ovGtHost, OVERLAY_PX);
        paintBlankCard(ovPredHost, OVERLAY_PX);
        ovGtHost.classList.remove('s9-visible');
        ovPredHost.classList.remove('s9-visible');
        predLabelEl.textContent = 'prediction';
      }

      // Caption + control widgets.
      caption.textContent = captionFor(step);
      stepInput.value = String(step);
      stepOut.textContent = step + ' / ' + (NUM_STEPS - 1);
      prevBtn.disabled = step <= 0;
      nextBtn.disabled = step >= NUM_STEPS - 1;

      // Re-position SVG arrows -- card sizes don't change but the diagram
      // itself can resize on viewport changes; cheap to re-run.
      requestAnimationFrame(layoutArrows);
    }

    function applyStep(c) {
      state.step = Math.max(0, Math.min(NUM_STEPS - 1, c));
      render();
    }

    /* Switch to a different sample. Resets to step 6 so the user immediately
       sees the new prediction; but keep current step if user is still walking
       through the encoder. */
    function switchSample(idx) {
      if (idx < 0 || idx >= D.samples.length) return;
      state.sampleIdx = idx;
      // If we are past step 6 we stay there; if before, jump to 6 so the
      // overlay updates. The behaviour the user expects is: "click a thumb,
      // see that scene's prediction".
      if (state.step < 6) state.step = 6;
      render();
    }

    /* Step 7 special-case: switch to the next sample and replay rapidly. */
    function maybeAutoSwap() {
      if (state.step !== 7) return;
      const next = (state.sampleIdx + 1) % D.samples.length;
      state.replayedIdx = next;
      state.sampleIdx = next;
      render();
    }

    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () {
      applyStep(state.step + 1);
      maybeAutoSwap();
    });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) {
        applyStep(v);
        maybeAutoSwap();
      }
    });

    /* Initial paint */
    renderThumbs();
    render();
    // First layout pass after the browser has measured. Belt-and-suspenders:
    // two RAFs *and* a microtask, so headless renderers (which sometimes do
    // not fire RAFs as expected) still pick up the correct geometry.
    requestAnimationFrame(function () {
      requestAnimationFrame(layoutArrows);
    });
    setTimeout(layoutArrows, 0);
    setTimeout(layoutArrows, 50);
    setTimeout(layoutArrows, 200);

    const onResize = function () { layoutArrows(); };
    window.addEventListener('resize', onResize);

    /* &run -> auto-advance to step 6 over ~5s, then pause. */
    let runTimer = null;
    function autoAdvance(target) {
      if (state.step >= target) { runTimer = null; return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(function () { autoAdvance(target); }, RUN_INTERVAL_MS);
    }

    if (readHashFlag('run')) {
      runTimer = setTimeout(function () { autoAdvance(6); }, 200);
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
        if (state.step < NUM_STEPS - 1) {
          applyStep(state.step + 1);
          maybeAutoSwap();
          return true;
        }
        return false;
      },
      onPrevKey: function () {
        if (state.step > 0) { applyStep(state.step - 1); return true; }
        return false;
      },
    };
  }

  /* ---------------------------------------------------------------------
     One U-Net level card: header + 2x2 feature thumbnail body.
     --------------------------------------------------------------------- */
  function makeLevelCard(parent, name, sub, px) {
    const card = el('div', { class: 's9-card' }, parent);
    const head = el('div', { class: 's9-card-head' }, card);
    el('span', { class: 's9-card-name', text: name }, head);
    el('span', { class: 's9-card-sub', text: sub }, head);
    const body = el('div', {
      class: 'canvas-host s9-card-body',
      style: 'width:' + px + 'px;height:' + px + 'px;',
    }, card);
    return { card: card, body: body };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene9 = function (root) { return buildScene(root); };
})();
