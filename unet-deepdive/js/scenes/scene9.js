/* Scene 9 — "What the bottleneck threw away".

   The pedagogical centerpiece for skip connections. Side-by-side:
   no-skip prediction vs. with-skip prediction on identical samples.

   IMPORTANT pedagogical reframe (from PLAN §2 Scene 9 empirical note):
   On this small synthetic dataset both models are very accurate in
   aggregate (with-skip ~99.98%, no-skip ~95.4%). The dramatic story is
   not the accuracy number — it is *where* the no-skip errors live.
   They concentrate on 1-pixel-wide object boundaries (tree trunk,
   person silhouette, sun edge), exactly the spatial detail skip
   connections are designed to preserve. So this scene leads with the
   diff overlay, includes a 4× zoom-in on the boundary regions, and
   carries an explicit honesty caption.

   Layout:
     - Hero (h1, italic subtitle, lede).
     - Sample selector strip: 6 thumbnails.
     - Four-panel display: input | ground truth | no-skip pred | with-skip pred.
     - Overlay toggles (diff, zoom rectangle).
     - Decoder channel-count diagram (skip vs. no-skip arithmetic).
     - 4× zoom-in viewer of a chosen 16×16 patch (focused by default on
       the tree-trunk/person region of the selected sample).
     - Receptive-field hover overlay on input when hovering the no-skip
       prediction (with-skip would have *additionally* a tiny enc1 RF;
       we annotate that in the caption).
     - Two pixel-accuracy badges + an honesty caption.
     - Step engine (6 steps): 0 input+GT only -> 1 no-skip pred -> 2
       with-skip pred -> 3 diff overlays -> 4 channel-count diagram ->
       5 receptive-field hover enabled.

   `&run` auto-advances 0 -> 5. */
(function () {
  'use strict';

  const NUM_STEPS = 6;
  const RUN_INTERVAL_MS = 800;

  const CLASS_NAMES = ['sky', 'grass', 'sun', 'tree', 'person'];

  const PANEL_PX = 168;       // each of the four big panels
  const THUMB_PX = 56;        // sample selector
  const ZOOM_PATCH = 16;      // 16x16 input patch -> 4x zoom
  const ZOOM_PX = 256;        // zoom viewer logical edge size (256 / 16 = 16x per patch pixel)
  const RF_INPUT_PX = PANEL_PX; // overlay size = panel size

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

  /* Pick the sample with the most foreground pixels concentrated on a
     thin object (tree trunk + person). Heuristic: maximise count of
     "tree" + "person" pixels that have at least one neighbour of a
     different class — those are the boundary pixels we want to showcase.
     Tie-break to lowest sample index for determinism. */
  function pickRichestSample(samples) {
    let best = 0, bestScore = -1;
    for (let k = 0; k < samples.length; k++) {
      const lbl = samples[k].label;
      const H = lbl.length, W = lbl[0].length;
      let score = 0;
      for (let i = 0; i < H; i++) {
        for (let j = 0; j < W; j++) {
          const c = lbl[i][j];
          if (c !== 3 && c !== 4) continue; // tree or person
          // boundary pixel?
          let isB = false;
          if (i > 0 && lbl[i-1][j] !== c) isB = true;
          else if (i < H-1 && lbl[i+1][j] !== c) isB = true;
          else if (j > 0 && lbl[i][j-1] !== c) isB = true;
          else if (j < W-1 && lbl[i][j+1] !== c) isB = true;
          if (isB) score++;
        }
      }
      if (score > bestScore) { bestScore = score; best = k; }
    }
    return best;
  }

  /* Find a 16x16 patch on the selected sample maximising tree+person
     boundary density, so the zoom viewer lands on something interesting
     by default. Returns {y0, x0}. */
  function pickZoomPatch(label) {
    const H = label.length, W = label[0].length;
    let bestY = 24, bestX = 24, bestS = -1;
    for (let y = 0; y <= H - ZOOM_PATCH; y += 4) {
      for (let x = 0; x <= W - ZOOM_PATCH; x += 4) {
        let s = 0;
        for (let i = 0; i < ZOOM_PATCH; i++) {
          for (let j = 0; j < ZOOM_PATCH; j++) {
            const c = label[y + i][x + j];
            if (c !== 3 && c !== 4) continue;
            // boundary?
            const ii = y + i, jj = x + j;
            const left  = (jj > 0)     ? label[ii][jj-1] : c;
            const right = (jj < W - 1) ? label[ii][jj+1] : c;
            const up    = (ii > 0)     ? label[ii-1][jj] : c;
            const down  = (ii < H - 1) ? label[ii+1][jj] : c;
            if (left !== c || right !== c || up !== c || down !== c) s++;
          }
        }
        if (s > bestS) { bestS = s; bestY = y; bestX = x; }
      }
    }
    return { y0: bestY, x0: bestX };
  }

  /* Diff mask: 1 where pred != gt, 0 otherwise. */
  function diffMask(pred, gt) {
    const H = pred.length, W = pred[0].length;
    const m = window.UNET.zeros2D(H, W);
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        m[i][j] = (pred[i][j] !== gt[i][j]) ? 1 : 0;
      }
    }
    return m;
  }

  function pixelAcc(pred, gt) {
    const H = pred.length, W = pred[0].length;
    let same = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) if (pred[i][j] === gt[i][j]) same++;
    }
    return same / (H * W);
  }

  /* Boundary pixel count on the GT label (for the honesty caption stats).
     A pixel is "on a boundary" if at least one 4-neighbour differs. */
  function boundaryCount(gt) {
    const H = gt.length, W = gt[0].length;
    let n = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        const c = gt[i][j];
        if ((i > 0 && gt[i-1][j] !== c) ||
            (i < H-1 && gt[i+1][j] !== c) ||
            (j > 0 && gt[i][j-1] !== c) ||
            (j < W-1 && gt[i][j+1] !== c)) n++;
      }
    }
    return n;
  }

  /* Of the predictions's errors, how many fall on GT boundary pixels? */
  function errorsOnBoundary(pred, gt) {
    const H = gt.length, W = gt[0].length;
    let onB = 0, total = 0;
    for (let i = 0; i < H; i++) {
      for (let j = 0; j < W; j++) {
        if (pred[i][j] === gt[i][j]) continue;
        total++;
        const c = gt[i][j];
        if ((i > 0 && gt[i-1][j] !== c) ||
            (i < H-1 && gt[i+1][j] !== c) ||
            (j > 0 && gt[i][j-1] !== c) ||
            (j < W-1 && gt[i][j+1] !== c) ||
            (i > 1 && gt[i-2][j] !== c) ||
            (i < H-2 && gt[i+2][j] !== c) ||
            (j > 1 && gt[i][j-2] !== c) ||
            (j < W-2 && gt[i][j+2] !== c)) onB++;
      }
    }
    return { onB: onB, total: total };
  }

  /* Paint a zoomed patch of an HxWx3 RGB array (input pixels) using
     nearest-neighbour scaling — pixel-perfect zoom. */
  function paintRGBPatch(host, rgb, y0, x0, patchSize, px) {
    host.innerHTML = '';
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const off = document.createElement('canvas');
    off.width = patchSize; off.height = patchSize;
    const offCtx = off.getContext('2d');
    const id = offCtx.createImageData(patchSize, patchSize);
    let p = 0;
    for (let i = 0; i < patchSize; i++) {
      for (let j = 0; j < patchSize; j++) {
        const r = rgb[y0 + i][x0 + j][0];
        const g = rgb[y0 + i][x0 + j][1];
        const b = rgb[y0 + i][x0 + j][2];
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

  /* Paint a zoomed patch of a label map. Optional diffMask outlines diffs. */
  function paintLabelPatch(host, lbl, y0, x0, patchSize, px, opts) {
    opts = opts || {};
    host.innerHTML = '';
    const setup = window.Drawing.setupCanvas(host, px, px);
    const ctx = setup.ctx;
    const colors = opts.colors || window.Drawing.readClassColors();
    const off = document.createElement('canvas');
    off.width = patchSize; off.height = patchSize;
    const offCtx = off.getContext('2d');
    const id = offCtx.createImageData(patchSize, patchSize);
    let p = 0;
    for (let i = 0; i < patchSize; i++) {
      for (let j = 0; j < patchSize; j++) {
        const c = lbl[y0 + i][x0 + j] | 0;
        const hex = colors[c] || '#888';
        const rgb = window.Drawing.parseHex(hex);
        id.data[p++] = rgb[0];
        id.data[p++] = rgb[1];
        id.data[p++] = rgb[2];
        id.data[p++] = 255;
      }
    }
    offCtx.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, px, px);

    if (opts.diffMask) {
      const cw = px / patchSize, ch = px / patchSize;
      const t = window.Drawing.tokens();
      ctx.strokeStyle = t.ink;
      ctx.lineWidth = 1.5;
      for (let i = 0; i < patchSize; i++) {
        for (let j = 0; j < patchSize; j++) {
          if (opts.diffMask[y0 + i][x0 + j]) {
            ctx.strokeRect(j * cw + 0.75, i * ch + 0.75, cw - 1.5, ch - 1.5);
          }
        }
      }
    }
    // Faint grid for the zoom viewer so individual pixels are countable.
    const t = window.Drawing.tokens();
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 0.5;
    ctx.globalAlpha = 0.5;
    const cw = px / patchSize;
    for (let k = 0; k <= patchSize; k++) {
      ctx.beginPath();
      ctx.moveTo(k * cw, 0); ctx.lineTo(k * cw, px); ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, k * cw); ctx.lineTo(px, k * cw); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64 || !window.DATA.noskip) {
      root.innerHTML = '<p style="opacity:0.5">scene9 data missing (need scene64 + noskip).</p>';
      return {};
    }
    const D  = window.DATA.scene64;
    const NS = window.DATA.noskip;
    const RF = window.DATA.rfields || null;

    // Build a quick lookup from sample index -> noskip pred. The noskip
    // dump uses .index keys matching scene64.samples[k].
    const noskipPredByIdx = {};
    for (let k = 0; k < NS.samples.length; k++) {
      noskipPredByIdx[NS.samples[k].index] = NS.samples[k].pred;
    }
    // Restrict the selector to samples for which we have a no-skip pred.
    const sampleIndices = [];
    for (let k = 0; k < D.samples.length; k++) {
      if (noskipPredByIdx[k]) sampleIndices.push(k);
    }

    root.innerHTML = '';
    root.classList.add('s9-root');
    const wrap = el('div', { class: 's9-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s9-hero' }, wrap);
    el('h1', { text: 'What the bottleneck threw away.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Two U-Nets, identical except for one wire. The wire is the punchline.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'We trained two U-Nets on the same data, with the same recipe. ' +
        'One has the encoder-to-decoder skip connections; the other does not. ' +
        'Both end up almost equally accurate <em>in aggregate</em>. ' +
        'But look at <em>where</em> the no-skip model is wrong.',
    }, hero);

    /* ---- Sample selector ------------------------------------------- */
    const selectorStrip = el('div', { class: 's9-selector' }, wrap);
    el('div', { class: 's9-selector-label', text: 'sample' }, selectorStrip);
    const selectorRow = el('div', { class: 's9-selector-row' }, selectorStrip);
    const thumbHosts = [];
    for (let i = 0; i < sampleIndices.length; i++) {
      const sIdx = sampleIndices[i];
      const btn = el('button', {
        type: 'button',
        class: 's9-thumb',
        'data-sample-index': String(sIdx),
        'aria-label': 'Select sample ' + (i + 1),
      }, selectorRow);
      const tHost = el('div', { class: 'canvas-host s9-thumb-host' }, btn);
      thumbHosts.push(tHost);
      btn.addEventListener('click', function () {
        const idx = parseInt(btn.getAttribute('data-sample-index'), 10);
        switchSample(idx);
      });
    }

    /* ---- Four-panel display ---------------------------------------- */
    const panels = el('div', { class: 's9-panels' }, wrap);

    function makePanel(parent, title, sub, klass) {
      const p = el('div', { class: 's9-panel ' + (klass || '') }, parent);
      el('div', { class: 's9-panel-title', text: title }, p);
      el('div', { class: 's9-panel-sub', text: sub || '' }, p);
      const host = el('div', { class: 'canvas-host s9-panel-host' }, p);
      const badge = el('div', { class: 's9-panel-badge' }, p);
      return { panel: p, host: host, badge: badge };
    }

    const pInput   = makePanel(panels, 'input', '64×64×3', 's9-p-input');
    const pGT      = makePanel(panels, 'ground truth', '64×64 label', 's9-p-gt');
    const pNoSkip  = makePanel(panels, 'no-skip pred', 'decoder uses up-only path', 's9-p-noskip');
    const pSkip    = makePanel(panels, 'with-skip pred', 'decoder concats encoder maps', 's9-p-skip');

    /* ---- Class legend ----------------------------------------------- */
    const legend = el('div', { class: 's9-legend' }, wrap);
    for (let c = 0; c < CLASS_NAMES.length; c++) {
      const item = el('div', { class: 's9-legend-item' }, legend);
      el('span', { class: 's9-swatch class-' + CLASS_NAMES[c] }, item);
      el('span', { class: 's9-legend-name', text: CLASS_NAMES[c] }, item);
    }

    /* ---- Zoom viewer ----------------------------------------------- */
    const zoomBox = el('div', { class: 's9-zoom-box' }, wrap);
    const zoomHead = el('div', { class: 's9-zoom-head' }, zoomBox);
    el('div', { class: 's9-zoom-title', text: '4× zoom · same patch on all four' }, zoomHead);
    const zoomCoords = el('div', { class: 's9-zoom-coords' }, zoomHead);
    el('div', {
      class: 's9-zoom-help',
      text: 'click any panel above to move the zoom window',
    }, zoomHead);

    const zoomGrid = el('div', { class: 's9-zoom-grid' }, zoomBox);
    function makeZoomCell(parent, label) {
      const cell = el('div', { class: 's9-zoom-cell' }, parent);
      el('div', { class: 's9-zoom-cell-label', text: label }, cell);
      const host = el('div', { class: 'canvas-host s9-zoom-host' }, cell);
      return host;
    }
    const zInput  = makeZoomCell(zoomGrid, 'input');
    const zGT     = makeZoomCell(zoomGrid, 'ground truth');
    const zNoSkip = makeZoomCell(zoomGrid, 'no-skip pred');
    const zSkip   = makeZoomCell(zoomGrid, 'with-skip pred');

    /* ---- Channel-count diagram (step 4) ----------------------------- */
    const channels = el('div', { class: 's9-channels' }, wrap);
    el('div', { class: 's9-channels-title', text: 'decoder input shapes' }, channels);
    el('div', {
      class: 's9-channels-sub',
      text: 'Same up-convolution. Different number of channels going into each conv-block.',
    }, channels);

    const chRow = el('div', { class: 's9-channels-row' }, channels);

    function makeChCell(parent, klass, headText) {
      const c = el('div', { class: 's9-ch-cell ' + klass }, parent);
      el('div', { class: 's9-ch-head', text: headText }, c);
      const body = el('div', { class: 's9-ch-body' }, c);
      return body;
    }

    const chNoSkipBody = makeChCell(chRow, 's9-ch-noskip', 'no-skip decoder');
    chNoSkipBody.innerHTML =
      '<div class="s9-ch-line"><span class="s9-ch-tile s9-ch-up">up2 · 32 ch</span>' +
      ' <span class="s9-ch-arrow">→</span>' +
      ' <span class="s9-ch-tile s9-ch-conv">conv-block(<strong>32</strong>→32)</span></div>' +
      '<div class="s9-ch-line"><span class="s9-ch-tile s9-ch-up">up1 · 16 ch</span>' +
      ' <span class="s9-ch-arrow">→</span>' +
      ' <span class="s9-ch-tile s9-ch-conv">conv-block(<strong>16</strong>→16)</span></div>' +
      '<div class="s9-ch-foot">decoder sees only the up-sampled bottleneck.</div>';

    const chSkipBody = makeChCell(chRow, 's9-ch-skip', 'with-skip decoder');
    chSkipBody.innerHTML =
      '<div class="s9-ch-line"><span class="s9-ch-tile s9-ch-up">up2 · 32 ch</span>' +
      ' <span class="s9-ch-plus">⊕</span>' +
      ' <span class="s9-ch-tile s9-ch-skip-tile">enc2 · 32 ch</span>' +
      ' <span class="s9-ch-arrow">→</span>' +
      ' <span class="s9-ch-tile s9-ch-conv">conv-block(<strong>64</strong>→32)</span></div>' +
      '<div class="s9-ch-line"><span class="s9-ch-tile s9-ch-up">up1 · 16 ch</span>' +
      ' <span class="s9-ch-plus">⊕</span>' +
      ' <span class="s9-ch-tile s9-ch-skip-tile">enc1 · 16 ch</span>' +
      ' <span class="s9-ch-arrow">→</span>' +
      ' <span class="s9-ch-tile s9-ch-conv">conv-block(<strong>32</strong>→16)</span></div>' +
      '<div class="s9-ch-foot">decoder also sees the early, sharp encoder maps.</div>';

    /* ---- Receptive-field overlay (step 5) -------------------------- */
    // The RF overlay is drawn directly on top of the input panel using a
    // floating absolutely-positioned rectangle inside the panel host.
    const rfOverlay = document.createElement('div');
    rfOverlay.className = 's9-rf-overlay';
    rfOverlay.style.display = 'none';
    pInput.host.style.position = 'relative';
    pInput.host.appendChild(rfOverlay);

    const rfHelp = el('div', {
      class: 's9-rf-help',
      html:
        '<strong>step 5 · hover the no-skip panel.</strong> ' +
        'The yellow box on the input shows the bottleneck cell that "owns" that pixel — ' +
        'a roughly 32×32 patch (after three pools). ' +
        'The with-skip decoder additionally sees a tiny 1×1 enc1 cell at the same location, ' +
        'which is where the sharp boundary information comes from.',
    }, wrap);

    /* ---- Honesty caption / accuracy badges ------------------------- */
    const honesty = el('div', { class: 's9-honesty' }, wrap);
    const honestyTitle = el('div', { class: 's9-honesty-title' }, honesty);
    honestyTitle.textContent = 'pixel accuracy on the test set';
    const honestyRow = el('div', { class: 's9-honesty-row' }, honesty);

    const accBadgeNS = el('div', { class: 's9-acc s9-acc-ns' }, honestyRow);
    el('span', { class: 's9-acc-label', text: 'no-skip' }, accBadgeNS);
    el('span', {
      class: 's9-acc-value',
      text: (NS.meanPixelAccuracy * 100).toFixed(2) + '%',
    }, accBadgeNS);

    const accBadgeWS = el('div', { class: 's9-acc s9-acc-ws' }, honestyRow);
    el('span', { class: 's9-acc-label', text: 'with-skip' }, accBadgeWS);
    el('span', {
      class: 's9-acc-value',
      text: (D.meanPixelAccuracy * 100).toFixed(2) + '%',
    }, accBadgeWS);

    el('p', {
      class: 's9-honesty-note',
      html:
        '<strong>An honest note.</strong> Both numbers look good because this dataset ' +
        'is small and synthetic. The interesting story is <em>where</em> the no-skip model is wrong: ' +
        'its errors cluster on 1-pixel-wide object boundaries — exactly the spatial detail that ' +
        'pooling threw away and that the skip connection puts back. ' +
        'On harder data (street scenes, biomedical slices) the gap becomes much larger.',
    }, honesty);

    /* ---- Caption + step controls ----------------------------------- */
    const caption = el('p', { class: 'caption s9-caption' }, wrap);

    const controls = el('div', { class: 'controls s9-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's9-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's9-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', {
      class: 'control-value', text: '0 / ' + (NUM_STEPS - 1),
    }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State ----------------------------------------------------- */
    const initialIdx = sampleIndices.length
      ? sampleIndices[0]
      : pickRichestSample(D.samples);
    const initSample = D.samples[initialIdx];
    const initPatch = pickZoomPatch(initSample.label);

    const state = {
      step: 0,
      sampleIdx: initialIdx,
      zoomY: initPatch.y0,
      zoomX: initPatch.x0,
      hoverI: -1,
      hoverJ: -1,
    };

    /* ---- Painters / renderers -------------------------------------- */

    function renderThumbs() {
      for (let i = 0; i < sampleIndices.length; i++) {
        window.Drawing.paintRGB(thumbHosts[i], D.samples[sampleIndices[i]].input, THUMB_PX);
      }
      updateThumbActive();
    }
    function updateThumbActive() {
      const btns = selectorRow.querySelectorAll('.s9-thumb');
      btns.forEach(function (b) {
        const idx = parseInt(b.getAttribute('data-sample-index'), 10);
        b.classList.toggle('active', idx === state.sampleIdx);
      });
    }

    function captionFor(step) {
      switch (step) {
        case 0: return 'A test sample and its ground-truth label. Both U-Nets see this same input.';
        case 1: return 'No-skip prediction. The shapes are mostly there, but the boundaries are smudged. The decoder is trying to draw thin objects from a 16×16 summary.';
        case 2: return 'With-skip prediction. Boundaries snap into place. Same encoder, same decoder — only the skip wires were added.';
        case 3: return 'Diff overlay: every cell where the prediction disagrees with the ground truth is outlined. The no-skip errors form 1-pixel halos around objects. The with-skip diff is almost empty.';
        case 4: return 'Where does the difference come from? The no-skip decoder takes only the upsampled tensor. The with-skip decoder concatenates the matching encoder feature map alongside it.';
        case 5: return 'Hover the no-skip panel. Each pixel’s prediction depends on a ~32×32 receptive field in the input — far too coarse for a 1-pixel-thin tree trunk. The skip wires add a 1×1 path from the early encoder that fixes exactly this.';
        default: return '';
      }
    }

    function withSkipSample() { return D.samples[state.sampleIdx]; }
    function noSkipPred() { return noskipPredByIdx[state.sampleIdx]; }

    function renderPanels() {
      const step = state.step;
      const sample = withSkipSample();
      const nsPred = noSkipPred();

      // Always: input & GT
      window.Drawing.paintRGB(pInput.host, sample.input, PANEL_PX);
      // Re-attach the absolute RF overlay element after host innerHTML is rewritten.
      pInput.host.style.position = 'relative';
      pInput.host.appendChild(rfOverlay);

      window.Drawing.paintLabelMap(pGT.host, sample.label, PANEL_PX);

      const showNS = step >= 1;
      const showWS = step >= 2;
      const showDiff = step >= 3;

      if (showNS) {
        const dm = showDiff ? diffMask(nsPred, sample.label) : null;
        window.Drawing.paintLabelMap(pNoSkip.host, nsPred, PANEL_PX, { diffMask: dm });
        const acc = pixelAcc(nsPred, sample.label);
        const eb = errorsOnBoundary(nsPred, sample.label);
        const onBPct = eb.total ? (100 * eb.onB / eb.total) : 0;
        pNoSkip.badge.innerHTML =
          '<span class="s9-bdg-acc">' + (acc * 100).toFixed(2) + '% match</span>' +
          (eb.total ? ('<span class="s9-bdg-bnd">' + onBPct.toFixed(0) +
            '% of errors on boundary</span>') : '');
        pNoSkip.panel.classList.add('s9-visible');
      } else {
        window.Drawing.paintBlankCard(pNoSkip.host, PANEL_PX);
        pNoSkip.badge.textContent = '';
        pNoSkip.panel.classList.remove('s9-visible');
      }

      if (showWS) {
        const dm = showDiff ? diffMask(sample.pred, sample.label) : null;
        window.Drawing.paintLabelMap(pSkip.host, sample.pred, PANEL_PX, { diffMask: dm });
        const acc = pixelAcc(sample.pred, sample.label);
        pSkip.badge.innerHTML =
          '<span class="s9-bdg-acc">' + (acc * 100).toFixed(2) + '% match</span>';
        pSkip.panel.classList.add('s9-visible');
      } else {
        window.Drawing.paintBlankCard(pSkip.host, PANEL_PX);
        pSkip.badge.textContent = '';
        pSkip.panel.classList.remove('s9-visible');
      }

      // Click-to-move-zoom on any of the four panels (only after step >= 1).
      attachPanelClicks();

      // RF overlay visibility + position
      const rfEnabled = step >= 5 && RF;
      if (!rfEnabled) {
        rfOverlay.style.display = 'none';
      }
      // (positioning happens on hover; we hide otherwise)
      attachRFHover(rfEnabled);

      // Diagram + RF help reveal classes
      channels.classList.toggle('s9-visible', step >= 4);
      rfHelp.classList.toggle('s9-visible', step >= 5);
      panels.classList.toggle('s9-diff-on', step >= 3);
    }

    function renderZoom() {
      const sample = withSkipSample();
      const nsPred = noSkipPred();
      const y0 = state.zoomY, x0 = state.zoomX;
      paintRGBPatch(zInput, sample.input, y0, x0, ZOOM_PATCH, ZOOM_PX);

      const dmGT = null;
      paintLabelPatch(zGT, sample.label, y0, x0, ZOOM_PATCH, ZOOM_PX);

      const showDiff = state.step >= 3;
      const dmNS = showDiff ? diffMask(nsPred, sample.label) : null;
      paintLabelPatch(zNoSkip, nsPred, y0, x0, ZOOM_PATCH, ZOOM_PX, { diffMask: dmNS });

      const dmWS = showDiff ? diffMask(sample.pred, sample.label) : null;
      paintLabelPatch(zSkip, sample.pred, y0, x0, ZOOM_PATCH, ZOOM_PX, { diffMask: dmWS });

      zoomCoords.textContent =
        'window: rows ' + y0 + '–' + (y0 + ZOOM_PATCH - 1) +
        ' · cols ' + x0 + '–' + (x0 + ZOOM_PATCH - 1);

      // Draw the zoom rectangle on each big panel so the relationship is
      // explicit. We do this by overlaying an SVG-free DOM rectangle.
      drawZoomRect();
    }

    function drawZoomRect() {
      [pInput, pGT, pNoSkip, pSkip].forEach(function (P) {
        // Remove old rect
        const old = P.host.querySelector('.s9-zoom-rect');
        if (old) old.remove();
        const r = document.createElement('div');
        r.className = 's9-zoom-rect';
        const left = (state.zoomX / 64) * PANEL_PX;
        const top  = (state.zoomY / 64) * PANEL_PX;
        const w    = (ZOOM_PATCH / 64) * PANEL_PX;
        r.style.cssText =
          'position:absolute;left:' + left + 'px;top:' + top + 'px;' +
          'width:' + w + 'px;height:' + w + 'px;pointer-events:none;';
        P.host.style.position = 'relative';
        P.host.appendChild(r);
      });
    }

    /* Click on any of the four panels to recenter the zoom window. */
    function attachPanelClicks() {
      [pInput, pGT, pNoSkip, pSkip].forEach(function (P) {
        const cv = P.host.querySelector('canvas');
        if (!cv) return;
        cv.style.cursor = 'crosshair';
        cv.onclick = function (ev) {
          const rect = cv.getBoundingClientRect();
          const x = ev.clientX - rect.left;
          const y = ev.clientY - rect.top;
          const cj = Math.floor(x / rect.width * 64);
          const ci = Math.floor(y / rect.height * 64);
          // Clamp the patch so it stays inside 0..64-PATCH
          const x0 = Math.max(0, Math.min(64 - ZOOM_PATCH, cj - ZOOM_PATCH / 2));
          const y0 = Math.max(0, Math.min(64 - ZOOM_PATCH, ci - ZOOM_PATCH / 2));
          state.zoomX = x0; state.zoomY = y0;
          renderZoom();
        };
      });
    }

    /* Hover on the no-skip panel (step >= 5) shows the bottleneck-cell
       receptive field as a yellow rectangle on the input panel. */
    function attachRFHover(enabled) {
      const cv = pNoSkip.host.querySelector('canvas');
      if (!cv) return;
      if (!enabled || !RF) {
        cv.onmousemove = null;
        cv.onmouseleave = null;
        rfOverlay.style.display = 'none';
        pNoSkip.panel.classList.remove('s9-rf-on');
        return;
      }
      pNoSkip.panel.classList.add('s9-rf-on');
      cv.onmousemove = function (ev) {
        const rect = cv.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        const j = Math.max(0, Math.min(63, Math.floor(x / rect.width * 64)));
        const i = Math.max(0, Math.min(63, Math.floor(y / rect.height * 64)));
        if (i === state.hoverI && j === state.hoverJ) return;
        state.hoverI = i; state.hoverJ = j;
        // Map (i, j) in 64x64 input coords to the bottleneck cell that
        // owns it. Bottleneck is 16x16 after 3 pools of factor 2 (=8).
        // Clamp; rfields.cells is 256 entries (16x16) row-major in (i, j).
        const bi = Math.max(0, Math.min(15, Math.floor(i / 4)));
        const bj = Math.max(0, Math.min(15, Math.floor(j / 4)));
        const cell = RF.cells[bi * 16 + bj];
        if (!cell) return;
        // Draw the RF box on the INPUT panel.
        const sx = PANEL_PX / 64;
        rfOverlay.style.display = 'block';
        rfOverlay.style.left   = (cell.x0 * sx) + 'px';
        rfOverlay.style.top    = (cell.y0 * sx) + 'px';
        rfOverlay.style.width  = ((cell.x1 - cell.x0) * sx) + 'px';
        rfOverlay.style.height = ((cell.y1 - cell.y0) * sx) + 'px';
      };
      cv.onmouseleave = function () {
        // Keep last frame visible for a moment; pure hide feels jumpy.
        rfOverlay.style.opacity = '0.45';
        setTimeout(function () { rfOverlay.style.opacity = ''; }, 200);
      };
    }

    function render() {
      const step = state.step;
      updateThumbActive();
      renderPanels();
      renderZoom();
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
      if (!noskipPredByIdx[idx]) return;
      state.sampleIdx = idx;
      const patch = pickZoomPatch(D.samples[idx].label);
      state.zoomY = patch.y0; state.zoomX = patch.x0;
      // If the user is still in the "no overlays" steps, jump to step 2 so
      // they immediately see the contrast.
      if (state.step < 2) state.step = 2;
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

    /* &run -> auto-advance to last step. */
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
      onEnter: function () { render(); },
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
  window.scenes.scene9 = function (root) { return buildScene(root); };
})();
