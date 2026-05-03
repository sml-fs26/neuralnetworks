/* Scene 6 -- "Receptive fields -- the cone of vision."

   The structural beat. Each column shows a layer; each layer shows a few
   of its feature-map channels as small canvases. The student clicks any
   cell in any feature map; the corresponding receptive field is drawn
   as a translucent box on the input image, and a soft connecting line
   walks from the input rect to the selected cell. The cone widens from
   layer to layer.

   Data: the cross sample (samples[0]) -- most legible activation pattern.
   Channels shown per layer are picked once based on activation energy
   so the maps actually carry a signal.

   `&run` pre-selects a conv3 neuron at (3, 3) so the screenshot shows
   the largest cone. */
(function () {
  'use strict';

  // --- Layer table. row/col are in feature-map coords; size_in is the
  //     side length of one feature map for that layer. ---
  const LAYERS = [
    { key: 'conv1', label: 'conv1',  size: 28, mapsTotal: 8,  panelPx: 84,  cell: 3 },
    { key: 'pool1', label: 'pool1',  size: 14, mapsTotal: 8,  panelPx: 84,  cell: 6 },
    { key: 'conv2', label: 'conv2',  size: 14, mapsTotal: 16, panelPx: 84,  cell: 6 },
    { key: 'pool2', label: 'pool2',  size: 7,  mapsTotal: 16, panelPx: 84,  cell: 12 },
    { key: 'conv3', label: 'conv3',  size: 7,  mapsTotal: 24, panelPx: 84,  cell: 12 },
  ];

  // Captions per layer, italic, beneath each column. Shows the actual RF
  // size from CNN.receptiveField.
  const LAYER_CAPTIONS = {
    conv1: 'Sees 5x5.',
    pool1: 'Sees 6x6 (max-pool widens slightly).',
    conv2: 'Sees 14x14.',
    pool2: 'Sees 16x16.',
    conv3: 'Sees 24x24 -- almost everything.',
  };

  // Operation per layer — what the network actually does at that step.
  // Shown above each column so the audience knows the filter / pool
  // dimensions that produce the cone widening.
  const LAYER_OPS = {
    conv1: '5×5 conv · 8 filters · pad 2',
    pool1: '2×2 max-pool · stride 2',
    conv2: '5×5 conv · 16 filters · pad 2',
    pool2: '2×2 max-pool · stride 2',
    conv3: '3×3 conv · 24 filters · pad 1',
  };

  const INPUT_PX = 220;       // 28 cells * ~7.85 px logical -- hero panel
  const CHANNELS_PER_LAYER = 4;

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

  /* Pick the top-K channels of a feature stack by total absolute activation.
     Stable ordering: ties break by channel index. */
  function topChannels(stack, k) {
    const scored = stack.map((map, idx) => {
      let s = 0;
      for (const row of map) for (const v of row) s += Math.abs(v);
      return { idx, s };
    });
    scored.sort((a, b) => (b.s - a.s) || (a.idx - b.idx));
    return scored.slice(0, k).map(o => o.idx).sort((a, b) => a - b);
  }

  function symmetricMax(map) {
    let m = 0;
    for (const row of map) for (const v of row) {
      const a = Math.abs(v);
      if (a > m) m = a;
    }
    return m || 1;
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.shapelets || !window.DATA.shapelets.samples) {
      root.innerHTML = '<p style="opacity:0.5">Scene 6: shapelets data missing.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s6-root');
    const wrap = el('div', { class: 's6-wrap' }, root);

    // ---- Hero ---------------------------------------------------------
    const hero = el('header', { class: 'hero s6-hero' }, wrap);
    el('h1', { text: 'Receptive fields -- the cone of vision.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'Click a neuron. See what it can see.',
    }, hero);
    el('p', {
      class: 'lede',
      text: 'Each layer sees a wider patch of the input than the last. A conv1 neuron looks through a 5x5 window. A conv3 neuron looks through almost the entire image.',
    }, hero);

    // ---- Layout shell --------------------------------------------------
    // Two-row main layout:
    //   row 1: input panel | columns of feature-map tiles
    //   row 2: readout strip below (layer / row,col / RF size)
    const board = el('div', { class: 's6-board' }, wrap);

    // SVG overlay for the connection line from input rect to selected tile.
    // Sits absolutely over the board; pointer-events:none.
    const overlaySvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    overlaySvg.setAttribute('class', 's6-overlay');
    overlaySvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    overlaySvg.setAttribute('aria-hidden', 'true');
    board.appendChild(overlaySvg);

    // -- Input column --
    const inputCol = el('div', { class: 's6-col s6-col-input' }, board);
    el('div', { class: 's6-col-label', text: 'input (28x28)' }, inputCol);
    const inputHost = el('div', { class: 'canvas-host s6-input-host' }, inputCol);
    const inputCaption = el('div', { class: 's6-input-caption caption' }, inputCol);

    // -- One column per layer --
    const layerHosts = {};
    LAYERS.forEach((layer) => {
      const col = el('div', { class: 's6-col s6-col-layer' }, board);
      el('div', { class: 's6-col-label', text: layer.label }, col);
      el('div', {
        class: 's6-layer-op',
        text: LAYER_OPS[layer.key],
      }, col);
      const tiles = el('div', { class: 's6-tiles' }, col);
      el('div', {
        class: 'caption s6-layer-caption',
        text: LAYER_CAPTIONS[layer.key],
      }, col);
      layerHosts[layer.key] = { col, tiles, panels: [] };
    });

    // ---- Readout ------------------------------------------------------
    const readout = el('div', { class: 's6-readout card' }, wrap);
    const readoutLayer = el('span', { class: 's6-readout-layer', text: '-' }, readout);
    el('span', { class: 's6-readout-sep', text: '·' }, readout);
    const readoutCell = el('span', { class: 's6-readout-cell', text: '(-, -)' }, readout);
    el('span', { class: 's6-readout-sep', text: '·' }, readout);
    const readoutRF = el('span', { class: 's6-readout-rf', text: 'RF -' }, readout);

    el('p', {
      class: 'caption s6-bottom-caption',
      text: 'A neuron deep in the network has a wide window onto the input. Stacking convolutions pushes it open, layer by layer.',
    }, wrap);

    // ---- Data ---------------------------------------------------------
    const sample = window.DATA.shapelets.samples[0]; // cross
    const inputData = sample.input;
    const stacks = {
      conv1: sample.conv1Out,
      pool1: sample.pool1Out,
      conv2: sample.conv2Out,
      pool2: sample.pool2Out,
      conv3: sample.conv3Out,
    };
    // Pick channels once.
    const layerChannels = {};
    LAYERS.forEach((layer) => {
      layerChannels[layer.key] = topChannels(stacks[layer.key], CHANNELS_PER_LAYER);
    });
    // Per-channel symmetric range for diverging color.
    const layerVMax = {};
    LAYERS.forEach((layer) => {
      let m = 0;
      for (const ch of layerChannels[layer.key]) {
        const v = symmetricMax(stacks[layer.key][ch]);
        if (v > m) m = v;
      }
      layerVMax[layer.key] = m || 1;
    });

    // ---- State --------------------------------------------------------
    const state = {
      // selection: { layer, channel, row, col } or null
      selection: null,
    };

    // ---- Painters -----------------------------------------------------
    function paintInput() {
      inputHost.innerHTML = '';
      const { ctx, w, h } = window.Drawing.setupCanvas(inputHost, INPUT_PX, INPUT_PX);
      const t = window.Drawing.tokens();
      ctx.fillStyle = t.bg;
      ctx.fillRect(0, 0, w, h);
      window.Drawing.drawGrid(ctx, inputData, 0, 0, w, h, {
        diverging: false, valueRange: [0, 1],
      });
      ctx.lineWidth = 1;
      ctx.strokeStyle = t.rule;
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

      // Receptive-field rectangle, if a neuron is selected.
      if (state.selection) {
        const sel = state.selection;
        const rf = window.CNN.receptiveField(sel.layer, sel.row, sel.col);
        const cellPx = w / 28;
        // Clip to the visible 28x28; rf can extend negative or past 28 due
        // to padding. For drawing, intersect with the panel.
        const x0 = rf.left * cellPx;
        const y0 = rf.top  * cellPx;
        const wPx = rf.size * cellPx;
        const hPx = rf.size * cellPx;
        // Translucent fill -- accent on top of the input.
        ctx.fillStyle = withAlpha(t.pos, 0.18);
        ctx.fillRect(x0, y0, wPx, hPx);
        ctx.lineWidth = 2;
        ctx.strokeStyle = t.pos;
        ctx.strokeRect(x0 + 0.5, y0 + 0.5, wPx - 1, hPx - 1);
      }

      // The italic input caption tells us which sample we're staring at.
      inputCaption.textContent =
        'A cross. Pick a neuron at right; the box at left shows its window.';
    }

    function paintLayerPanels() {
      LAYERS.forEach((layer) => {
        const host = layerHosts[layer.key];
        host.tiles.innerHTML = '';
        host.panels = [];
        const channels = layerChannels[layer.key];
        const vmax = layerVMax[layer.key];

        channels.forEach((chIdx) => {
          const map = stacks[layer.key][chIdx];
          const panel = el('div', { class: 's6-panel' }, host.tiles);
          panel.dataset.layer = layer.key;
          panel.dataset.channel = String(chIdx);

          const cvHost = el('div', { class: 'canvas-host s6-panel-canvas' }, panel);
          const cv = window.Drawing.setupCanvas(cvHost, layer.panelPx, layer.panelPx);
          // Draw the heatmap.
          const t = window.Drawing.tokens();
          cv.ctx.fillStyle = t.bg;
          cv.ctx.fillRect(0, 0, layer.panelPx, layer.panelPx);
          window.Drawing.drawGrid(cv.ctx, map, 0, 0, layer.panelPx, layer.panelPx, {
            diverging: true, valueRange: [-vmax, vmax],
          });
          // Faint border.
          cv.ctx.lineWidth = 1;
          cv.ctx.strokeStyle = t.rule;
          cv.ctx.strokeRect(0.5, 0.5, layer.panelPx - 1, layer.panelPx - 1);

          // Highlight the selected cell.
          if (state.selection &&
              state.selection.layer === layer.key &&
              state.selection.channel === chIdx) {
            const cellPx = layer.panelPx / layer.size;
            const sx = state.selection.col * cellPx;
            const sy = state.selection.row * cellPx;
            cv.ctx.lineWidth = 2;
            cv.ctx.strokeStyle = t.pos;
            cv.ctx.strokeRect(sx + 0.5, sy + 0.5, cellPx - 1, cellPx - 1);
            panel.classList.add('selected');
          } else {
            panel.classList.remove('selected');
          }

          el('div', { class: 's6-panel-tag', text: 'ch ' + chIdx }, panel);

          // Click handler: pick a neuron.
          panel.addEventListener('click', (ev) => {
            const rect = cv.canvas.getBoundingClientRect();
            const x = ev.clientX - rect.left;
            const y = ev.clientY - rect.top;
            const col = Math.max(0, Math.min(layer.size - 1,
              Math.floor((x / rect.width) * layer.size)));
            const row = Math.max(0, Math.min(layer.size - 1,
              Math.floor((y / rect.height) * layer.size)));
            select(layer.key, chIdx, row, col);
          });

          host.panels.push({ chIdx, canvas: cv.canvas, panel });
        });
      });
    }

    function paintReadout() {
      if (!state.selection) {
        readoutLayer.textContent = 'no selection';
        readoutCell.textContent = '';
        readoutRF.textContent = 'click any cell to choose a neuron';
        return;
      }
      const { layer, channel, row, col } = state.selection;
      const rf = window.CNN.receptiveField(layer, row, col);
      readoutLayer.textContent = layer + '  ch ' + channel;
      readoutCell.textContent = '(row ' + row + ', col ' + col + ')';
      readoutRF.textContent = 'RF ' + rf.size + 'x' + rf.size +
        '  at input (top ' + rf.top + ', left ' + rf.left + ')';
    }

    /* Draw a soft curve from the input panel's RF rect to the selected tile,
       using SVG so it sits above all canvases. */
    function paintConnector() {
      // Clear old connectors.
      while (overlaySvg.firstChild) overlaySvg.removeChild(overlaySvg.firstChild);
      if (!state.selection) return;

      const boardRect = board.getBoundingClientRect();
      // Resize SVG to match board dimensions.
      overlaySvg.setAttribute('viewBox',
        '0 0 ' + boardRect.width + ' ' + boardRect.height);
      overlaySvg.setAttribute('width', boardRect.width);
      overlaySvg.setAttribute('height', boardRect.height);

      // Source: the RF rectangle on the input canvas.
      const inputCanvas = inputHost.querySelector('canvas');
      if (!inputCanvas) return;
      const inputRect = inputCanvas.getBoundingClientRect();
      const sel = state.selection;
      const rf = window.CNN.receptiveField(sel.layer, sel.row, sel.col);
      const cellPxIn = inputRect.width / 28;
      const srcX = inputRect.right - boardRect.left;
      const srcY = inputRect.top + (rf.top + rf.size / 2) * cellPxIn - boardRect.top;

      // Target: the selected panel's selected cell centre.
      const layerSpec = LAYERS.find(L => L.key === sel.layer);
      const targetPanel = layerHosts[sel.layer].panels.find(
        p => p.chIdx === sel.channel
      );
      if (!targetPanel) return;
      const tCanvas = targetPanel.canvas;
      const tRect = tCanvas.getBoundingClientRect();
      const cellPxOut = tRect.width / layerSpec.size;
      const tgtX = tRect.left - boardRect.left;
      const tgtY = tRect.top + (sel.row + 0.5) * cellPxOut - boardRect.top;

      // Bezier control points: pull horizontally to keep the cone shape.
      const dx = tgtX - srcX;
      const c1x = srcX + dx * 0.55;
      const c1y = srcY;
      const c2x = srcX + dx * 0.45;
      const c2y = tgtY;

      // Two paths: a translucent "cone" wedge from RF top/bottom and a
      // crisp centre line.
      const srcYTop = inputRect.top + rf.top * cellPxIn - boardRect.top;
      const srcYBot = inputRect.top + (rf.top + rf.size) * cellPxIn - boardRect.top;
      const tgtYTop = tRect.top + sel.row * cellPxOut - boardRect.top;
      const tgtYBot = tRect.top + (sel.row + 1) * cellPxOut - boardRect.top;

      const wedge = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      wedge.setAttribute('class', 's6-cone');
      wedge.setAttribute('d',
        'M ' + srcX + ' ' + srcYTop +
        ' C ' + c1x + ' ' + srcYTop + ', ' + c2x + ' ' + tgtYTop + ', ' + tgtX + ' ' + tgtYTop +
        ' L ' + tgtX + ' ' + tgtYBot +
        ' C ' + c2x + ' ' + tgtYBot + ', ' + c1x + ' ' + srcYBot + ', ' + srcX + ' ' + srcYBot +
        ' Z'
      );
      overlaySvg.appendChild(wedge);

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      line.setAttribute('class', 's6-conn');
      line.setAttribute('d',
        'M ' + srcX + ' ' + srcY +
        ' C ' + c1x + ' ' + c1y + ', ' + c2x + ' ' + c2y + ', ' + tgtX + ' ' + tgtY
      );
      overlaySvg.appendChild(line);
    }

    function render() {
      paintInput();
      paintLayerPanels();
      paintReadout();
      // Connector must run after layout (panels just got innerHTML'd).
      // Use requestAnimationFrame so DOM has settled.
      requestAnimationFrame(paintConnector);
    }

    function select(layer, channel, row, col) {
      state.selection = { layer, channel, row, col };
      render();
    }

    // ---- &run ---------------------------------------------------------
    // Pre-select a conv3 neuron at (3, 3); pick the top conv3 channel as
    // the most informative for the screenshot.
    function defaultSelect() {
      const layer = 'conv3';
      const channel = layerChannels.conv3[0];
      select(layer, channel, 3, 3);
    }

    // Initial paint.
    if (readHashFlag('run')) {
      defaultSelect();
    } else {
      // Show *something*: pre-select a conv1 neuron near the centre so
      // the cone idea is visible on entry.
      const channel = layerChannels.conv1[0];
      select('conv1', channel, 14, 14);
    }

    // Recompute the connector on resize (the line is positioned against
    // live DOM rects, not stored coords).
    let resizeRaf = null;
    function onResize() {
      if (resizeRaf) cancelAnimationFrame(resizeRaf);
      resizeRaf = requestAnimationFrame(() => {
        resizeRaf = null;
        paintConnector();
      });
    }
    window.addEventListener('resize', onResize);

    return {
      onEnter() { render(); },
      onLeave() {
        window.removeEventListener('resize', onResize);
      },
    };
  }

  /* Convert a hex or rgb color to rgba with given alpha. Tolerant of both. */
  function withAlpha(color, alpha) {
    if (color.startsWith('rgb(')) {
      return color.replace('rgb(', 'rgba(').replace(')', ',' + alpha + ')');
    }
    if (color.startsWith('#')) {
      let h = color.slice(1);
      if (h.length === 3) h = h.split('').map(c => c + c).join('');
      const r = parseInt(h.slice(0, 2), 16);
      const g = parseInt(h.slice(2, 4), 16);
      const b = parseInt(h.slice(4, 6), 16);
      return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }
    return color;
  }

  window.scenes = window.scenes || {};
  window.scenes.scene6 = function (root) { return buildScene(root); };
})();
