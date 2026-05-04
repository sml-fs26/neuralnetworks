/* Scene 5 -- "16x16 of pure semantics."

   The bottleneck. 16x16 spatial cells, 64 channels each. Each cell looks
   at a wide patch of the input but knows nothing precise about *where*
   inside that patch things are.

   Layout:
     hero
     sample selector strip
     grid: input (with optional RF overlay) | bottleneck-channel grid (16 of 64)
     "click a bottleneck cell" interaction: shows receptive field, plus a
     stack of up to three 64-d activation profile bar charts side by side.

   Step engine:
     0 = bottleneck channel grid only
     1 = receptive-field overlay enabled (click any cell)
     2 = three-region comparison panel (click up to three cells, see profiles) */
(function () {
  'use strict';

  const NUM_STEPS = 3;
  const RUN_INTERVAL_MS = 900;

  const INPUT_PX = 320;     // big input canvas so RF rectangles read clearly
  const CHANNEL_GRID_PX = 360;  // outer size of the 4x4 channel grid
  const PROFILE_W = 200;
  const PROFILE_H = 88;

  // Highlight palette for the up-to-three picked cells in step 2.
  const PICK_COLOR_VARS = ['--cnn-pos', '--cnn-accent', '--cnn-purple'];

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

  function readVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim() || '#888';
  }

  /* Decode the 64-channel bottleneck into a {channels: [c][i][j], ...}
     structure that the painter can consume. */
  function decodeFullBottleneck(sampleIdx) {
    const interObj = window.DATA.intermediates.samples[sampleIdx].enc3;
    const decoded = window.DATA._b64decode(interObj);
    const C = decoded.shape[0], H = decoded.shape[1], W = decoded.shape[2];
    const arr = new Array(C);
    for (let c = 0; c < C; c++) {
      const ch = new Array(H);
      for (let i = 0; i < H; i++) {
        const row = new Array(W);
        for (let j = 0; j < W; j++) {
          row[j] = decoded.data[c * H * W + i * W + j];
        }
        ch[i] = row;
      }
      arr[c] = ch;
    }
    return { channels: arr, C, H, W };
  }

  /* Pick the k channels with highest variance (most informative). */
  function pickTopVarChannels(channels, k) {
    const stats = channels.map(function (ch, idx) {
      let n = 0, sum = 0, sumSq = 0;
      for (let i = 0; i < ch.length; i++) {
        for (let j = 0; j < ch[0].length; j++) {
          const v = ch[i][j]; n++; sum += v; sumSq += v * v;
        }
      }
      const m = sum / n;
      return { idx: idx, variance: sumSq / n - m * m };
    });
    stats.sort(function (a, b) { return b.variance - a.variance; });
    return stats.slice(0, k).map(function (s) { return s.idx; });
  }

  function buildScene(root) {
    if (!window.DATA || !window.DATA.scene64 || !window.DATA.intermediates ||
        !window.DATA.rfields) {
      root.innerHTML = '<p style="opacity:0.5">scene5 data missing.</p>';
      return {};
    }
    const D = window.DATA.scene64;
    const RF = window.DATA.rfields;
    const Drawing = window.Drawing;

    root.innerHTML = '';
    root.classList.add('s5-root');
    const wrap = el('div', { class: 's5-wrap' }, root);

    /* ---- Hero ------------------------------------------------------- */
    const hero = el('header', { class: 'hero s5-hero' }, wrap);
    el('h1', { text: '16×16 of pure semantics.' }, hero);
    el('p', {
      class: 'subtitle',
      text: 'The bottleneck: a tiny spatial map with deep, wide-receptive-field channels.',
    }, hero);
    el('p', {
      class: 'lede',
      html:
        'Three pools later, the feature map is 16×16. Each cell now ' +
        'sees a roughly <em>30×30</em> patch of the input. It knows ' +
        '<em>what kind of stuff</em> is there. It does not know exactly ' +
        '<em>where</em>. Fixing that "where" is the decoder&#39;s job.',
    }, hero);

    /* ---- Sample selector ------------------------------------------- */
    const selectorStrip = el('div', { class: 's5-selector' }, wrap);
    el('div', { class: 's5-selector-label', text: 'sample' }, selectorStrip);
    const selectorRow = el('div', { class: 's5-selector-row' }, selectorStrip);
    const thumbHosts = [];
    for (let i = 0; i < D.samples.length; i++) {
      const btn = el('button', {
        type: 'button',
        class: 's5-thumb',
        'data-sample-index': String(i),
        'aria-label': 'Select sample ' + (i + 1),
      }, selectorRow);
      const tHost = el('div', { class: 'canvas-host s5-thumb-host' }, btn);
      thumbHosts.push(tHost);
      btn.addEventListener('click', function () {
        const idx = parseInt(btn.getAttribute('data-sample-index'), 10);
        switchSample(idx);
      });
    }

    /* ---- "You are here" mini-map ------------------------------------- */
    /* Show the full U-Net diagram with the bottleneck (enc3) highlighted,
       so the viewer sees where in the architecture this scene focuses. */
    const miniHost = el('div', { class: 's5-mini-host' }, wrap);
    if (window.UNET && typeof window.UNET.mountUNetMiniMap === 'function') {
      const mm = window.UNET.mountUNetMiniMap(miniHost, {
        width: 280, title: 'you are here',
        label: 'enc3 / bottleneck — 16 × 16 grid, 64 channels per cell',
      });
      mm.setHighlight(['enc3']);
    }

    /* ---- Main grid -------------------------------------------------- */
    const main = el('div', { class: 's5-main' }, wrap);

    // Left: input image + RF overlay layer.
    const inputCol = el('div', { class: 's5-input-col' }, main);
    el('div', { class: 's5-col-label', text: 'input · 64×64' }, inputCol);
    const inputBox = el('div', {
      class: 's5-input-box',
      style: 'width:' + INPUT_PX + 'px;height:' + INPUT_PX + 'px;',
    }, inputCol);
    const inputHost = el('div', { class: 'canvas-host s5-input-host' }, inputBox);
    // SVG overlay for the RF rectangles. Drawn over the canvas.
    const rfSvg = el('div', { class: 's5-rf-overlay' }, inputBox);
    rfSvg.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" preserveAspectRatio="none"></svg>';
    el('div', {
      class: 's5-input-note',
      text: 'each cell\'s receptive field ≈ 30×30 px',
    }, inputCol);

    // Right: 4x4 channel grid.
    const chanCol = el('div', { class: 's5-chan-col' }, main);
    el('div', {
      class: 's5-col-label',
      text: 'bottleneck · 16×16, 16 of 64 channels (top variance)',
    }, chanCol);
    const chanGrid = el('div', {
      class: 's5-chan-grid',
    }, chanCol);
    const channelTiles = [];
    const CHAN_TILE_PX = Math.floor((CHANNEL_GRID_PX - 12) / 4);
    for (let t = 0; t < 16; t++) {
      const tile = el('div', { class: 's5-chan-tile' }, chanGrid);
      const tHead = el('div', { class: 's5-chan-tile-head' }, tile);
      const tName = el('span', { class: 's5-chan-tile-name', text: 'ch ?' }, tHead);
      const tBody = el('div', {
        class: 'canvas-host s5-chan-tile-body',
        style: 'width:' + CHAN_TILE_PX + 'px;height:' + CHAN_TILE_PX + 'px;',
      }, tile);
      channelTiles.push({
        tile: tile, head: tHead, nameEl: tName,
        body: tBody, clickSvg: null,
      });
    }

    /* ---- "What is a 64-dim profile?" primer ----------------------- */
    const primer = el('div', { class: 's5-primer' }, wrap);
    const primerText = el('div', { class: 's5-primer-text' }, primer);
    el('div', {
      class: 's5-primer-eyebrow',
      text: 'what is a 64-dim profile?',
    }, primerText);
    el('p', {
      class: 's5-primer-body',
      html:
        'The bottleneck is a <strong>16×16 grid of cells</strong>. ' +
        'At every cell the network has computed <strong>64 numbers</strong> — ' +
        'one per channel, because this layer applies <strong>64 conv filters</strong> ' +
        '(scene 10 visualizes them). That is 16 × 16 × 64 = ' +
        '<strong>16,384</strong> numbers in this single feature map.',
    }, primerText);
    el('p', {
      class: 's5-primer-body',
      html:
        'A cell&#39;s <em>64-dim profile</em> is just that vector of 64 ' +
        'activations. Cells that look at similar stuff (two patches of sky) ' +
        'should have similar profiles. Cells in different regions ' +
        '(sky vs. tree) should have very different ones.',
    }, primerText);

    // Tiny visual: one cell highlighted in a 4x4 mini-grid → 64 mini-bars.
    const primerVisual = el('div', { class: 's5-primer-visual' }, primer);
    const primerVisualSvg =
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 96" ' +
      'preserveAspectRatio="xMidYMid meet" class="s5-primer-svg"></svg>';
    primerVisual.innerHTML = primerVisualSvg;
    (function buildPrimerVisual() {
      const svg = primerVisual.querySelector('svg');
      const NS = 'http://www.w3.org/2000/svg';

      // Mini-grid (left): 4x4 to hint at 16x16 without crowding.
      const gridX = 6, gridY = 14, gridSize = 64;
      const cellSize = gridSize / 4;
      const gridLabel = document.createElementNS(NS, 'text');
      gridLabel.setAttribute('x', String(gridX + gridSize / 2));
      gridLabel.setAttribute('y', String(gridY - 4));
      gridLabel.setAttribute('text-anchor', 'middle');
      gridLabel.setAttribute('class', 's5-primer-label');
      gridLabel.textContent = '16×16 grid';
      svg.appendChild(gridLabel);

      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          const r = document.createElementNS(NS, 'rect');
          r.setAttribute('x', String(gridX + j * cellSize));
          r.setAttribute('y', String(gridY + i * cellSize));
          r.setAttribute('width', String(cellSize));
          r.setAttribute('height', String(cellSize));
          r.setAttribute('class', 's5-primer-cell');
          if (i === 1 && j === 2) r.setAttribute('class', 's5-primer-cell s5-primer-cell-pick');
          svg.appendChild(r);
        }
      }
      const oneLabel = document.createElementNS(NS, 'text');
      oneLabel.setAttribute('x', String(gridX + gridSize / 2));
      oneLabel.setAttribute('y', String(gridY + gridSize + 12));
      oneLabel.setAttribute('text-anchor', 'middle');
      oneLabel.setAttribute('class', 's5-primer-label s5-primer-label-pick');
      oneLabel.textContent = 'this one cell';
      svg.appendChild(oneLabel);

      // Arrow.
      const arrow = document.createElementNS(NS, 'path');
      const arrowY = gridY + gridSize / 2;
      arrow.setAttribute('d',
        'M ' + (gridX + gridSize + 6) + ' ' + arrowY +
        ' L ' + (gridX + gridSize + 36) + ' ' + arrowY);
      arrow.setAttribute('class', 's5-primer-arrow');
      svg.appendChild(arrow);
      const arrowHead = document.createElementNS(NS, 'path');
      const ax = gridX + gridSize + 36;
      arrowHead.setAttribute('d',
        'M ' + ax + ' ' + arrowY +
        ' L ' + (ax - 5) + ' ' + (arrowY - 4) +
        ' L ' + (ax - 5) + ' ' + (arrowY + 4) + ' Z');
      arrowHead.setAttribute('class', 's5-primer-arrowhead');
      svg.appendChild(arrowHead);

      // Bars (right): 64 little vertical bars. Use a deterministic
      // pseudo-random shape so it reads as an "activation profile".
      const barsX0 = gridX + gridSize + 46;
      const barsY = gridY;
      const barsW = 320 - barsX0 - 8;
      const barsH = gridSize;
      const barsLabel = document.createElementNS(NS, 'text');
      barsLabel.setAttribute('x', String(barsX0 + barsW / 2));
      barsLabel.setAttribute('y', String(barsY - 4));
      barsLabel.setAttribute('text-anchor', 'middle');
      barsLabel.setAttribute('class', 's5-primer-label');
      barsLabel.textContent = '64 activations (one per channel)';
      svg.appendChild(barsLabel);

      // Baseline.
      const baseLine = document.createElementNS(NS, 'line');
      const midY = barsY + barsH / 2;
      baseLine.setAttribute('x1', String(barsX0));
      baseLine.setAttribute('y1', String(midY));
      baseLine.setAttribute('x2', String(barsX0 + barsW));
      baseLine.setAttribute('y2', String(midY));
      baseLine.setAttribute('class', 's5-primer-baseline');
      svg.appendChild(baseLine);

      const N = 64;
      const bw = barsW / N;
      // Deterministic "profile" shape — sums of a few sines + a phase.
      for (let c = 0; c < N; c++) {
        const t = c / N;
        const v = Math.sin(t * 11.3 + 0.7) * 0.55 +
                  Math.sin(t * 4.1 + 1.9) * 0.35 +
                  Math.cos(t * 23.0 + 0.3) * 0.25;
        const h = (barsH / 2 - 2) * Math.max(-0.95, Math.min(0.95, v));
        const r = document.createElementNS(NS, 'rect');
        r.setAttribute('x', String(barsX0 + c * bw + 0.4));
        r.setAttribute('y', String(h >= 0 ? midY - h : midY));
        r.setAttribute('width', String(Math.max(0.8, bw - 0.6)));
        r.setAttribute('height', String(Math.abs(h)));
        r.setAttribute('class', 's5-primer-bar');
        svg.appendChild(r);
      }

      const barsCaption = document.createElementNS(NS, 'text');
      barsCaption.setAttribute('x', String(barsX0 + barsW / 2));
      barsCaption.setAttribute('y', String(barsY + barsH + 12));
      barsCaption.setAttribute('text-anchor', 'middle');
      barsCaption.setAttribute('class', 's5-primer-label');
      barsCaption.textContent = "= the cell's 64-dim profile";
      svg.appendChild(barsCaption);
    })();

    /* ---- Profile panel (step 2) ----------------------------------- */
    const profilePanel = el('div', { class: 's5-profile-panel' }, wrap);
    el('div', {
      class: 's5-profile-head',
      text: 'click up to three bottleneck cells — compare their 64-dim profiles',
    }, profilePanel);
    el('div', {
      class: 's5-profile-sub',
      html:
        'Each chart below shows the <strong>64 activation values</strong> at the ' +
        'cell you clicked (one bar per channel, above/below the baseline = ' +
        'positive/negative). <em>Different shapes ⇒ the bottleneck encoded ' +
        'different information at that location.</em>',
    }, profilePanel);
    const profileRow = el('div', { class: 's5-profile-row' }, profilePanel);
    const profileSlots = [];
    for (let s = 0; s < 3; s++) {
      const slot = el('div', { class: 's5-profile-slot' }, profileRow);
      const slotHead = el('div', { class: 's5-profile-slot-head' }, slot);
      const slotSwatch = el('span', { class: 's5-profile-swatch' }, slotHead);
      const slotLabel = el('span', { class: 's5-profile-slot-label', text: '(empty)' }, slotHead);
      const slotHost = el('div', {
        class: 'canvas-host s5-profile-host',
        style: 'width:' + PROFILE_W + 'px;height:' + PROFILE_H + 'px;',
      }, slot);
      profileSlots.push({
        slot: slot, swatch: slotSwatch, label: slotLabel, host: slotHost,
      });
    }
    const clearPicksBtn = el('button', {
      type: 'button',
      class: 's5-clear-picks',
      text: 'clear picks',
    }, profilePanel);

    /* ---- Caption + controls --------------------------------------- */
    const caption = el('p', { class: 'caption s5-caption' }, wrap);
    const controls = el('div', { class: 'controls s5-controls' }, wrap);
    const stepGroup = el('div', { class: 'control-group' }, controls);
    el('label', { text: 'step', for: 's5-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's5-step', type: 'range', min: '0', max: String(NUM_STEPS - 1),
      step: '1', value: '0',
    }, stepGroup);
    const stepOut = el('output', { class: 'control-value', text: '0 / ' + (NUM_STEPS - 1) }, stepGroup);

    const navGroup = el('div', { class: 'control-group' }, controls);
    const prevBtn = el('button', { type: 'button', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 'primary', text: 'next' }, navGroup);
    const resetBtn = el('button', { type: 'button', text: 'reset' }, navGroup);

    /* ---- State ------------------------------------------------------ */
    const initialIdx = pickRichestSample(D.samples);
    const state = {
      step: 0,
      sampleIdx: initialIdx,
      hoverCell: null,    // {i, j} -- transient hover, just shows the rect
      stickyCell: null,   // {i, j} -- last clicked, persistent
      pickedCells: [],    // step 2: array of {i, j} (max 3)
      bottleneck: null,
      topChannels: null,
    };

    function loadBottleneck() {
      state.bottleneck = decodeFullBottleneck(state.sampleIdx);
      state.topChannels = pickTopVarChannels(state.bottleneck.channels, 16);
    }

    function profileAt(i, j) {
      const bn = state.bottleneck;
      const out = new Array(bn.C);
      for (let c = 0; c < bn.C; c++) out[c] = bn.channels[c][i][j];
      return out;
    }

    function gridRange() {
      let m = 0;
      for (let k = 0; k < state.topChannels.length; k++) {
        const ch = state.bottleneck.channels[state.topChannels[k]];
        for (let i = 0; i < ch.length; i++) {
          for (let j = 0; j < ch[0].length; j++) {
            const a = Math.abs(ch[i][j]);
            if (a > m) m = a;
          }
        }
      }
      if (!m) m = 1;
      return m;
    }

    /* Paint one channel into a tile body using the diverging colormap.
       The clearing of innerHTML drops the previous click overlay; we
       rebuild it after drawGrid. */
    function paintChannelTile(tile, channelData, valueMax) {
      tile.body.innerHTML = '';
      const setup = Drawing.setupCanvas(tile.body, CHAN_TILE_PX, CHAN_TILE_PX);
      const ctx = setup.ctx;
      Drawing.drawGrid(ctx, channelData, 0, 0, CHAN_TILE_PX, CHAN_TILE_PX, {
        diverging: true, valueRange: [-valueMax, valueMax],
      });
      const clickSvg = el('div', { class: 's5-chan-click' }, tile.body);
      clickSvg.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" preserveAspectRatio="none"></svg>';
      tile.clickSvg = clickSvg;
    }

    /* Wire click + hover handlers on a freshly created click-layer SVG. */
    function wireClickLayer(tile) {
      const svg = tile.clickSvg.querySelector('svg');
      function cellFromEvent(ev) {
        const rect = svg.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        let j = Math.floor((x / rect.width) * 16);
        let i = Math.floor((y / rect.height) * 16);
        if (i < 0 || j < 0 || i > 15 || j > 15) return null;
        return { i: i, j: j };
      }
      svg.addEventListener('mousemove', function (ev) {
        if (state.step < 1) return;
        const c = cellFromEvent(ev);
        if (!c) return;
        if (!state.hoverCell || state.hoverCell.i !== c.i || state.hoverCell.j !== c.j) {
          state.hoverCell = c;
          renderRfOverlay();
        }
      });
      svg.addEventListener('mouseleave', function () {
        if (state.hoverCell) { state.hoverCell = null; renderRfOverlay(); }
      });
      svg.addEventListener('click', function (ev) {
        if (state.step < 1) return;
        const c = cellFromEvent(ev);
        if (!c) return;
        if (state.step >= 2) {
          const existing = state.pickedCells.findIndex(function (p) {
            return p.i === c.i && p.j === c.j;
          });
          if (existing >= 0) {
            state.pickedCells.splice(existing, 1);
          } else {
            state.pickedCells.push(c);
            if (state.pickedCells.length > 3) state.pickedCells.shift();
          }
          state.stickyCell = c;
          renderProfiles();
          renderRfOverlay();
        } else {
          state.stickyCell = c;
          renderRfOverlay();
        }
      });
    }

    /* Update the receptive-field overlay on the input image. */
    function renderRfOverlay() {
      const svg = rfSvg.querySelector('svg');
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      if (state.step < 1) return;

      const NS = 'http://www.w3.org/2000/svg';

      function drawRect(cellIdx, color, opacity) {
        const cell = RF.cells[cellIdx];
        const r = document.createElementNS(NS, 'rect');
        r.setAttribute('x', String(cell.x0));
        r.setAttribute('y', String(cell.y0));
        r.setAttribute('width', String(cell.x1 - cell.x0));
        r.setAttribute('height', String(cell.y1 - cell.y0));
        r.setAttribute('fill', color);
        r.setAttribute('fill-opacity', String(opacity));
        r.setAttribute('stroke', color);
        r.setAttribute('stroke-width', '0.9');
        r.setAttribute('vector-effect', 'non-scaling-stroke');
        svg.appendChild(r);
      }

      // Picked cells (step 2): one rect per pick, colored by slot.
      if (state.step >= 2) {
        for (let p = 0; p < state.pickedCells.length; p++) {
          const c = state.pickedCells[p];
          const color = readVar(PICK_COLOR_VARS[p]);
          drawRect(c.i * 16 + c.j, color, 0.25);
        }
      }
      // Sticky / hover: solid amber rect.
      const c = state.hoverCell || state.stickyCell;
      if (c) {
        const accent = readVar('--cnn-accent');
        drawRect(c.i * 16 + c.j, accent, 0.30);
      }
    }

    /* Paint a 64-bar profile chart into a host div. */
    function paintProfile(host, profile, color) {
      host.innerHTML = '';
      const setup = Drawing.setupCanvas(host, PROFILE_W, PROFILE_H);
      const ctx = setup.ctx;
      const t = Drawing.tokens();
      ctx.fillStyle = t.bg;
      ctx.fillRect(0, 0, PROFILE_W, PROFILE_H);

      let m = 0;
      for (let c = 0; c < profile.length; c++) {
        const a = Math.abs(profile[c]);
        if (a > m) m = a;
      }
      if (!m) m = 1;

      const padX = 4, padY = 6;
      const innerW = PROFILE_W - 2 * padX;
      const innerH = PROFILE_H - 2 * padY;
      const midY = padY + innerH / 2;
      const barW = innerW / profile.length;

      ctx.strokeStyle = t.rule;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padX, midY);
      ctx.lineTo(padX + innerW, midY);
      ctx.stroke();

      ctx.fillStyle = color;
      for (let c = 0; c < profile.length; c++) {
        const v = profile[c] / m;
        const h = (innerH / 2) * Math.abs(v);
        const x = padX + c * barW;
        if (v >= 0) {
          ctx.fillRect(x + 0.5, midY - h, Math.max(1, barW - 1), h);
        } else {
          ctx.fillRect(x + 0.5, midY, Math.max(1, barW - 1), h);
        }
      }

      ctx.strokeStyle = t.rule;
      ctx.strokeRect(0.5, 0.5, PROFILE_W - 1, PROFILE_H - 1);
    }

    function renderProfiles() {
      for (let s = 0; s < profileSlots.length; s++) {
        const slot = profileSlots[s];
        const pick = state.pickedCells[s];
        if (!pick) {
          slot.host.innerHTML = '';
          slot.label.textContent = '(empty)';
          slot.swatch.style.background = 'transparent';
          slot.slot.classList.remove('s5-profile-filled');
          continue;
        }
        const color = readVar(PICK_COLOR_VARS[s]);
        slot.swatch.style.background = color;
        slot.label.textContent = 'cell (' + pick.i + ', ' + pick.j + ')';
        const prof = profileAt(pick.i, pick.j);
        paintProfile(slot.host, prof, color);
        slot.slot.classList.add('s5-profile-filled');
      }
    }

    function renderThumbs() {
      for (let i = 0; i < D.samples.length; i++) {
        Drawing.paintRGB(thumbHosts[i], D.samples[i].input, 56);
      }
      updateThumbActive();
    }
    function updateThumbActive() {
      const btns = selectorRow.querySelectorAll('.s5-thumb');
      btns.forEach(function (b, i) {
        b.classList.toggle('active', i === state.sampleIdx);
      });
    }

    function captionFor(step) {
      switch (step) {
        case 0: return 'Sixteen of the bottleneck\'s sixty-four channels at 16×16. Each tile is one channel.';
        case 1: return 'Click any cell. The amber rectangle is the patch of the input that cell can "see" — its receptive field.';
        case 2: return 'Pick three cells in semantically different regions. Each chart shows that cell\'s 64 activation values — its 64-dim profile. Cells in different regions have visibly different shapes; that is the bottleneck\'s vocabulary.';
        default: return '';
      }
    }

    function paintBottleneck() {
      const valueMax = gridRange();
      for (let t = 0; t < channelTiles.length; t++) {
        const tile = channelTiles[t];
        const chIdx = state.topChannels[t];
        tile.nameEl.textContent = 'ch ' + chIdx;
        const channelData = state.bottleneck.channels[chIdx];
        paintChannelTile(tile, channelData, valueMax);
        wireClickLayer(tile);
      }
    }

    function render() {
      const step = state.step;
      const sample = D.samples[state.sampleIdx];

      updateThumbActive();
      Drawing.paintRGB(inputHost, sample.input, INPUT_PX);

      paintBottleneck();
      renderRfOverlay();

      profilePanel.classList.toggle('s5-visible', step >= 2);
      chanGrid.classList.toggle('s5-clickable', step >= 1);

      if (step < 2) state.pickedCells = [];
      renderProfiles();

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
      state.hoverCell = null;
      state.stickyCell = null;
      state.pickedCells = [];
      loadBottleneck();
      render();
    }

    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () {
      state.hoverCell = null;
      state.stickyCell = null;
      state.pickedCells = [];
      applyStep(0);
    });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });
    clearPicksBtn.addEventListener('click', function () {
      state.pickedCells = [];
      renderProfiles();
      renderRfOverlay();
    });

    /* Initial paint */
    loadBottleneck();
    renderThumbs();
    render();

    /* &run -> auto-advance through the steps. */
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
  window.scenes.scene5 = function (root) { return buildScene(root); };
})();
