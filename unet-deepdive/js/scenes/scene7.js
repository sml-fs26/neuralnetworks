/* Scene 7 — Transposed convolution: stamp instead of slide.

   The conceptual centerpiece of the deepdive.

   Pedagogical flow:
     1. Hero + formula.
     2. Shape-derivation block: WHY is the output (N-1)*s + K?
        For N=3, K=3, s=1: 3 + 3 - 1 = 5.
     3. Cheat-sheet: current input/filter/stride/output shapes.
     4. Per-cell stamp animation. Walk through every input cell in
        row-major order. For each cell (i, j) with value v:
          - highlight that input cell;
          - show the scaled stamp = v × K;
          - place it into the output at rows [i*s .. i*s+K-1],
            cols [j*s .. j*s+K-1] (sum with what's already there).
     5. After all 9 stamps, hover any output cell to see which input
        cells contributed and the formula for that cell.
     6. Knobs: stride (1 or 2), filter preset (plus / edge / blur).
        Changing a knob restarts the animation from step 0.

   Step engine: 0 = nothing stamped; 1..N*N = after the k-th stamp.
   Total steps = N*N + 1 = 10 for the default 3×3 input. */
(function () {
  'use strict';

  // ---- DOM helper -------------------------------------------------------
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

  // ---- Constants --------------------------------------------------------
  const N_INPUT = 3;
  const K_FILT = 3;
  function outDim(s) { return (N_INPUT - 1) * s + K_FILT; }

  // Default input — chosen with several non-zero values so most stamps
  // make a visible contribution. The viewer also sees one zero cell
  // (a "this contributes nothing" beat).
  const DEFAULT_INPUT = [
    [1, 0, 2],
    [1, 3, 0],
    [0, 2, 1],
  ];

  const FILTER_PRESETS = {
    'plus':     [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    'edge':     [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
    'gaussian': [[1, 2, 1], [2, 2, 2], [1, 2, 1]],
  };
  const FILTER_LABELS = { plus: 'plus  ＋', edge: 'edge ↓', gaussian: 'blur' };

  const INPUT_PX  = 168;     // 56 px / cell at 3×3
  const FILT_PX   = 168;
  const STAMP_PX  = 168;
  // Output canvas px — keep cell size constant at 56 px so stride 2 (7×7) is wider
  const OUT_CELL  = 50;

  const RUN_STEP_MS = 900;

  // ---- Numerical utilities ---------------------------------------------
  function zeros2D(h, w) {
    const out = new Array(h);
    for (let i = 0; i < h; i++) {
      const row = new Array(w);
      for (let j = 0; j < w; j++) row[j] = 0;
      out[i] = row;
    }
    return out;
  }
  function cloneGrid(g) {
    const out = new Array(g.length);
    for (let i = 0; i < g.length; i++) out[i] = g[i].slice();
    return out;
  }
  function scaleGrid(g, s) {
    const out = new Array(g.length);
    for (let i = 0; i < g.length; i++) {
      out[i] = new Array(g[i].length);
      for (let j = 0; j < g[i].length; j++) out[i][j] = g[i][j] * s;
    }
    return out;
  }
  function maxAbs(g) {
    let m = 0;
    for (const row of g) for (const v of row) {
      const a = Math.abs(v);
      if (a > m) m = a;
    }
    return m || 1;
  }

  /* Build the per-stamp trace.
     Returns frames[0..N*N] where frames[k] is the output canvas after
     the first k input cells (row-major order) have stamped, plus
     metadata for each stamp. */
  function buildStampTrace(input, filter, stride) {
    const N = input.length;
    const K = filter.length;
    const O = (N - 1) * stride + K;
    const canvas = zeros2D(O, O);
    const frames = [cloneGrid(canvas)];
    const meta = [];
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const v = input[i][j];
        for (let u = 0; u < K; u++) {
          for (let w = 0; w < K; w++) {
            canvas[i * stride + u][j * stride + w] += v * filter[u][w];
          }
        }
        frames.push(cloneGrid(canvas));
        meta.push({
          k: meta.length + 1,            // 1-based step index
          inputCell: [i, j],
          inputValue: v,
          stamp: scaleGrid(filter, v),   // v × K
          targetR: [i * stride, i * stride + K - 1],
          targetC: [j * stride, j * stride + K - 1],
        });
      }
    }
    return { frames, meta };
  }

  // ---- Painter for a value grid (input / filter / stamp / output) ------
  function paintGrid(host, data, opts) {
    host.innerHTML = '';
    opts = opts || {};
    const rows = data.length, cols = data[0].length;
    const cell = opts.cell || Math.floor((opts.px || 168) / Math.max(rows, cols));
    const W = cell * cols;
    const H = cell * rows;
    const setup = window.Drawing.setupCanvas(host, W, H);
    const ctx = setup.ctx;
    const t = window.Drawing.tokens();

    ctx.fillStyle = t.bg;
    ctx.fillRect(0, 0, W, H);

    const m = (opts.vmax != null) ? opts.vmax : Math.max(1, maxAbs(data));

    // fills
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const v = data[i][j];
        ctx.fillStyle = window.Drawing.divergingColor(v / m, t);
        ctx.fillRect(j * cell, i * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }

    // dim cells
    if (opts.dimMask) {
      ctx.fillStyle = t.bg;
      ctx.globalAlpha = 0.55;
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          if (opts.dimMask[i][j]) ctx.fillRect(j * cell, i * cell, cell, cell);
        }
      }
      ctx.globalAlpha = 1;
    }

    // grid lines
    ctx.strokeStyle = t.rule;
    ctx.lineWidth = 1;
    for (let i = 0; i <= rows; i++) {
      ctx.beginPath();
      ctx.moveTo(0, i * cell);
      ctx.lineTo(W, i * cell);
      ctx.stroke();
    }
    for (let j = 0; j <= cols; j++) {
      ctx.beginPath();
      ctx.moveTo(j * cell, 0);
      ctx.lineTo(j * cell, H);
      ctx.stroke();
    }

    // labels
    if (opts.labels !== false) {
      ctx.font = `${Math.max(11, Math.floor(cell * 0.34))}px "SF Mono", Menlo, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          const v = data[i][j];
          if (opts.dimMask && opts.dimMask[i][j]) continue;
          const intensity = Math.min(1, Math.abs(v) / m);
          ctx.fillStyle = intensity > 0.55 ? t.bg : t.ink;
          const s = Math.abs(v) < 1e-9 ? '0'
                    : (Number.isInteger(v) ? String(v) : v.toFixed(1));
          ctx.fillText(s, (j + 0.5) * cell, (i + 0.5) * cell);
        }
      }
    }

    // pulsing highlight for the active input cell
    if (opts.highlight) {
      const h = opts.highlight;
      const accent = h.color || t.accent || '#d97a1f';
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = accent;
      ctx.strokeRect(
        h.col * cell + 1.75,
        h.row * cell + 1.75,
        (h.cols || 1) * cell - 3.5,
        (h.rows || 1) * cell - 3.5
      );
    }

    // dashed target box on the output (the 3×3 region the stamp lands in)
    if (opts.targetBox) {
      const b = opts.targetBox;
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = b.color || '#d97a1f';
      ctx.setLineDash([5, 4]);
      ctx.strokeRect(
        b.col * cell + 1.25,
        b.row * cell + 1.25,
        (b.cols || 1) * cell - 2.5,
        (b.rows || 1) * cell - 2.5
      );
      ctx.setLineDash([]);
    }

    return { W, H, cell, rows, cols };
  }

  // ---- Scene builder ---------------------------------------------------
  function buildScene(root) {
    if (!window.Drawing || !window.UNET) {
      root.innerHTML = '<p style="opacity:0.5">Scene 7: missing globals.</p>';
      return {};
    }

    root.innerHTML = '';
    root.classList.add('s7-root');
    const wrap = el('div', { class: 's7-wrap' }, root);

    // ---- Hero --------------------------------------------------------
    const hero = el('header', { class: 'hero s7-hero' }, wrap);
    el('h1', { text: 'Transposed convolution: stamp instead of slide.' }, hero);
    el('p', {
      class: 'subtitle',
      text:
        'A regular conv slides the filter and asks "how much of this pattern is here?". ' +
        'A transposed conv does the dual: each input value stamps the filter into the output. ' +
        'Where stamps overlap, they sum.',
    }, hero);

    // ---- Shape derivation block --------------------------------------
    const shape = el('section', { class: 's7-shape' }, wrap);
    el('div', { class: 's7-shape-eyebrow', text: 'why is the output 5 × 5?' }, shape);
    const shapeBody = el('div', { class: 's7-shape-body' }, shape);
    const shapeDiag = el('div', { class: 's7-shape-diag' }, shapeBody);
    const shapeProse = el('div', { class: 's7-shape-prose' }, shapeBody);

    // The diagram: row of 3 input cells, each contributing a 3-cell stamp,
    // and the final output extent of 5 cells. Drawn inline as SVG so it
    // recolors with the theme.
    function buildShapeDiagram() {
      shapeDiag.innerHTML = '';
      const cellPx = 36;
      const gap = 8;
      const inputW = 3 * cellPx;
      const totalH = 4 * cellPx + 3 * gap;
      const outputCells = 5;
      const outputW = outputCells * cellPx;
      const leftPad = 110;            // room for left-side text labels
      const rightPad = 16;
      const innerW = Math.max(inputW, outputW);
      const totalW = leftPad + innerW + rightPad;
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);
      svg.setAttribute('width', String(totalW));
      svg.setAttribute('height', String(totalH));

      const t = window.Drawing.tokens();
      const accent = '#d97a1f';

      // Top: 3 input cells, centered horizontally inside the inner area
      const inputY = 0;
      const inputX0 = leftPad + (innerW - inputW) / 2;
      for (let j = 0; j < 3; j++) {
        const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        r.setAttribute('x', String(inputX0 + j * cellPx));
        r.setAttribute('y', String(inputY));
        r.setAttribute('width', String(cellPx));
        r.setAttribute('height', String(cellPx));
        r.setAttribute('fill', t.bg);
        r.setAttribute('stroke', t.ink);
        r.setAttribute('stroke-width', '1.5');
        svg.appendChild(r);
        const lab = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        lab.setAttribute('x', String(inputX0 + j * cellPx + cellPx / 2));
        lab.setAttribute('y', String(inputY + cellPx / 2 + 4));
        lab.setAttribute('text-anchor', 'middle');
        lab.setAttribute('fill', t.ink);
        lab.setAttribute('font-family', 'SF Mono, Menlo, monospace');
        lab.setAttribute('font-size', '12');
        lab.textContent = `X${j}`;
        svg.appendChild(lab);
      }
      const inputCap = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      inputCap.setAttribute('x', String(inputX0 - 8));
      inputCap.setAttribute('y', String(inputY + cellPx / 2 + 4));
      inputCap.setAttribute('text-anchor', 'end');
      inputCap.setAttribute('fill', t.inkSecondary || t.ink);
      inputCap.setAttribute('font-size', '11');
      inputCap.textContent = 'input (3 cells)';
      svg.appendChild(inputCap);

      // Middle: 3 stamps, each a 3-wide bar at offsets 0, 1, 2
      const outputX0 = leftPad;
      const palette = [accent, '#3a8fb7', '#7a4eb7'];
      for (let k = 0; k < 3; k++) {
        const stampY = inputY + cellPx + gap + k * (cellPx + gap);
        for (let c = 0; c < 3; c++) {
          const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
          r.setAttribute('x', String(outputX0 + (k + c) * cellPx));
          r.setAttribute('y', String(stampY));
          r.setAttribute('width', String(cellPx));
          r.setAttribute('height', String(cellPx));
          r.setAttribute('fill', palette[k]);
          r.setAttribute('fill-opacity', '0.45');
          r.setAttribute('stroke', palette[k]);
          r.setAttribute('stroke-width', '1.2');
          svg.appendChild(r);
        }
        const cap = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        cap.setAttribute('x', String(outputX0 - 8));
        cap.setAttribute('y', String(stampY + cellPx / 2 + 4));
        cap.setAttribute('text-anchor', 'end');
        cap.setAttribute('fill', t.inkSecondary || t.ink);
        cap.setAttribute('font-size', '11');
        cap.textContent = `stamp from X${k}`;
        svg.appendChild(cap);
      }

      // Bottom: 5-cell output extent
      const outY = inputY + cellPx + gap + 3 * (cellPx + gap);
      // (drawn as the union of the three stamps above, but we leave the
      //  visual to do that work — we just draw a thin outline below)
      const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      r.setAttribute('x', String(outputX0));
      r.setAttribute('y', String(outY - 4));
      r.setAttribute('width', String(outputW));
      r.setAttribute('height', '0.1');
      r.setAttribute('stroke', accent);
      r.setAttribute('stroke-width', '1');
      svg.appendChild(r);
      const ocap = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      ocap.setAttribute('x', String(outputX0 + outputW / 2));
      ocap.setAttribute('y', String(outY + 6));
      ocap.setAttribute('text-anchor', 'middle');
      ocap.setAttribute('fill', accent);
      ocap.setAttribute('font-size', '11');
      ocap.setAttribute('font-weight', '600');
      ocap.textContent = 'output extent = 5 cells';
      svg.appendChild(ocap);

      shapeDiag.appendChild(svg);
    }

    // Prose half: explanation + KaTeX formula
    el('p', {
      class: 's7-shape-text',
      html:
        'Each of the <strong>3</strong> input cells stamps a <strong>3</strong>-wide ' +
        'pattern. The first stamp covers output positions <code>0–2</code>, the second ' +
        '<code>1–3</code>, the third <code>2–4</code>. Together they reach from ' +
        '<code>0</code> to <code>4</code> — that\'s <strong>5</strong> output cells.',
    }, shapeProse);
    const formulaHost = el('div', { class: 's7-shape-formula' }, shapeProse);
    function renderFormula() {
      formulaHost.innerHTML = '';
      window.Katex.render(
        'Y_{\\text{side}} \\;=\\; (X_{\\text{side}} - 1)\\cdot s \\;+\\; K',
        formulaHost, true
      );
    }

    // ---- Cheat-sheet -------------------------------------------------
    const cheat = el('section', { class: 's7-cheat' }, wrap);
    function cheatItem(label, valueSpan) {
      const it = el('div', { class: 's7-cheat-item' }, cheat);
      el('span', { class: 's7-cheat-k', text: label }, it);
      it.appendChild(valueSpan);
      return it;
    }
    const cInputV  = el('span', { class: 's7-cheat-v', text: '3 × 3' });
    const cFiltV   = el('span', { class: 's7-cheat-v', text: '3 × 3 (plus)' });
    const cStrideV = el('span', { class: 's7-cheat-v', text: '1' });
    const cOutV    = el('span', { class: 's7-cheat-v s7-cheat-v-em', text: '5 × 5' });
    cheatItem('input X', cInputV);
    cheatItem('filter K', cFiltV);
    cheatItem('stride s', cStrideV);
    cheatItem('output Y', cOutV);
    el('span', { class: 's7-cheat-formula', text: '(X−1)·s + K' }, cheat);

    // ---- Main animation panel ---------------------------------------
    const panel = el('section', { class: 's7-panel' }, wrap);
    const panelTitle = el('div', { class: 's7-panel-title' }, panel);
    el('span', { class: 's7-panel-eyebrow', text: 'how the output is built' }, panelTitle);
    el('span', { class: 's7-panel-progress', text: 'press → to begin' }, panelTitle);

    const grid = el('div', { class: 's7-panel-grid' }, panel);

    // -- Input column
    const inputCol = el('div', { class: 's7-col' }, grid);
    el('div', { class: 's7-col-cap', text: 'input X' }, inputCol);
    const inputHost = el('div', { class: 'canvas-host s7-canvas-host' }, inputCol);
    el('div', { class: 's7-col-sub', text: 'one cell lights up at a time' }, inputCol);

    el('div', { class: 's7-op', text: '×' }, grid);

    // -- Filter column
    const filtCol = el('div', { class: 's7-col' }, grid);
    el('div', { class: 's7-col-cap', text: 'filter K' }, filtCol);
    const filterHost = el('div', { class: 'canvas-host s7-canvas-host' }, filtCol);
    el('div', { class: 's7-col-sub', text: 'fixed for the whole animation' }, filtCol);

    el('div', { class: 's7-op', text: '=' }, grid);

    // -- Stamp column
    const stampCol = el('div', { class: 's7-col' }, grid);
    const stampCap = el('div', { class: 's7-col-cap', text: 'stamp = X × K' }, stampCol);
    const stampHost = el('div', { class: 'canvas-host s7-canvas-host' }, stampCol);
    const stampSub = el('div', { class: 's7-col-sub', text: 'before the first stamp' }, stampCol);

    el('div', { class: 's7-op s7-op-arrow', text: '→' }, grid);

    // -- Output column (separate row beneath, so it can be wider)
    const outRow = el('div', { class: 's7-out-row' }, panel);
    const outCol = el('div', { class: 's7-col s7-col-out' }, outRow);
    const outCap = el('div', { class: 's7-col-cap', text: 'output Y (running sum)' }, outCol);
    const outHost = el('div', { class: 'canvas-host s7-canvas-host s7-out-host' }, outCol);
    el('div', {
      class: 's7-col-sub',
      text: 'each stamp lands at (i·s, j·s) and adds to whatever is already there',
    }, outCol);

    // -- Narration card
    const narration = el('section', { class: 's7-narration' }, wrap);
    const narrTitle = el('div', { class: 's7-narration-title', text: 'press → to start the first stamp' }, narration);
    const narrBody  = el('div', { class: 's7-narration-body' }, narration);

    // -- Hover chip (only after all stamps done)
    const hoverChip = el('div', { class: 's7-hover-chip' }, wrap);
    hoverChip.style.display = 'none';

    // ---- Step controls ----------------------------------------------
    const controls = el('section', { class: 's7-controls' }, wrap);
    const stepGroup = el('div', { class: 's7-control-group' }, controls);
    el('label', { class: 's7-control-label', text: 'step', for: 's7-step' }, stepGroup);
    const stepInput = el('input', {
      id: 's7-step',
      type: 'range',
      min: '0',
      max: '9',  // updated when stride/filter change
      value: '0',
      step: '1',
    }, stepGroup);
    const stepOut = el('output', { class: 's7-control-value', text: '0 / 9' }, stepGroup);

    const navGroup = el('div', { class: 's7-control-group' }, controls);
    const prevBtn = el('button', { type: 'button', class: 's7-btn', text: 'prev' }, navGroup);
    const nextBtn = el('button', { type: 'button', class: 's7-btn s7-btn-primary', text: 'next stamp' }, navGroup);
    const resetBtn = el('button', { type: 'button', class: 's7-btn', text: 'reset' }, navGroup);
    const playBtn  = el('button', { type: 'button', class: 's7-btn', text: 'play full ▶' }, navGroup);

    // ---- Knobs -------------------------------------------------------
    const knobs = el('section', { class: 's7-knobs' }, wrap);

    // Stride
    const strideGroup = el('div', { class: 's7-knob' }, knobs);
    el('span', { class: 's7-knob-label', text: 'stride' }, strideGroup);
    const stride1Btn = el('button', {
      type: 'button', class: 's7-knob-btn s7-active', 'data-stride': '1', text: '1',
    }, strideGroup);
    const stride2Btn = el('button', {
      type: 'button', class: 's7-knob-btn', 'data-stride': '2', text: '2',
    }, strideGroup);

    // Filter pattern
    const filtGroup = el('div', { class: 's7-knob' }, knobs);
    el('span', { class: 's7-knob-label', text: 'filter' }, filtGroup);
    const filtBtns = {};
    Object.keys(FILTER_PRESETS).forEach(function (k) {
      const b = el('button', {
        type: 'button',
        class: 's7-knob-btn' + (k === 'plus' ? ' s7-active' : ''),
        'data-filt': k,
        text: FILTER_LABELS[k],
      }, filtGroup);
      filtBtns[k] = b;
    });

    // ---- Caption -----------------------------------------------------
    const caption = el('p', { class: 'caption s7-caption' }, wrap);

    // ---- State -------------------------------------------------------
    const state = {
      step: 0,
      stride: 1,
      filterKey: 'plus',
      input: DEFAULT_INPUT.map(function (r) { return r.slice(); }),
      hoverCell: null,
    };

    function currentFilter() { return FILTER_PRESETS[state.filterKey]; }
    function maxStep() { return N_INPUT * N_INPUT; }

    let trace = null;
    function rebuildTrace() {
      trace = buildStampTrace(state.input, currentFilter(), state.stride);
    }

    function captionFor(step) {
      if (step === 0) {
        return 'The output is empty. We will fill it in 9 stamps — one per input cell.';
      }
      if (step >= maxStep()) {
        return 'All 9 stamps are placed. Hover any output cell to see which input cells contributed and the formula for that cell.';
      }
      return 'Stamp ' + step + ' of 9. The active input cell (orange ring) multiplies the filter, then the result is added to the output inside the dashed box.';
    }

    function renderShapeDerivation() {
      buildShapeDiagram();
      renderFormula();
    }

    function renderCheat() {
      cInputV.textContent = N_INPUT + ' × ' + N_INPUT;
      cFiltV.textContent = K_FILT + ' × ' + K_FILT + ' (' + FILTER_LABELS[state.filterKey].split(' ')[0] + ')';
      cStrideV.textContent = String(state.stride);
      const od = outDim(state.stride);
      cOutV.textContent = od + ' × ' + od;
    }

    function renderInputAndFilter() {
      // Active cell from current step
      let highlight = null;
      if (state.step >= 1 && state.step <= maxStep()) {
        const m = trace.meta[state.step - 1];
        highlight = { row: m.inputCell[0], col: m.inputCell[1] };
      }
      paintGrid(inputHost, state.input, {
        px: INPUT_PX, vmax: 4,
        highlight: highlight,
      });
      paintGrid(filterHost, currentFilter(), {
        px: FILT_PX, vmax: 2,
      });
    }

    function renderStamp() {
      // Stamp panel: shows v × K for the current step. If step = 0 or step
      // is past the last stamp, show a faint placeholder.
      stampHost.innerHTML = '';
      if (state.step >= 1 && state.step <= maxStep()) {
        const m = trace.meta[state.step - 1];
        if (m.inputValue === 0) {
          // Show a "0" panel — explicit
          stampCap.textContent = 'stamp = 0 × K = 0';
          stampSub.textContent = 'this input cell is 0, so this stamp adds nothing';
          paintGrid(stampHost, zeros2D(K_FILT, K_FILT), { px: STAMP_PX, vmax: 1 });
        } else {
          stampCap.textContent = 'stamp = ' + m.inputValue + ' × K';
          stampSub.textContent = 'each cell of K is multiplied by ' + m.inputValue;
          paintGrid(stampHost, m.stamp, { px: STAMP_PX, vmax: Math.max(2, Math.abs(m.inputValue) * 2) });
        }
      } else {
        stampCap.textContent = 'stamp = X × K';
        stampSub.textContent = state.step === 0 ? 'before the first stamp' : 'all stamps placed';
        paintGrid(stampHost, zeros2D(K_FILT, K_FILT), { px: STAMP_PX, vmax: 1, dimMask: zeros2D(K_FILT, K_FILT).map(function(r){ return r.map(function(){return 1;}); }) });
      }
    }

    function renderOutput() {
      const od = outDim(state.stride);
      const cell = OUT_CELL;
      const data = trace.frames[Math.min(state.step, trace.frames.length - 1)];
      const opts = { cell: cell, vmax: Math.max(2, maxAbs(trace.frames[trace.frames.length - 1])) };
      if (state.step >= 1 && state.step <= maxStep()) {
        const m = trace.meta[state.step - 1];
        opts.targetBox = {
          row: m.targetR[0], col: m.targetC[0],
          rows: K_FILT, cols: K_FILT,
        };
      }
      paintGrid(outHost, data, opts);
      // Sizing the host — just match canvas dims
      outHost.style.width = (cell * od) + 'px';
      outHost.style.height = (cell * od) + 'px';
    }

    function renderNarration() {
      if (state.step === 0) {
        narrTitle.textContent = 'press → to start the first stamp';
        narrBody.innerHTML = '';
        return;
      }
      if (state.step > maxStep()) {
        narrTitle.textContent = 'all 9 stamps placed';
        narrBody.innerHTML = '';
        return;
      }
      const m = trace.meta[state.step - 1];
      const [i, j] = m.inputCell;
      const v = m.inputValue;
      const tr = m.targetR, tc = m.targetC;
      narrTitle.textContent = 'Stamp ' + m.k + ' of 9 — input cell (' + i + ', ' + j + ')';
      narrBody.innerHTML = '';
      const lines = [];
      lines.push('<span class="s7-n-k">value</span>'
        + '<span class="s7-n-v">X[' + i + '][' + j + '] = ' + v + '</span>');
      if (v === 0) {
        lines.push('<span class="s7-n-k">stamp</span>'
          + '<span class="s7-n-v">0 × K = all zeros — nothing added</span>');
        lines.push('<span class="s7-n-k">target</span>'
          + '<span class="s7-n-v">rows ' + tr[0] + '–' + tr[1]
          + ', cols ' + tc[0] + '–' + tc[1]
          + ' (skipped because the stamp is zero)</span>');
      } else {
        lines.push('<span class="s7-n-k">stamp</span>'
          + '<span class="s7-n-v">' + v + ' × K (orange panel above)</span>');
        lines.push('<span class="s7-n-k">target</span>'
          + '<span class="s7-n-v">rows ' + tr[0] + '–' + tr[1]
          + ', cols ' + tc[0] + '–' + tc[1]
          + ' &nbsp;←&nbsp; dashed box on output</span>');
        lines.push('<span class="s7-n-k">action</span>'
          + '<span class="s7-n-v">add the stamp values into those output cells (sum with what was already there)</span>');
      }
      lines.forEach(function (h) {
        const row = el('div', { class: 's7-n-row' }, narrBody);
        row.innerHTML = h;
      });
    }

    function renderHoverChip() {
      hoverChip.innerHTML = '';
      if (state.step < maxStep() || !state.hoverCell) {
        hoverChip.style.display = 'none';
        return;
      }
      const [r, c] = state.hoverCell;
      const od = outDim(state.stride);
      if (r < 0 || r >= od || c < 0 || c >= od) {
        hoverChip.style.display = 'none';
        return;
      }
      // Find all input cells (i, j) that contributed: r - i*s in [0, K-1]
      // and c - j*s in [0, K-1].
      const contribs = [];
      let total = 0;
      for (let i = 0; i < N_INPUT; i++) {
        for (let j = 0; j < N_INPUT; j++) {
          const u = r - i * state.stride;
          const w = c - j * state.stride;
          if (u >= 0 && u < K_FILT && w >= 0 && w < K_FILT) {
            const v = state.input[i][j] * currentFilter()[u][w];
            if (state.input[i][j] !== 0 && currentFilter()[u][w] !== 0) {
              contribs.push({ i: i, j: j, x: state.input[i][j], k: currentFilter()[u][w], v: v });
            }
            total += v;
          }
        }
      }
      const head = el('div', { class: 's7-hover-head' }, hoverChip);
      head.textContent = 'Y[' + r + '][' + c + '] = ' + total;
      const list = el('div', { class: 's7-hover-list' }, hoverChip);
      if (!contribs.length) {
        el('div', { class: 's7-hover-row', text: '(no input cells contributed; all-zero stamps)' }, list);
      } else {
        contribs.forEach(function (c) {
          const row = el('div', { class: 's7-hover-row' }, list);
          row.textContent = 'X[' + c.i + '][' + c.j + ']·K = ' + c.x + '·' + c.k + ' = ' + c.v;
        });
      }
      hoverChip.style.display = '';
    }

    function render() {
      // Cap step in case of stride change
      if (state.step > maxStep()) state.step = maxStep();

      renderCheat();
      renderInputAndFilter();
      renderStamp();
      renderOutput();
      renderNarration();
      renderHoverChip();

      // Progress label
      const labelTxt = state.step === 0
        ? 'press → to begin (0 of ' + maxStep() + ')'
        : (state.step >= maxStep()
            ? 'all ' + maxStep() + ' stamps placed — hover output cells'
            : 'stamp ' + state.step + ' of ' + maxStep());
      panelTitle.querySelector('.s7-panel-progress').textContent = labelTxt;

      // Step input
      stepInput.max = String(maxStep());
      stepInput.value = String(state.step);
      stepOut.textContent = state.step + ' / ' + maxStep();

      prevBtn.disabled = state.step <= 0;
      nextBtn.disabled = state.step >= maxStep();
      resetBtn.disabled = state.step === 0;
      playBtn.disabled = state.step >= maxStep();
      nextBtn.textContent = state.step === 0 ? 'first stamp →' : (state.step + 1 === maxStep() ? 'last stamp →' : 'next stamp →');

      caption.textContent = captionFor(state.step);
    }

    function applyStep(n) {
      state.step = Math.max(0, Math.min(maxStep(), n));
      render();
    }

    // ---- Wire controls ----------------------------------------------
    prevBtn.addEventListener('click', function () { applyStep(state.step - 1); });
    nextBtn.addEventListener('click', function () { applyStep(state.step + 1); });
    resetBtn.addEventListener('click', function () { applyStep(0); });
    stepInput.addEventListener('input', function () {
      const v = parseInt(stepInput.value, 10);
      if (Number.isFinite(v)) applyStep(v);
    });

    // Play full
    let runTimer = null;
    function clearRun() { if (runTimer) { clearTimeout(runTimer); runTimer = null; } }
    function autoStep() {
      if (state.step >= maxStep()) { clearRun(); return; }
      applyStep(state.step + 1);
      runTimer = setTimeout(autoStep, RUN_STEP_MS);
    }
    playBtn.addEventListener('click', function () {
      clearRun();
      applyStep(0);
      runTimer = setTimeout(autoStep, 500);
    });

    // Knob handlers
    function setStride(s) {
      state.stride = s;
      stride1Btn.classList.toggle('s7-active', s === 1);
      stride2Btn.classList.toggle('s7-active', s === 2);
      rebuildTrace();
      applyStep(0);
    }
    stride1Btn.addEventListener('click', function () { setStride(1); });
    stride2Btn.addEventListener('click', function () { setStride(2); });

    function setFilter(k) {
      state.filterKey = k;
      Object.keys(filtBtns).forEach(function (kk) {
        filtBtns[kk].classList.toggle('s7-active', kk === k);
      });
      rebuildTrace();
      applyStep(0);
    }
    Object.keys(filtBtns).forEach(function (k) {
      filtBtns[k].addEventListener('click', function () { setFilter(k); });
    });

    // Output hover (only meaningful at step = maxStep())
    outHost.addEventListener('mousemove', function (ev) {
      if (state.step < maxStep()) return;
      const rect = outHost.getBoundingClientRect();
      const x = ev.clientX - rect.left;
      const y = ev.clientY - rect.top;
      const od = outDim(state.stride);
      const cell = rect.width / od;
      const c = Math.floor(x / cell);
      const r = Math.floor(y / cell);
      const newH = (r >= 0 && r < od && c >= 0 && c < od) ? [r, c] : null;
      const old = state.hoverCell;
      if (!newH && !old) return;
      if (newH && old && newH[0] === old[0] && newH[1] === old[1]) return;
      state.hoverCell = newH;
      renderHoverChip();
    });
    outHost.addEventListener('mouseleave', function () {
      state.hoverCell = null;
      renderHoverChip();
    });

    // ---- Initial paint ----------------------------------------------
    rebuildTrace();
    renderShapeDerivation();
    render();

    if (readHashFlag('run')) {
      runTimer = setTimeout(function () {
        applyStep(0);
        runTimer = setTimeout(autoStep, 500);
      }, 200);
    }

    return {
      onEnter: function () {
        renderShapeDerivation();
        render();
      },
      onLeave: function () { clearRun(); },
      onNextKey: function () {
        if (state.step < maxStep()) { applyStep(state.step + 1); return true; }
        return false;
      },
      onPrevKey: function () {
        if (state.step > 0) { applyStep(state.step - 1); return true; }
        return false;
      },
    };
  }

  window.scenes = window.scenes || {};
  window.scenes.scene7 = function (root) { return buildScene(root); };
})();
